import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import memory
import tokens
from llm import complete, stream_chat
from memory import Memory, Window, counted, summarized
from storage import Storage

SYSTEM_PROMPT = """\
Ты — вежливый и краткий ассистент.
Ты помнишь предыдущие реплики диалога и опираешься на них: если вопрос ссылается
на сказанное раньше, отвечай по контексту и не переспрашивай очевидное.
Начало давнего разговора может прийти не репликами, а кратким содержанием —
это твоя собственная память о нём, и факты из неё такие же надёжные.
Отвечай на языке пользователя.
"""

TITLE_PROMPT = """\
Ты называешь диалоги. По первому сообщению пользователя сформулируй суть запроса
как заголовок из 2–5 слов на языке этого сообщения.
Не отвечай на сам запрос, не добавляй кавычки, точку и пояснения.
"""

# Заголовок — не ответ, а ярлык: температура низкая, чтобы одна и та же тема не
# называлась каждый раз по-новому, а лимит токенов отсекает попытку разговориться.
TITLE_TEMPERATURE = 0.2
TITLE_MAX_TOKENS = 32

# Панель узкая, длинный заголовок в ней всё равно обрежется многоточием.
TITLE_LIMIT = 60

DEFAULT_PROMPT = "Объясни в трёх предложениях, что такое идемпотентность в HTTP."

DEFAULT_TEMPERATURE = 0.7

# Потолок одного ответа: диалог остаётся диалогом, а не полотном на весь экран.
MAX_TOKENS = 2000

# Бюджет входа в токенах: сколько агент готов отдать под system prompt, окно
# истории и новый вопрос. Со сжатием он работает страховкой — основной объём
# убирает свёртка, а бюджет ловит тот случай, когда и хвост не влез.
CONTEXT_BUDGET = 4000

# Значения для переключателя в шапке: на 500 токенах обрезка наступает через
# пару ходов, и потерю памяти видно вживую.
BUDGET_OPTIONS = (500, 2000, 4000, 16000)

MIN_BUDGET = 100

# Сжатие включено по умолчанию: выключатель в шапке оставлен, чтобы тот же диалог
# можно было провести на полной истории и сравнить ответы и счёт.
COMPRESSION = True


class ContextOverflow(RuntimeError):
    """Запрос не влезает в контекст модели: до API он не доходит."""

    def __init__(self, request_tokens: int, max_tokens: int, limit: int) -> None:
        self.request_tokens = request_tokens
        self.max_tokens = max_tokens
        self.limit = limit
        super().__init__(
            f"Переполнение контекста: {request_tokens} токенов запроса плюс {max_tokens} "
            f"на ответ против лимита модели {limit}. Сократите сообщение.",
        )


@dataclass(frozen=True)
class AgentPlan:
    """Что уйдёт в модель — посчитано локально, до запроса."""

    question_tokens: int
    context_messages: int
    context_tokens: int
    request_tokens: int
    history_messages: int
    history_tokens: int
    dropped_messages: int
    budget: int
    over_budget: bool
    # Память в сжатом виде: конспект вместо выпавших из запроса реплик.
    compression: bool
    summary_tokens: int
    summary_messages: int
    pending_messages: int


@dataclass(frozen=True)
class AgentDelta:
    content: str = ""
    finish_reason: str | None = None


@dataclass(frozen=True)
class AgentTurn:
    """Что получилось — факт из usage вместе с оценкой, которую он проверяет."""

    usage: tokens.Usage
    estimated_tokens: int
    drift_percent: float | None
    context_messages: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class AgentCompaction:
    """Свёртка: блок реплик заменён конспектом, и это тоже стоило токенов."""

    messages: int
    summary_messages: int
    summary_tokens: int
    # Сколько токенов дословной истории теперь не уходит в каждый следующий запрос.
    replaced_tokens: int
    usage: tokens.Usage


AgentEvent = AgentPlan | AgentDelta | AgentTurn | AgentCompaction


def _title_messages(prompt: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=TITLE_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]


def clean_title(text: str) -> str:
    """Модель просили обойтись без кавычек и точки, но просьба — не гарантия."""
    return " ".join(text.split()).strip("\"'«»`.")[:TITLE_LIMIT]


class Agent:
    """Собеседник, привязанный к сессии в базе: сам объект — временный, память — нет."""

    def __init__(
        self,
        session_id: str,
        storage: Storage,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        context_budget: int = CONTEXT_BUDGET,
        compression: bool = COMPRESSION,
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_budget = context_budget
        self.compression = compression
        self._storage = storage
        self._memory = Memory()
        self._loaded = False
        self._lock = asyncio.Lock()

    async def plan(
        self,
        prompt: str = "",
        budget: int | None = None,
        compression: bool | None = None,
    ) -> AgentPlan:
        """Числа до запроса: пустой prompt — просто состояние памяти для шапки."""
        async with self._lock:
            await self._load()
            _, plan = self._plan(prompt, budget or self.context_budget, self._compression(compression))
            return plan

    async def turns(self) -> list[dict[str, Any]]:
        return await self._storage.load_turns(self.session_id)

    async def transcript(self) -> list[dict[str, Any]]:
        return await self._storage.load_all(self.session_id)

    async def summary(self) -> dict[str, Any] | None:
        """Действующий конспект: то, что агент помнит вместо свёрнутых реплик."""
        async with self._lock:
            await self._load()
            if self._memory.summary is None:
                return None
            return {
                "text": self._memory.summary.text,
                "messages": self._memory.summary.messages,
                "tokens": self._memory.summary.tokens,
            }

    async def summaries(self) -> list[dict[str, Any]]:
        return await self._storage.load_summaries(self.session_id)

    async def ask(
        self,
        prompt: str,
        budget: int | None = None,
        compression: bool | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            await self._load()

            budget = budget or self.context_budget
            compressing = self._compression(compression)
            window, plan = self._plan(prompt, budget, compressing)
            # Числа запроса уходят на страницу до первого токена ответа: сколько
            # стоит вопрос, видно ещё до того, как модель начнёт отвечать.
            yield plan

            # Бюджет — политика агента, лимит модели — физика: превысить его нельзя,
            # и мёртвый запрос незачем отправлять в API.
            if plan.request_tokens + self.max_tokens > tokens.MODEL_CONTEXT_LIMIT:
                raise ContextOverflow(plan.request_tokens, self.max_tokens, tokens.MODEL_CONTEXT_LIMIT)

            question = counted("user", prompt)
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(role="system", content=self.system_prompt),
                *window.params(),
                question.as_param(),
            ]

            # Имя диалогу даётся один раз, по первому вопросу. Отдельный запрос
            # уходит в модель одновременно с ответом и к концу стрима уже готов —
            # ждать заголовок пользователю не приходится.
            title = asyncio.create_task(self._title(prompt)) if not self._memory.size else None

            try:
                parts: list[str] = []
                finish_reason: str | None = None
                raw_usage: Any = None

                async for delta in stream_chat(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    if delta.usage is not None:
                        raw_usage = delta.usage
                    if delta.content:
                        parts.append(delta.content)
                        yield AgentDelta(content=delta.content)
                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
                        yield AgentDelta(finish_reason=delta.finish_reason)

                # Ход попадает в память только целиком: оборванный стрим (ошибка или
                # остановка пользователем) не оставляет ни вопроса, ни полуответа.
                answer = "".join(parts)
                if not answer:
                    return

                usage = tokens.usage_from(raw_usage)
                question_id, answer_id = await self._storage.save_turn(
                    self.session_id,
                    prompt,
                    answer,
                    {
                        "kind": "turn",
                        "prompt_tokens": usage.prompt_tokens,
                        "cached_tokens": usage.cached_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "estimated_tokens": plan.request_tokens,
                        "context_messages": plan.context_messages,
                        "cost_usd": usage.cost_usd,
                    },
                )
                self._memory.add(
                    counted("user", prompt, question_id),
                    counted("assistant", answer, answer_id),
                )
                await self._save_title(title, prompt)

                yield AgentTurn(
                    usage=usage,
                    estimated_tokens=plan.request_tokens,
                    drift_percent=tokens.drift_percent(plan.request_tokens, usage.prompt_tokens),
                    context_messages=plan.context_messages,
                    finish_reason=finish_reason,
                )

                # Свёртка идёт после ответа, а не перед ним: пользователь уже
                # прочитал реплику, пока агент пересказывает себе старое. Ждать
                # приходится только итоговых чисел хода.
                if compressing and self._memory.crowded():
                    compaction = await self._compact()
                    if compaction is not None:
                        yield compaction
            finally:
                # Ход не сохранился — сессия осталась пустой, и называть её нечем.
                if title is not None:
                    title.cancel()

    def _compression(self, compression: bool | None) -> bool:
        return self.compression if compression is None else compression

    def _plan(self, prompt: str, budget: int, compression: bool) -> tuple[Window, AgentPlan]:
        question = counted("user", prompt)
        system = ChatCompletionSystemMessageParam(role="system", content=self.system_prompt)
        # Неснимаемая часть запроса: system prompt, новый вопрос и разметка чата.
        base = tokens.count_messages([system, question.as_param()])

        window = self._memory.select(budget, base, compression=compression)
        used = base + window.cost
        summary = window.summary

        return window, AgentPlan(
            question_tokens=question.tokens,
            context_messages=len(window.messages),
            context_tokens=window.cost,
            request_tokens=used,
            history_messages=self._memory.size,
            history_tokens=self._memory.cost,
            # Сообщения, которых в запросе нет ни дословно, ни в виде конспекта:
            # со сжатием такие появляются только когда хвост не влез в бюджет.
            dropped_messages=self._memory.size - len(window.messages) - (summary.messages if summary else 0),
            budget=budget,
            # Один вопрос может не влезть в бюджет целиком: обрезать историю
            # дальше некуда, и агент отправляет запрос, помечая перерасход.
            over_budget=used > budget,
            compression=compression,
            summary_tokens=summary.tokens if summary else 0,
            summary_messages=summary.messages if summary else 0,
            # Сколько реплик ждёт следующей свёртки: по ним видно, когда она будет.
            pending_messages=len(self._memory.block()) if compression else 0,
        )

    async def _compact(self) -> AgentCompaction | None:
        """Блок реплик за хвостом уходит в модель и возвращается конспектом."""
        block = self._memory.block()
        previous = self._memory.summary
        request = memory.summary_request(previous, block)

        try:
            answer = await complete(
                request,
                temperature=memory.SUMMARY_TEMPERATURE,
                max_tokens=memory.SUMMARY_MAX_TOKENS,
            )
        except Exception:
            # Свёртка не удалась — история остаётся дословной: блок никуда не
            # делся и уйдёт в модель на следующем ходу вместе с новыми репликами.
            return None

        text = memory.tidy(answer.text)
        if not text:
            return None

        summary = summarized(
            text,
            covered_id=block[-1].id,
            messages=(previous.messages if previous else 0) + len(block),
        )
        usage = tokens.usage_from(answer.usage)
        await self._storage.save_summary(
            self.session_id,
            summary.text,
            summary.covered_id,
            summary.messages,
            summary.tokens,
            {
                "kind": "summary",
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "completion_tokens": usage.completion_tokens,
                "estimated_tokens": tokens.count_messages(request),
                "context_messages": len(block),
                "cost_usd": usage.cost_usd,
            },
        )
        self._memory.summary = summary

        return AgentCompaction(
            messages=len(block),
            summary_messages=summary.messages,
            summary_tokens=summary.tokens,
            replaced_tokens=sum(message.cost for message in block) + (previous.cost if previous else 0),
            usage=usage,
        )

    async def _title(self, prompt: str) -> tuple[str, Any]:
        try:
            answer = await complete(
                _title_messages(prompt),
                temperature=TITLE_TEMPERATURE,
                max_tokens=TITLE_MAX_TOKENS,
            )
        except Exception:
            # Заголовок — украшение панели, а не часть разговора: не вышло — в списке
            # останется начало первого вопроса, ответ пользователь получит в любом случае.
            return "", None
        return clean_title(answer.text), answer.usage

    async def _save_title(self, title: asyncio.Task[tuple[str, Any]] | None, prompt: str) -> None:
        if title is None:
            return

        name, raw_usage = await title
        if name:
            await self._storage.set_title(self.session_id, name)
        if raw_usage is None:
            return

        # Служебный запрос платный, как и любой другой: в панели он отдельной
        # строкой, чтобы стоимость диалога сходилась с выставленным счётом.
        usage = tokens.usage_from(raw_usage)
        estimated = tokens.count_messages(_title_messages(prompt))
        await self._storage.save_service_turn(
            self.session_id,
            {
                "kind": "title",
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "completion_tokens": usage.completion_tokens,
                "estimated_tokens": estimated,
                "context_messages": 0,
                "cost_usd": usage.cost_usd,
            },
        )

    async def _load(self) -> None:
        # Агент создаётся пустым, в том числе после перезапуска процесса: история
        # и конспект поднимаются из базы при первом же ходе, а токены каждой реплики
        # считаются один раз — окно потом набирается из готовых чисел.
        if self._loaded:
            return

        history = await self._storage.load_all(self.session_id)
        self._memory.add(
            *(counted(message["role"], message["content"], message["id"]) for message in history),
        )

        stored = await self._storage.load_summary(self.session_id)
        if stored is not None:
            self._memory.summary = summarized(
                stored["text"],
                covered_id=stored["covered_id"],
                messages=stored["messages"],
            )
        self._loaded = True
