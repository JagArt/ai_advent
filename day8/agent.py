import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import tokens
from llm import complete, stream_chat
from storage import Storage

SYSTEM_PROMPT = """\
Ты — вежливый и краткий ассистент.
Ты помнишь предыдущие реплики диалога и опираешься на них: если вопрос ссылается
на сказанное раньше, отвечай по контексту и не переспрашивай очевидное.
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
# истории и новый вопрос. В day7 окно мерилось сообщениями, но 20 коротких реплик
# и 20 длинных — это разные деньги, а платят именно за токены.
CONTEXT_BUDGET = 4000

# Значения для переключателя в шапке: на 500 токенах обрезка наступает через
# пару ходов, и потерю памяти видно вживую.
BUDGET_OPTIONS = (500, 2000, 4000, 16000)

MIN_BUDGET = 100


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


AgentEvent = AgentPlan | AgentDelta | AgentTurn


@dataclass(frozen=True)
class Counted:
    """Реплика вместе со своей ценой в токенах: считаем её один раз."""

    role: str
    content: str
    tokens: int

    @property
    def cost(self) -> int:
        return tokens.MESSAGE_OVERHEAD + self.tokens

    def as_param(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return ChatCompletionUserMessageParam(role="user", content=self.content)
        return ChatCompletionAssistantMessageParam(role="assistant", content=self.content)


def counted(role: str, content: str) -> Counted:
    return Counted(role=role, content=content, tokens=tokens.count_text(content))


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
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_budget = context_budget
        self._storage = storage
        self._history: list[Counted] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def plan(self, prompt: str = "", budget: int | None = None) -> AgentPlan:
        """Числа до запроса: пустой prompt — просто состояние памяти для шапки."""
        async with self._lock:
            await self._load()
            _, plan = self._plan(prompt, budget or self.context_budget)
            return plan

    async def turns(self) -> list[dict[str, Any]]:
        return await self._storage.load_turns(self.session_id)

    async def transcript(self) -> list[dict[str, str]]:
        return await self._storage.load_all(self.session_id)

    async def ask(
        self,
        prompt: str,
        budget: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            await self._load()

            budget = budget or self.context_budget
            window, plan = self._plan(prompt, budget)
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
                *(message.as_param() for message in window),
                question.as_param(),
            ]

            # Имя диалогу даётся один раз, по первому вопросу. Отдельный запрос
            # уходит в модель одновременно с ответом и к концу стрима уже готов —
            # ждать заголовок пользователю не приходится.
            title = asyncio.create_task(self._title(prompt)) if not self._history else None

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
                await self._storage.save_turn(
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
                self._history.append(question)
                self._history.append(counted("assistant", answer))
                await self._save_title(title, prompt)

                yield AgentTurn(
                    usage=usage,
                    estimated_tokens=plan.request_tokens,
                    drift_percent=tokens.drift_percent(plan.request_tokens, usage.prompt_tokens),
                    context_messages=plan.context_messages,
                    finish_reason=finish_reason,
                )
            finally:
                # Ход не сохранился — сессия осталась пустой, и называть её нечем.
                if title is not None:
                    title.cancel()

    def _plan(self, prompt: str, budget: int) -> tuple[list[Counted], AgentPlan]:
        question = counted("user", prompt)
        system = ChatCompletionSystemMessageParam(role="system", content=self.system_prompt)
        # Неснимаемая часть запроса: system prompt, новый вопрос и разметка чата.
        base = tokens.count_messages([system, question.as_param()])

        window: list[Counted] = []
        used = base
        # Окно набирается с конца: свежие реплики важнее, старые уходят первыми.
        for message in reversed(self._history):
            if used + message.cost > budget:
                break
            used += message.cost
            window.append(message)
        window.reverse()

        # Диалог всегда начинается с вопроса пользователя, иначе первым в контексте
        # окажется ответ на реплику, которой модель уже не видит.
        if window and window[0].role == "assistant":
            used -= window[0].cost
            del window[0]

        return window, AgentPlan(
            question_tokens=question.tokens,
            context_messages=len(window),
            context_tokens=used - base,
            request_tokens=used,
            history_messages=len(self._history),
            history_tokens=sum(message.cost for message in self._history),
            dropped_messages=len(self._history) - len(window),
            budget=budget,
            # Один вопрос может не влезть в бюджет целиком: обрезать историю
            # дальше некуда, и агент отправляет запрос, помечая перерасход.
            over_budget=used > budget,
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
        # поднимается из базы при первом же ходе, а токены каждой реплики считаются
        # один раз — окно потом набирается из готовых чисел.
        if self._loaded:
            return
        history = await self._storage.load_all(self.session_id)
        self._history = [counted(message["role"], message["content"]) for message in history]
        self._loaded = True
