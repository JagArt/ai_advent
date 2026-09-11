import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import facts as facts_policy
import tokens
from llm import complete, stream_chat
from memory import (
    DEFAULT_STRATEGY,
    FACTS,
    WINDOW_MESSAGES,
    Memory,
    Window,
    counted,
)
from storage import Storage

SYSTEM_PROMPT = """\
Ты — вежливый и краткий ассистент.
Ты помнишь предыдущие реплики диалога и опираешься на них: если вопрос ссылается
на сказанное раньше, отвечай по контексту и не переспрашивай очевидное.
Начало давнего разговора может прийти не репликами, а списком фактов —
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
# истории и новый вопрос. Основной объём убирает сама стратегия, а бюджет ловит
# тот случай, когда и окно из нескольких реплик оказалось слишком длинным.
CONTEXT_BUDGET = 4000

# Значения для переключателя в шапке: на 500 токенах обрезка наступает через
# пару ходов, и потерю памяти видно вживую.
BUDGET_OPTIONS = (500, 2000, 4000, 16000)

MIN_BUDGET = 100

# Имя новой ветки по умолчанию: корневая называется «основная», остальные —
# по номеру, чтобы вкладки в интерфейсе отличались без участия пользователя.
BRANCH_NAME = "ветка {number}"


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
    # Стратегия и её настройка: от них зависит, что попало в окно.
    strategy: str
    window_messages: int
    facts_count: int
    facts_tokens: int


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
    # id реплик хода: по ним страница ставит checkpoint на только что сказанном.
    question_id: int
    answer_id: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class AgentFactsUpdate:
    """Картотека после хода: разбор реплики — отдельный запрос со своим счётом."""

    added: int
    changed: int
    removed: int
    items: tuple[tuple[str, str], ...]
    tokens: int
    usage: tokens.Usage


AgentEvent = AgentPlan | AgentDelta | AgentTurn | AgentFactsUpdate


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
        strategy: str = DEFAULT_STRATEGY,
        window_messages: int = WINDOW_MESSAGES,
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_budget = context_budget
        self.strategy = strategy
        self.window_messages = window_messages
        self._storage = storage
        self._memory = Memory()
        self._branch_id = 0
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def branch_id(self) -> int:
        return self._branch_id

    async def plan(
        self,
        prompt: str = "",
        budget: int | None = None,
        strategy: str | None = None,
        window: int | None = None,
    ) -> AgentPlan:
        """Числа до запроса: пустой prompt — просто состояние памяти для шапки."""
        async with self._lock:
            await self._load()
            _, plan = self._plan(
                prompt,
                budget or self.context_budget,
                strategy or self.strategy,
                window or self.window_messages,
            )
            return plan

    async def turns(self) -> list[dict[str, Any]]:
        return await self._storage.load_turns(self.session_id)

    async def transcript(self) -> list[dict[str, Any]]:
        """Лента текущей ветки: своё и унаследованное до checkpoint."""
        async with self._lock:
            await self._load()
            return await self._storage.load_messages(self.session_id, self._branch_id)

    async def facts(self) -> list[dict[str, str]]:
        """Картотека ветки: то, что агент помнит помимо окна сообщений."""
        async with self._lock:
            await self._load()
            if self._memory.facts is None:
                return []
            return [{"key": key, "value": value} for key, value in self._memory.facts.items]

    async def branches(self) -> list[dict[str, Any]]:
        async with self._lock:
            await self._load()
            branches = await self._storage.list_branches(self.session_id)
            return [{**branch, "active": branch["id"] == self._branch_id} for branch in branches]

    async def fork(self, message_id: int, name: str = "") -> int:
        """Checkpoint становится веткой: разговор продолжается с этого места заново."""
        async with self._lock:
            await self._load()
            branches = await self._storage.list_branches(self.session_id)
            branch_id = await self._storage.create_branch(
                self.session_id,
                message_id,
                name.strip() or BRANCH_NAME.format(number=len(branches) + 1),
            )
            await self._switch(branch_id)
            return branch_id

    async def switch(self, branch_id: int) -> None:
        async with self._lock:
            await self._load()
            if branch_id == self._branch_id:
                return
            await self._switch(branch_id)

    async def ask(
        self,
        prompt: str,
        budget: int | None = None,
        strategy: str | None = None,
        window: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            await self._load()

            budget = budget or self.context_budget
            strategy = strategy or self.strategy
            window = window or self.window_messages
            frame, plan = self._plan(prompt, budget, strategy, window)
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
                *frame.params(),
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
                    self._branch_id,
                    prompt,
                    answer,
                    {
                        "branch_id": self._branch_id,
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
                    question_id=question_id,
                    answer_id=answer_id,
                    finish_reason=finish_reason,
                )

                # Картотека обновляется после ответа, а не перед ним: пользователь
                # уже читает реплику, пока агент разбирает её на факты. Ход всё
                # равно лежит в окне дословно, так что до следующего вопроса факты
                # из него ничего не решают.
                if strategy == FACTS:
                    update = await self._update_facts(prompt, answer)
                    if update is not None:
                        yield update
            finally:
                # Ход не сохранился — сессия осталась пустой, и называть её нечем.
                if title is not None:
                    title.cancel()

    def _plan(self, prompt: str, budget: int, strategy: str, window: int) -> tuple[Window, AgentPlan]:
        question = counted("user", prompt)
        system = ChatCompletionSystemMessageParam(role="system", content=self.system_prompt)
        # Неснимаемая часть запроса: system prompt, новый вопрос и разметка чата.
        base = tokens.count_messages([system, question.as_param()])

        frame = self._memory.select(budget, base, strategy=strategy, window=window)
        used = base + frame.cost

        return frame, AgentPlan(
            question_tokens=question.tokens,
            context_messages=len(frame.messages),
            context_tokens=frame.cost,
            request_tokens=used,
            history_messages=self._memory.size,
            history_tokens=self._memory.cost,
            # Реплики ветки, которых в запросе нет: у окна это всё, что старше N,
            # у фактов — то же самое, но пересказанное картотекой.
            dropped_messages=self._memory.size - len(frame.messages),
            budget=budget,
            # Один вопрос может не влезть в бюджет целиком: обрезать историю
            # дальше некуда, и агент отправляет запрос, помечая перерасход.
            over_budget=used > budget,
            strategy=strategy,
            window_messages=window,
            facts_count=frame.facts.count if frame.facts else 0,
            facts_tokens=frame.facts.tokens if frame.facts else 0,
        )

    async def _update_facts(self, prompt: str, answer: str) -> AgentFactsUpdate | None:
        """Ход уходит в модель и возвращается изменениями к картотеке."""
        current = self._memory.facts
        request = facts_policy.update_request(current, prompt, answer)

        try:
            reply = await complete(
                request,
                temperature=facts_policy.FACTS_TEMPERATURE,
                max_tokens=facts_policy.FACTS_MAX_TOKENS,
            )
        except Exception:
            # Запрос не дошёл — картотека остаётся прежней: следующий ход разберут
            # вместе с этим, ничего не потеряется.
            return None

        usage = tokens.usage_from(reply.usage)
        try:
            update = facts_policy.parse(reply.text)
        except ValueError:
            # Ответ не по форме, но запрос уже оплачен: счёт пишем, картотеку — нет.
            await self._storage.save_service_turn(
                self.session_id,
                self._metrics("facts", usage, tokens.count_messages(request)),
            )
            return None

        items, added, changed, removed = facts_policy.apply(
            current.items if current else (),
            update,
        )
        self._memory.facts = facts_policy.block(items)
        await self._storage.save_facts(
            self.session_id,
            self._branch_id,
            items,
            self._metrics("facts", usage, tokens.count_messages(request)),
        )

        return AgentFactsUpdate(
            added=added,
            changed=changed,
            removed=removed,
            items=items,
            tokens=self._memory.facts.tokens if self._memory.facts else 0,
            usage=usage,
        )

    def _metrics(self, kind: str, usage: tokens.Usage, estimated: int) -> dict[str, Any]:
        """Строка счёта за служебный запрос: платный, как и любой другой."""
        return {
            "branch_id": self._branch_id,
            "kind": kind,
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": usage.cached_tokens,
            "completion_tokens": usage.completion_tokens,
            "estimated_tokens": estimated,
            "context_messages": 0,
            "cost_usd": usage.cost_usd,
        }

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
        await self._storage.save_service_turn(
            self.session_id,
            self._metrics(
                "title",
                tokens.usage_from(raw_usage),
                tokens.count_messages(_title_messages(prompt)),
            ),
        )

    async def _switch(self, branch_id: int) -> None:
        """Переключение ветки: память пересобирается, потому что история другая."""
        await self._storage.set_branch(self.session_id, branch_id)
        self._loaded = False
        await self._load()

    async def _load(self) -> None:
        # Агент создаётся пустым, в том числе после перезапуска процесса: история
        # активной ветки и её картотека поднимаются из базы при первом же ходе, а
        # токены каждой реплики считаются один раз — окно потом набирается из
        # готовых чисел.
        if self._loaded:
            return

        self._branch_id = await self._storage.active_branch(self.session_id)
        history = await self._storage.load_messages(self.session_id, self._branch_id)
        self._memory = Memory()
        self._memory.add(
            *(counted(message["role"], message["content"], message["id"]) for message in history),
        )
        self._memory.facts = facts_policy.block(tuple(await self._storage.load_facts(self._branch_id)))
        self._loaded = True
