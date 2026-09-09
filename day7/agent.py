import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

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

# Сколько реплик агент отдаёт модели: 20 сообщений — это 10 ходов. База хранит
# весь диалог, лимитом ограничено только окно контекста.
HISTORY_LIMIT = 20


@dataclass(frozen=True)
class AgentDelta:
    content: str = ""
    finish_reason: str | None = None


def as_param(message: dict[str, str]) -> ChatCompletionMessageParam:
    if message["role"] == "user":
        return ChatCompletionUserMessageParam(role="user", content=message["content"])
    return ChatCompletionAssistantMessageParam(role="assistant", content=message["content"])


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
        history_limit: int = HISTORY_LIMIT,
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_limit = history_limit
        self._storage = storage
        self._context: list[ChatCompletionMessageParam] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def context_size(self) -> int:
        """Сколько сообщений уйдёт в модель следующим ходом."""
        async with self._lock:
            await self._load()
            return len(self._context)

    async def history_size(self) -> int:
        return await self._storage.count(self.session_id)

    async def transcript(self) -> list[dict[str, str]]:
        return await self._storage.load_all(self.session_id)

    async def ask(self, prompt: str) -> AsyncIterator[AgentDelta]:
        async with self._lock:
            await self._load()

            user = ChatCompletionUserMessageParam(role="user", content=prompt)
            messages = [
                ChatCompletionSystemMessageParam(role="system", content=self.system_prompt),
                *self._context,
                user,
            ]

            # Имя диалогу даётся один раз, по первому вопросу. Отдельный запрос
            # уходит в модель одновременно с ответом и к концу стрима уже готов —
            # ждать заголовок пользователю не приходится.
            title = asyncio.create_task(self._title(prompt)) if not self._context else None

            try:
                parts: list[str] = []
                async for delta in stream_chat(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    if delta.content:
                        parts.append(delta.content)
                    yield AgentDelta(content=delta.content, finish_reason=delta.finish_reason)

                # Ход попадает в память только целиком: оборванный стрим (ошибка или
                # остановка пользователем) не оставляет ни вопроса, ни полуответа.
                answer = "".join(parts)
                if answer:
                    await self._storage.save_turn(self.session_id, prompt, answer)
                    self._context.append(user)
                    self._context.append(
                        ChatCompletionAssistantMessageParam(role="assistant", content=answer),
                    )
                    self._trim()
                    await self._save_title(title)
            finally:
                # Ход не сохранился — сессия осталась пустой, и называть её нечем.
                if title is not None:
                    title.cancel()

    async def _title(self, prompt: str) -> str:
        try:
            answer = await complete(
                [
                    ChatCompletionSystemMessageParam(role="system", content=TITLE_PROMPT),
                    ChatCompletionUserMessageParam(role="user", content=prompt),
                ],
                temperature=TITLE_TEMPERATURE,
                max_tokens=TITLE_MAX_TOKENS,
            )
        except Exception:
            # Заголовок — украшение панели, а не часть разговора: не вышло — в списке
            # останется начало первого вопроса, ответ пользователь получит в любом случае.
            return ""
        return clean_title(answer)

    async def _save_title(self, title: asyncio.Task[str] | None) -> None:
        if title is None:
            return
        name = await title
        if name:
            await self._storage.set_title(self.session_id, name)

    async def _load(self) -> None:
        # Агент создаётся пустым, в том числе после перезапуска процесса: окно
        # контекста поднимается из базы при первом же ходе.
        if self._loaded:
            return
        tail = await self._storage.load_tail(self.session_id, self.history_limit)
        self._context = [as_param(message) for message in tail]
        self._trim()
        self._loaded = True

    def _trim(self) -> None:
        del self._context[: max(0, len(self._context) - self.history_limit)]
        # Диалог всегда начинается с вопроса пользователя, иначе первым в контексте
        # окажется ответ на реплику, которой модель уже не видит.
        if self._context and self._context[0]["role"] == "assistant":
            del self._context[0]
