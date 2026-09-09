import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from llm import stream_chat

SYSTEM_PROMPT = """\
Ты — вежливый и краткий ассистент.
Ты помнишь предыдущие реплики диалога и опираешься на них: если вопрос ссылается
на сказанное раньше, отвечай по контексту и не переспрашивай очевидное.
Отвечай на языке пользователя.
"""

DEFAULT_PROMPT = "Объясни в трёх предложениях, что такое идемпотентность в HTTP."

DEFAULT_TEMPERATURE = 0.7

# Потолок одного ответа: диалог остаётся диалогом, а не полотном на весь экран.
MAX_TOKENS = 2000

# Сколько реплик диалога агент держит в контексте: 20 сообщений — это 10 ходов.
HISTORY_LIMIT = 20


@dataclass(frozen=True)
class AgentDelta:
    content: str = ""
    finish_reason: str | None = None


class Agent:
    """Собеседник с собственной памятью: снаружи виден только вопрос и поток ответа."""

    def __init__(
        self,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        history_limit: int = HISTORY_LIMIT,
    ) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_limit = history_limit
        self._history: list[ChatCompletionMessageParam] = []
        self._lock = asyncio.Lock()

    @property
    def history_size(self) -> int:
        return len(self._history)

    def transcript(self) -> list[dict[str, str]]:
        return [{"role": message["role"], "content": message["content"]} for message in self._history]

    def reset(self) -> None:
        self._history.clear()

    async def ask(self, prompt: str) -> AsyncIterator[AgentDelta]:
        async with self._lock:
            user = ChatCompletionUserMessageParam(role="user", content=prompt)
            messages = [
                ChatCompletionSystemMessageParam(role="system", content=self.system_prompt),
                *self._history,
                user,
            ]

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
                self._history.append(user)
                self._history.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=answer),
                )
                self._trim()

    def _trim(self) -> None:
        del self._history[: max(0, len(self._history) - self.history_limit)]
        # Диалог всегда начинается с вопроса пользователя, иначе первым в контексте
        # окажется ответ на реплику, которой модель уже не видит.
        if self._history and self._history[0]["role"] == "assistant":
            del self._history[0]
