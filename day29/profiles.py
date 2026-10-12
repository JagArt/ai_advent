"""Два профиля одной модели: как в day28 и настроенный под задачу «ответ по выдержкам».

Профиль — всё, что отличает оптимизированную модель от базовой, в одном месте:
параметры генерации, потолок ответа, бюджет контекста, промпт. `llm.py` только
исполняет профиль, страница и прогон получают его описание отсюда же, и чекбокс
«оптимизация» на странице выбирает между `BASE` и `OPTIMIZED` — ничего другого
он не делает.

`BASE` — ровно day28: температура 0.2, 600 токенов ответа, пять выдержек целиком,
промпт day28, рассуждение по галочке. Это точка отсчёта, с ней сравниваются
таблицы в README.

`OPTIMIZED` собран под то, что в day28 было слабым у `qwen3-4b` на этой задаче:

- Пять с половиной секунд до первого токена — чтение промпта в 2000 токенов.
  Бюджет контекста режет выдержки до ~1200 токенов, см. `search.Context.fit`.
- Имя не процитировано дословно (#4), числа и имена потеряны в пересказе (#3).
  Правило о цитировании переписано с «приводи как в выдержках» на «приводи
  сами имена и числа, а не описывай их словами», и перед правилами стоит
  короткий пример формы ответа с выдуманными значениями.
- Английское слово посреди русской фразы (#8) промптом не лечится: запрет на
  латиницу в системной строке и отдельным правилом проверен на трёх
  формулировках — «whereas» и «эмbedding» остались на месте во всех. Это
  свойство самой модели в этом кванте, а не промпта, поэтому в профиле запрета
  нет: лишние токены без эффекта.
- 600 токенов ответа нужны были только рассуждениям, медианный ответ — 56.
  Потолок 220: ответ в три предложения с номерами помещается с запасом, а
  сорвавшаяся в болтовню генерация обрывается в четыре раза раньше.
- Рассуждение выключено всегда: в day27 и day28 оно стоило в 2.5 раза по времени
  и фактов не прибавляло, кроме одного вопроса.
- Температура 0.1 и `top_p 0.8`, `top_k 20` — рекомендованные Qwen значения
  отсечения для режима без рассуждений при низкой температуре под извлечение
  фактов, а не под сочинение.

Порядок частей промпта — вопрос, выдержки, правила, вопрос ещё раз — у обоих
профилей один: в day28 именно он заставил 4B отвечать по делу вместо «[1] В
выдержках этого нет.» на каждую выдержку. Оптимизация его не трогает.
"""

from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from search import Context

REFUSAL = "В выдержках этого нет."
NO_THINK = "/no_think"

BASE_SYSTEM = (
    "Ты отвечаешь на вопросы о проекте AI Advent — репозитории с дневником итераций, "
    "где есть документация, код на Python и PDF.\n"
    "Отвечай по-русски, коротко и по делу: два-четыре предложения, без вступлений "
    "и без пересказа вопроса.\n"
    "Если чего-то не знаешь — скажи об этом прямо, не придумывай."
)

BASE_RULES = (
    "Правила ответа:\n"
    "1. Отвечай только по выдержкам, не добавляй ничего от себя.\n"
    "2. После каждого утверждения ставь номер выдержки в квадратных скобках, например [2].\n"
    "3. Числа, имена файлов и названия приводи ровно так, как в выдержках.\n"
    f"4. Только если ответа в выдержках нет — напиши одной строкой «{REFUSAL}»."
)

OPTIMIZED_SYSTEM = (
    "Ты — справочник по проекту AI Advent: репозиторий-дневник с документацией, кодом на Python и PDF. "
    "Отвечаешь только по выдержкам из базы проекта, по-русски, "
    "двумя-тремя предложениями, без вступлений и без повторения вопроса."
)

OPTIMIZED_RULES = (
    "Правила:\n"
    "1. Используй только выдержки выше; свои знания не добавляй.\n"
    "2. Числа, порты, имена файлов, моделей и инструментов приводи сами, дословно, как они написаны "
    "в выдержке, — не описывай их словами и не переводи.\n"
    "3. После каждого утверждения — номер выдержки в квадратных скобках: [2].\n"
    f"4. Только если ни одна выдержка не отвечает на вопрос — одной строкой: «{REFUSAL}»\n\n"
    "Пример формы ответа (имена и числа выдуманы): «Сервер слушает порт 9100 и запускается "
    "командой `python tools/serve.py` [2]. Индекс хранится в `store.db` [4].»"
)


@dataclass(frozen=True)
class Profile:
    """Всё, что отличает один режим модели от другого; `llm.py` только исполняет."""

    name: str
    title: str
    temperature: float
    max_tokens: int
    top_p: float | None
    top_k: int | None
    context_budget: int | None
    think_allowed: bool
    system: str
    rules: str

    def context(self, found: Context) -> Context:
        """Выдержки для этого профиля: целиком или уложенные в бюджет."""
        return found if self.context_budget is None else found.fit(self.context_budget)

    def messages(self, question: str, block: str, *, think: bool) -> list[ChatCompletionMessageParam]:
        user = (
            f"Вопрос: {question}\n\n"
            f"Выдержки из базы проекта, каждая под своим номером:\n\n{block}\n\n"
            f"{self.rules}\n\n"
            f"Ответь на вопрос: {question}"
        )
        if not self.thinks(think):
            user = f"{user} {NO_THINK}"
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def thinks(self, think: bool) -> bool:
        return think and self.think_allowed

    def sampling(self) -> dict[str, object]:
        """Именованные аргументы для `chat.completions.create`."""
        extra = {key: value for key, value in (("top_k", self.top_k),) if value is not None}
        kwargs: dict[str, object] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    @property
    def summary(self) -> str:
        parts = [f"temperature {self.temperature}", f"max_tokens {self.max_tokens}"]
        if self.top_p is not None:
            parts.append(f"top_p {self.top_p}")
        if self.top_k is not None:
            parts.append(f"top_k {self.top_k}")
        parts.append("контекст целиком" if self.context_budget is None else f"контекст ≤ {self.context_budget} ток.")
        parts.append("рассуждение по галочке" if self.think_allowed else f"всегда {NO_THINK}")
        parts.append("промпт day28" if self.name == "base" else "промпт под выдержки")
        return " · ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "context_budget": self.context_budget,
            "think_allowed": self.think_allowed,
            "summary": self.summary,
        }


BASE = Profile(
    name="base",
    title="базовая · как в day28",
    temperature=0.2,
    max_tokens=600,
    top_p=None,
    top_k=None,
    context_budget=None,
    think_allowed=True,
    system=BASE_SYSTEM,
    rules=BASE_RULES,
)

OPTIMIZED = Profile(
    name="optimized",
    title="оптимизированная · под ответ по выдержкам",
    temperature=0.1,
    max_tokens=220,
    top_p=0.8,
    top_k=20,
    context_budget=1200,
    think_allowed=False,
    system=OPTIMIZED_SYSTEM,
    rules=OPTIMIZED_RULES,
)

PROFILES = {profile.name: profile for profile in (BASE, OPTIMIZED)}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"Профиля {name!r} нет. Есть: {', '.join(PROFILES)}.")
    return PROFILES[name]


def for_flag(optimized: bool) -> Profile:
    return OPTIMIZED if optimized else BASE
