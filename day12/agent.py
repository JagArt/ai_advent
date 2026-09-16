import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import remember
from llm import complete, stream_chat
from memory import (
    LONGTERM,
    TIERS,
    WINDOW_MESSAGES,
    WORKING,
    Memory,
    Message,
    Store,
    Window,
    block,
)
from profile import BY_KEY, PROFILE, Profile
from profile import build as build_profile
from remember import SCALE, SECTIONS_BY_TARGET, SKIP
from storage import Storage

SYSTEM_PROMPT = """\
Ты — ассистент, который помогает пользователю довести задачу до внятного
технического задания.

Первым блоком приходит профиль пользователя: с кем ты говоришь и в каком виде он
ждёт ответ. Это требования к каждому твоему ответу, а не пожелания: соблюдай их
молча — не переспрашивай, подходит ли такая форма, и не упоминай сам профиль.
Тон, длину и вид ответа задаёт он, а не привычка отвечать подробно.

Память приходит к тебе тремя частями, и они разные.
Долговременная память — отдельный блок: решения команды и знания о её окружении.
Это верно всегда, переспрашивать его не нужно.
Рабочая память — второй блок: техническое задание текущей задачи, всё, что
пользователь уже подтвердил. Это опора, а не черновик: не предлагай заново то,
что там уже записано, и не противоречь ему.
Краткосрочная память — последние реплики диалога дословно. Того, что было раньше
и не попало в блоки памяти, ты не помнишь вовсе.

Помогай доводить требования до формулировок: уточняй неясное, предлагай варианты,
называй пропущенное. Не выдумывай за пользователя решения, которых он не принимал.
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

DEFAULT_PROMPT = "Нужен сервис выгрузки отчётов по складским остаткам. С чего начнём?"

DEFAULT_TEMPERATURE = 0.7

# Потолок одного ответа: диалог остаётся диалогом, а не полотном на весь экран.
MAX_TOKENS = 2000

# Потолок ответа в сравнении профилей. Он выше, чем нужно любому из ответов: обрезка
# по лимиту испортила бы главную мерку — длину, — и профиль «подробно» проиграл бы
# не по своей вине.
COMPARE_MAX_TOKENS = 1200

# Автосохранение выключено: смысл задания в том, что уровень выбирают явно, и по
# умолчанию это делает человек. Переключатель в шапке отдаёт выбор агенту.
DEFAULT_AUTOSAVE = False


@dataclass
class Proposal:
    """Кандидат в память или в профиль: предложение записать, а не запись.

    Живёт в агенте, а не в базе, и это не экономия на таблице: неразобранный
    кандидат — часть краткосрочной памяти, состояние текущего разговора. До
    решения пользователя он не память ни о чём.
    """

    id: int
    text: str
    # Куда советует положить агент: с этого адреса и раздела открыта карточка.
    tier: str
    section: str
    why: str = ""
    # Куда легло на самом деле. Пусто — решения ещё нет.
    saved: str | None = None
    item_id: int = 0
    # Ответ, из которого достали находку: карточка стоит в ленте под ним и
    # возвращается на то же место после перезагрузки страницы.
    answer_id: int = 0
    # Вид находки: формулировка в раздел или правка шкалы профиля.
    kind: str = remember.ITEM
    scale: str = ""
    value: str = ""
    # Значение шкалы до правки: у пункта отмена — это удаление, а у шкалы —
    # возврат прежнего значения, и помнить его больше негде.
    previous: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tier": self.tier,
            "section": self.section,
            "why": self.why,
            "saved": self.saved,
            "answer_id": self.answer_id,
            "kind": self.kind,
            "scale": self.scale,
            "value": self.value,
            "previous": self.previous,
        }


@dataclass(frozen=True)
class AgentPlan:
    """Что уйдёт в модель: профиль, сколько реплик и сколько пунктов с каждого уровня."""

    context_messages: int
    history_messages: int
    dropped_messages: int
    window_messages: int
    # Каждый уровень отдельно: из чего сложился запрос, видно по строке.
    longterm_count: int
    working_count: int
    # Профиль уходит в запрос всегда, поэтому в плане он не числом, а именем: по
    # нему видно, от чьего лица пойдёт ход.
    profile_id: str = ""
    profile_name: str = ""
    profile_count: int = 0


@dataclass(frozen=True)
class AgentDelta:
    content: str = ""
    finish_reason: str | None = None


@dataclass(frozen=True)
class AgentTurn:
    """Ход, который дошёл до конца и лёг в краткосрочную память."""

    context_messages: int
    # id реплик хода: по ним страница отмечает границу окна в ленте.
    question_id: int
    answer_id: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class AgentProposals:
    """Кандидаты после хода: разбор реплики — отдельный запрос к модели.

    Снимок памяти едет вместе с событием, а не запрашивается отдельно: событие
    приходит изнутри ask(), где агент ещё держит свою блокировку, и обратный
    вызов в него встал бы намертво.
    """

    items: tuple[Proposal, ...] = ()
    saved: int = 0
    memory: dict[str, Any] = field(default_factory=dict)


AgentEvent = AgentPlan | AgentDelta | AgentTurn | AgentProposals


def _title_messages(prompt: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=TITLE_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]


def clean_title(text: str) -> str:
    """Модель просили обойтись без кавычек и точки, но просьба — не гарантия."""
    return " ".join(text.split()).strip("\"'«»`.")[:TITLE_LIMIT]


def request(
    system_prompt: str,
    profile: Profile | None,
    frame: Window,
    prompt: str,
) -> list[ChatCompletionMessageParam]:
    """Запрос целиком: кто ты, с кем говоришь, что помните, о чём речь сейчас.

    Профиль стоит первым из блоков и до памяти: он не про содержание, а про форму
    всего, что дальше. Порядок здесь единственный на всё приложение — и ход
    диалога, и одиночный ответ в сравнении профилей собираются этой функцией,
    иначе сравнение мерило бы не то, что видит пользователь в чате.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        *([profile.as_param()] if profile else []),
        *frame.params(),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]


async def answer_as(
    profile: Profile | None,
    longterm: Store | None,
    prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = COMPARE_MAX_TOKENS,
) -> str:
    """Один ответ от лица профиля, вне диалога.

    Ни ленты, ни рабочей памяти в запросе нет намеренно: сравнивать профили можно
    только тогда, когда всё остальное совпадает. Долговременная память остаётся —
    она общая для всех профилей, и без неё агент отвечал бы про другую команду.
    """
    frame = Window(longterm=longterm, working=None, messages=[])
    reply = await complete(
        request(system_prompt, profile, frame, prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return reply.text


class Agent:
    """Собеседник с профилем и тремя уровнями памяти: объект временный, они — нет."""

    def __init__(
        self,
        session_id: str,
        storage: Storage,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        window_messages: int = WINDOW_MESSAGES,
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.window_messages = window_messages
        self._storage = storage
        self._memory = Memory()
        # Профиль лежит рядом с памятью, а не внутри неё: у него другой охват (один
        # человек, а не один диалог) и другое назначение — форма ответа, не факты.
        self._profile: Profile | None = None
        self._profile_id = ""
        self._pending: dict[int, Proposal] = {}
        self._proposals = 0
        self._loaded = False
        self._longterm_version = -1
        self._profile_version = -1
        self._lock = asyncio.Lock()

    async def plan(self, window: int | None = None) -> AgentPlan:
        """Состав запроса до самого запроса: он же — состояние памяти для шапки."""
        async with self._lock:
            await self._load()
            _, plan = self._plan(window or self.window_messages)
            return plan

    async def transcript(self) -> list[dict[str, Any]]:
        """Краткосрочная память как лента: весь диалог, окно отмечается отдельно."""
        async with self._lock:
            await self._load()
            return await self._storage.load_messages(self.session_id)

    async def memory(self) -> dict[str, Any]:
        """Снимок профиля, обоих блоков и очереди кандидатов — всё, что рисует страница."""
        async with self._lock:
            await self._load()
            return self._snapshot()

    async def switch(self, profile_id: str) -> dict[str, Any]:
        """Сменить профиль диалога: следующий запрос уйдёт с другим первым блоком.

        Профиль меняется у диалога, а не у приложения: два диалога рядом могут идти
        от лица разных людей, и один и тот же вопрос в них отличается только этим.
        Память при этом не трогается — ни общая, ни рабочая: сменился собеседник, а
        не то, о чём договорились.
        """
        async with self._lock:
            await self._load()

            if await self._storage.load_profile(profile_id) is None:
                raise LookupError(f"Профиля {profile_id} нет")

            await self._storage.set_session_profile(self.session_id, profile_id)
            await self._reload_profile()
            return self._snapshot()

    async def tune(self, scale: str, value: str) -> dict[str, Any]:
        """Правка шкалы вручную: значение только из её списка.

        Шкалы правятся прямо в блоке профиля, без карточки: выбор из трёх слов не
        требует ни формулировки, ни решения, куда его положить.
        """
        async with self._lock:
            await self._load()

            known = BY_KEY.get(scale)
            if known is None or value not in known.values:
                raise ValueError(f"Шкала {scale} не принимает значение {value}")
            if self._profile is None:
                raise LookupError("У диалога нет профиля")

            await self._storage.set_scale(self._profile_id, scale, value)
            await self._reload_profile()
            return self._snapshot()

    async def commit(
        self,
        proposal_id: int,
        target: str,
        section: str = "",
        text: str = "",
        origin: str = "user",
    ) -> dict[str, Any]:
        """Решение по кандидату: записать по адресу, перенести на другой или отклонить.

        Один метод на все случаи, потому что случай всегда один и тот же — у
        кандидата меняется место. Уже записанный пункт сначала снимается с прежнего
        адреса: иначе перенос оставил бы копию. У правки шкалы «снять» значит
        вернуть прежнее значение, но решение от этого не перестаёт быть одним.
        """
        async with self._lock:
            await self._load()
            return await self._commit(proposal_id, target, section, text, origin)

    async def forget(self, target: str, item_id: int) -> dict[str, Any]:
        """Удаление пункта из памяти или из профиля: адрес указывается явно, как и при записи."""
        async with self._lock:
            await self._load()

            if target == LONGTERM:
                removed = await self._storage.delete_longterm(item_id)
            elif target == WORKING:
                removed = await self._storage.delete_working(self.session_id, item_id)
            elif target == PROFILE:
                removed = await self._storage.delete_profile_item(self._profile_id, item_id)
            else:
                raise ValueError(f"Неизвестный адрес записи: {target}")

            if not removed:
                raise LookupError(f"Пункта {item_id} нет по адресу {target}")

            # Карточка удалённого пункта возвращается в нерешённые: предложение
            # никуда не делось, а ответ на него отменили.
            for proposal in self._pending.values():
                if proposal.saved == target and proposal.item_id == item_id:
                    proposal.saved = None
                    proposal.item_id = 0

            await self._refresh({target})
            return self._snapshot()

    async def ask(
        self,
        prompt: str,
        window: int | None = None,
        autosave: bool = DEFAULT_AUTOSAVE,
    ) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            await self._load()

            window = window or self.window_messages
            frame, plan = self._plan(window)
            # Состав запроса уходит на страницу до первого токена ответа: что
            # именно агент взял из памяти, видно ещё до начала ответа.
            yield plan

            messages = request(self.system_prompt, self._profile, frame, prompt)

            # Имя диалогу даётся один раз, по первому вопросу. Отдельный запрос
            # уходит в модель одновременно с ответом и к концу стрима уже готов —
            # ждать заголовок пользователю не приходится.
            title = asyncio.create_task(self._title(prompt)) if not self._memory.size else None

            try:
                parts: list[str] = []
                finish_reason: str | None = None

                async for delta in stream_chat(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
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

                question_id, answer_id = await self._storage.save_turn(
                    self.session_id,
                    prompt,
                    answer,
                )
                # Краткосрочная память пополняется сама и сразу: разрешения на это
                # никто не спрашивает — тем она и отличается от двух других.
                self._memory.add(
                    Message(role="user", content=prompt, id=question_id),
                    Message(role="assistant", content=answer, id=answer_id),
                )
                await self._save_title(title)

                yield AgentTurn(
                    context_messages=plan.context_messages,
                    question_id=question_id,
                    answer_id=answer_id,
                    finish_reason=finish_reason,
                )

                # Разбор хода идёт после ответа, а не перед ним: пользователь уже
                # читает реплику, пока агент решает, что из неё стоит запомнить.
                proposals = await self._propose(prompt, answer, answer_id, autosave)
                if proposals is not None:
                    yield proposals
            finally:
                # Ход не сохранился — сессия осталась пустой, и называть её нечем.
                if title is not None:
                    title.cancel()

    def _plan(self, window: int) -> tuple[Window, AgentPlan]:
        frame = self._memory.select(window)
        longterm, working = frame.longterm, frame.working

        return frame, AgentPlan(
            context_messages=len(frame.messages),
            history_messages=self._memory.size,
            # Реплики, которых в запросе нет: всё, что старше окна. Часть из них
            # пересказана блоками памяти, часть забыта насовсем.
            dropped_messages=self._memory.size - len(frame.messages),
            window_messages=window,
            longterm_count=longterm.count if longterm else 0,
            working_count=working.count if working else 0,
            profile_id=self._profile.id if self._profile else "",
            profile_name=self._profile.name if self._profile else "",
            profile_count=self._profile.count if self._profile else 0,
        )

    async def _propose(
        self,
        prompt: str,
        answer: str,
        answer_id: int,
        autosave: bool,
    ) -> AgentProposals | None:
        """Ход уходит в модель и возвращается кандидатами на запись."""
        review = remember.request(
            self._profile,
            self._memory.longterm,
            self._memory.working,
            prompt,
            answer,
        )

        try:
            reply = await complete(
                review,
                temperature=remember.TEMPERATURE,
                max_tokens=remember.MAX_TOKENS,
            )
        except Exception:
            # Запрос не дошёл — память осталась прежней: следующий ход разберут
            # вместе с этим, ничего не потеряется.
            return None

        try:
            candidates = remember.parse(reply.text)
        except ValueError:
            # Ответ не по форме: находок нет, память не тронута.
            return AgentProposals(memory=self._snapshot())

        items = []
        saved = 0
        found = remember.fresh(
            candidates,
            self._memory.longterm,
            self._memory.working,
            self._profile,
        )
        for candidate in found:
            self._proposals += 1
            proposal = Proposal(
                id=self._proposals,
                text=candidate.text,
                tier=candidate.tier,
                section=candidate.section,
                why=candidate.why,
                answer_id=answer_id,
                kind=candidate.kind,
                scale=candidate.scale,
                value=candidate.value,
            )
            self._pending[proposal.id] = proposal
            items.append(proposal)

        if autosave:
            # Пользователь отдал выбор агенту: тот пишет по своей же рекомендации,
            # но карточка всё равно появляется — с пометкой, куда ушло.
            for proposal in items:
                await self._commit(proposal.id, proposal.tier, proposal.section, proposal.text, "agent")
                saved += 1

        return AgentProposals(items=tuple(items), saved=saved, memory=self._snapshot())

    async def _commit(
        self,
        proposal_id: int,
        target: str,
        section: str,
        text: str,
        origin: str,
    ) -> dict[str, Any]:
        proposal = self._pending.get(proposal_id)
        if proposal is None:
            raise LookupError(f"Предложения {proposal_id} больше нет")

        touched = set()
        if proposal.saved:
            # Перенос и отмена начинаются одинаково: решение снимается оттуда, где
            # оно сейчас. Иначе в памяти осталось бы две копии одного пункта.
            await self._undo(proposal)
            touched.add(proposal.saved)
            proposal.saved = None
            proposal.item_id = 0

        if target == SKIP:
            del self._pending[proposal_id]
            await self._refresh(touched)
            return self._snapshot()

        if target not in (*TIERS, PROFILE):
            raise ValueError(f"Неизвестный адрес записи: {target}")

        if proposal.kind == SCALE:
            # Правку шкалы никуда, кроме профиля, положить нельзя: у неё нет
            # формулировки, которая легла бы в раздел памяти.
            if target != PROFILE or self._profile is None:
                raise ValueError("Правка шкалы применяется только к профилю")

            # Прежнее значение запоминается на записи, а не на разборе хода: между
            # находкой и решением шкалу могли поправить руками.
            proposal.previous = self._profile.value(proposal.scale)
            await self._storage.set_scale(self._profile_id, proposal.scale, proposal.value)
            proposal.saved = PROFILE
            await self._refresh({*touched, PROFILE})
            return self._snapshot()

        proposal.text = remember.clean(text) or proposal.text
        proposal.tier = target
        sections = SECTIONS_BY_TARGET[target]
        proposal.section = section if section in sections else remember.section_of(target, section or proposal.section)
        proposal.item_id = await self._storage.add(
            target,
            self.session_id,
            proposal.section,
            proposal.text,
            origin,
            self._profile_id,
        )
        proposal.saved = target

        await self._refresh({*touched, target})
        return self._snapshot()

    async def _undo(self, proposal: Proposal) -> None:
        """Снять решение по карточке.

        У пункта это удаление, у шкалы — возврат прежнего значения: стереть шкалу
        нельзя, у профиля она есть всегда.
        """
        if proposal.kind == SCALE:
            if proposal.previous:
                await self._storage.set_scale(self._profile_id, proposal.scale, proposal.previous)
        elif proposal.saved == LONGTERM:
            await self._storage.delete_longterm(proposal.item_id)
        elif proposal.saved == PROFILE:
            await self._storage.delete_profile_item(self._profile_id, proposal.item_id)
        else:
            await self._storage.delete_working(self.session_id, proposal.item_id)

    async def _refresh(self, touched: set[str]) -> None:
        """Перечитать то, что изменилось: перенос задевает сразу два адреса."""
        for target in touched:
            if target == PROFILE:
                await self._reload_profile()
            else:
                await self._reload(target)

    async def _reload(self, tier: str) -> None:
        """Уровень перечитывается из базы: id пунктов нужны странице для удаления."""
        if tier == LONGTERM:
            self._longterm_version = self._storage.longterm_version
            self._memory.longterm = block(LONGTERM, await self._storage.load_longterm())
        else:
            self._memory.working = block(WORKING, await self._storage.load_working(self.session_id))

    async def _reload_profile(self) -> None:
        """Профиль перечитывается целиком: и шкалы, и пункты, и его имя.

        Заодно перечитывается и то, какой профиль у диалога: сменить его мог и
        соседний запрос, пока этот агент лежал в кэше.
        """
        self._profile_version = self._storage.profile_version
        self._profile_id = await self._storage.session_profile(self.session_id)
        row = await self._storage.load_profile(self._profile_id)
        self._profile = build_profile(row) if row else None

    def _snapshot(self) -> dict[str, Any]:
        longterm, working = self._memory.longterm, self._memory.working
        return {
            # Профиль едет снимком целиком: страница рисует его блок и селектор из
            # одного и того же ответа, что и память.
            "profile": self._profile.as_dict() if self._profile else None,
            "longterm": [item.as_dict() for item in longterm.items] if longterm else [],
            "working": [item.as_dict() for item in working.items] if working else [],
            # Нерешённые карточки переживают перезагрузку страницы, но не
            # перезапуск процесса: они часть краткосрочной памяти.
            "proposals": [proposal.as_dict() for proposal in self._pending.values()],
        }

    async def _title(self, prompt: str) -> str:
        try:
            answer = await complete(
                _title_messages(prompt),
                temperature=TITLE_TEMPERATURE,
                max_tokens=TITLE_MAX_TOKENS,
            )
        except Exception:
            # Заголовок — украшение панели, а не часть разговора: не вышло — в списке
            # останется начало первого вопроса, ответ пользователь получит в любом случае.
            return ""
        return clean_title(answer.text)

    async def _save_title(self, title: asyncio.Task[str] | None) -> None:
        if title is None:
            return

        name = await title
        if name:
            await self._storage.set_title(self.session_id, name)

    async def _load(self) -> None:
        # Агент создаётся пустым, в том числе после перезапуска процесса. Уровни
        # поднимаются из разных мест: лента и ТЗ — по session_id, долговременная
        # память — без него, она общая, а профиль — по ссылке из строки сессии.
        if self._loaded:
            # Общая память могла измениться в соседнем диалоге: там её пополнили,
            # здесь блок остался прежним. Счётчик изменений это и ловит.
            if self._longterm_version != self._storage.longterm_version:
                await self._reload(LONGTERM)
            # У профиля причина та же, но повод чаще: один человек ведёт несколько
            # диалогов, и предпочтение, записанное в одном, верно во всех.
            if self._profile_version != self._storage.profile_version:
                await self._reload_profile()
            return

        history = await self._storage.load_messages(self.session_id)
        self._memory = Memory()
        self._memory.add(
            *(
                Message(role=message["role"], content=message["content"], id=message["id"])
                for message in history
            ),
        )
        await self._reload_profile()
        await self._reload(LONGTERM)
        await self._reload(WORKING)
        self._loaded = True
