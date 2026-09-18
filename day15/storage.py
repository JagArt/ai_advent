import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import invariants
import task
from invariants import GLOBAL, TASK
from memory import LONGTERM, WORKING
from profile import PROFILE, SCALE_KEYS
from seed import DEFAULT_PROFILE, INVARIANTS, PROFILES, SEED

DB_PATH = Path(__file__).parent / "history.db"

# Запасной заголовок диалога — начало первого вопроса: длинную реплику незачем
# тащить из базы целиком, в строку списка всё равно попадёт пара слов.
TITLE_LIMIT = 80

# Три уровня памяти — три разных места хранения, и это главное, что видно в схеме.
# Краткосрочная лежит в messages: реплики диалога, из которых агент берёт окно.
# Рабочая — в working, по строке на пункт ТЗ, с session_id: задача кончилась
# вместе с диалогом. Долговременная — в longterm, и session_id у неё нет вовсе:
# она старше любого диалога и переживает каждый из них.
#
# Профиль лежит в своих трёх таблицах и в схеме отличается от памяти сразу двумя
# вещами: он принадлежит человеку, а не диалогу, и половина его — не текст, а
# значение шкалы из фиксированного списка.
#
# Состояние задачи — в tasks и transitions. В схеме оно устроено не как память и не
# как профиль: там пункты, а здесь одна строка на диалог, потому что состояние не
# накапливается — оно меняется. История переходов лежит рядом отдельным журналом:
# текущее состояние отвечает на вопрос «где мы», журнал — «как сюда пришли».
#
# Инварианты — в invariants и violations, и в схеме у них ровно та особенность, из
# которой сделан весь день: они не принадлежат диалогу. У общего инварианта нет
# session_id, как у долговременной памяти, а у инварианта задачи он есть, но правило
# всё равно лежит не в реплике и не в ТЗ, а отдельной строкой со своим детектором.
# Журнал нарушений рядом устроен как transitions: инварианты отвечают на вопрос «что
# нельзя», журнал — «где на это наткнулись и чем поймали».
#
# Гейты в схеме не лежат вовсе: они описаны кодом в gates.py, как этапы и шаги. Лежат
# решения о них — approvals: утверждение не часть состояния задачи, а решение
# пользователя о её движении, и хранить его флагом в tasks значило бы стереть разницу.
# Рядом overruns — журнал забегов вперёд, устроенный как violations: «где агент взялся
# за работу следующего этапа и чем его на этом поймали».
SCHEMA = """
-- Профиль: кто собеседник. Таблица идёт первой не для порядка — на неё ссылается
-- сессия, а профиль существует раньше любого диалога.
CREATE TABLE IF NOT EXISTS profiles (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    -- Порядок в селекторе: список профилей — не история, сортировать его по id
    -- значило бы менять порядок при переименовании.
    position   INTEGER NOT NULL DEFAULT 0
);

-- Шкалы предпочтений: по строке на шкалу, значение — слово из её списка. Пара
-- (профиль, шкала) первичный ключ, поэтому правка шкалы — это upsert, а не
-- накопление истории: у профиля одна длина ответа, а не пять.
CREATE TABLE IF NOT EXISTS profile_scales (
    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    scale      TEXT NOT NULL,
    value      TEXT NOT NULL,
    PRIMARY KEY (profile_id, scale)
);

-- Свободные пункты профиля: то, что списком значений не описать. Устроены как
-- пункты памяти — раздел, формулировка, origin, — но принадлежат человеку.
CREATE TABLE IF NOT EXISTS profile_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- seed — было до начала общения, user — отправил пользователь кнопкой,
    -- agent — записал сам при включённом автосохранении.
    origin     TEXT NOT NULL,
    -- В каком диалоге это узнали. Внешнего ключа нет намеренно: удалённый диалог
    -- не должен уносить предпочтение, которое из него вынесли.
    source_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS profile_items_idx ON profile_items(profile_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    -- От чьего лица идёт этот диалог. Профиль не принадлежит диалогу, диалог
    -- только ссылается на него: один и тот же профиль ведёт сколько угодно задач.
    profile_id TEXT REFERENCES profiles(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Краткосрочная память: реплики текущего диалога. Окно последних N из них уходит
-- в модель дословно, остальные остаются здесь как лента для чтения.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);

-- Долговременная память: профиль, решения, знания. Без session_id — она общая
-- для всех диалогов и существует до того, как начался первый.
CREATE TABLE IF NOT EXISTS longterm (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- Откуда пункт взялся: seed — было до начала общения, agent — записал сам
    -- при включённом автосохранении, user — отправил пользователь кнопкой.
    origin     TEXT NOT NULL,
    -- В каком диалоге это узнали. У seed источника нет, и внешнего ключа тоже:
    -- удалённый диалог не должен уносить знание, которое из него вынесли.
    source_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Рабочая память: ТЗ текущей задачи. Принадлежит сессии и уходит вместе с ней.
CREATE TABLE IF NOT EXISTS working (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    origin     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS working_session_idx ON working(session_id, id);

-- Состояние задачи: одна строка на диалог, потому что задача = диалог, как и её
-- рабочая память. Пара (этап, шаг) хранится словами из task.py, а не числами:
-- порядок этапов в коде может поменяться, а имя останется собой.
CREATE TABLE IF NOT EXISTS tasks (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    stage      TEXT NOT NULL,
    step       TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Журнал переходов: чем автомат отличается от подписи под ответом. По нему видно
-- не только где задача сейчас, но и как она сюда пришла — включая откаты.
CREATE TABLE IF NOT EXISTS transitions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    from_stage TEXT NOT NULL,
    from_step  TEXT NOT NULL,
    to_stage   TEXT NOT NULL,
    to_step    TEXT NOT NULL,
    -- Кто двинул автомат: user — кнопкой, agent — по своему же предложению при
    -- включённом автосохранении. Предложение, которое отклонили, сюда не попадает.
    origin     TEXT NOT NULL,
    -- Через какой гейт подтверждения прошёл переход. Пусто — гейта подтверждения на
    -- этом ребре нет: между этапами стоят не только они.
    gate       TEXT NOT NULL DEFAULT '',
    why        TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS transitions_session_idx ON transitions(session_id, id);

-- Утверждения: гейты подтверждения, которые прошёл пользователь. Отдельная таблица,
-- а не флаг в tasks, и причина та же, по которой инвариант не лежит в разделе памяти:
-- утверждение не часть состояния, а решение о нём. Состояние меняется переходом,
-- утверждение — только решением человека, и живут они по разным правилам.
--
-- Строка не удаляется и при снятии: откат отменяет действие утверждения, но не факт,
-- что его когда-то дали. По журналу видно, что план утверждали дважды, — а это ровно
-- то, что стоит знать о задаче, которую переоткрывали.
CREATE TABLE IF NOT EXISTS approvals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Ключ гейта из gates.py. Внешнего ключа нет: гейты описаны кодом, а не строками.
    gate       TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    -- Всегда user. Колонка есть ровно для того, чтобы в схеме было видно: другого
    -- значения здесь не бывает, и агент в эту таблицу не пишет.
    origin     TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- Когда утверждение перестало действовать и почему: снял пользователь или унёс
    -- откат. Пусто — действует.
    revoked_at TEXT,
    revoked_why TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS approvals_session_idx ON approvals(session_id, id);

-- Журнал забегов вперёд: то же место в схеме, что у violations, и та же роль. Не
-- «какая работа закрыта», а «где агент взялся за неё раньше времени и чем поймали».
CREATE TABLE IF NOT EXISTS overruns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Где стояла задача и чья работа была сделана: пара этапов и есть забег.
    from_stage TEXT NOT NULL,
    from_step  TEXT NOT NULL,
    ahead_stage TEXT NOT NULL,
    work       TEXT NOT NULL,
    -- Гейт, который эту работу держал. Без него забег — придирка к словам.
    gate       TEXT NOT NULL,
    -- Кто поймал: scan — код по детектору, audit — модель-аудитор после хода.
    caught_by  TEXT NOT NULL,
    quote      TEXT NOT NULL,
    -- Ушёл ли забег пользователю. Единица значит «не ушёл»: проход оборван детектором,
    -- а на экран попал переписанный ответ или отказ, дописанный кодом.
    rewritten  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS overruns_session_idx ON overruns(session_id, id);

-- Инварианты: то, что агент не имеет права нарушить. Хранятся отдельно от диалога, и
-- это не про удобство схемы — про то, чем инвариант отличается от реплики: разговор
-- его не создаёт, не меняет и не отменяет.
CREATE TABLE IF NOT EXISTS invariants (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    -- global — общий для приложения, task — принадлежит одной задаче.
    scope      TEXT NOT NULL,
    -- У общих его нет вовсе: они старше любого диалога, ровно как долговременная
    -- память. У инварианта задачи есть, и CASCADE унесёт его вместе с ней.
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- Что делать вместо. Без этого поля отказ агента превращается в тупик, а «нельзя»
    -- без открытого пути в этом приложении не встречается нигде.
    instead    TEXT NOT NULL DEFAULT '',
    -- Детектор: по строке на паттерн. Формат знает invariants.py, а не эта таблица.
    -- Пустой законен и означает правило, которое кодом не проверить.
    banned     TEXT NOT NULL DEFAULT '',
    -- Снять инвариант может только пользователь, и способов у него два: отключить
    -- (правило остаётся на виду и его видно в панели) и удалить.
    enabled    INTEGER NOT NULL DEFAULT 1,
    -- seed — было до первого диалога, user — добавил пользователь руками,
    -- spec — зафиксирован из пункта ТЗ.
    origin     TEXT NOT NULL,
    -- Из какого пункта рабочей памяти вырос. Внешнего ключа нет намеренно: пункт ТЗ
    -- можно удалить, а зафиксированный из него инвариант остаётся в силе.
    source_item_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS invariants_scope_idx ON invariants(scope, session_id, id);

-- Журнал нарушений: то же место в схеме, что у transitions. По нему видно не что
-- запрещено, а где на запрет наткнулись — и чем поймали: детектором или аудитором.
CREATE TABLE IF NOT EXISTS violations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Без внешнего ключа: удалённый инвариант не должен уносить историю о том, что
    -- его когда-то нарушили. Текст правила поэтому копируется строкой.
    invariant_id INTEGER NOT NULL,
    label        TEXT NOT NULL,
    rule         TEXT NOT NULL,
    -- Где наткнулись: answer — в ответе агента, item — в пункте, который просили
    -- записать в память.
    source       TEXT NOT NULL,
    -- Кто поймал: scan — код по детектору, audit — модель-аудитор после хода.
    caught_by    TEXT NOT NULL,
    quote        TEXT NOT NULL,
    -- Ушёл ли нарушающий ответ пользователю. Единица значит «не ушёл»: проход оборван
    -- детектором, а на экран попал либо переписанный ответ, либо отказ, дописанный
    -- кодом. Ноль стоит у нарушений аудитора: он читает уже отданный ответ, и
    -- переписать его задним числом нельзя.
    rewritten    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS violations_session_idx ON violations(session_id, id);
"""

ITEM_FIELDS = ("id", "section", "text", "origin")

INVARIANT_FIELDS = (
    "id",
    "scope",
    "kind",
    "text",
    "instead",
    "banned",
    "enabled",
    "origin",
    "source_item_id",
)

VIOLATION_FIELDS = (
    "invariant_id",
    "label",
    "rule",
    "source",
    "caught_by",
    "quote",
    "rewritten",
    "created_at",
)

TRANSITION_FIELDS = (
    "from_stage",
    "from_step",
    "to_stage",
    "to_step",
    "origin",
    "gate",
    "why",
    "created_at",
)

APPROVAL_FIELDS = (
    "id",
    "gate",
    "note",
    "origin",
    "created_at",
    "revoked_at",
    "revoked_why",
)

OVERRUN_FIELDS = (
    "from_stage",
    "from_step",
    "ahead_stage",
    "work",
    "gate",
    "caught_by",
    "quote",
    "rewritten",
    "created_at",
)


class Storage:
    """Память, профиль и состояние задачи в SQLite: агент живёт в процессе, они — в файле."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        # Долговременная память общая, а агенты живут в процессе по одному на
        # диалог и держат её блок у себя. Счётчик изменений — способ им об этом
        # узнать: записали в одном диалоге, в остальных блок устарел.
        self.longterm_version = 0
        # То же и с профилем, только причина сильнее: один профиль ведёт несколько
        # диалогов, и правка предпочтения в одном из них должна дойти до всех.
        self.profile_version = 0
        # И с общими инвариантами: снятый в одном диалоге инвариант перестаёт
        # действовать во всех, иначе «отдельно от диалога» было бы неправдой.
        self.invariants_version = 0

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Соединение на операцию: вызовы уходят в отдельный поток, поэтому одно
        # соединение на всех делить не приходится — и блокировку заводить не нужно.
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    async def init(self) -> None:
        await asyncio.to_thread(self._init)

    def _init(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            # WAL живёт в самой базе: пишущий ход не блокирует чтение ленты.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            self._plant(connection)
        finally:
            connection.close()

    @staticmethod
    def _plant(connection: sqlite3.Connection) -> None:
        """Профили и долговременная память заполняются до первого диалога, а не в нём.

        Кто собеседник, что команда решила и что она знает, агент не выясняет
        заново при каждом запуске: это уже есть, когда пользователь открывает
        страницу. Заливка идёт при создании базы и при сбросе — дальше и профилем,
        и памятью распоряжается пользователь.

        Два набора проверяются по отдельности: пустая память при заполненных
        профилях — это база, которую уже почистили руками, и восстанавливать в ней
        нужно только опустевшую половину.
        """
        # executescript выше закоммитился сам, как всякий DDL, а вот строки — нет:
        # без явного commit seed исчезал бы при каждом запуске.
        if not connection.execute("SELECT count(*) FROM profiles").fetchone()[0]:
            for position, preset in enumerate(PROFILES):
                connection.execute(
                    "INSERT INTO profiles (id, name, position) VALUES (?, ?, ?)",
                    (preset.id, preset.name, position),
                )
                connection.executemany(
                    "INSERT INTO profile_scales (profile_id, scale, value) VALUES (?, ?, ?)",
                    [(preset.id, scale, value) for scale, value in preset.scales.items()],
                )
                connection.executemany(
                    "INSERT INTO profile_items (profile_id, section, text, origin) VALUES (?, ?, ?, 'seed')",
                    [(preset.id, section, text) for section, text in preset.items],
                )

        if not connection.execute("SELECT count(*) FROM longterm").fetchone()[0]:
            connection.executemany(
                "INSERT INTO longterm (section, text, origin) VALUES (?, ?, 'seed')",
                SEED,
            )

        # Общие инварианты заливаются здесь по той же причине, что и память: рамки
        # существуют раньше первого разговора. Уровень задачи в seed не попадает
        # никогда — его наполняет пользователь, фиксируя пункты ТЗ.
        if not connection.execute(
            "SELECT count(*) FROM invariants WHERE scope = ?", (GLOBAL,)
        ).fetchone()[0]:
            connection.executemany(
                """
                INSERT INTO invariants (scope, kind, text, instead, banned, origin)
                VALUES (?, ?, ?, ?, ?, 'seed')
                """,
                [
                    (GLOBAL, rule.kind, rule.text, rule.instead, invariants.join(rule.banned))
                    for rule in INVARIANTS
                ],
            )
        connection.commit()

    async def reset(self) -> None:
        """Вернуть базу к состоянию после первого запуска: seed и ни одного диалога."""
        await asyncio.to_thread(self._reset)

    def _reset(self) -> None:
        with self._connect() as connection:
            # CASCADE унесёт реплики и ТЗ вместе с сессиями, шкалы и пункты — вместе
            # с профилями. Долговременную память чистим сами: у неё нет внешнего
            # ключа, и в этом как раз смысл.
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM longterm")
            # Инварианты задачи ушли вместе с сессиями, общие чистим сами: у них нет
            # внешнего ключа, и в этом как раз смысл.
            connection.execute("DELETE FROM invariants")
            # Профили удаляются после сессий, иначе внешний ключ не даст: диалог,
            # который на них ссылается, к этому моменту должен быть уже удалён.
            connection.execute("DELETE FROM profiles")
            # Счётчики AUTOINCREMENT тоже к нулю: иначе seed получит id 10+
            # после пары прогонов, и «исходное» состояние этим выдаст себя.
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'"
            ).fetchone():
                connection.execute("DELETE FROM sqlite_sequence")
            self._plant(connection)
        self.longterm_version += 1
        self.profile_version += 1
        self.invariants_version += 1

    async def create_session(self, profile_id: str = DEFAULT_PROFILE) -> str:
        """Новый диалог всегда от чьего-то лица: профиль выбирается до первой реплики."""
        session_id = uuid4().hex
        await asyncio.to_thread(self._create_session, session_id, profile_id)
        return session_id

    def _create_session(self, session_id: str, profile_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, profile_id) VALUES (?, ?)",
                (session_id, profile_id),
            )
            # Состояние появляется вместе с диалогом, а не с первой репликой: чего
            # агент ждёт от пользователя, известно до того, как тот заговорил.
            connection.execute(
                "INSERT INTO tasks (session_id, stage, step) VALUES (?, ?, ?)",
                (session_id, task.START_STAGE, task.START_STEP),
            )

    async def session_exists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._session_exists, session_id)

    def _session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    async def set_title(self, session_id: str, title: str) -> None:
        await asyncio.to_thread(self._set_title, session_id, title)

    def _set_title(self, session_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Все диалоги для боковой панели: заголовок — суть первого запроса."""
        return await asyncio.to_thread(self._list_sessions)

    def _list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id,
                       (SELECT count(*) FROM messages WHERE session_id = s.id) AS size,
                       -- Размер ТЗ прямо в списке: по нему видно, в какой задаче
                       -- рабочая память уже собрана, а какая только началась.
                       (SELECT count(*) FROM working WHERE session_id = s.id) AS working,
                       -- Сколько решений этой задачи пользователь успел зафиксировать
                       -- инвариантами: по этому числу видно, где рамки уже поставлены.
                       (SELECT count(*) FROM invariants WHERE session_id = s.id) AS invariants,
                       -- От чьего лица шёл разговор: в панели один и тот же вопрос
                       -- от двух профилей иначе не отличить.
                       (SELECT name FROM profiles WHERE id = s.profile_id) AS profile,
                       -- Этап задачи прямо в списке: по нему видно, какая задача
                       -- ещё планируется, а какая уже закрыта.
                       (SELECT stage FROM tasks WHERE session_id = s.id) AS stage,
                       -- Заголовок пишет агент после первого хода; пока его нет
                       -- (старый диалог, неудачный запрос) строка живёт началом вопроса.
                       coalesce(s.title,
                                (SELECT substr(content, 1, ?) FROM messages
                                  WHERE session_id = s.id AND role = 'user'
                                  ORDER BY id LIMIT 1)) AS title,
                       -- У пустой сессии сообщений нет: без coalesce она осталась бы
                       -- без времени и уехала в конец списка сразу после создания.
                       coalesce((SELECT max(created_at) FROM messages WHERE session_id = s.id),
                                s.created_at) AS updated_at
                FROM sessions s
                -- created_at с точностью до секунды не разводит ходы внутри одной
                -- секунды, порядок создания сессий добирается из rowid.
                ORDER BY updated_at DESC, s.rowid DESC
                """,
                (TITLE_LIMIT,),
            ).fetchall()
        return [
            {
                "id": session_id,
                "size": size,
                "working": working,
                "invariants": rules,
                "profile": profile,
                # Имя этапа, а не его ключ: в панели стоит то же слово, что в блоке.
                "stage": task.BY_KEY[stage].name if stage in task.BY_KEY else "",
                "title": title,
                "updated_at": updated_at,
            }
            for session_id, size, working, rules, profile, stage, title, updated_at in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    def _delete_session(self, session_id: str) -> None:
        # Реплики и рабочую память уносит ON DELETE CASCADE — внешние ключи
        # включены в _connect. Долговременная остаётся: она не принадлежит диалогу.
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def list_profiles(self) -> list[dict[str, Any]]:
        """Профили для селектора: только имена, без шкал и пунктов."""
        return await asyncio.to_thread(self._list_profiles)

    def _list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name FROM profiles ORDER BY position, id",
            ).fetchall()
        return [{"id": profile_id, "name": name} for profile_id, name in rows]

    async def load_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Профиль целиком: имя, значения шкал и свободные пункты по разделам."""
        return await asyncio.to_thread(self._load_profile, profile_id)

    def _load_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name FROM profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if row is None:
                return None

            scales = connection.execute(
                "SELECT scale, value FROM profile_scales WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
            items = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM profile_items WHERE profile_id = ? ORDER BY id",
                (profile_id,),
            ).fetchall()

        return {
            "id": row[0],
            "name": row[1],
            # Шкала, которой в базе нет, не подставляется здесь: значение по
            # умолчанию знает модель профиля, а не таблица.
            "scales": {scale: value for scale, value in scales if scale in SCALE_KEYS},
            "items": [dict(zip(ITEM_FIELDS, item)) for item in items],
        }

    async def session_profile(self, session_id: str) -> str:
        """Профиль диалога. Пустой — значит база старше этой колонки: берём первый."""
        return await asyncio.to_thread(self._session_profile, session_id)

    def _session_profile(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile_id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return (row[0] if row else None) or DEFAULT_PROFILE

    async def set_session_profile(self, session_id: str, profile_id: str) -> None:
        await asyncio.to_thread(self._set_session_profile, session_id, profile_id)

    def _set_session_profile(self, session_id: str, profile_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET profile_id = ? WHERE id = ?",
                (profile_id, session_id),
            )

    async def set_scale(self, profile_id: str, scale: str, value: str) -> None:
        """Правка шкалы: у профиля одно значение на шкалу, поэтому это upsert."""
        await asyncio.to_thread(self._set_scale, profile_id, scale, value)
        self.profile_version += 1

    def _set_scale(self, profile_id: str, scale: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO profile_scales (profile_id, scale, value) VALUES (?, ?, ?)
                ON CONFLICT (profile_id, scale) DO UPDATE SET value = excluded.value
                """,
                (profile_id, scale, value),
            )

    async def add_profile_item(
        self,
        profile_id: str,
        section: str,
        text: str,
        origin: str,
        source_id: str | None = None,
    ) -> int:
        item_id = await asyncio.to_thread(
            self._add_profile_item,
            profile_id,
            section,
            text,
            origin,
            source_id,
        )
        self.profile_version += 1
        return item_id

    def _add_profile_item(
        self,
        profile_id: str,
        section: str,
        text: str,
        origin: str,
        source_id: str | None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO profile_items (profile_id, section, text, origin, source_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (profile_id, section, text, origin, source_id),
            )
        return cursor.lastrowid or 0

    async def delete_profile_item(self, profile_id: str, item_id: int) -> bool:
        removed = await asyncio.to_thread(self._delete_profile_item, profile_id, item_id)
        if removed:
            self.profile_version += 1
        return removed

    def _delete_profile_item(self, profile_id: str, item_id: int) -> bool:
        # profile_id в условии не формальность: чужое предпочтение из этого диалога
        # удалить нельзя, даже если угадать его id.
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM profile_items WHERE id = ? AND profile_id = ?",
                (item_id, profile_id),
            )
        return cursor.rowcount > 0

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Краткосрочная память целиком: лента диалога в порядке разговора."""
        return await asyncio.to_thread(self._load_messages, session_id)

    def _load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"id": message_id, "role": role, "content": content}
            for message_id, role, content in rows
        ]

    async def save_turn(self, session_id: str, prompt: str, answer: str) -> tuple[int, int]:
        """Ход пишется целиком: вопрос без ответа в истории не остаётся."""
        return await asyncio.to_thread(self._save_turn, session_id, prompt, answer)

    def _save_turn(self, session_id: str, prompt: str, answer: str) -> tuple[int, int]:
        # Обе реплики — одна транзакция: половины хода в ленте не бывает.
        with self._connect() as connection:
            ids = []
            for role, content in (("user", prompt), ("assistant", answer)):
                cursor = connection.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content),
                )
                ids.append(cursor.lastrowid or 0)
        return ids[0], ids[1]

    async def load_longterm(self) -> list[dict[str, Any]]:
        """Долговременная память: одна на все диалоги, читается без session_id."""
        return await asyncio.to_thread(self._load_longterm)

    def _load_longterm(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM longterm ORDER BY id",
            ).fetchall()
        return [dict(zip(ITEM_FIELDS, row)) for row in rows]

    async def add_longterm(
        self,
        section: str,
        text: str,
        origin: str,
        source_id: str | None = None,
    ) -> int:
        item_id = await asyncio.to_thread(self._add_longterm, section, text, origin, source_id)
        self.longterm_version += 1
        return item_id

    def _add_longterm(self, section: str, text: str, origin: str, source_id: str | None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO longterm (section, text, origin, source_id) VALUES (?, ?, ?, ?)",
                (section, text, origin, source_id),
            )
        return cursor.lastrowid or 0

    async def delete_longterm(self, item_id: int) -> bool:
        removed = await asyncio.to_thread(self._delete_longterm, item_id)
        if removed:
            self.longterm_version += 1
        return removed

    def _delete_longterm(self, item_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM longterm WHERE id = ?", (item_id,))
        return cursor.rowcount > 0

    async def load_working(self, session_id: str) -> list[dict[str, Any]]:
        """Рабочая память: ТЗ одной задачи, читается только вместе с её сессией."""
        return await asyncio.to_thread(self._load_working, session_id)

    def _load_working(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM working WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(zip(ITEM_FIELDS, row)) for row in rows]

    async def add_working(self, session_id: str, section: str, text: str, origin: str) -> int:
        return await asyncio.to_thread(self._add_working, session_id, section, text, origin)

    def _add_working(self, session_id: str, section: str, text: str, origin: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO working (session_id, section, text, origin) VALUES (?, ?, ?, ?)",
                (session_id, section, text, origin),
            )
        return cursor.lastrowid or 0

    async def delete_working(self, session_id: str, item_id: int) -> bool:
        return await asyncio.to_thread(self._delete_working, session_id, item_id)

    def _delete_working(self, session_id: str, item_id: int) -> bool:
        # session_id в условии не формальность: пункт чужого ТЗ из этого диалога
        # удалить нельзя, даже если угадать его id.
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM working WHERE id = ? AND session_id = ?",
                (item_id, session_id),
            )
        return cursor.rowcount > 0

    async def load_task(self, session_id: str) -> dict[str, str]:
        """Состояние задачи. Строки нет — диалог старше этой таблицы: начало автомата."""
        return await asyncio.to_thread(self._load_task, session_id)

    def _load_task(self, session_id: str) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT stage, step FROM tasks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {"stage": task.START_STAGE, "step": task.START_STEP}
        return {"stage": row[0], "step": row[1]}

    async def set_task(
        self,
        session_id: str,
        stage: str,
        step: str,
        *,
        origin: str = "user",
        why: str = "",
        previous: tuple[str, str] | None = None,
        gate: str = "",
    ) -> None:
        """Переход: новое состояние и строка в журнале — одной транзакцией.

        Журнал ведётся здесь, а не в агенте, по той же причине, по которой здесь
        лежит и само состояние: перехода без записи о нём не бывает, и разъехаться
        они не должны даже при падении посреди хода. Отмена карточки журналируется
        так же, как и сам переход: возврат — это тоже переход, а не стирание следа.

        Гейт пишется в ту же строку: по журналу должно быть видно не только кто двинул
        задачу, но и через что она прошла. Иначе утверждение плана и переход, который
        оно открыло, лежали бы в базе как два несвязанных события.
        """
        await asyncio.to_thread(self._set_task, session_id, stage, step, origin, why, previous, gate)

    def _set_task(
        self,
        session_id: str,
        stage: str,
        step: str,
        origin: str,
        why: str,
        previous: tuple[str, str] | None,
        gate: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (session_id, stage, step) VALUES (?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE
                    SET stage = excluded.stage,
                        step = excluded.step,
                        updated_at = datetime('now')
                """,
                (session_id, stage, step),
            )
            if previous is None:
                return
            connection.execute(
                """
                INSERT INTO transitions
                    (session_id, from_stage, from_step, to_stage, to_step, origin, gate, why)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, previous[0], previous[1], stage, step, origin, gate, why),
            )

    async def load_transitions(self, session_id: str) -> list[dict[str, Any]]:
        """Журнал переходов диалога в порядке, в котором они случились."""
        return await asyncio.to_thread(self._load_transitions, session_id)

    def _load_transitions(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(TRANSITION_FIELDS)} FROM transitions
                 WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [dict(zip(TRANSITION_FIELDS, row)) for row in rows]

    async def load_approvals(self, session_id: str) -> list[dict[str, Any]]:
        """Утверждения диалога: и действующие, и снятые.

        Снятые приходят вместе с действующими намеренно, как отключённые инварианты в
        day14: спрятанное утверждение выглядело бы так, будто его и не давали, а
        снятое откатом утверждение — это как раз то, что объясняет, почему путь
        вперёд снова закрыт.
        """
        return await asyncio.to_thread(self._load_approvals, session_id)

    def _load_approvals(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(APPROVAL_FIELDS)} FROM approvals
                 WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [dict(zip(APPROVAL_FIELDS, row)) for row in rows]

    async def add_approval(self, session_id: str, gate: str, note: str = "") -> int:
        """Утверждение гейта. Origin в запросе не бывает: сюда пишет только пользователь."""
        return await asyncio.to_thread(self._add_approval, session_id, gate, note)

    def _add_approval(self, session_id: str, gate: str, note: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO approvals (session_id, gate, note) VALUES (?, ?, ?)",
                (session_id, gate, note),
            )
        return int(cursor.lastrowid or 0)

    async def revoke_approvals(self, session_id: str, keys: tuple[str, ...], why: str) -> int:
        """Снять действующие утверждения. Строки остаются: они были.

        Одной ручкой снимается и одно утверждение, и все, что унёс откат: разница
        только в списке ключей, а причина в обоих случаях пишется словами — по ней
        потом видно, кто именно отменил решение.
        """
        if not keys:
            return 0
        return await asyncio.to_thread(self._revoke_approvals, session_id, keys, why)

    def _revoke_approvals(self, session_id: str, keys: tuple[str, ...], why: str) -> int:
        marks = ", ".join("?" for _ in keys)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE approvals
                   SET revoked_at = datetime('now'), revoked_why = ?
                 WHERE session_id = ? AND revoked_at IS NULL AND gate IN ({marks})
                """,
                (why, session_id, *keys),
            )
        return cursor.rowcount

    async def add_overrun(
        self,
        session_id: str,
        from_stage: str,
        from_step: str,
        ahead_stage: str,
        work: str,
        gate: str,
        caught_by: str,
        quote: str,
        rewritten: bool = False,
    ) -> None:
        """Строка в журнал забегов вперёд. Работа и гейт копируются текстом, как правило
        инварианта в violations: область этапа описана кодом, а журнал должен читаться
        и после того, как код изменится."""
        await asyncio.to_thread(
            self._add_overrun,
            session_id,
            from_stage,
            from_step,
            ahead_stage,
            work,
            gate,
            caught_by,
            quote,
            rewritten,
        )

    def _add_overrun(
        self,
        session_id: str,
        from_stage: str,
        from_step: str,
        ahead_stage: str,
        work: str,
        gate: str,
        caught_by: str,
        quote: str,
        rewritten: bool,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO overruns
                    (session_id, from_stage, from_step, ahead_stage, work, gate,
                     caught_by, quote, rewritten)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    from_stage,
                    from_step,
                    ahead_stage,
                    work,
                    gate,
                    caught_by,
                    quote,
                    1 if rewritten else 0,
                ),
            )

    async def load_overruns(self, session_id: str) -> list[dict[str, Any]]:
        """Журнал забегов диалога в порядке, в котором на них наткнулись."""
        return await asyncio.to_thread(self._load_overruns, session_id)

    def _load_overruns(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(OVERRUN_FIELDS)} FROM overruns
                 WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [dict(zip(OVERRUN_FIELDS, row)) for row in rows]

    async def load_invariants(self, scope: str, session_id: str = "") -> list[dict[str, Any]]:
        """Инварианты одного уровня: общие читаются без session_id, задачи — только с ним."""
        return await asyncio.to_thread(self._load_invariants, scope, session_id)

    def _load_invariants(self, scope: str, session_id: str) -> list[dict[str, Any]]:
        fields = ", ".join(INVARIANT_FIELDS)
        with self._connect() as connection:
            if scope == GLOBAL:
                rows = connection.execute(
                    f"SELECT {fields} FROM invariants WHERE scope = ? ORDER BY id",
                    (GLOBAL,),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    SELECT {fields} FROM invariants
                     WHERE scope = ? AND session_id = ? ORDER BY id
                    """,
                    (TASK, session_id),
                ).fetchall()
        return [dict(zip(INVARIANT_FIELDS, row)) for row in rows]

    async def add_invariant(
        self,
        scope: str,
        kind: str,
        text: str,
        instead: str,
        banned: str,
        origin: str,
        session_id: str = "",
        source_item_id: int = 0,
    ) -> int:
        """Новый инвариант: общий или задачи. Заводит его только пользователь."""
        item_id = await asyncio.to_thread(
            self._add_invariant,
            scope,
            kind,
            text,
            instead,
            banned,
            origin,
            session_id,
            source_item_id,
        )
        self.invariants_version += 1
        return item_id

    def _add_invariant(
        self,
        scope: str,
        kind: str,
        text: str,
        instead: str,
        banned: str,
        origin: str,
        session_id: str,
        source_item_id: int,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO invariants
                    (scope, session_id, kind, text, instead, banned, origin, source_item_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    # У общего инварианта session_id нет: он старше любого диалога.
                    session_id if scope == TASK else None,
                    kind,
                    text,
                    instead,
                    banned,
                    origin,
                    source_item_id or None,
                ),
            )
        return cursor.lastrowid or 0

    async def set_invariant(self, invariant_id: int, enabled: bool) -> bool:
        """Отключить или вернуть инвариант. Правило остаётся на виду — меняется сила."""
        changed = await asyncio.to_thread(self._set_invariant, invariant_id, enabled)
        if changed:
            self.invariants_version += 1
        return changed

    def _set_invariant(self, invariant_id: int, enabled: bool) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE invariants SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, invariant_id),
            )
        return cursor.rowcount > 0

    async def delete_invariant(self, invariant_id: int) -> bool:
        removed = await asyncio.to_thread(self._delete_invariant, invariant_id)
        if removed:
            self.invariants_version += 1
        return removed

    def _delete_invariant(self, invariant_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM invariants WHERE id = ?", (invariant_id,))
        return cursor.rowcount > 0

    async def add_violation(
        self,
        session_id: str,
        invariant_id: int,
        label: str,
        rule: str,
        source: str,
        caught_by: str,
        quote: str,
        rewritten: bool = False,
    ) -> None:
        """Строка в журнал нарушений. Текст правила копируется: инвариант могут удалить."""
        await asyncio.to_thread(
            self._add_violation,
            session_id,
            invariant_id,
            label,
            rule,
            source,
            caught_by,
            quote,
            rewritten,
        )

    def _add_violation(
        self,
        session_id: str,
        invariant_id: int,
        label: str,
        rule: str,
        source: str,
        caught_by: str,
        quote: str,
        rewritten: bool,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO violations
                    (session_id, invariant_id, label, rule, source, caught_by, quote, rewritten)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    invariant_id,
                    label,
                    rule,
                    source,
                    caught_by,
                    quote,
                    1 if rewritten else 0,
                ),
            )

    async def load_violations(self, session_id: str) -> list[dict[str, Any]]:
        """Журнал нарушений диалога в порядке, в котором на них наткнулись."""
        return await asyncio.to_thread(self._load_violations, session_id)

    def _load_violations(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(VIOLATION_FIELDS)} FROM violations
                 WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [dict(zip(VIOLATION_FIELDS, row)) for row in rows]

    async def add(
        self,
        target: str,
        session_id: str,
        section: str,
        text: str,
        origin: str,
        profile_id: str = "",
    ) -> int:
        """Запись по выбранному адресу: сам выбор делает пользователь или агент.

        Адресов три, и session_id значит для каждого своё: рабочая память ему
        принадлежит, долговременная только помнит, где это узнали, а профилю он
        нужен ровно за этим же — сам пункт лежит у человека.
        """
        if target == LONGTERM:
            return await self.add_longterm(section, text, origin, session_id)
        if target == WORKING:
            return await self.add_working(session_id, section, text, origin)
        if target == PROFILE:
            return await self.add_profile_item(profile_id, section, text, origin, session_id)
        raise ValueError(f"Неизвестный адрес записи: {target}")
