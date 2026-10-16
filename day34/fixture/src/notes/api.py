"""Публичный API библиотеки заметок."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


@dataclass
class Note:
    """Заметка: заголовок, тело и необязательные теги."""

    id: int
    title: str
    body: str
    tags: tuple[str, ...] = ()
    archived: bool = False
    created_at: datetime = field(default_factory=datetime.now)


_NOTES: dict[int, Note] = {}
_NEXT_ID = 1


def create_note(title: str, body: str, tags: Iterable[str] | None = None) -> Note:
    """Создаёт заметку с заголовком, телом и необязательными тегами."""
    global _NEXT_ID
    note = Note(id=_NEXT_ID, title=title, body=body, tags=tuple(tags or ()))
    _NOTES[note.id] = note
    _NEXT_ID += 1
    return note


def get_note(note_id: int) -> Note | None:
    """Возвращает заметку по идентификатору или None, если её нет."""
    return _NOTES.get(note_id)


def list_notes(*, include_archived: bool = False) -> list[Note]:
    """Список заметок, по умолчанию без архивных."""
    notes = list(_NOTES.values())
    if include_archived:
        return notes
    return [note for note in notes if not note.archived]


def archive_note(note_id: int) -> Note | None:
    """Помечает заметку архивной. Возвращает её или None, если идентификатора нет."""
    note = _NOTES.get(note_id)
    if note is None:
        return None
    note.archived = True
    return note
