"""Хранилище поверх публичного API: создаёт заметку и читает её обратно."""

from notes.api import Note, create_note, get_note


def add_plain(title: str, body: str) -> Note:
    """Кладёт заметку без тегов и сразу проверяет, что она читается."""
    created = create_note(title, body)
    stored = get_note(created.id)
    if stored is None:
        raise RuntimeError(f"заметка {created.id} не сохранилась")
    return stored
