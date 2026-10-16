"""Команды над списком заметок: показать и отправить в архив."""

from notes.api import archive_note, list_notes


def show(include_archived: bool = False) -> str:
    """Печатает идентификаторы и заголовки заметок по одной на строку."""
    lines = []
    for note in list_notes(include_archived=include_archived):
        mark = "архив" if note.archived else "открыта"
        lines.append(f"{note.id}\t{mark}\t{note.title}")
    return "\n".join(lines)


def archive(note_id: int) -> str:
    """Архивирует заметку и возвращает короткую сводку."""
    note = archive_note(note_id)
    if note is None:
        return f"заметки {note_id} нет"
    return f"архивирована {note.id}: {note.title}"
