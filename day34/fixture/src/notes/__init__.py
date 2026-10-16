"""Библиотека заметок: публичный API реэкспортируется отсюда."""

from notes.api import Note, archive_note, create_note, get_note, list_notes

__all__ = ["Note", "archive_note", "create_note", "get_note", "list_notes"]
