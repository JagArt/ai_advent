# notes

Крошечная библиотека заметок. Код лежит в `src/notes/`.

## Быстрый старт

```python
from notes.api import create_note, get_note, list_notes

note = create_note("покупка", text="молоко")
print(get_note(note.id).title)
print(list_notes())
```

## API

Публичные функции модуля `notes.api`:

- `create_note(title, text)` — создаёт заметку. Второй аргумент называется `text`.
- `get_note(note_id)` — возвращает заметку или `None`.
- `list_notes()` — все заметки списком.

Класс `Note` хранит `id`, `title` и `text`.
