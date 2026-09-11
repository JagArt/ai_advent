# day2 — ограничения формата ответа

Установка, ключи и команда запуска — в [корневом README](../README.md).

Один промпт уходит в модель дважды: без ограничений и с ограничениями из `day2/constraints.json`. Обе колонки показывают количество слов и `finish_reason`, чтобы разницу было видно.

Ограничения правятся только в файле, без изменений кода:

```json
{
  "system_prompt": "You are a helpful assistant.",
  "sections": [
    {
      "title": "Формат ответа",
      "rules": [
        "Используй ровно 3 пункта.",
        "Каждый пункт должен содержать одно предложение.",
        "Используй формат нумерованного списка: 1., 2., 3."
      ]
    }
  ],
  "params": { "max_tokens": 200, "stop": ["4."] }
}
```

`system_prompt` и `sections` собираются в system prompt, `params` уходят в `chat.completions.create` как есть. `stop: ["4."]` обрывает генерацию, если модель всё же начнёт четвёртый пункт.

| Файл | Назначение |
| --- | --- |
| `day2/main.py` | FastAPI-приложение, роуты `/`, `GET /api/constraints`, `POST /api/ask` |
| `day2/llm.py` | Клиент DeepSeek, стрим чанков |
| `day2/constraints.json` | Ограничения: формат, длина, условие завершения, параметры API |
| `day2/constraints.py` | Pydantic-модели ограничений и сборка system prompt |
| `day2/static/` | Страница интерфейса: HTML, CSS, JS |

## API

`POST /api/ask` принимает `{"prompt": "...", "constrained": false}` и отдаёт `text/event-stream`: кадры `data: <json-строка>` с частями ответа, затем `event: done` с `{finish_reason, word_count}` либо `event: error` с текстом ошибки. При `constrained: true` применяются ограничения из файла.

`GET /api/constraints` возвращает содержимое `constraints.json` — интерфейс рисует из него блок ограничений.
