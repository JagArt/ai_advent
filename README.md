# AI Advent

Итерации проекта по дням, каждая в своей папке.

| Папка | Итерация |
| --- | --- |
| `day1/` | CLI: запрос из терминала, потоковый вывод в stdout |
| `day2/` | Веб-интерфейс: сравнение свободного ответа и ответа по ограничениям из JSON |
| `day3/` | Веб-интерфейс: одна задача, один запрос — способы рассуждения из чекбоксов |

## Установка

Общее виртуальное окружение в корне, зависимости — под конкретный день:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r day2/requirements.txt
```

Зависимости `day2` и `day3` совпадают.

## Настройка

Скопируйте `.env.example` в `.env` и укажите ключ:

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-v4-flash
```

`.env` лежит в корне и подхватывается из папок day1–day3. `DEEPSEEK_MODEL` необязателен и читается в `day2` и `day3`, по умолчанию используется `deepseek-v4-flash`; в `day1` модель задана в коде.

## day1 — CLI

```bash
python day1/main.py
```

Скрипт спрашивает запрос в терминале и печатает ответ модели по мере поступления чанков.

## day2 — ограничения формата ответа

```bash
uvicorn main:app --reload --app-dir day2
```

Интерфейс доступен на http://127.0.0.1:8000

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

### API

`POST /api/ask` принимает `{"prompt": "...", "constrained": false}` и отдаёт `text/event-stream`: кадры `data: <json-строка>` с частями ответа, затем `event: done` с `{finish_reason, word_count}` либо `event: error` с текстом ошибки. При `constrained: true` применяются ограничения из файла.

`GET /api/constraints` возвращает содержимое `constraints.json` — интерфейс рисует из него блок ограничений.

## day3 — способы рассуждения

```bash
uvicorn main:app --reload --app-dir day3
```

Интерфейс доступен на http://127.0.0.1:8000

Один запрос: без чекбоксов это прямой ответ, включённые опции складываются в один промпт.

| Чекбокс | Что происходит |
| --- | --- |
| Решай пошагово | В system prompt добавляется «Решай пошагово» |
| Сначала промпт | Модель сначала составляет промпт, затем отвечает по нему |
| Группа экспертов | В запрос добавляется список ролей (по умолчанию аналитик, инженер, критик) |

| Файл | Назначение |
| --- | --- |
| `day3/main.py` | FastAPI-приложение, роуты `/`, `GET /api/defaults`, `POST /api/ask` |
| `day3/llm.py` | Клиент DeepSeek, задача и роли по умолчанию, сборка system prompt, стрим чанков |
| `day3/static/` | Страница интерфейса: HTML, CSS, JS |

### API

`POST /api/ask` принимает `{"prompt": "...", "step_by_step": false, "meta_prompt": false, "experts": []}` и отдаёт `text/event-stream`: кадры `data: <json-строка>` с частями ответа, затем `event: done` с `{finish_reason, word_count}` либо `event: error` с текстом ошибки.

Пустой `experts` — эксперты выключены. Задача и роли по умолчанию задаются в `day3/llm.py` (`DEFAULT_PROMPT`, `DEFAULT_EXPERTS`) и отдаются через `GET /api/defaults`. Если `meta_prompt: true`, перед чанками ответа приходит `event: stage` со значением `"prompt"` или `"answer"`: сначала стримится сгенерированный промпт, затем решение.
