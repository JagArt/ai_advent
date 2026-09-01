# AI Advent

Итерации проекта по дням, каждая в своей папке.

| Папка | Итерация |
| --- | --- |
| `day1/` | CLI: запрос из терминала, потоковый вывод в stdout |
| `day2/` | Веб-интерфейс: страница с полем промпта и потоковым ответом |

## Установка

Общее виртуальное окружение в корне, зависимости — под конкретный день:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r day2/requirements.txt
```

## Настройка

Скопируйте `.env.example` в `.env` и укажите ключ:

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-v4-flash
```

`.env` лежит в корне и подхватывается из обеих папок. `DEEPSEEK_MODEL` необязателен и читается только в `day2`, по умолчанию используется `deepseek-v4-flash`; в `day1` модель задана в коде.

## day1 — CLI

```bash
python day1/main.py
```

Скрипт спрашивает запрос в терминале и печатает ответ модели по мере поступления чанков.

## day2 — веб-интерфейс

```bash
uvicorn main:app --reload --app-dir day2
```

Интерфейс доступен на http://127.0.0.1:8000

| Файл | Назначение |
| --- | --- |
| `day2/main.py` | FastAPI-приложение, роуты `/` и `POST /api/ask` |
| `day2/llm.py` | Клиент DeepSeek, async-генератор чанков ответа |
| `day2/static/` | Страница интерфейса: HTML, CSS, JS |

### API

`POST /api/ask` принимает `{"prompt": "..."}` и отдаёт `text/event-stream`: кадры `data: <json-строка>` с частями ответа, затем `event: done` либо `event: error` с текстом ошибки.
