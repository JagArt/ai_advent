# DeepSeek Web

Веб-интерфейс для запросов к DeepSeek: многострочное поле промпта, кнопка отправки и потоковый вывод ответа.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка

Скопируйте `.env.example` в `.env` и укажите ключ:

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-v4-flash
```

`DEEPSEEK_MODEL` необязателен, по умолчанию используется `deepseek-v4-flash`.

## Запуск

```bash
uvicorn main:app --reload
```

Интерфейс доступен на http://127.0.0.1:8000

## Структура

| Файл | Назначение |
| --- | --- |
| `main.py` | FastAPI-приложение, роуты `/` и `POST /api/ask` |
| `llm.py` | Клиент DeepSeek, async-генератор чанков ответа |
| `static/` | Страница интерфейса: HTML, CSS, JS |

## API

`POST /api/ask` принимает `{"prompt": "..."}` и отдаёт `text/event-stream`: кадры `data: <json-строка>` с частями ответа, затем `event: done` либо `event: error` с текстом ошибки.
