"""Настройки шлюза и страницы чата.

Оба процесса читают одно и то же окружение: страница должна звать тот адрес
и тот ключ, с которыми поднялся сервис. Облачные ключи день не использует.
`load_dotenv` берёт корневой `.env`, если команда запущена из корня репозитория.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_KEY = "day30-local"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")

SERVICE_HOST = os.environ.get("SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8780"))
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", DEFAULT_API_KEY)
SERVICE_BASE_URL = os.environ.get("SERVICE_BASE_URL", f"http://127.0.0.1:{SERVICE_PORT}").rstrip("/")

SLOTS = int(os.environ.get("SERVICE_SLOTS", "1"))
QUEUE_DEPTH = int(os.environ.get("SERVICE_QUEUE", "8"))
RATE_LIMIT = int(os.environ.get("SERVICE_RATE_LIMIT", "12"))
RATE_WINDOW = float(os.environ.get("SERVICE_RATE_WINDOW", "60"))
MAX_CONTEXT = int(os.environ.get("SERVICE_MAX_CONTEXT", "2048"))
MAX_TOKENS = int(os.environ.get("SERVICE_MAX_TOKENS", "512"))


def refuses_public_default(host: str | None = None, key: str | None = None) -> bool:
    """Слушать не-loopback с ключом `day30-local` нельзя: иначе сервис открыт всем."""
    host = SERVICE_HOST if host is None else host
    key = SERVICE_API_KEY if key is None else key
    return host not in LOOPBACK_HOSTS and key == DEFAULT_API_KEY
