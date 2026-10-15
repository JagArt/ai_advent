"""Две модели за одним вызовом с JSON-ответом и выбор между ними по доступности.

Провайдеры те же, что в [day31](../day31/llm.py): `qwen3-4b` в LM Studio и
DeepSeek в облаке, OpenAI-совместимый API у обоих. Отличие — где стоит LM
Studio: не на этой машине, а на соседней в локальной сети, и адрес по
умолчанию — её, `192.168.77.47:1234`. Self-hosted раннер GitHub Actions в той
же сети до неё дотянется; раннер GitHub — нет, и тогда ревью уходит в DeepSeek.

Маршрут здесь не выбирает модель по вопросу, как в day31: ревью — всегда одна и
та же работа. Решает доступность: LM Studio отвечает — локально, иначе облако
по ключу, и причина печатается строкой перед ревью. Флаги `--local` и `--cloud`
перекрывают.

Ответ модели — JSON по схеме. LM Studio принимает `response_format` вида
`json_schema` и держит модель в рамках грамматики; DeepSeek — `json_object`,
схема ему уходит текстом в промпте. Невалидный ответ не роняет ревью: модели
показывают её же ответ и просят вернуть только JSON — один раз.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = ("local", "cloud")
TITLES = {"local": "локальная · LM Studio", "cloud": "облачная · DeepSeek"}

LOCAL_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://192.168.77.47:1234/v1")
LOCAL_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")
CLOUD_BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MODELS = {"local": LOCAL_MODEL, "cloud": CLOUD_MODEL}

PROBE_TIMEOUT = 3
TIMEOUTS = {"local": 300, "cloud": 120}
NO_THINK = "/no_think"
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
RETRIES = 1

_local = OpenAI(api_key="lm-studio", base_url=LOCAL_BASE_URL)
_cloud: OpenAI | None = None


class ModelError(RuntimeError):
    """Модель не ответила или ответила не JSON и после повтора."""


def cloud_configured() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _cloud_client() -> OpenAI:
    global _cloud
    if _cloud is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ModelError("В окружении нет DEEPSEEK_API_KEY: облачная модель недоступна.")
        _cloud = OpenAI(api_key=key, base_url=CLOUD_BASE_URL)
    return _cloud


def _client(provider: str) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Модели {provider!r} нет. Есть: {', '.join(PROVIDERS)}.")
    return _local if provider == "local" else _cloud_client()


# --- доступность и маршрут ---------------------------------------------------


@dataclass(frozen=True)
class Availability:
    local_online: bool
    local_models: tuple[str, ...]
    local_error: str | None
    local_context: int | None
    cloud_configured: bool

    @property
    def local_model_known(self) -> bool:
        return LOCAL_MODEL in self.local_models


def _loaded_context() -> int | None:
    """Окно загруженной модели из REST LM Studio — чтобы бюджеты промпта были не
    на глаз. Эндпоинт вне `/v1`, и его может не быть: тогда None."""
    root = LOCAL_BASE_URL.rsplit("/v1", 1)[0]
    try:
        payload = httpx.get(f"{root}/api/v0/models", timeout=PROBE_TIMEOUT).json()
    except Exception:  # noqa: BLE001
        return None
    for model in payload.get("data", []):
        if model.get("id") == LOCAL_MODEL:
            return model.get("loaded_context_length") or model.get("max_context_length")
    return None


def probe() -> Availability:
    try:
        page = _local.with_options(timeout=PROBE_TIMEOUT).models.list()
    except Exception as exc:  # noqa: BLE001
        return Availability(False, (), f"LM Studio {LOCAL_BASE_URL} не отвечает ({type(exc).__name__})", None, cloud_configured())
    return Availability(True, tuple(model.id for model in page.data), None, _loaded_context(), cloud_configured())


@dataclass(frozen=True)
class Route:
    provider: str
    reason: str
    forced: bool = False

    @property
    def model(self) -> str:
        return MODELS[self.provider]

    @property
    def title(self) -> str:
        return TITLES[self.provider]


def route(availability: Availability, forced: str | None = None) -> Route:
    """Доступность решает: LM Studio на связи — локально, иначе облако по ключу."""
    if forced:
        if forced == "local" and not availability.local_online:
            raise ModelError(f"Запрошена локальная модель, но {availability.local_error}.")
        if forced == "cloud" and not availability.cloud_configured:
            raise ModelError("Запрошена облачная модель, но в окружении нет DEEPSEEK_API_KEY.")
        return Route(forced, f"флаг --{forced}", forced=True)

    if availability.local_online and availability.local_model_known:
        return Route("local", f"LM Studio {LOCAL_BASE_URL} на связи")
    if availability.cloud_configured:
        if availability.local_online:
            return Route("cloud", f"в LM Studio нет модели {LOCAL_MODEL}, ревью уходит в облако")
        return Route("cloud", f"{availability.local_error}, ревью уходит в облако")
    if availability.local_online:
        raise ModelError(f"В LM Studio нет модели {LOCAL_MODEL}, а DEEPSEEK_API_KEY не задан.")
    raise ModelError(f"{availability.local_error}, и DEEPSEEK_API_KEY не задан: ревьюить некому.")


# --- вызов с JSON-ответом ----------------------------------------------------


@dataclass(frozen=True)
class Completion:
    data: dict[str, Any]
    raw: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ms: int
    attempts: int


def _for_provider(messages: list[dict[str, str]], provider: str) -> list[dict[str, str]]:
    if provider != "local":
        return messages
    copied = [dict(message) for message in messages]
    for message in copied:
        if message["role"] == "system" and NO_THINK not in message["content"]:
            message["content"] = f"{message['content']}\n{NO_THINK}"
            break
    return copied


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = FENCE.sub("", THINK.sub("", text)).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    body = cleaned[start : end + 1]
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        # Место ошибки — в сообщении: модели оно нужно, чтобы исправить, человеку —
        # чтобы понять, что именно сломалось (обычно кавычка внутри цитаты кода).
        around = body[max(0, exc.pos - 60) : exc.pos + 60].replace("\n", " ")
        raise ValueError(f"{exc.msg} на символе {exc.pos}, рядом: …{around}…") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON-ответ не объект")
    return data


def complete_json(
    messages: list[dict[str, str]],
    *,
    provider: str,
    schema: dict[str, Any],
    schema_name: str,
    max_tokens: int,
    temperature: float = 0.1,
) -> Completion:
    """Один вызов без стрима, ответ — объект по схеме. Невалидный JSON — один повтор."""
    client = _client(provider).with_options(timeout=TIMEOUTS[provider])
    if provider == "local":
        extra: dict[str, Any] = {
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        }
    else:
        extra = {"response_format": {"type": "json_object"}, "extra_body": {"thinking": {"type": "disabled"}}}

    history = _for_provider(messages, provider)
    started = time.perf_counter()
    prompt_tokens = completion_tokens = None
    raw = ""
    error = ""
    for attempt in range(1, RETRIES + 2):
        try:
            response = client.chat.completions.create(
                model=MODELS[provider],
                messages=history,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"{MODELS[provider]} не ответила: {type(exc).__name__}: {str(exc)[:200]}") from exc

        usage = response.usage
        if usage is not None:
            prompt_tokens = (prompt_tokens or 0) + (usage.prompt_tokens or 0)
            completion_tokens = (completion_tokens or 0) + (usage.completion_tokens or 0)
        if not response.choices:
            raise ModelError(f"{MODELS[provider]} вернула ответ без choices.")
        raw = response.choices[0].message.content or ""
        try:
            data = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            history = [
                *history,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Ответ не разобрался как JSON: {error}. Верни тот же ответ валидным JSON-объектом по схеме, "
                        "без текста вокруг; двойные кавычки внутри строк экранируй как \\\" или замени одинарными."
                    ),
                },
            ]
            continue
        return Completion(data, raw, prompt_tokens, completion_tokens, round((time.perf_counter() - started) * 1000), attempt)

    raise ModelError(f"{MODELS[provider]} и после повтора не вернула JSON: {error}. Начало ответа: {raw[:160]!r}")
