"""Тонкий слой над REST LM Studio: что за модель на диске и с каким окном она загружена.

OpenAI-совместимый `/v1` ничего не знает ни о квантовании, ни о контекстном окне:
`/v1/models` отдаёт только идентификаторы. Это знает собственный REST LM Studio,
`/api/v1/models`: у каждой модели там `quantization`, `max_context_length` и список
`loaded_instances`, у каждого — `config.context_length`, с которым экземпляр реально
загружен. Там же `POST /api/v1/models/load` принимает `context_length` — так
контекстное окно задаётся из кода, а не кликом в приложении и не руками через
`lms load -c`.

Одна тонкость: `load` для уже загруженной модели не меняет окно, а поднимает второй
экземпляр `qwen3-4b:2` рядом с первым, и `/v1` запросы к `qwen3-4b` продолжают идти
в старый. Поэтому перед загрузкой с новым окном все экземпляры модели выгружаются.

Модуль не держит состояния и не решает, что делать: он отвечает на два вопроса
(«что загружено?», «загрузи с таким окном») и отдаёт ответ как есть.
"""

import os
from dataclasses import dataclass

import httpx

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").removesuffix("/v1")
TIMEOUT = httpx.Timeout(10.0, read=180.0)


@dataclass(frozen=True)
class Instance:
    id: str
    context_length: int | None
    parallel: int | None = None
    context_checkpoints: int | None = None


@dataclass(frozen=True)
class ModelInfo:
    """Модель на диске: квантование и то, с каким окном она сейчас загружена, если загружена."""

    key: str
    quantization: str | None
    bits_per_weight: float | None
    size_bytes: int | None
    max_context_length: int | None
    instances: tuple[Instance, ...]

    @property
    def loaded(self) -> bool:
        return bool(self.instances)

    @property
    def loaded_context_length(self) -> int | None:
        """Окно экземпляра, который отвечает на `/v1`-запросы по ключу модели: первого."""
        return self.instances[0].context_length if self.instances else None

    @property
    def cache(self) -> str:
        """Коротко о KV-кэше загруженного экземпляра: слоты и снимки."""
        if not self.instances:
            return ""
        first = self.instances[0]
        return f"parallel {first.parallel} · checkpoints {first.context_checkpoints}"

    def as_dict(self) -> dict[str, object]:
        first = self.instances[0] if self.instances else None
        return {
            "key": self.key,
            "quantization": self.quantization,
            "bits_per_weight": self.bits_per_weight,
            "size_gb": round(self.size_bytes / 1e9, 2) if self.size_bytes else None,
            "max_context_length": self.max_context_length,
            "loaded": self.loaded,
            "loaded_context_length": self.loaded_context_length,
            "parallel": first.parallel if first else None,
            "context_checkpoints": first.context_checkpoints if first else None,
            "instances": len(self.instances),
        }


def _parse(item: dict) -> ModelInfo:
    quant = item.get("quantization") or {}
    instances = []
    for instance in item.get("loaded_instances") or []:
        config = instance.get("config") or {}
        instances.append(
            Instance(
                id=instance.get("id", ""),
                context_length=config.get("context_length"),
                parallel=config.get("parallel"),
                context_checkpoints=config.get("context_checkpoints"),
            )
        )
    return ModelInfo(
        key=item.get("key") or item.get("id", ""),
        quantization=quant.get("name") if isinstance(quant, dict) else quant,
        bits_per_weight=quant.get("bits_per_weight") if isinstance(quant, dict) else None,
        size_bytes=item.get("size_bytes"),
        max_context_length=item.get("max_context_length"),
        instances=tuple(instances),
    )


async def models() -> list[ModelInfo]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.get("/api/v1/models")
        response.raise_for_status()
    data = response.json()
    return [_parse(item) for item in data.get("models", []) if item.get("type") == "llm"]


async def model(key: str) -> ModelInfo | None:
    for info in await models():
        if info.key == key:
            return info
    return None


async def load(
    key: str,
    *,
    context_length: int,
    parallel: int | None = None,
    context_checkpoints: int | None = None,
) -> dict:
    """Загрузить модель с заданным окном, выгрузив перед этим её прежние экземпляры.

    `parallel` — сколько запросов модель обслуживает одновременно, у каждого свой
    KV-кэш; `context_checkpoints` — сколько снимков кэша LM Studio хранит внутри
    одного диалога. По умолчанию 4 и 32. Кэш уже виденных промптов ни один из них
    не выключает — проверено: с `parallel=1, context_checkpoints=0` повторный
    промпт всё равно читается за 200 мс. Сбрасывает кэш только перезагрузка,
    поэтому `load` — это ещё и способ получить «холодные» замеры.
    """
    current = await model(key)
    if current is None:
        raise RuntimeError(f"Модели {key!r} нет в LM Studio.")
    body: dict[str, object] = {"model": key, "context_length": context_length}
    if parallel is not None:
        body["parallel"] = parallel
    if context_checkpoints is not None:
        body["context_checkpoints"] = context_checkpoints
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for instance in current.instances:
            await client.post("/api/v1/models/unload", json={"instance_id": instance.id})
        response = await client.post("/api/v1/models/load", json=body)
        response.raise_for_status()
        return response.json()
