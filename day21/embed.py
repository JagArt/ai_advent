"""Эмбеддинги локальной моделью: тот же токенайзер режет чанки и считает векторы.

Ни DeepSeek, ни OpenRouter эндпоинта эмбеддингов не дают, поэтому вектора считает
модель на машине. Выбрана `model2vec` — статические эмбеддинги: у неё вместо
прогона трансформера обычный просмотр таблицы и усреднение, ей не нужен torch, и
она разом снимает вопрос про стоимость индексации. Из моделей взята
`potion-multilingual-128M`, потому что корпус и запросы здесь по-русски, а базовая
`potion-base-8M` знает только английский.

Модель грузится при первом обращении: страница, CLI и генератор вопросов
импортируют этот модуль, а 512 МБ из кеша разворачиваются секунды. Векторы сразу
нормируются по L2, поэтому косинусная близость в поиске — это просто скалярное
произведение, и матрицу можно перемножить одним вызовом numpy.
"""

import time
from dataclasses import dataclass
from functools import cache

import numpy as np

MODEL_NAME = "minishlab/potion-multilingual-128M"
BATCH = 256


@cache
def model():
    """Модель одна на процесс: 512 МБ таблицы векторов незачем держать дважды."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained(MODEL_NAME)


@cache
def dimensions() -> int:
    return int(model().dim)


class ModelTokenizer:
    """Границы токенов и их число — тем же токенайзером, что считает эмбеддинги.

    Мерить чанки сторонней линейкой (символами или `tiktoken`) здесь не за чем:
    важно, сколько токенов увидит именно эта модель.
    """

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = model().tokenizer.encode(text, add_special_tokens=False)
        return [
            (start, end) for start, end in encoded.offsets if end > start
        ]

    def count(self, text: str) -> int:
        return len(model().tokenizer.encode(text, add_special_tokens=False).ids)


@dataclass(frozen=True)
class Encoded:
    """Матрица векторов и время, которое она стоила: время идёт в сравнение стратегий."""

    vectors: np.ndarray
    seconds: float

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Пустой вектор бывает у чанка из одних знаков препинания: делить на ноль нельзя.
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def encode(texts: list[str]) -> Encoded:
    """Векторы для списка текстов, нормированные, в порядке входа."""
    if not texts:
        return Encoded(np.zeros((0, dimensions()), dtype=np.float32), 0.0)

    started = time.perf_counter()
    vectors = model().encode(texts, batch_size=BATCH, show_progress_bar=False)
    return Encoded(_normalize(np.asarray(vectors)), time.perf_counter() - started)


def encode_one(text: str) -> np.ndarray:
    """Вектор запроса: та же модель и та же нормировка, что у чанков."""
    return encode([text]).vectors[0]


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blobs(blobs: list[bytes]) -> np.ndarray:
    """Блобы из SQLite обратно в матрицу `(N, dim)` одним куском памяти."""
    if not blobs:
        return np.zeros((0, dimensions()), dtype=np.float32)

    return np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), -1)
