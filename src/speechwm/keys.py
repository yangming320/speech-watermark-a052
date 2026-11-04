from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .utils import write_json


def validate_key(values: np.ndarray, size: int = 63) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (size,) or not np.isin(values, [-1, 1]).all():
        raise ValueError(f"Watermark must contain {size} elements, each -1 or +1")
    return values.astype(np.int8)


def key_hash(values: np.ndarray) -> str:
    return hashlib.sha256(validate_key(values).tobytes()).hexdigest()


def make_key(seed: int = 20261017) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=63)


def save_key(path: str | Path, seed: int = 20261017) -> dict:
    values = make_key(seed)
    record = {"seed": seed, "bit_generator": "PCG64", "length": 63,
              "values": values.tolist(), "sha256_int8": key_hash(values)}
    write_json(path, record)
    return record


def load_key(path: str | Path) -> np.ndarray:
    record = json.loads(Path(path).read_text())
    values = validate_key(record["values"])
    if key_hash(values) != record["sha256_int8"]:
        raise ValueError("Watermark sequence hash mismatch")
    return values

