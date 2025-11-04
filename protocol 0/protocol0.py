# protocol0.py
from __future__ import annotations
import json
import os
import time
from hashlib import blake2b
from typing import List, Tuple
import numpy as np

_REGISTRY_FILE = "p0_registry.json"
_SECRET_KEY = "P0_DEMO_SECRET_V1"

def _hash_to_uint64(s: str) -> int:
    h = blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)

def _rng_from_tuple(*items) -> np.random.Generator:
    text = "|".join(str(x) for x in items) + "|" + _SECRET_KEY
    seed = _hash_to_uint64(text)
    return np.random.default_rng(seed)

def _flatten_params(nds: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]], List[np.dtype]]:
    shapes = [arr.shape for arr in nds]
    dtypes = [arr.dtype for arr in nds]
    flat = np.concatenate([arr.reshape(-1).astype(np.float32, copy=False) for arr in nds], axis=0)
    return flat, shapes, dtypes

def _unflatten_params(flat: np.ndarray, shapes: List[Tuple[int, ...]], dtypes: List[np.dtype]) -> List[np.ndarray]:
    out = []
    idx = 0
    for shp, dt in zip(shapes, dtypes):
        size = int(np.prod(shp))
        part = flat[idx: idx + size].reshape(shp).astype(dt, copy=False)
        out.append(part)
        idx += size
    return out

def _preview(nds: List[np.ndarray], k: int = 6) -> str:
    vec, _, _ = _flatten_params(nds)
    k = min(k, vec.size)
    with np.printoptions(precision=5, suppress=True):
        return np.array2string(vec[:k], separator=", ")

def assign_rank(num_clients: int = 3) -> int:
    if not os.path.exists(_REGISTRY_FILE):
        reg = {"assigned": [], "ts": time.time()}
        with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(reg, f)
    for _ in range(20):
        try:
            with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
                reg = json.load(f)
            assigned = set(reg.get("assigned", []))
            candidates = [r for r in range(num_clients) if r not in assigned]
            if not candidates:
                assigned = set()
                candidates = [0]
            rank = min(candidates)
            reg["assigned"] = sorted(list(assigned | {rank}))
            with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(reg, f)
            return rank
        except Exception:
            time.sleep(0.05)
    return 0

class Protocol0Masker:
    def __init__(self, rank: int, num_clients: int = 3):
        assert 0 <= rank < num_clients, "rank không hợp lệ"
        self.rank = rank
        self.n = num_clients
        self._round = 0

    def next_round(self) -> int:
        self._round += 1
        return self._round

    def mask_parameters(self, nds: List[np.ndarray], round_id: int) -> List[np.ndarray]:
        flat, shapes, dtypes = _flatten_params(nds)
        total_len = flat.size
        total_mask = np.zeros_like(flat, dtype=np.float32)
        for j in range(self.n):
            if j == self.rank:
                continue
            i, k = min(self.rank, j), max(self.rank, j)
            rng = _rng_from_tuple("pair", i, k, "round", round_id)
            pair_mask = rng.standard_normal(total_len, dtype=np.float32)
            if self.rank == i:
                total_mask += pair_mask
            else:
                total_mask -= pair_mask
        masked = flat + total_mask
        return _unflatten_params(masked, shapes, dtypes)

def preview_params(nds: List[np.ndarray], k: int = 6) -> str:
    return _preview(nds, k=k)

def l2_diff(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    fa, _, _ = _flatten_params(a)
    fb, _, _ = _flatten_params(b)
    return float(np.linalg.norm(fa - fb))

def flatten_list(nds: List[np.ndarray]) -> np.ndarray:
    flat, _, _ = _flatten_params(nds)
    return flat
