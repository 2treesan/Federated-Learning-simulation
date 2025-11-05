# protocol1.py
from __future__ import annotations
import json, os, time
from hashlib import blake2b
from typing import List, Tuple
import numpy as np

PRIME = (1 << 61) - 1
_SECRET_PAIR = "P1_PAIR_SEED_V1"

def _h64(*items) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    return int.from_bytes(blake2b(s, digest_size=8).digest(), "big")

def _rng_from_int(seed_int: int) -> np.random.Generator:
    return np.random.default_rng(seed_int & ((1 << 63) - 1))

def pair_base_seed(a: int, b: int) -> int:
    a, b = (a, b) if a < b else (b, a)
    return _h64("pair_base", a, b, _SECRET_PAIR) % PRIME

def seed_from_base(base_seed: int, round_id: int) -> int:
    return _h64("pair_round", base_seed, round_id) % PRIME

def mask_vec_from_seed(total_len: int, seed_int: int) -> np.ndarray:
    rng = _rng_from_int(seed_int)
    return rng.standard_normal(total_len, dtype=np.float32)

def flatten_params(nds: List[np.ndarray]) -> tuple[np.ndarray, List[tuple[int, ...]], List[np.dtype]]:
    shapes = [a.shape for a in nds]
    dtypes = [a.dtype for a in nds]
    flat = np.concatenate([a.reshape(-1).astype(np.float32, copy=False) for a in nds], 0)
    return flat, shapes, dtypes

def unflatten_params(flat: np.ndarray, shapes: List[tuple[int, ...]], dtypes: List[np.dtype]) -> List[np.ndarray]:
    out, i = [], 0
    for shp, dt in zip(shapes, dtypes):
        sz = int(np.prod(shp))
        out.append(flat[i:i+sz].reshape(shp).astype(dt, copy=False))
        i += sz
    return out

def headtail_str(arr: np.ndarray | None, k1: int = 5, k2: int = 5, nd: int = 6) -> str:
    if arr is None: return "[]"
    v = np.asarray(arr, dtype=np.float32).ravel()
    if nd is not None: v = np.round(v, nd)
    if v.size <= k1 + k2:
        elems = [f"{float(x):.{nd}f}" for x in v]
    else:
        elems = [f"{float(x):.{nd}f}" for x in v[:k1]] + ['"..."'] + [f"{float(x):.{nd}f}" for x in v[-k2:]]
    return "[" + ", ".join(elems) + "]"

# Rank registry: cấp tuần tự 0,1,2,...
_REG = "p0_registry.json"
def assign_rank(_n_ignored: int | None = None) -> int:
    if not os.path.exists(_REG):
        with open(_REG, "w", encoding="utf-8") as f:
            json.dump({"assigned": [], "ts": time.time()}, f)
    for _ in range(50):
        try:
            with open(_REG, "r", encoding="utf-8") as f:
                reg = json.load(f)
            used = set(reg.get("assigned", []))
            r = 0
            while r in used:
                r += 1
            reg["assigned"] = sorted(list(used | {r}))
            with open(_REG, "w", encoding="utf-8") as f:
                json.dump(reg, f)
            return r
        except Exception:
            time.sleep(0.05)
    return 0
