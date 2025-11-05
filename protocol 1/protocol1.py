# protocol1.py
from __future__ import annotations
import json, os, time
from hashlib import blake2b
from typing import List, Tuple, Dict
import numpy as np
from shamir import share_secret, reconstruct_secret, P as PRIME

_SECRET_PAIR = "P1_PAIR_SEED_V1"  # demo: hằng chung; paper: ECDH/HKDF

def _h64(*items) -> int:
    s = "|".join(map(str, items)).encode()
    return int.from_bytes(blake2b(s, digest_size=8).digest(), "big")

def _rng_from_seed(seed_int: int) -> np.random.Generator:
    return np.random.default_rng(seed_int & ((1<<63)-1))

def pair_seed(a: int, b: int, round_id: int) -> int:
    a, b = (a, b) if a < b else (b, a)
    return _h64("pair", a, b, "round", round_id, _SECRET_PAIR) % PRIME

def mask_vec_from_seed(total_len: int, seed_int: int) -> np.ndarray:
    rng = _rng_from_seed(seed_int)
    return rng.standard_normal(total_len, dtype=np.float32)

def flatten_params(nds: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]], List[np.dtype]]:
    shapes = [a.shape for a in nds]
    dtypes = [a.dtype for a in nds]
    flat = np.concatenate([a.reshape(-1).astype(np.float32, copy=False) for a in nds], axis=0)
    return flat, shapes, dtypes

def unflatten_params(flat: np.ndarray, shapes: List[Tuple[int, ...]], dtypes: List[np.dtype]) -> List[np.ndarray]:
    out, i = [], 0
    for shp, dt in zip(shapes, dtypes):
        sz = int(np.prod(shp))
        out.append(flat[i:i+sz].reshape(shp).astype(dt, copy=False))
        i += sz
    return out

def headtail_str(arr: np.ndarray, k1=5, k2=5, nd=6) -> str:
    if arr is None: return "[]"
    v = np.asarray(arr, dtype=np.float32).ravel()
    if nd is not None: v = np.round(v, nd)
    if v.size <= k1 + k2:
        elems = [f"{float(x):.{nd}f}" for x in v]
    else:
        elems = [f"{float(x):.{nd}f}" for x in v[:k1]] + ['"..."'] + [f"{float(x):.{nd}f}" for x in v[-k2:]]
    return "[" + ", ".join(elems) + "]"

# Rank registry (tái dùng)
_REG = "p0_registry.json"
def assign_rank(n: int) -> int:
    if not os.path.exists(_REG):
        with open(_REG, "w", encoding="utf-8") as f:
            json.dump({"assigned": [], "ts": time.time()}, f)
    for _ in range(20):
        try:
            with open(_REG, "r", encoding="utf-8") as f:
                reg = json.load(f)
            used = set(reg.get("assigned", []))
            cand = [r for r in range(n) if r not in used] or [0]
            rnk = min(cand)
            reg["assigned"] = sorted(list(used | {rnk}))
            with open(_REG, "w", encoding="utf-8") as f:
                json.dump(reg, f)
            return rnk
        except Exception:
            time.sleep(0.05)
    return 0
