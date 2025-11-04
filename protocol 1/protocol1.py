# protocol1.py
from __future__ import annotations
import json
import os
import time
from hashlib import blake2b
from typing import List, Tuple, Dict
import numpy as np

# ====== Thiết lập chung ======
_SECRET_KEY_P1 = "P1_DEMO_SECRET_V1"  # bí mật nội bộ (demo) để tạo PRG theo cặp/round

def _hash_to_uint64(s: str) -> int:
    h = blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)

def _rng_from_tuple(*items) -> np.random.Generator:
    text = "|".join(str(x) for x in items) + "|" + _SECRET_KEY_P1
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

def preview_params(nds: List[np.ndarray], k: int = 6) -> str:
    flat, _, _ = _flatten_params(nds)
    k = min(k, flat.size)
    with np.printoptions(precision=5, suppress=True):
        return np.array2string(flat[:k], separator=", ")

def l2_diff(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    fa, _, _ = _flatten_params(a)
    fb, _, _ = _flatten_params(b)
    return float(np.linalg.norm(fa - fb))

def flatten_list(nds: List[np.ndarray]) -> np.ndarray:
    flat, _, _ = _flatten_params(nds)
    return flat

# ====== Cấp phát rank (dùng chung file như P0) ======
_REGISTRY_FILE = "p0_registry.json"  # tái dùng cho tiện
def assign_rank(num_clients: int) -> int:
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

# ====== Protocol 1 Masker ======
class Protocol1Masker:
    """
    - Tương tự P0: mỗi cặp (i,j) tạo một mặt nạ p_{i,j} = - p_{j,i} theo PRG(pair, round).
    - Ngoài ra, client i xuất "contribution" cho từng peer j (vector mask mà i đã cộng, có dấu),
      để phía server có thể cộng dồn contributions của các client sống sót và KHỬ phần mask
      phát sinh do các client rớt (dropouts).
    """
    def __init__(self, rank: int, num_clients: int):
        assert 0 <= rank < num_clients
        self.rank = rank
        self.n = num_clients
        self._round = 0

    def next_round(self) -> int:
        self._round += 1
        return self._round

    def _pair_mask_vec(self, total_len: int, i: int, j: int, round_id: int) -> np.ndarray:
        """Tạo vector mặt nạ cho cặp {i,j} theo PRG; i<->j đối xứng, chỉ khác dấu ở phía người dùng."""
        a, b = (i, j) if i < j else (j, i)
        rng = _rng_from_tuple("P1_pair", a, b, "round", round_id)
        return rng.standard_normal(total_len, dtype=np.float32)

    def mask_and_contribs(self, nds: List[np.ndarray], round_id: int, roster: List[int]) -> Tuple[List[np.ndarray], Dict[int, List[float]]]:
        """Trả về (tham số đã mask, contributions theo peer).
        contributions[j] là vector đã được i cộng vào (có dấu), để server dùng khi j bị dropout.
        """
        flat, shapes, dtypes = _flatten_params(nds)
        total_len = flat.size
        total_mask = np.zeros_like(flat, dtype=np.float32)
        contribs: Dict[int, List[float]] = {}

        for j in roster:
            if j == self.rank:
                continue
            base = self._pair_mask_vec(total_len, self.rank, j, round_id)
            sign = +1.0 if self.rank < j else -1.0
            vec = sign * base
            total_mask += vec
            # Lưu "contribution" cho peer j
            contribs[j] = vec.astype(np.float32).tolist()

        masked = flat + total_mask
        return _unflatten_params(masked, shapes, dtypes), contribs
