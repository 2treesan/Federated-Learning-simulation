# shamir.py
from __future__ import annotations
from typing import List, Tuple
import secrets
from hashlib import blake2b

# Prime lớn an toàn
P = (1 << 61) - 1  # 2^61 - 1, Mersenne prime

def _modinv(a: int, p: int = P) -> int:
    return pow(a % p, p - 2, p)

def share_secret(secret: int, n: int, t: int, p: int = P) -> List[Tuple[int, int]]:
    """Bản random (giữ lại nếu muốn thử nghiệm)."""
    secret %= p
    coeffs = [secret] + [secrets.randbelow(p) for _ in range(t - 1)]
    def f(x: int) -> int:
        res, xp = 0, 1
        for a in coeffs:
            res = (res + a * xp) % p
            xp = (xp * x) % p
        return res
    return [(i, f(i)) for i in range(1, n + 1)]

def det_share_secret(secret: int, n: int, t: int, salt: int, p: int = P) -> List[Tuple[int, int]]:
    """
    Chia sẻ Shamir *deterministic*:
    - secret: seed cơ sở (đã mod p)
    - salt: dùng để sinh hệ số đa thức cố định cho cặp (u,v)
    """
    secret %= p
    coeffs = [secret]
    for k in range(1, t):
        h = blake2b(f"{salt}|coeff|{k}".encode(), digest_size=16).digest()
        coeffs.append(int.from_bytes(h, "big") % p)

    def f(x: int) -> int:
        res, xp = 0, 1
        for a in coeffs:
            res = (res + a * xp) % p
            xp = (xp * x) % p
        return res

    return [(i, f(i)) for i in range(1, n + 1)]

def reconstruct_secret(shares: List[Tuple[int, int]], p: int = P) -> int:
    """Khôi phục f(0) từ >=t shares bằng nội suy Lagrange."""
    total = 0
    k = len(shares)
    for j in range(k):
        xj, yj = shares[j]
        num, den = 1, 1
        for m in range(k):
            if m == j: 
                continue
            xm, _ = shares[m]
            num = (num * (-xm % p)) % p      # (0 - xm)
            den = (den * (xj - xm)) % p
        total = (total + yj * num * _modinv(den, p)) % p
    return total
