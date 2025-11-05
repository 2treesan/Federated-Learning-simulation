# shamir.py
from __future__ import annotations
from typing import List, Tuple
import secrets

# Prime lớn an toàn cho số hạt giống 64-bit
P = (1 << 61) - 1  # 2^61-1, Mersenne prime

def _modinv(a: int, p: int = P) -> int:
    # Fermat vì p prime
    return pow(a % p, p - 2, p)

def share_secret(secret: int, n: int, t: int, p: int = P) -> List[Tuple[int, int]]:
    """Trả về danh sách (x_i, y_i) cho i=1..n."""
    secret %= p
    coeffs = [secret] + [secrets.randbelow(p) for _ in range(t - 1)]
    def f(x: int) -> int:
        res = 0
        xp = 1
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
