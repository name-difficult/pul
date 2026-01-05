import numpy as np
import secrets

from KarneySampler import karney_sampler

def secure_rng():
    # 如果 trapGen 会被多次调用，建议把 rng 放到外面复用（见下方备注）
    seed = secrets.randbits(256)
    return np.random.default_rng(seed)

def gen_G_Matrix_fast(n: int, k: int) -> np.ndarray:
    """G = I_n ⊗ g^T, g=(1,2,4,...,2^{k-1})  shape=(n, n*k)"""
    g = (1 << np.arange(k, dtype=np.int64))          # (k,)
    G = np.zeros((n, n * k), dtype=np.int64)
    for i in range(n):
        G[i, i * k:(i + 1) * k] = g
    return G

def sample_R_fast(bar_m: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    R entries: -1 with prob 1/4, +1 with prob 1/4, 0 with prob 1/2
    用整数采样避免生成巨大 float 矩阵
    """
    u = rng.integers(0, 4, size=(bar_m, w), dtype=np.uint8)  # {0,1,2,3}
    R = np.zeros((bar_m, w), dtype=np.int64)
    R[u == 0] = -1
    R[u == 1] = 1
    # u==2 or 3 -> 0
    return R

def trapGen_fast(n: int, k: int, q: int, rng: np.random.Generator | None = None):
    w = n * k
    bar_m = n * k
    if rng is None:
        rng = secure_rng()

    # G / HG（这里 H=I，所以 HG=G）
    G = gen_G_Matrix_fast(n, k)     # (n, w)
    HG = G % q

    # bar_A, R
    bar_A = rng.integers(0, q, size=(n, bar_m), dtype=np.int64)
    R = sample_R_fast(bar_m, w, rng)  # (bar_m, w), entries in {-1,0,1}

    # 关键加速点：不做 (R % q) 的大矩阵拷贝
    AR = (bar_A @ R) % q
    A_right = (HG - AR) % q
    A = np.hstack([bar_A, A_right]).astype(np.int64)

    return A, R

def disc_gauss_round(shape, s: float, rng: np.random.Generator) -> np.ndarray:
    """Teaching version: round(N(0, s^2)) -> Z."""
    return np.rint(rng.normal(0.0, s, size=shape)).astype(np.int64)

def Sk_for_power_of_two(k: int) -> np.ndarray:
    S = np.zeros((k, k), dtype=np.int64)
    for i in range(k - 1):
        S[i, i] = 2
        S[i, i + 1] = -1
    S[k - 1, k - 1] = 2
    return S

def bitdecomp_u(u: int, k: int) -> np.ndarray:
    return np.array([(u >> i) & 1 for i in range(k)], dtype=np.int64)

def sample_preimage_gadget_block(S: np.ndarray, v_i: int, k: int, s_t: float, rng: np.random.Generator, q: int) -> np.ndarray:
    z0 = bitdecomp_u(int(v_i) % q, k)          # g^T z0 = v_i (as integer)
    # t = karney_sampler(s_t, k)
    t = disc_gauss_round((k,), s_t, rng)
    z = z0 + (S.T @ t)                         # KEY FIX
    return z.astype(np.int64)

def sample_preimage_G(v: np.ndarray, n: int, k: int, s_t: float, rng: np.random.Generator, q: int) -> np.ndarray:
    S = Sk_for_power_of_two(k)
    v = np.asarray(v, dtype=np.int64) % q
    blocks = [sample_preimage_gadget_block(S, int(v[i]), k, s_t, rng, q) for i in range(n)]
    return np.concatenate(blocks, axis=0).astype(np.int64)  # length w=nk

def sampleD_34(A: np.ndarray, R: np.ndarray, q: int, u: np.ndarray, s_p: float, s_t: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.int64)
    R = np.asarray(R, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64) % q

    rng = secure_rng()

    n, m = A.shape
    bar_m, w = R.shape

    I_w = np.eye(w, dtype=np.int64)
    Rbar = np.vstack([R, I_w])               # (m, w)

    # 1) sample p
    # p = karney_sampler(s_p, m)
    p = disc_gauss_round((m,), s_p, rng)

    # 2) v = u - A p (mod q)
    v = (u - (A @ (p % q)) % q) % q

    # 3) sample z with G z = v (mod q)
    k = w // n
    z = sample_preimage_G(v, n=n, k=k, s_t=s_t, rng=rng, q=q)

    # 4) y = [R;I] z
    y = (Rbar @ z).astype(np.int64)

    # 5) x = p + y
    x = (p + y).astype(np.int64)

    return x

if __name__ == "__main__":
    n = 200
    k = 12
    q = 2**k

    A, R = trapGen_fast(n, k, q)

    print(A)
    # w = n * k
    # bar_m = n * k
    # Iw = np.eye(w, dtype=np.int64)
    # F = np.vstack([R % q, Iw]) % q
    #
    # ok = np.all((A @ F) % q == (G % q))
    # print("verify A*[R;I] == G (mod q):", ok)

    rng = secure_rng()

    u = rng.integers(0, q, size=(n,), dtype=np.int64)
    print(u)
    x = sampleD_34(A, R, q, u, s_p=2.0, s_t=2.0)
    print(x)

    Ax_mod_q = (A @ (x % q)) % q
    print("verify A x ≡ u (mod q):", np.array_equal(Ax_mod_q, u % q))

