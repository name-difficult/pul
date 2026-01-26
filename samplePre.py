import time
import math
from typing import Optional, Tuple

import numpy as np
from sympy import prevprime, isprime

def get_secure_param(n: int, k: int):
    q = int(prevprime(1 << k))   # prime close to 2^k
    w = n * k
    bar_m = 2 * n
    m = bar_m + w
    return n, k, q, w, bar_m, m

def get_secure_param_only_n(n: int, *, min_bits: int = 14):
    """
    Compute RLWE/Trapdoor-friendly parameters.

    Constraints:
      - n is a power of two
      - q is prime
      - q ≡ 1 (mod 2n)
      - k = ceil(log2 q)
      - w = n*k
      - bar_m = 2n
      - m = bar_m + w

    Args:
        n: ring dimension (power of two)
        min_bits: minimum bit-length lower bound for q (default 14).
                  Increase if you want a larger modulus.

    Returns:
        (n, k, q, w, bar_m, m)
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two.")

    modulus = 2 * n
    start_q = 1 << min_bits

    # smallest t s.t. q = t*(2n) + 1 >= start_q
    t = (start_q - 1 + modulus - 1) // modulus

    while True:
        q = t * modulus + 1
        if isprime(q):
            k = math.ceil(math.log2(q))
            w = n * k
            bar_m = 2 * n
            m = bar_m + w
            return n, k, q, w, bar_m, m
        t += 1

def get_secure_param_only_n_min(n: int, *, min_bits: int | None = None, add_bits: int = 0):
    """
    If min_bits is None, use min_bits = log2(n) + add_bits (since n is power of two).
    If you set add_bits=0, this is exactly your idea: min_bits = k_n.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two.")

    k_n = n.bit_length() - 1  # since n is power of two, this equals log2(n)

    if min_bits is None:
        min_bits = k_n + add_bits

    modulus = 2 * n
    start_q = 1 << min_bits

    t = (start_q - 1 + modulus - 1) // modulus

    while True:
        q = t * modulus + 1
        if isprime(q):
            k = math.ceil(math.log2(q))
            w = n * k
            bar_m = 2 * n
            m = bar_m + w
            return n, k, q, w, bar_m, m
        t += 1


def gen_gadget_G(n: int, k: int, q: int) -> np.ndarray:
    """
    G = I_n ⊗ g^T (mod q),  g=(1,2,4,...,2^{k-1})
    Shape: (n, n*k)
    """
    if n <= 0 or k <= 0 or q <= 1:
        raise ValueError("Require n>0, k>0, q>1.")

    g = (1 << np.arange(k, dtype=np.int64)) % np.int64(q)  # (k,)
    I = np.eye(n, dtype=np.int64)
    G = np.kron(I, g.reshape(1, k)).astype(np.int64) % np.int64(q)
    return G


def build_Sk_from_G(G: np.ndarray, q: int) -> np.ndarray:
    """
    Construct S_k (k x k) for g=(1,2,4,...,2^{k-1}):
      - columns 0..k-2: b_i = 2 e_i - e_{i+1}
      - last column    : b_k = q e_1
    """
    G = np.asarray(G)
    if G.ndim != 2:
        raise ValueError("G must be a 2D matrix.")
    n, w = G.shape
    if n <= 0 or w <= 0 or (w % n) != 0:
        raise ValueError("G must have shape (n, n*k) with w divisible by n.")
    k = w // n
    if k < 2:
        raise ValueError("Need k >= 2.")
    if q is None or q <= 1:
        raise ValueError("q must be provided and >= 2.")

    S_k = np.zeros((k, k), dtype=np.int64)
    for i in range(k - 1):
        S_k[i, i] = 2
        S_k[i + 1, i] = -1
    S_k[0, k - 1] = int(q)
    return S_k

def _matvec_mod_q_fast(A: np.ndarray, x: np.ndarray, q: int) -> np.ndarray:
    """
    Fast mat-vec modulo q using int64 path.
    Assumes values are in int64 range (true for your parameter regime).
    """
    q64 = np.int64(q)
    A = np.asarray(A, dtype=np.int64) % q64
    x = np.asarray(x, dtype=np.int64) % q64
    return (A @ x) % q64

def sample_R(
        bar_m: int,
        w: int,
        sigma: float,
        mu: float = 0.0,
        seed: Optional[int] = None
) -> np.ndarray:
    """
    R_ij = round(N(mu, sigma^2))
    """
    if bar_m <= 0 or w <= 0:
        raise ValueError("bar_m and w must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    rng = np.random.default_rng(seed)
    R = np.rint(rng.normal(loc=mu, scale=sigma, size=(bar_m, w))).astype(np.int64)
    return R


def sample_p(
        m: int,
        s: float,
        *,
        c: float = 1.0,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    p ≈ round( N(0, (c*s)^2 I_m) )
    """
    if rng is None:
        rng = np.random.default_rng()
    if m <= 0:
        raise ValueError("m must be positive.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if c < 1.0:
        raise ValueError("c must be >= 1.0.")

    sigma_p = c * s
    p_float = rng.normal(loc=0.0, scale=sigma_p, size=m)
    return np.rint(p_float).astype(np.int64)

def gen_trapdoor_G_trapdoor(
        n: int,
        q: int,
        k: int,
        bar_m: int,
        sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate:
      G = I_n ⊗ g^T
      S_k basis
      A = [Abar | G - Abar R] (mod q)
      R ~ round(N(0,sigma^2))
    Returns (A, R, G, S_k).
    """
    if n <= 0 or q <= 1 or bar_m <= 0:
        raise ValueError("Require n>0, q>1, bar_m>0.")
    if not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be a scalar.")
    if sigma <= 0:
        raise ValueError("Require sigma > 0.")

    q64 = np.int64(q)

    G = gen_gadget_G(n, k, q)
    S_k = build_Sk_from_G(G, q)

    w = G.shape[1]
    m = bar_m + w

    rng = np.random.default_rng()
    Abar = rng.integers(0, q, size=(n, bar_m), dtype=np.int64)

    R = sample_R(bar_m, w, sigma)

    # Fast int64 multiplication then mod q
    # (R entries are small, no need R%q beforehand)
    AbarR = (Abar @ R) % q64
    A1 = (G - AbarR) % q64

    A = np.concatenate([Abar % q64, A1], axis=1) % q64
    if A.shape != (n, m):
        raise RuntimeError("Internal shape error constructing A.")

    return A.astype(np.int64), R.astype(np.int64), G.astype(np.int64), S_k.astype(np.int64)

def samplePre(
        A: np.ndarray,
        R: np.ndarray,
        u: np.ndarray,
        S_k: np.ndarray,
        q: int,
        *,
        s: float = 3.0,
        c: float = 1.0,
        sigma_t: float = 3.0,
) -> np.ndarray:
    """
    Optimized prototype SampleD-like flow (engineering version).
    Goal: return x s.t. A x ≡ u (mod q) (validated by verify_preimage).
    """
    rng = np.random.default_rng()

    A = np.asarray(A, dtype=np.int64)
    R = np.asarray(R, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64)

    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    n, m = A.shape
    if u.shape != (n,):
        raise ValueError(f"u must have shape (n,), got {u.shape}.")
    if R.ndim != 2:
        raise ValueError("R must be 2D.")
    if q <= 1:
        raise ValueError("q must be >= 2.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if c < 1.0:
        raise ValueError("c must be >= 1.0.")

    bar_m, w = R.shape
    if m != bar_m + w:
        raise ValueError(f"Dimension mismatch: A has m={m}, but R implies bar_m+w={bar_m+w}.")
    if w % n != 0:
        raise ValueError(f"w must be divisible by n; got w={w}, n={n}.")
    k = w // n

    S_k = np.asarray(S_k, dtype=np.int64)
    if S_k.shape != (k, k):
        raise ValueError(f"S_k must have shape (k,k)=({k},{k}), got {S_k.shape}.")

    q64 = np.int64(q)

    # 1) sample p
    p = sample_p(m=m, s=s, c=c, rng=rng).astype(np.int64)

    # 2) v = u - A p (mod q)
    Ap = _matvec_mod_q_fast(A, p, q64).astype(np.int64)
    v = (u % q64 - Ap) % q64  # (n,)

    # 3) Build z block-wise but vectorized:
    #    z0 = binary expansion of each v[i] (little-endian), shape (n,k)
    shifts = np.arange(k, dtype=np.int64)  # (k,)
    z0 = ((v[:, None] >> shifts[None, :]) & 1).astype(np.int64)  # (n,k)

    #    t sampled as round(N(0,sigma_t^2)), shape (n,k)
    if sigma_t is None or sigma_t <= 0.0:
        t = np.zeros((n, k), dtype=np.int64)
    else:
        t = np.rint(rng.normal(loc=0.0, scale=float(sigma_t), size=(n, k))).astype(np.int64)

    #    z_blocks = z0 + S_k t_i  -> vectorize as t @ S_k^T
    z_blocks = z0 + (t @ S_k.T)  # (n,k)
    z = z_blocks.reshape(w).astype(np.int64)

    # 4) y = [R; I] z where y0 = R z (mod q), y1 = z
    y0 = _matvec_mod_q_fast(R, z, q64).astype(np.int64)  # (bar_m,)
    y = np.concatenate([y0, z], axis=0).astype(np.int64)  # (m,)

    # 5) x = p + y
    x = (p + y).astype(np.int64)
    return x

def ring_samplePre(
        A: np.ndarray,
        R: np.ndarray,
        U: np.ndarray,
        S_k: np.ndarray,
        q: int,
        *,
        s: float = 3.0,
        c: float = 1.0,
        sigma_t: float = 3.0
) -> np.ndarray:
    """
    Batched SamplePre (engineering version).
    U: (n,l)  -> X: (m,l) with A@X == U (mod q)
    """
    rng = np.random.default_rng()

    A = np.asarray(A, dtype=np.int64)
    R = np.asarray(R, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    S_k = np.asarray(S_k, dtype=np.int64)

    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    if R.ndim != 2:
        raise ValueError("R must be 2D.")
    if U.ndim != 2:
        raise ValueError("U must be 2D with shape (n,l).")
    if q <= 1:
        raise ValueError("q must be >= 2.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if c < 1.0:
        raise ValueError("c must be >= 1.0.")

    n, m = A.shape
    if U.shape[0] != n:
        raise ValueError(f"U must have shape (n,l); got {U.shape}, expected first dim n={n}.")
    l = U.shape[1]

    bar_m, w = R.shape
    if m != bar_m + w:
        raise ValueError(f"Dimension mismatch: A has m={m}, but R implies bar_m+w={bar_m+w}.")
    if w % n != 0:
        raise ValueError(f"w must be divisible by n; got w={w}, n={n}.")
    k = w // n

    if S_k.shape != (k, k):
        raise ValueError(f"S_k must have shape (k,k)=({k},{k}), got {S_k.shape}.")

    q64 = np.int64(q)

    # 1) Sample P: (m,l)
    sigma_p = float(c * s)
    P = np.rint(rng.normal(loc=0.0, scale=sigma_p, size=(m, l))).astype(np.int64)

    # 2) V = U - A P (mod q): (n,l)
    AP = (A @ (P % q64)) % q64
    V = (U % q64 - AP) % q64

    # 3) z0 bit-decomp: (n,k,l)
    shifts = np.arange(k, dtype=np.int64)  # (k,)
    z0 = ((V[:, None, :] >> shifts[None, :, None]) & 1).astype(np.int64)  # (n,k,l)

    # 4) t: (n,k,l)
    if sigma_t is None or sigma_t <= 0.0:
        t = np.zeros((n, k, l), dtype=np.int64)
    else:
        t = np.rint(rng.normal(loc=0.0, scale=float(sigma_t), size=(n, k, l))).astype(np.int64)

    # 5) z_blocks = z0 + t @ S_k^T  (batched over (n,l))
    #    Do: (n,l,k) @ (k,k) -> (n,l,k) then transpose back to (n,k,l)
    t_nlk = np.transpose(t, (0, 2, 1))                 # (n,l,k)
    tSk_nlk = t_nlk @ S_k.T                             # (n,l,k)
    tSk = np.transpose(tSk_nlk, (0, 2, 1))              # (n,k,l)

    z_blocks = z0 + tSk                                 # (n,k,l)
    Z = z_blocks.reshape(w, l).astype(np.int64)          # (w,l)

    # 6) Y = [R; I] Z : (m,l)
    Y0 = (R @ (Z % q64)) % q64                           # (bar_m,l)
    Y = np.vstack([Y0, Z % q64]).astype(np.int64)        # (m,l)

    # 7) X = P + Y
    X = (P + Y).astype(np.int64)
    return X

def deltrap_HI(
        A: np.ndarray,
        R: np.ndarray,
        A1: np.ndarray,
        G: np.ndarray,
        S_k: np.ndarray,
        q: int,
        *,
        s_prime: float = 3.0,
        c: float = 1.0,
        sigma_t: float = 3.0,
        center: bool = False,
        progress_every: int = 0,
) -> np.ndarray:
    """
    Engineering DelTrap for H' = I_n using samplePre as the "oracle".

    Goal:
        Return R' (m, w) such that
            A @ R' ≡ (G - A1) (mod q)

    This replaces the ideal coset discrete Gaussian oracle in Alg.3
    with your current samplePre implementation.

    Parameters
    ----------
    A : (n, m)
    R : (bar_m, w)   G-trapdoor of A (for samplePre)
    A1: (n, w)       extension block
    G : (n, w)       gadget matrix
    S_k: (k, k)      basis for Λ⊥(g^T)
    q : modulus (prime recommended)

    s_prime, c, sigma_t : forwarded into samplePre
    center : whether to output centered representatives in [-q/2, q/2)
    progress_every : if >0, print progress every this many columns

    Returns
    -------
    R_prime : (m, w) integer matrix (mod q) satisfying A R' = G - A1 (mod q)
    """
    if q <= 1:
        raise ValueError("q must be >= 2.")
    q64 = np.int64(q)

    A  = np.asarray(A,  dtype=np.int64) % q64
    A1 = np.asarray(A1, dtype=np.int64) % q64
    G  = np.asarray(G,  dtype=np.int64) % q64
    R  = np.asarray(R,  dtype=np.int64)  # samplePre expects int64 (not modded necessarily)
    S_k = np.asarray(S_k, dtype=np.int64)

    n, m = A.shape
    if A1.shape[0] != n:
        raise ValueError("A1 must have n rows.")
    if G.shape != A1.shape:
        raise ValueError("G must have same shape as A1.")
    w = A1.shape[1]

    # U = G - A1  (n, w)
    U = (G - A1) % q64

    # Allocate R'
    R_prime = np.empty((m, w), dtype=np.int64)

    # Column-by-column: r'_j = samplePre(A, R, u_j, ...)
    # so that A r'_j ≡ u_j (mod q)
    for j in range(w):
        u_j = U[:, j].copy()  # shape (n,)
        r_j = samplePre(
            A=A,
            R=R,
            u=u_j,
            S_k=S_k,
            q=q,
            s=s_prime,
            c=c,
            sigma_t=sigma_t,
        ).astype(np.int64)

        # Store mod-q representative
        R_prime[:, j] = r_j % q64

        # if progress_every and ((j + 1) % progress_every == 0):
        #     print(f"[deltrap_HI_using_samplePre] done {j+1}/{w} columns")

    if center:
        half = q // 2
        R_prime = ((R_prime + half) % q64) - half

    return R_prime

def samplePre_batch(
        A: np.ndarray,
        R: np.ndarray,
        U: np.ndarray,          # shape (n, B)
        S_k: np.ndarray,
        q: int,
        *,
        s: float = 3.0,
        c: float = 1.0,
        sigma_t: float = 3.0,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Batch version of samplePre.

    Inputs:
      A: (n, m)
      R: (bar_m, w)
      U: (n, B)   syndromes, each column is u_j
      S_k: (k, k)
      q: modulus
    Output:
      X: (m, B)   each column x_j satisfies A x_j ≡ u_j (mod q)

    Notes:
      - Engineering version, preserves your current samplePre logic
      - Much faster than calling samplePre B times in Python
    """
    if rng is None:
        rng = np.random.default_rng()
    if q <= 1:
        raise ValueError("q must be >= 2.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if c < 1.0:
        raise ValueError("c must be >= 1.0.")

    q64 = np.int64(q)

    A = np.asarray(A, dtype=np.int64) % q64
    R = np.asarray(R, dtype=np.int64)          # keep as int64
    U = np.asarray(U, dtype=np.int64) % q64
    S_k = np.asarray(S_k, dtype=np.int64)

    if A.ndim != 2 or R.ndim != 2 or U.ndim != 2:
        raise ValueError("A, R, U must be 2D.")
    n, m = A.shape
    n2, B = U.shape
    if n2 != n:
        raise ValueError("U must have shape (n, B).")

    bar_m, w = R.shape
    if m != bar_m + w:
        raise ValueError("Dimension mismatch: A has m but R implies bar_m+w.")
    if (w % n) != 0:
        raise ValueError("w must be divisible by n.")
    k = w // n
    if S_k.shape != (k, k):
        raise ValueError(f"S_k must be ({k},{k}).")

    # ---- (1) sample P: (m, B) ----
    sigma_p = c * s
    P = np.rint(rng.normal(loc=0.0, scale=float(sigma_p), size=(m, B))).astype(np.int64)

    # ---- (2) V = U - A P (mod q) ----
    # A:(n,m) @ P:(m,B) => (n,B)
    AP = (A @ (P % q64)) % q64
    V = (U - AP) % q64   # (n,B)

    # ---- (3) z0 = bit-decomposition of V entries (little-endian) ----
    # V: (n,B) -> z0: (n,B,k)
    shifts = np.arange(k, dtype=np.int64)
    # broadcast: (n,B,1) >> (k,) => (n,B,k)
    z0 = ((V[:, :, None] >> shifts[None, None, :]) & 1).astype(np.int64)

    # ---- (4) t and z_blocks = z0 + t @ S_k^T ----
    if sigma_t is None or sigma_t <= 0.0:
        t = np.zeros((n, B, k), dtype=np.int64)
    else:
        t = np.rint(rng.normal(loc=0.0, scale=float(sigma_t), size=(n, B, k))).astype(np.int64)

    # t @ S_k.T per (n,B) slice:
    # (n,B,k) x (k,k) -> (n,B,k)
    z_blocks = z0 + (t @ S_k.T)

    # ---- (5) flatten z: (w,B) ----
    # z_blocks currently (n,B,k). We need (n,k,B) then reshape to (w,B).
    z = np.transpose(z_blocks, (0, 2, 1)).reshape(w, B).astype(np.int64)

    # ---- (6) y0 = R z (mod q), y = [y0; z] ----
    y0 = ((R % q64) @ (z % q64)) % q64        # (bar_m,B)
    Y = np.vstack([y0, z % q64]).astype(np.int64)  # (m,B)

    # ---- (7) X = P + Y (mod q) ----
    X = (P + Y) % q64
    return X

def deltrap_HI_batched(
        A: np.ndarray,
        R: np.ndarray,
        A1: np.ndarray,
        G: np.ndarray,
        S_k: np.ndarray,
        q: int,
        *,
        s_prime: float = 3.0,
        c: float = 1.0,
        sigma_t: float = 3.0,
        batch_size: int = 32,
        center: bool = False,
        rng: Optional[np.random.Generator] = None,
        progress_every: int = 0,
) -> np.ndarray:
    """
    DelTrap for H'=I using samplePre in a batched manner.

    Returns R' (m,w) s.t. A R' ≡ G - A1 (mod q).
    """
    if rng is None:
        rng = np.random.default_rng()
    if q <= 1:
        raise ValueError("q must be >= 2.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    q64 = np.int64(q)
    A  = np.asarray(A,  dtype=np.int64) % q64
    A1 = np.asarray(A1, dtype=np.int64) % q64
    G  = np.asarray(G,  dtype=np.int64) % q64
    R  = np.asarray(R,  dtype=np.int64)
    S_k = np.asarray(S_k, dtype=np.int64)

    n, m = A.shape
    if A1.shape[0] != n or G.shape != A1.shape:
        raise ValueError("Shape mismatch: A1/G must be (n,w).")
    w = A1.shape[1]

    U = (G - A1) % q64  # (n,w)

    R_prime = np.empty((m, w), dtype=np.int64)

    # Process columns in batches
    for start in range(0, w, batch_size):
        end = min(start + batch_size, w)
        U_batch = U[:, start:end]  # (n,B)

        X_batch = samplePre_batch(
            A=A,
            R=R,
            U=U_batch,
            S_k=S_k,
            q=q,
            s=s_prime,
            c=c,
            sigma_t=sigma_t,
            rng=rng,
        )  # (m,B)

        R_prime[:, start:end] = X_batch  # already mod q

        if progress_every and (end % progress_every == 0):
            print(f"[deltrap_batched] done {end}/{w} columns")

    if center:
        half = q // 2
        R_prime = ((R_prime + half) % q64) - half

    return R_prime

def verify_G_S(G: np.ndarray, S_k: np.ndarray, q: int):
    n = G.shape[0]
    S = np.kron(np.eye(n, dtype=np.int64), S_k)
    GS = (G @ S) % np.int64(q)
    assert np.all(GS == 0), "FAILED: G·(I⊗S_k) ≢ 0 (mod q)"
    print("✔ G · (I_n ⊗ S_k) ≡ 0 (mod q)")


def verify_G_trapdoor_H_is_I(A: np.ndarray, R: np.ndarray, G: np.ndarray, q: int):
    """
    Verify: A [R; I] == G (mod q)
    """
    A = np.asarray(A, dtype=np.int64) % np.int64(q)
    R = np.asarray(R, dtype=np.int64)
    G = np.asarray(G, dtype=np.int64) % np.int64(q)

    n, m = A.shape
    bar_m, w = R.shape
    if m != bar_m + w or G.shape != (n, w):
        print("verify A*[R;I] == G (mod q):", False)
        return

    RI = np.vstack([R % np.int64(q), np.eye(w, dtype=np.int64) % np.int64(q)]) % np.int64(q)
    left = (A @ RI) % np.int64(q)
    ok = np.array_equal(left, G)
    print("verify A*[R;I] == G (mod q):", ok)



def verify_preimage(A: np.ndarray, X: np.ndarray, U: np.ndarray, q: int) -> None:
    """
    Verify and PRINT whether:
        A @ X ≡ U (mod q)

    A: (n,m), X: (m,l) or (m,), U: (n,l) or (n,)
    """
    if q <= 1:
        raise ValueError("q must be >= 2.")
    q64 = np.int64(q)

    A = np.asarray(A, dtype=np.int64) % q64
    X = np.asarray(X, dtype=np.int64) % q64
    U = np.asarray(U, dtype=np.int64) % q64

    AX = (A @ X) % q64
    r = (AX - U) % q64
    ok = bool(np.all(r == 0))
    print(f"[Verify] A@X == U (mod q): {ok}")
    if not ok:
        idx = np.argwhere(r != 0)[0]
        if r.ndim == 1:
            i = int(idx[0])
            print(f"First mismatch row={i}: AX={int(AX[i])}, U={int(U[i])}")
        else:
            i, j = map(int, idx)
            print(f"First mismatch row={i}, col={j}: AX={int(AX[i,j])}, U={int(U[i,j])}")

if __name__ == '__main__':
    n = 128

    # n, k, q, w, bar_m, m = get_secure_param(256, 16)
    # n, k, q, w, bar_m, m = get_secure_param_only_n(n)
    n, k, q, w, bar_m, m = get_secure_param_only_n_min(n)

    # 验证环上的采样
    A, R, G, S_k = gen_trapdoor_G_trapdoor(n, q, k, bar_m, sigma=3.0)
    u1 = np.random.default_rng().integers(0, q, size=(n,), dtype=np.int64)
    u2 = np.random.default_rng().integers(0, q, size=(n,), dtype=np.int64)
    u3 = np.random.default_rng().integers(0, q, size=(n,), dtype=np.int64)
    u4 = np.random.default_rng().integers(0, q, size=(n,), dtype=np.int64)
    # 拼接 U=[u1|u2|u3]
    U = np.column_stack([u1, u2, u3, u4]).astype(np.int64)
    # 使用 ring_samplePre生成X
    for i in range(5):
        t0 = time.perf_counter()
        X = ring_samplePre(
            A=A,
            R=R,
            U=U,
            S_k=S_k,
            q=q,
            s=3.0,
            c=1.0,
            sigma_t=3.0
        )
        t1 = time.perf_counter()
        print(f"[ring_samplePre]  {(t1 - t0) * 1000:.3f} ms")
        # 验证 AX=U mod q
        verify_preimage(A, X, U, q)

    # for i in range(1):
    #
    #     # ---- GenTrap timing
    #     t0 = time.perf_counter()
    #     A, R, G, S_k = gen_trapdoor_G_trapdoor(n, q, k, bar_m, sigma=3.0)
    #     t1 = time.perf_counter()
    #     print(f"[gen_trapdoor_G_trapdoor]  {(t1 - t0) * 1000:.3f} ms")
    #
    #     # Optional checks (not timed)
    #     # verify_G_trapdoor_H_is_I(A, R, G, q)
    #     # verify_G_S(G, S_k, q)
    #
    #     rng = np.random.default_rng()
    #     u = rng.integers(0, q, size=(n,), dtype=np.int64)
    #
    #     # ---- samplePre timing
    #     t0 = time.perf_counter()
    #     x = samplePre(A, R, u, S_k, q, s=3.0, c=1.0, sigma_t=3.0)
    #     t1 = time.perf_counter()
    #     print(f"[samplePre]  {(t1 - t0) * 1000:.3f} ms")
    #
    #     # Optional verify (not timed)
    #     # verify_preimage(A, x, u, q)
    #
    #     H_prime = np.eye(n, dtype=np.int64) % np.int64(q)  # 你可以不再需要它
    #     A1 = rng.integers(0, q, size=(n, w), dtype=np.int64)
    #
    #     # t0 = time.perf_counter()
    #     # R_prime = deltrap_HI(
    #     #     A=A,
    #     #     R=R,
    #     #     A1=A1,
    #     #     G=G,
    #     #     S_k=S_k,
    #     #     q=q,
    #     #     s_prime=3.0,
    #     #     c=1.0,
    #     #     sigma_t=3.0,
    #     #     progress_every=200,   # 可选：打印进度
    #     # )
    #     # t1 = time.perf_counter()
    #     # print(f"[deltrap]  {(t1 - t0) * 1000:.3f} ms")
    #
    #     t0 = time.perf_counter()
    #     R_prime = deltrap_HI_batched(
    #         A=A, R=R, A1=A1, G=G, S_k=S_k, q=q,
    #         s_prime=3.0, c=1.0, sigma_t=3.0,
    #         batch_size=64,
    #     )
    #     t1 = time.perf_counter()
    #     print(f"[deltrap_batched]  {(t1 - t0) * 1000:.3f} ms")
    #
    #     # A_prime = np.concatenate([A % q, A1 % q], axis=1).astype(np.int64)
    #     # verify_G_trapdoor_H_is_I(A_prime, R_prime, G, q)