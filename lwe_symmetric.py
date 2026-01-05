import math
import secrets
import numpy as np
from typing import Tuple

def str_to_bits_numpy(msg: str) -> np.ndarray:
    """
    Convert a string to a bit array using ASCII encoding.
    Each character -> 8-bit binary (big-endian).

    Returns:
        bits (np.ndarray): shape (L,), entries in {0,1}
        L (int): length of the bit array
    """
    # Encode string to bytes using ASCII
    msg_bytes = msg.encode('ascii')  # raises error if non-ASCII

    # Convert each byte to 8 bits
    bits_list = []
    for b in msg_bytes:
        # format to 8-bit binary, big-endian
        bits_list.extend([(b >> i) & 1 for i in range(7, -1, -1)])

    bits = np.array(bits_list, dtype=np.int64)
    return bits

def bits_numpy_to_str(bits: np.ndarray) -> str:
    """
    Convert a bit array (0/1) to a string using ASCII decoding.
    Every 8 bits are interpreted as one ASCII character (big-endian).
    """
    bits = np.asarray(bits, dtype=np.int64)

    if bits.ndim != 1:
        raise ValueError("bits must be a 1-D array")
    if bits.size % 8 != 0:
        raise ValueError("length of bits must be a multiple of 8")

    chars = []
    for i in range(0, bits.size, 8):
        byte_bits = bits[i:i+8]

        # big-endian: most significant bit first
        value = 0
        for b in byte_bits:
            value = (value << 1) | int(b)

        chars.append(chr(value))

    return "".join(chars)


# ============================================================
# 0) CSPRNG helpers (scheme B): uniform -> Gaussian (Box-Muller)
# ============================================================

def csprng_uniform_0_1() -> float:
    """
    Cryptographically secure U in [0,1).
    Use 53 random bits to match double precision mantissa.
    """
    return secrets.randbits(53) / (1 << 53)


def csprng_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """
    Cryptographically secure Normal(mean, std^2) via Box-Muller transform.
    """
    # u1 must be in (0,1], avoid log(0)
    u1 = csprng_uniform_0_1()
    while u1 == 0.0:
        u1 = csprng_uniform_0_1()
    u2 = csprng_uniform_0_1()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z0


def csprng_normal_vec(std: float, size: int) -> np.ndarray:
    """
    Vector of cryptographically secure normals N(0, std^2).
    Uses Box-Muller; generates in pairs for efficiency.
    """
    out = np.empty(size, dtype=np.float64)
    i = 0
    while i < size:
        u1 = csprng_uniform_0_1()
        while u1 == 0.0:
            u1 = csprng_uniform_0_1()
        u2 = csprng_uniform_0_1()

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)

        out[i] = z0 * std
        i += 1
        if i < size:
            out[i] = z1 * std
            i += 1
    return out


# ============================================================
# 1) Psi_alpha noise sampling (strictly per literature)
# ============================================================

def sample_psi_alpha(alpha: float, q: int, size: int) -> np.ndarray:
    r"""
    Sample noise from Psi_alpha as defined in Regev-style LWE:

      x ~ N(0, alpha^2 / (2*pi))   over R (conceptually over T = R/Z)
      e = round(q * x) mod q       in Z_q

    Returns centered representatives in (-q/2, q/2].
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if q <= 0 or (q & (q - 1)) != 0:
        raise ValueError("This implementation expects q = 2^k (power of two).")

    sigma_T = alpha / math.sqrt(2.0 * math.pi)          # std on T (before scaling by q)
    # Sample x ~ N(0, sigma_T^2) using CSPRNG-based Gaussian
    x = csprng_normal_vec(std=sigma_T, size=size)       # float64

    # e = round(q*x) mod q
    e = np.rint(q * x).astype(np.int64) % q

    # Map to centered reps in (-q/2, q/2]
    e = (e + q // 2) % q - q // 2
    return e


# ============================================================
# 2) KeyGen / Message / Random a using CSPRNG (secrets)
# ============================================================

def lwe_sym_keygen(q: int) -> int:
    """Scalar secret key s in Z_q using CSPRNG."""
    return secrets.randbelow(q)


def csprng_rand_Zq_vec(q: int, size: int) -> np.ndarray:
    """Vector in Z_q^size using CSPRNG."""
    return np.array([secrets.randbelow(q) for _ in range(size)], dtype=np.int64)


def csprng_rand_bits_vec(size: int) -> np.ndarray:
    """Vector of bits in {0,1}^size using CSPRNG."""
    # Efficient-ish: use randbits blocks
    out = np.empty(size, dtype=np.int64)
    i = 0
    while i < size:
        block = secrets.randbits(64)
        for j in range(64):
            if i >= size:
                break
            out[i] = (block >> j) & 1
            i += 1
    return out


# ============================================================
# 3) Encrypt / Decrypt
# ============================================================

def lwe_sym_encrypt(
        m: np.ndarray,      # bits {0,1}^l
        s: int,             # scalar secret key in Z_q
        q: int,
        alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Encrypt m ∈ {0,1}^l (bitwise packing) using scalar-key LWE:
        gamma <- Z_q^l
        e <- Psi_alpha^l
        c = gamma*s + e + (q/2)*m  (mod q)
    Returns (a,b).
    """
    m = np.asarray(m, dtype=np.int64)
    l = m.size
    if np.any((m != 0) & (m != 1)):
        raise ValueError("m must be a bit-vector (entries 0/1).")

    gamma = csprng_rand_Zq_vec(q, l)
    e = sample_psi_alpha(alpha, q, l)

    Delta = q // 2
    c = (gamma * int(s) + e + Delta * m) % q
    return gamma, c


def lwe_sym_decrypt(gamma: np.ndarray, c: np.ndarray, s: int, q: int) -> np.ndarray:
    """
    Decrypt with decision threshold q/4:
      v = (c - gamma*s) mod q
      m_hat = 1 if v in [q/4, 3q/4), else 0
    """
    gamma = np.asarray(gamma, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)

    v = (c - gamma * int(s)) % q
    lower = q // 4
    upper = 3 * q // 4
    return np.where((v >= lower) & (v < upper), 1, 0).astype(np.int64)