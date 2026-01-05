import secrets
import time
from math import gcd

import numpy as np

from LSSS import gen_LSSS_from_policy_str, num_to_all_and_policy_str, get_omega_from_attr_list_fast
from lwe_symmetric import str_to_bits_numpy, lwe_sym_encrypt, lwe_sym_decrypt, bits_numpy_to_str
from samplePre import trapGen_fast, sampleD_34


def setup(attr_num, n, k, q):
    A_list = []
    R_list = []
    for i in range(attr_num):
        A, R = trapGen_fast(n, k, q)
        A_list.append(A)
        R_list.append(R)
    return A_list, R_list

def setup_fast(attr_num, n, k, q):
    A_list = []
    R_list = []
    A, R = trapGen_fast(n, k, q)
    for i in range(attr_num):
        A_list.append(A)
        R_list.append(R)
    return A_list, R_list

def random_binary_vector_secure(n: int) -> np.ndarray:
    # generate first n-1 bits
    z = np.fromiter((secrets.randbits(1) for _ in range(n - 1)),
                    dtype=np.int8, count=n - 1)

    # set last bit to enforce odd Hamming weight
    parity = int(z.sum() & 1)          # 0 if even, 1 if odd
    last_bit = 1 - parity             # make total sum odd
    z = np.append(z, np.int8(last_bit))
    return z

def keyGenDo(attr_num, n):
    z_list = []
    for i in range(attr_num):
        z_i = random_binary_vector_secure(n)
        z_list.append(z_i)
    return z_list

def random_vector_mod_q_secure(length: int, q: int) -> np.ndarray:
    seed = secrets.randbits(256)
    rng = np.random.default_rng(seed)
    return rng.integers(0, q, size=length, dtype=np.int64)

def random_nonzero_mod_q(q: int) -> int:
    if q <= 1:
        raise ValueError("q must be >= 2")
    # sample from {1,3,5,...}
    return secrets.randbelow(q // 2) * 2 + 1

def modinv(a: int, q: int) -> int:
    """Return a^{-1} mod q if it exists; raise ValueError otherwise."""
    a %= q
    if a == 0 or gcd(a, q) != 1:
        raise ValueError(f"no modular inverse for a={a} mod q={q} (gcd={gcd(a,q)})")
    # Python 3.8+ supports modular inverse via pow
    return pow(a, -1, q)


def encrypt(msg, s, s_index, q, leaf_node_v_list, z_list, A_list):
    # 1、处理msg
    msg_bits = str_to_bits_numpy(msg)

    # 2、获取主密文 c = s*gamma+M[q/2]+noise
    gamma, c = lwe_sym_encrypt(msg_bits, s, q, 0.01)
    
    # 3、根据秘密s与s_index，生成向量v，其中v[s_index]=s，并计算lambda_vector = LSSS_matrix*v
    v_length = len(leaf_node_v_list[0])
    v = random_vector_mod_q_secure(v_length, q)
    v[s_index] = s % q
    LSSS_matrix = np.asarray(leaf_node_v_list, dtype=np.int64)
    lambda_vector = (LSSS_matrix @ v) % q

    # 4、计算c_1
    beta = random_nonzero_mod_q(q)
    c_1 = (beta * lambda_vector) % q

    # 5、计算c_2
    c_2 = []
    beta_inv = modinv(beta, q)
    for i in range(len(A_list)):
        A_i = np.asarray(A_list[i], dtype=np.int64) % q
        z_i = np.asarray(z_list[i], dtype=np.int64).reshape(-1) % q  # (n,)

        # z_int = z_i^T z_i (scalar) mod q
        z_int = int((z_i @ z_i) % q)
        z_int_inv = modinv(z_int, q)

        # u_i = z_i^T A_i  -> shape (m,)
        u_i = (z_i.T @ A_i) % q

        # c_2_i = u_i * z_int_inv * beta_inv mod q
        c_2_i = (u_i * z_int_inv) % q
        c_2_i = (c_2_i * beta_inv) % q

        c_2.append(c_2_i.astype(np.int64))
    return gamma, c, c_1, c_2, beta

def random_binary_vector_secure_even_weight(n: int) -> np.ndarray:
    z = np.fromiter((secrets.randbits(1) for _ in range(n - 1)),
                    dtype=np.int8, count=n - 1)
    last_bit = int(z.sum() & 1)   # 若前面是奇数，最后补 1；若偶数，补 0
    return np.append(z, np.int8(last_bit))

def random_binary_vector_secure_odd_weight(n: int) -> np.ndarray:
    """
    Generate a cryptographically secure binary vector z ∈ {0,1}^n
    with ODD Hamming weight.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    # 先随机生成前 n-1 位
    z = np.fromiter(
        (secrets.randbits(1) for _ in range(n - 1)),
        dtype=np.int8,
        count=n - 1
    )

    # 根据前 n-1 位的奇偶性，决定最后一位
    # 如果前面是偶数个 1，就补 1；如果是奇数个 1，就补 0
    last_bit = 1 - (int(z.sum()) & 1)

    return np.append(z, np.int8(last_bit))

def keyGenUsers(user_attr_list, A_list, T_list, q, z_list):
    n = A_list[0].shape[0]

    sk_list = []
    t = random_nonzero_mod_q(q)
    h_list = []

    num  = len(user_attr_list)
    for i in range(num):
        A_i = A_list[i]
        T_i = T_list[i]
        z_i = z_list[i]
        e_i = sampleD_34(A_i, T_i, q, z_i, s_p=2.0, s_t=2.0)

        # 确保后续的b_j是奇数
        if i == num - 1:
            # 最后一个用奇权重，保证总权重为奇数
            h_i = random_binary_vector_secure_odd_weight(n)
        else:
            h_i = random_binary_vector_secure_even_weight(n)
        h_list.append(h_i.astype(np.int64))
        r_i = sampleD_34(A_i, T_i, q, h_i, s_p=2.0, s_t=2.0)

        # 计算sk_i = t*(e_i)*(r_i^T)
        e = np.asarray(e_i, dtype=np.int64).reshape(-1, 1) % q   # (m, 1)
        rT = np.asarray(r_i, dtype=np.int64).reshape(1, -1) % q # (1, m)
        sk_i = (t * (e @ rT)) % q    # shape: (m, m)
        sk_list.append(sk_i)

    # 计算 h_sum = h_i的累加
    h_sum = np.sum(np.stack(h_list, axis=0), axis=0) % q  # shape (n,)

    b_j = int(np.sum(h_sum * h_sum) % q)

    # 计算p = h_sum*t^-1*b_j^-1
    t_inv = modinv(t, q)
    b_inv = modinv(b_j, q)
    p = (h_sum * t_inv) % q
    p = (p * b_inv) % q

    return sk_list, p

def decrypt(gamma, c, c_1, c_2, sk_list, p, A_list , leaf_node_name_list,leaf_node_v_list, user_attr_list, s_index):
    ptc_1_list = []

    for i in range(len(A_list)):
        c2_i = np.asarray(c_2[i], dtype=np.int64) % q        # shape (m,)
        sk_i = np.asarray(sk_list[i], dtype=np.int64) % q   # shape (m, m)
        A_i  = np.asarray(A_list[i], dtype=np.int64) % q    # shape (n, m)

        # c2_i @ sk_i @ A_i.T  -> shape (n,)
        ptc_1_i = (c2_i @ sk_i @ A_i.T) % q
        ptc_1_list.append(ptc_1_i)

    # 聚合
    ptc_1 = np.sum(np.stack(ptc_1_list, axis=0), axis=0) % q   # shape (n,)

    # 计算 ptc = ptc_1 * p
    ptc = int(np.dot(ptc_1, p) % q)

    # 获取omega
    omega_omega_vector = get_omega_from_attr_list_fast(leaf_node_name_list,leaf_node_v_list, user_attr_list,s_index)

    # ptc_2 = omega_vector * c_1
    omega_vec = np.asarray(omega_omega_vector, dtype=np.int64) % q
    c1_vec    = np.asarray(c_1, dtype=np.int64) % q
    ptc_2 = int(np.dot(omega_vec, c1_vec) % q)

    # 计算s = ptc*ptc_2
    s = int((ptc * ptc_2) % q)

    m_bits = lwe_sym_decrypt(gamma, c, s, q)
    m_str = bits_numpy_to_str(m_bits)
    return m_str

def test_attr_num(attr_num, n, k, q):
    m = 2*n*k
    policy_str = num_to_all_and_policy_str(attr_num)
    s = 50
    s_index = 1
    msg = "hello"

    start = time.time()
    A_list, T_list = setup_fast(attr_num, n, k, q)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"setup耗时: {elapsed_ms:.3f} ms")

    start = time.time()
    z_list = keyGenDo(attr_num, n)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"keyGenDo耗时: {elapsed_ms:.3f} ms")

    start = time.time()
    leaf_node_name_list, leaf_node_v_list = gen_LSSS_from_policy_str(policy_str)
    gamma, c, c_1, c_2, beta = encrypt(msg, s, s_index, q, leaf_node_v_list, z_list, A_list)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"encrypt耗时: {elapsed_ms:.3f} ms")

    user_attr_list = leaf_node_name_list
    start = time.time()
    sk_list, p = keyGenUsers(user_attr_list, A_list, T_list, q, z_list)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"keyGenUsers耗时: {elapsed_ms:.3f} ms")

    start = time.time()
    msg_1 = decrypt(gamma, c, c_1, c_2, sk_list, p, A_list ,leaf_node_name_list,leaf_node_v_list, user_attr_list, s_index)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"decrypt耗时: {elapsed_ms:.3f} ms")

    print(msg_1)

if __name__ == '__main__':
    n = 100
    k = 12
    q = 2**k
    m = 2*n*k

    test_attr_num(100, n, k, q)