import time
from collections import deque
from typing import Tuple

import numpy as np


def num_to_all_and_policy_str(n: int) -> str:
    # 1、准备属性数组
    attr_list = []
    for i in range(n):
        attr_str = "a"+str(i)
        attr_list.append(attr_str)

    # 2、两两连接属性，形成策略字符串
    while len(attr_list)>1:
        tmp_list = []
        for i in range(len(attr_list)):
            if i % 2 ==0:
                if i != len(attr_list)-1:
                    this_attr = attr_list[i]
                    next_attr = attr_list[i+1]
                    tmp_attr = "( "+this_attr+" and "+next_attr+" )"
                    tmp_list.append(tmp_attr)
                else:
                    tmp_list.append(attr_list[i])
        attr_list = tmp_list

    return attr_list[0]

class TreeNode:
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        self.v = []
    def set_v(self, v):
        self.v = list(v)

    def get_v(self):
        return self.v

    def get_data(self):
        return self.data


def create_tree_from_three_value(v1,v2,v3):
    if isinstance(v1, str):
        node1 = TreeNode(v1)
    else:
        node1 = v1
    node2 = TreeNode(v2)
    if isinstance(v3, str):
        node3 = TreeNode(v3)
    else:
        node3 = v3
    node2.left = node1
    node2.right = node3
    return node2

def gen_policy_tree_from_policy(policy):
    tmp_list = policy.split(" ")
    stack = []
    for tmp_i in tmp_list:
        if tmp_i != ")":
            stack.append(tmp_i)
        else:
            v3=stack.pop()
            v2=stack.pop()
            v1=stack.pop()
            stack.pop()
            node = create_tree_from_three_value(v1,v2,v3)
            stack.append(node)
    return stack[0]

def set_left_v_from_father(father_node, this_node):
    father_node_v = father_node.get_v()
    if father_node.get_data() == "or":
        this_node.set_v(father_node_v)
    else:
        this_node_v = []
        for i in range(len(father_node_v)):
            this_node_v.append(0)
        this_node_v.append(-1)
        this_node.set_v(this_node_v)

def set_right_v_from_father(father_node, this_node):
    father_node_v = father_node.get_v()
    if father_node.get_data() == "or":
        this_node.set_v(father_node_v)
    else:
        this_node_v = list(father_node_v)
        this_node_v.append(1)
        this_node.set_v(this_node_v)


def set_v_level_order(root):
    if root is None:
        return
    if root.left is None and root.right is None:
        return

    # 1、初始化准备
    root.set_v([1])
    tmp_queue = deque()
    tmp_queue.append(root)
    c=1
    leaf_node_name_list = []
    leaf_node_v_list = []

    # 2、层级遍历设置v
    while len(tmp_queue)>0:
        this_node = tmp_queue.popleft()
        # 2.1、将父节点的v长度补齐至c
        this_node_v = this_node.get_v()
        if len(this_node_v)!=c:
            for i in range(c-len(this_node_v)):
                this_node_v.append(0)
            this_node.set_v(this_node_v)
        # 2.2、设置左子节点的向量
        this_node_left = this_node.left
        if this_node_left is not None:
            set_left_v_from_father(this_node, this_node_left)
            tmp_queue.append(this_node_left)
        # 2.3、设置右子节点的向量
        this_node_right = this_node.right
        if this_node_right is not None:
            set_right_v_from_father(this_node, this_node_right)
            tmp_queue.append(this_node_right)
        # 2.4、更新c
        if this_node.data == "and":
            c+=1
        # 2.5、获取叶子节点的名称与对应的向量
        if this_node.left is None and this_node.right is None:
            leaf_node_name_list.append(this_node.data)
            leaf_node_v_list.append(this_node_v)
    for v_list in leaf_node_v_list:
        if len(v_list)!=c:
            for i in range(c-len(v_list)):
                v_list.append(0)
    return leaf_node_name_list, leaf_node_v_list

def gen_LSSS_from_policy_str(policy_str: str) -> Tuple[list, list]:
    root = gen_policy_tree_from_policy(policy_str)
    leaf_node_name_list, leaf_node_v_list = set_v_level_order(root)
    return leaf_node_name_list, leaf_node_v_list

def get_omega_from_attr_list(
        leaf_node_name_list,
        leaf_node_v_list,
        user_attr_list,
        i_for_e,
        *,
        strict: bool = True,   # True: 缺失属性直接报错；False: 忽略不在叶子里的属性
):
    """
    输入：
      - leaf_node_name_list: 所有叶子属性名列表（全属性）
      - leaf_node_v_list:   与之对应的向量列表，每个向量长度为 n
      - user_attr_list:     用户拥有的属性名列表（可能乱序、可能重复）
      - i_for_e:            e_i 中 1 的位置（0-based）

    输出：
      - omega_list: 全属性长度的 omega（未拥有属性位置为 0）
      - index_list: 用户属性在叶子列表中的索引（排序后）
      - omega_user: 与 index_list 对齐的 omega 子向量
    """

    # 0) 建立 name -> index 映射（O(n)）
    name2idx = {name: i for i, name in enumerate(leaf_node_name_list)}

    # 1) 用户属性 -> index（去重 + 可选严格模式）
    index_set = set()
    missing = []
    for a in user_attr_list:
        idx = name2idx.get(a, None)
        if idx is None:
            missing.append(a)
            continue
        index_set.add(idx)

    if missing and strict:
        raise ValueError(f"user_attr_list 中存在不在 leaf_node_name_list 的属性: {missing}")

    index_list = sorted(index_set)
    if not index_list:
        raise ValueError("用户属性在 leaf_node_name_list 中一个都没匹配到，无法求 omega。")

    # 2) 组装 A（rows = 用户拥有的叶子向量）
    v_list_user_has = [leaf_node_v_list[i] for i in index_list]
    A = np.asarray(v_list_user_has, dtype=np.int64)  # shape: (k, n)

    if A.ndim != 2:
        raise ValueError(f"leaf_node_v_list 的元素应为等长向量，得到 A.ndim={A.ndim}")

    k, n = A.shape
    if not (0 <= i_for_e < n):
        raise ValueError(f"i_for_e 超界：i_for_e={i_for_e}, n={n}")

    # 3) 解 A^T x = e_i
    e_i = np.zeros(n, dtype=np.float64)
    e_i[i_for_e] = 1.0

    # 你原逻辑隐含要求：A.T 必须是方阵 => k == n
    if k != n:
        # 两种选择：
        # (a) 严格：直接报错（更符合“必须可逆才能求唯一 omega”）
        # (b) 兜底：最小二乘
        raise ValueError(f"A.T 不是方阵：A.shape={A.shape}，需要 k==n 才能用 solve。")

        # 如果你想兜底（注释掉上面 raise，改用下面）：
        # x, *_ = np.linalg.lstsq(A.T.astype(np.float64), e_i, rcond=None)

    x = np.linalg.solve(A.T.astype(np.float64), e_i)  # shape: (k,)

    # 4) 浮点 -> 整数
    x_round = np.rint(x)

    omega_user = x_round.astype(np.int64)             # 与 index_list 对齐
    omega_list_user_has = omega_user.tolist()

    # 5) 映射回全属性 omega
    omega_list = [0] * len(leaf_node_name_list)
    for pos, idx in enumerate(index_list):
        omega_list[idx] = int(omega_user[pos])

    return omega_list

def get_omega_from_attr_list_fast(
        leaf_node_name_list,
        leaf_node_v_list,
        user_attr_list,
        i_for_e,
        *,
        strict: bool = True,
):
    # ---------- 0) 懒初始化缓存 ----------
    # 用函数属性做 cache（不污染全局命名空间）
    if not hasattr(get_omega_from_attr_list_fast, "_cache"):
        get_omega_from_attr_list_fast._cache = {}

    # 用 leaf_node_name_list 的 id 作为 key（假定其内容在生命周期内不变）
    key = id(leaf_node_name_list)
    cache = get_omega_from_attr_list_fast._cache

    if key not in cache:
        name2idx = {name: i for i, name in enumerate(leaf_node_name_list)}
        V = np.asarray(leaf_node_v_list, dtype=np.int64)
        cache[key] = (name2idx, V)
    else:
        name2idx, V = cache[key]

    # ---------- 1) 用户属性 -> index ----------
    index_set = set()
    missing = []
    for a in user_attr_list:
        idx = name2idx.get(a)
        if idx is None:
            missing.append(a)
        else:
            index_set.add(idx)

    if missing and strict:
        raise ValueError(f"user_attr_list 中存在不在叶子里的属性: {missing}")

    index_list = sorted(index_set)
    if not index_list:
        raise ValueError("用户属性未命中任何叶子，无法求 omega。")

    # ---------- 2) 取行构造 A ----------
    A = V[index_list, :]
    k, n = A.shape
    if not (0 <= i_for_e < n):
        raise ValueError(f"i_for_e 超界：i_for_e={i_for_e}, n={n}")
    if k != n:
        raise ValueError(f"A.T 不是方阵：A.shape={A.shape}，需要 k==n 才能 solve。")

    # ---------- 3) 解 A^T x = e_i ----------
    AT = A.T.astype(np.float64, copy=False)
    e_i = np.zeros(n, dtype=np.float64)
    e_i[i_for_e] = 1.0

    x = np.linalg.solve(AT, e_i)
    omega_user = np.rint(x).astype(np.int64)

    # ---------- 4) 映射回全属性 omega ----------
    omega_all = np.zeros(len(leaf_node_name_list), dtype=np.int64)
    omega_all[index_list] = omega_user
    return omega_all

def get_s_from_omega(v_list, omega_vector):
    V = np.asarray(v_list, dtype=np.int64)      # (k, d)
    omega = np.asarray(omega_vector, dtype=np.int64)  # (k,)
    s = omega @ V                               # (d,)
    return s


if __name__ == '__main__':
    policy_str = num_to_all_and_policy_str(50)
    print(policy_str)
    leaf_node_name_list, leaf_node_v_list = gen_LSSS_from_policy_str(policy_str)
    print(leaf_node_name_list, leaf_node_v_list)

    start = time.time()
    omega_list = get_omega_from_attr_list(leaf_node_name_list,leaf_node_v_list, leaf_node_name_list,1)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"get_omega_from_attr_list 耗时: {elapsed_ms:.3f} ms")

    start = time.time()
    omega_vector_fast = get_omega_from_attr_list_fast(leaf_node_name_list,leaf_node_v_list, leaf_node_name_list,1)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"get_omega_from_attr_list_fast 耗时: {elapsed_ms:.3f} ms")

    # 验证结果一致
    print("omega_list      =", omega_list)
    print("omega_list_fast =", omega_vector_fast)

    # policy_str = "( ( a0 or a1 ) and ( a2 and a3 ) )"
    # leaf_node_name_list, leaf_node_v_list = gen_LSSS_from_policy_str(policy_str)
    # print(leaf_node_name_list, leaf_node_v_list)
    # omega_list_fast = get_omega_from_attr_list_fast(leaf_node_name_list,leaf_node_v_list, ["a3","a0","a2"],1)
    # print("omega_list_fast =", omega_list_fast)

    s = get_s_from_omega(leaf_node_v_list, omega_vector_fast)
    print(s)