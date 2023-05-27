from itertools import combinations
import numpy as np


class PartialMps:
    def __init__(self, error, mat, prefix):
        self.error = error
        self.mat = mat
        self.prefix = prefix

    def __lt__(self, other):
        return self.error < other.error


def mps_uniform(data, bond_dimension = 4) -> [int]:
    """
    :param data: data
    :param bond_dimension:
    :return: optimal permutation with minimal error
    """
    from heapq import heappush, heappop
    pq = []
    num_qubits = int(np.log2(data.size))
    l = list(range(num_qubits))
    origin = data.reshape((2,)*num_qubits)
    num_operand = int(np.log2(bond_dimension)) + 1
    for prefix in combinations(l, num_operand):
        prefix = list(prefix)
        last = [e for e in l if e not in prefix]
        d = origin.transpose(list(prefix)+last)
        left_size = 2**num_operand
        u, lamb, vt = np.linalg.svd(d.reshape(left_size, -1), full_matrices=False)
        error = np.linalg.norm(lamb[bond_dimension:])**2
        mat = (vt[:bond_dimension].T * lamb[:bond_dimension]).T
        node = PartialMps(error, mat, prefix)
        heappush(pq, node)
    while True:
        node = heappop(pq)
        n = len(node.prefix)
        if n == num_qubits - num_operand:
            ret = node.prefix[:num_operand]
            indices = [i for i in range(num_qubits) if i not in ret]
            for nth in node.prefix[num_operand:]:
                ret.append(indices[nth])
                indices.pop(nth)
            for i in range(num_qubits):
                if i not in ret:
                    ret.append(i)
            return ret

        for i in range(num_qubits - n):
            prefix = node.prefix + [i]
            last = [e + 1 for e in range(num_qubits-n) if e != i]
            d = node.mat.reshape([bond_dimension] + [2]*(num_qubits-n))
            d = d.transpose([0, i+1] + last).reshape(bond_dimension * 2, -1)
            u, lamb, vt = np.linalg.svd(d, full_matrices=False)
            error = node.error + np.linalg.norm(lamb[bond_dimension:]) ** 2
            mat = (vt[:bond_dimension].T * lamb[:bond_dimension]).T
            next_node = PartialMps(error, mat, prefix)
            heappush(pq, next_node)


if __name__ == '__main__':
    data = np.random.random((32,32))
    perm = mps_uniform(data, 4)
    print(perm)
