from typing import List, Dict
import numpy as np
import dwave_networkx as dnx

M = 16
L = 4

chimera = dnx.chimera_graph(M, M, L)

LP = {i: [] for i in range(M * L)}
left_start = []
RP = {i: [] for i in range(M * L)}
right_start = []

for i in range(0, 2 * M * L - 1, 8):
    for j in range(L):
        left_start.append(i + j)
for i in range(4, 2 * M * M * L + 3, 2 * M * L):
    for j in range(L):
        right_start.append(i + j)


def create_chains(start_nodes: List[int], partition: Dict[int, List[int]]):
    count = 0
    for node in start_nodes:
        partition[count].append(node)
        next_node = node
        for i in range(M - 1):
            next_node = list(chimera.adj.get(next_node).keys())[-1]
            partition[count].append(next_node)
        count += 1


create_chains(left_start, LP)
create_chains(right_start, RP)

A = np.ones((M * L, M * L), dtype=int)

print(A)
print(RP)
print(LP)
