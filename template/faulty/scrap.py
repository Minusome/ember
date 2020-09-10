from collections import defaultdict

import dwave_networkx as dnx
import numpy as np
from ortools.sat.python import cp_model

from template.util import Chimera


def append_nonempty(super, sub):
    if sub:
        super.append(sub)


C = Chimera(2, 4).random_faulty(0)
M, L, faulty = C.M, C.L, C.faulty
to_linear = dnx.chimera_coordinates(M, t=L).chimera_to_linear
U1 = []
for i in range(int(M * L / 2)):
    chain = []
    cell, unit = i // L, i % L
    for j in range(M):
        ln = to_linear((cell, j, 1, unit))
        if ln in faulty:
            append_nonempty(U1, chain)
            chain = []
        else:
            chain.append(ln)
    append_nonempty(U1, chain)

U2 = []
for i in range(M * L):
    chain = []
    cell, unit = i // L, i % L
    for j in range(int(M/2)):
        ln = to_linear((j, cell, 0, unit))
        if ln in faulty:
            append_nonempty(U2, chain)
            chain = []
        else:
            chain.append(ln)
    append_nonempty(U2, chain)

U3 = []
for i in range(M * L):
    chain = []
    cell, unit = i // L, i % L
    for j in range(int(M/2), M):
        ln = to_linear((j, cell, 0, unit))
        if ln in faulty:
            append_nonempty(U3, chain)
            chain = []
        else:
            chain.append(ln)
    append_nonempty(U3, chain)

U4 = []
for i in range(int(M * L / 2), M * L):
    chain = []
    cell, unit = i // L, i % L
    for j in range(M):
        ln = to_linear((cell, j, 1, unit))
        if ln in faulty:
            append_nonempty(U4, chain)
            chain = []
        else:
            chain.append(ln)
    append_nonempty(U4, chain)

print(U1)
print(U2)
print(U3)
print(U4)
