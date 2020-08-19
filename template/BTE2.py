from pulp import *

import dwave_networkx as dnx
import networkx as nx

m = 16
l = 4
chimera_graph = dnx.chimera_graph(m, m, l)

G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])

nodes = G.nodes
set_k = [1, 2]
set_I = range(G.number_of_nodes())

model = LpProblem("BTE", LpMaximize)
y = LpVariable.dicts("Y", [(i, k) for i in set_I for k in set_k], lowBound=0, upBound=1, cat='Binary')
y_p = LpVariable.dicts("Y_P", [i for i in set_I], lowBound=0, upBound=1, cat='Binary')
model += lpSum([y_p[i] for i in set_I])
model += lpSum(y[(i, 1)] for i in set_I) <= m * l
model += lpSum(y[(i, 2)] for i in set_I) <= m * l
for i in set_I:
    model += y_p[i] <= y[(i, 1)] + y[(i, 2)]
for (i, j) in G.edges:
    model += y[(i, 1)] + y[(j, 1)] - y[(i, 2)] - y[(j, 2)] <= 1
    model += y[(i, 2)] + y[(j, 2)] - y[(i, 1)] - y[(j, 1)] <= 1
model.solve()

print("Status:", LpStatus[model.status])
for v in model.variables():
    print(v.name, "=", v.varValue)
