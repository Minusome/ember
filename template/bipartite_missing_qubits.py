import pulp
import networkx as nx

M = 16
L = 4

G = nx.generators.complete_graph(M * L)

# bipartite nodes
K = range(2 * M * L)
# problem nodes
I = range(len(G))
# bipartite edges
L = [(a, b) for a in range(M * L) for b in range(M * L)]
# problem edges
E = G.edges

model = pulp.LpProblem("MIP_Model", pulp.LpMaximize)

# Decision variables
y = pulp.LpVariable.dicts("y", (I, K), cat="Binary")
y_edge = pulp.LpVariable.dicts("y", (E, L), cat="Binary")
# y_prime = pulp.LpVariable.dict("y_prime", E, cat="Binary")

# Objective
model += pulp.lpSum(y_edge)

# Constraints
for e in E:
    for l in L:
        model += 2 * y_edge[e][l] <= y[e[0]][l[0]] + y[e[1]][l[1]] + y[e[0]][l[1]] + y[e[1]][l[0]]

for i in I:
    model += pulp.lpSum([y[i][k] for k in K]) <= 1

for k in K:
    model += pulp.lpSum([y[i][k] for i in I]) <= 1

gurobi = pulp.apis.GUROBI()

if gurobi.available():
    model.solve(solver=gurobi)
else:
    model.solve()

# Check embedding found
if model.objective.value() == len(G):
    print("Found valid embedding")
else:
    print("Valid embedding not found")
