import pulp
import networkx as nx

M = 16
L = 4

# Clique graph as input
G = nx.generators.complete_graph(M * L)

K = [1, 2]
I = range(len(G))

model = pulp.LpProblem("MIP_Model", pulp.LpMaximize)

# Decision variables
y = pulp.LpVariable.dicts("y", (I, K), cat="Binary")
y_prime = pulp.LpVariable.dict("y_prime", I, cat="Binary")

# Objective
model += pulp.lpSum(y_prime)

# Constraints
for i in I:
    model += y_prime[i] <= y[i][1] + y[i][2]

for k in K:
    model += pulp.lpSum([y[i][k] for i in I]) <= M * L

for i, j in G.edges:
    model += y[i][1] + y[j][1] - y[i][2] - y[j][2] <= 1
    model += y[i][2] + y[j][2] - y[i][1] - y[j][1] <= 1

gurobi = pulp.apis.GUROBI()

if gurobi.available():
    model.solve(solver=gurobi)
else:
    model.solve()

# Check embedding found
if model.objective.value() == len(G):
    print("Found valid embedding")

# Check clique
for i in I:
    assert y[i][1].value() == 1
    assert y[i][2].value() == 1
