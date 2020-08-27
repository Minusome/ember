import cvxpy as cvx
import networkx as nx

M = 16
L = 4

# Clique graph as input
# G = nx.generators.complete_graph(M * L)
G = nx.generators.gnp_random_graph(75, 0.3, seed=10)

# Decision variables
y = cvx.Variable((2, len(G)), "y", boolean=True)
y_prime = cvx.Variable(len(G), "y_prime", boolean=True)

# Objective
obj = cvx.Maximize(cvx.sum(y_prime))

# Constraints
constr = []
constr.append(y_prime <= y[0] + y[1])
constr.append(cvx.sum(y, axis=1) <= M * L)

for i, j in G.edges:
    constr.append(y[0][i] + y[0][j] - y[1][i] - y[1][j] <= 1)
    constr.append(y[1][i] + y[1][j] - y[0][i] - y[0][j] <= 1)

prob = cvx.Problem(obj, constr)
prob.solve(verbose=True, solver=cvx.GUROBI)

# Check embedding found
if prob.value == len(G):
    print("Found valid embedding")
else:
    print("Valid embedding not found")

# Check clique
for i in range(len(G)):
    assert y[0][i].value == 1
    assert y[1][i].value == 1
