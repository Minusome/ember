import cvxpy as cvx
import networkx as nx

M = 16
L = 4

# Clique graph as input
G = nx.generators.complete_graph(M * L)

# Decision variables
y = cvx.Variable((len(G), 2), "y", boolean=True)
y_prime = cvx.Variable(len(G), "y_prime", boolean=True)

# Objective
obj = cvx.Maximize(cvx.sum(y_prime))

# Constraints
constr = []
for i in range(len(G)):
    constr.append(y_prime[i] <= y[i][0] + y[i][1])

constr.append(cvx.sum(y, axis=0) <= M * L)

for i, j in G.edges:
    constr.append(y[i][0] + y[j][0] - y[i][1] - y[j][1] <= 1)
    constr.append(y[i][1] + y[j][1] - y[i][0] - y[j][0] <= 1)

prob = cvx.Problem(obj, constr)
prob.solve(verbose=True, solver=cvx.GUROBI)

# Check embedding found
if prob.value == len(G):
    print("Found valid embedding")

# Check clique
for i in range(len(G)):
    assert y[i][0].value == 1
    assert y[i][1].value == 1
