import pulp as plp
import pandas as pd

import dwave_networkx as dnx
import networkx as nx

m = 16
l = 4
chimera_graph = dnx.chimera_graph(m, m, l)

G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])

print(G.edges)
print(G.nodes)

nodes = G.nodes
set_k = [1, 2]
set_I = range(G.number_of_nodes())

opt_model = plp.LpProblem(name="MIP_Model")
y = {(i, k): plp.LpVariable(cat=plp.LpBinary, name="y_{}{}".format(i, k), lowBound=0, upBound=1) for i in set_I for k in
     set_k}
y_prime = {i: plp.LpVariable(cat=plp.LpBinary, name="y_prime{}".format(i), lowBound=0, upBound=1) for i in set_I}

# print(y[])
# print(y_prime)
constraints = {
    "yp" + str(i): opt_model.addConstraint(
        plp.LpConstraint(e=y_prime[i], sense=plp.LpConstraintLE, rhs=plp.lpSum([y[(i, 1)], y[(i, 2)]]).value(),
                         name="yp_{0}".format(i))) for i in set_I
}
constraints["sum_y"] = opt_model.addConstraint(
    plp.LpConstraint(e=plp.lpSum(y[(i, k)] for i in set_I for k in set_k), sense=plp.LpConstraintLE, rhs=m * l,
                     name="sum_y"))
for (i, j) in G.edges:
    constraints["edge1_{}{}".format(i, j)] = opt_model.addConstraint(
        plp.LpConstraint(e=y[(i, 1)] + y[(j, 1)] - y[(i, 2)] - y[(j, 2)], sense=plp.LpConstraintLE, rhs=1,
                         name="edge1_{}{}".format(i, j)))
    constraints["edge2_{}{}".format(i, j)] = opt_model.addConstraint(
        plp.LpConstraint(e=y[(i, 2)] + y[(j, 2)] - y[(i, 1)] - y[(j, 1)], sense=plp.LpConstraintLE, rhs=1,
                         name="edge2_{}{}".format(i, j)))

objective = plp.lpSum(y_prime[i] for i in set_I)
opt_model.sense = plp.LpMaximize

opt_model.solve()

for i in set_I:
    print(y[(i, 1)].value(), y[(i, 2)].value(), y_prime[i].value())
# opt_df = pd.DataFrame.from_dict(y_prime, orient="index", columns=["variable_object"])
# opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
# opt_df.reset_index(inplace=True)
#
# opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
# opt_df.drop(columns=["variable_object"], inplace=True)
# opt_df.to_csv("./optimization_solution.csv")
