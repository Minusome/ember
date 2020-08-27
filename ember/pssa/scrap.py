import dwave_networkx as dnx

m = 1
l = 4
cg = dnx.chimera_graph(m, m, l)
print(cg[1])
# for g_to_nb in iter(cg[5]):
#     print(g_to_nb)
