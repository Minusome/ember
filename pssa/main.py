import networkx as nx

# embed = find_clique_embedding(k=12, m=3)
from pssa.graph import FastMutableGraph

# embed = triangle_semi_clique_embed(10, 4)
# embed = divide_guiding_pattern(embed, 100, strategy="balanced")
# guide = {}
# for k, ll in embed.items():
#     for l in ll:
#         guide[l] = k
#
# guide = {l: k for k, ll in embed.items() for l in ll}
#
# pprint(guide)

input = nx.generators.complete_graph(5)

input.edges
input.remove_edge(0, 1)

t = FastMutableGraph(input)

print(t.edges)

t.remove_edge(1, 4)

print(t.edges)

# print(t)
#
# t.swap(0, 4)
#
# print(t)
#
# t.add_edge(1, 4)
#
# print(t)
#
# t.add_edge(4, 1)
#
# print(t)
#
# t.remove_edge(1, 4)
#
# print(t)

#
#     pprint(node)
#     pprint(adj_dict)
#
# pprint(input.nodes(data=True))
# pprint(input.edges)

# input = nx.relabel_nodes(input, {0: 4, 4: 0}, copy=False)

# pprint(input.nodes)
# pprint(input.edges)

# e = {e for e in input.edges}
#
# pprint(e)

# G = dnx.chimera_graph(10, 10, 4)
# pprint(G.is_directed())
#
# plt.ion()
# plt.figure(figsize=(20, 20))
# dnx.draw_chimera_embedding(G, embed)
# dnx.draw_chimera(G, with_labels=True)
# nx.draw(input)

# small = dnx.chimera_graph(3, 3, 4)
# print(list(small.edges))

# plt.ion()
# dnx.draw_pegasus(prg)
