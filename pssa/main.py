# embed = find_clique_embedding(k=12, m=3)
# from pssa.initializer import triangle_semi_clique_embed, divide_guiding_pattern
#
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt

# from pssa.initializer import triangle_semi_clique_embed, divide_guiding_pattern

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

# a = dnx.chimera_graph(2,2,4)


# input = nx.generators.complete_graph(5)
# input.add_edge(5, 6, weight=4)
#
# input.edges
# input.remove_edge(0, 1)
#
# t = FastMutableGraph(input)
#
# print(t.edges)
#
# t.remove_edge(1, 4)
#
# print(t.edges)
#
# random.sample()

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

G = dnx.chimera_graph(10, 10, 4)


def chimera_shortest_path_length(g1, g2):
    if g1 == g2:
        return 0
    lc = dnx.chimera_coordinates(10, t=4).linear_to_chimera
    (i1, j1, u1, k1) = lc(g1)
    (i2, j2, u2, k2) = lc(g2)
    dist = abs(i1 - i2) + abs(j1 - j2)
    dist += 2 if u1 == u2 else 1
    if u1 == 0 and u2 == 0 and (j1 - j2) == 0 and k1 == k2:
        dist -= 2
    if u1 == 1 and u2 == 1 and (i1 - i2) == 0 and k1 == k2:
        dist -= 2
    return dist

length = {}

for g1 in range(len(G)):
    length[g1] = {}
    for g2 in range(len(G)):
        if g2 < g1:
            length[g1][g2] = length[g2][g1]
        else:
            length[g1][g2] = chimera_shortest_path_length(g1, g2)

length2 = dict(nx.algorithms.all_pairs_shortest_path_length(G))

print(length == length2)

for i in range(len(length[0])):
    if length[0][i] != length2[0][i]:
        print(i)
        print(length[0][i])
        print(length2[0][i])

# pprint(G.is_directed())

# def boundary_bfs(chimera, chain):
#     chain_set = set(chain)
#     result = set()
#     for c in chain:
#         for nb in iter(G[c]):
#             if nb not in chain_set:
#                 result.add(nb)
#     return result


# r = boundary_bfs(G, [12, 8, 88])
# assert len(r) == 13
# pprint(r)
#
# pprint(G[0])
#

# target = nx.empty_graph(6)
# target.add_edge(0, 1)
# target.add_edge(0, 2)
# target.add_edge(1, 3)
# target.add_edge(2, 3)
# target.add_edge(2, 4)
# target.add_edge(3, 5)
# target.add_edge(4, 5)
# plt.ion()
# plt.figure(figsize=(20, 20))
# dnx.draw_chimera_embedding(G, embed)
# dnx.draw_chimera(G, with_labels=True)
# nx.draw(target, with_labels=True)

# small = dnx.chimera_graph(3, 3, 4)
# print(list(small.edges))

# plt.ion()
# dnx.draw_pegasus(prg)
