# embed = find_clique_embedding(k=12, m=3)
# from pssa.initializer import triangle_semi_clique_embed, divide_guiding_pattern
#
import dwave_networkx as dnx
import networkx as nx
from dwave import embedding as de
import matplotlib.pyplot as plt
from minorminer import find_embedding
import pickle
import random

from pssa.annealer import run_simulated_annealing
from pssa.context import OptimizationContext
from pssa.initializer import triangle_semi_clique_embed, divide_guiding_pattern

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

# input = nx.readwrite.read_adjlist("../test_graphs/DWaveTwo.alist")

# G = nx.readwrite.read_adjlist("")

random.seed(1)

G = dnx.chimera_graph(16, 16, 4)  # Dwave 2000q arch
input = nx.generators.fast_gnp_random_graph(66, 0.2, seed=1)

guiding_pattern = triangle_semi_clique_embed(16,4)

print(len(guiding_pattern))
print(len(input))

initial = divide_guiding_pattern(guiding_pattern, len(input))

print(len(initial))

# print(initial)

# context = OptimizationContext(16, 4, input, guiding_pattern)
#
# embed = run_simulated_annealing(context, initial)
#
# print(embed)




# emb = find_embedding(input, G)



# with open("../emb", "rb") as file:
#     emb = pickle.load(file)
#
# diag = de.diagnose_embedding(emb, input, G)




# with open("../emb", "wb") as file:
#     pickle.dump(emb, file)

# def read_graph(infile):  # CS220 adjacency list format
#     n=int(infile.readline().strip())
#     G=nx.empty_graph(n,create_using=nx.Graph())
#     for u in range(n):
#         neighbors=infile.readline().split()
#         for v in neighbors: G.add_edge(u,int(v))
#     return G
#
# with open("../test_graphs/guests2/Q6.alist") as file:
#     G = read_graph(file)

# with open("../test_graphs/guests2/Ljubljana.alist", "r") as file:
#     input = file.read()
#     input = input.split("\n")[1:]
#     input = [s for s in input if s]
#     G = nx.readwrite.parse_adjlist(input)
#


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
plt.ion()
plt.figure(figsize=(20, 20))
dnx.draw_chimera_embedding(G, initial)
# dnx.draw_chimera(G, with_labels=True)
# nx.draw(input, with_labels=True)

# small = dnx.chimera_graph(3, 3, 4)
# print(list(small.edges))

# plt.ion()
# dnx.draw_pegasus(prg)
