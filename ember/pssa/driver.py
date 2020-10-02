import random

import dwave_networkx as dnx
import networkx as nx
from dwave import embedding
from matplotlib import pyplot as plt

from ember.pssa.annealer import run_simulated_annealing
from ember.pssa.context import OptimizationContext
from ember.pssa.initializer import triangle_semi_clique_embed, divide_guiding_pattern
from ember.template.util import plot_chimera_embedding

random.seed(1)

G = dnx.chimera_graph(16, 16, 4)  # Dwave 2000q arch
input = nx.generators.fast_gnp_random_graph(70, 0.2, seed=1)

print(input.number_of_edges())

# embed = find_embedding(input, G, verbose=1)

gp = triangle_semi_clique_embed(16, 4)

plot_chimera_embedding(gp, G)
exit()

initial = divide_guiding_pattern(gp, len(input))

context = OptimizationContext(16, 4, input, gp)
embed = run_simulated_annealing(context, initial)

embed = {i: embed[i] for i in range(len(embed))}

diag = embedding.diagnose_embedding(embed, input, G)
missing = 0
for prob in diag:
    missing += 1
    print(prob)

print(f"Double check cost: {input.number_of_edges() - missing}")

_, cost = context.create_contact_graph(embed)

print("Double check 2: {}", cost)

plt.ion()
plt.figure(figsize=(20, 20))
dnx.draw_chimera_embedding(G, embed)
