from random import seed

import networkx as nx

from template.faulty.bipartite_sat import BipartiteSAT
from template.faulty.quadripartite import Quadripartite
from template.util import *

times = []
found = 0
problems = 0
for i in range(0, 8):

    seed(10 + i)

    G = nx.generators.complete_graph(10)
    C = Chimera(16, 4).random_faulty(0)

    problems += 1

    # start = time.process_time()
    # em = find_embedding(G, C, verbose=True)
    # walltime = time.process_time() - start

    em, walltime = BipartiteSAT(G, C).solve(return_walltime=True)
    # em, walltime = Quadripartite(G, C).solve(return_walltime=True)

    if check_embedding(em, G, C):
        found += 1
        times.append(walltime)
        plot_chimera_embedding(em, C)

print("-------------------")
print("\n".join([str(t) for t in times]))
print(f"AVG: {sum(times) / len(times)}")
print(f"found {found} out of {problems}")
