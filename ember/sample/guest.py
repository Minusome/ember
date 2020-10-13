import networkx as nx
import random
"""
Reference: T. D. Goodrich, B. D. Sullivan, and T. S. Humble, 
“Optimizing adiabatic quantum program compilation using a graph-theoretic framework,” 
Quantum Information Processing, vol. 17, no. 5, p. 118, 2018.
URL: https://github.com/TheoryInPractice
"""


def barabasi_albert_graph(n: int, density: str, seed=0):
    density_ratio = {"low": 0.25, "medium": 0.50, "high": 0.75}
    m = max(round(density_ratio[density] * n), 1)
    guest = nx.barabasi_albert_graph(n, m, seed=seed)
    return guest


def d_regular_graph(n: int, density: str, seed=0):
    density_ratio = {"low": 0.25, "medium": 0.50, "high": 0.75}
    d = density_ratio[density] * n

    if (n * round(d)) % 2 == 1:
        if d - int(d) > 0.5:
            d = int(d)
        else:
            d = int(d) + 1
    else:
        d = round(d)

    guest = nx.random_regular_graph(d, n, seed=seed)
    return guest


def erdos_reyni_graph(n: int, density: str, seed=0):
    density_ratio = {"low": 0.25, "medium": 0.50, "high": 0.75}
    p = density_ratio[density]
    guest = nx.gnp_random_graph(n, p, seed=seed)
    return guest


def noisy_bipartite_graph(n: int, density: str, seed=0):
    density_ratio = {"low": 0.25, "medium": 0.5, "high": 0.75}
    p = density_ratio[density]
    guest = _random_noisy_bipartite_graph(n, p, seed=seed)
    return guest


def _random_noisy_bipartite_graph(n, p, seed):
    """Generates a base bipartite graph with evently split nodes.
    Adds a bipatite edge with probability p.
    Add every edge with probability 1/10.
    Parameters
    ----------
    n : int
        Number of nodes
    p : double
        Probability of bipartite edge
    seed : int
        Seed for pseudorandom number generator
    Returns
    -------
    NeworkX Graph
        The random noisy bipartite graph
    """
    random.seed(seed)
    noisy_p = p / 5
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            # Random bipartite edge
            temp1 = random.random()
            if temp1 < p and (i - j) % 2 == 1:
                G.add_edge(i, j)
            # Noisy edge
            temp2 = random.random()
            if temp2 < noisy_p:
                G.add_edge(i, j)
    return G
