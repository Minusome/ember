import math
from typing import Dict, List

import dwave_networkx as dnx
import numpy as np

from ember.hardware.chimera import ChimeraGraph


def overlap_clique(chimera_graph: ChimeraGraph):
    m,l = chimera_graph.params
    to_linear = dnx.chimera_coordinates(m, t=l).chimera_to_linear

    # Embed the clique major
    top_embed = [[] for _ in range(m * l)]
    for i in range(m * l):
        cell, unit = i // l, i % l
        # Add the nodes above diagonal cell
        for j in range(cell):
            top_embed[i].append(to_linear((j, cell, 0, unit)))
        # Add the two nodes in the diagonal cell
        top_embed[i].extend((to_linear((cell, cell, 0, unit)),
                             to_linear((cell, cell, 1, unit))))
        # Add the entire row
        for j in range(0, m):
            top_embed[i].append(to_linear((cell, j, 1, unit)))

    # Embed the clique minor
    bot_embed = [[] for _ in range((m - 1) * l)]
    for i in range((m - 1) * l):
        cell, unit = i // l, i % l
        for j in range(cell, m - 1):
            bot_embed[i].append(to_linear((j + 1, cell, 0, unit)))

    combined = top_embed + bot_embed

    return {i: combined[i] for i in range(len(combined))}


def double_triangle_clique(chimera_graph: ChimeraGraph) -> Dict[int, List[int]]:
    """
    Performs a double-sided triangle embedding in a similar fashion as described by the PSSA paper.
    'Graph Minors from Simulated Annealing for Annealing Machines with Sparse Connectivity' by
    Sugie et al.

    :param m: Number of rows and columns of bi-cliques in the chimera hardware G (i.e. G_{m,m,l})
    :param l: The number of nodes in one half of the bi-clique (i.e. k_{l, l})

    :return: an Embedding. !NOTE: The vertex sets are represented as lists but the ordering
    matters. The nodes should be ordered according to the chain formed on the hardware hardware.
    """
    m, l = chimera_graph.params
    to_linear = dnx.chimera_coordinates(m, t=l).chimera_to_linear

    # Embed the upper triangular
    top_embed = [[] for _ in range(m * l)]
    for i in range(m * l):
        cell, unit = i // l, i % l
        # Add the nodes above diagonal cell
        for j in range(cell):
            top_embed[i].append(to_linear((j, cell, 0, unit)))
        # Add the two nodes in the diagonal cell
        top_embed[i].extend((to_linear((cell, cell, 0, unit)),
                             to_linear((cell, cell, 1, unit))))
        # Add the nodes to right of diagonal cell
        for j in range(cell + 1, m):
            top_embed[i].append(to_linear((cell, j, 1, unit)))

    # Embed the lower triangular
    bot_embed = [[] for _ in range((m - 1) * l)]
    for i in range((m - 1) * l):
        cell, unit = i // l, i % l
        # Add the nodes to left of diagonal cell
        for j in range(cell):
            bot_embed[i].append(to_linear((cell + 1, j, 1, unit)))
        # Add the two nodes in the diagonal cell
        bot_embed[i].extend((to_linear((cell + 1, cell, 1, unit)),
                             to_linear((cell + 1, cell, 0, unit))))
        # Add the nodes below diagonal cell
        for j in range(cell + 1, m - 1):
            bot_embed[i].append(to_linear((j + 1, cell, 0, unit)))

    combined = top_embed

    return {i: combined[i] for i in range(len(combined))}


def klymko_max_clique(chimera_graph: ChimeraGraph) -> Dict[int, List[int]]:
    """
    Algorithm adapted from 'Adiabatic Quantum Computing: Minor Embedding with Hard Faults' by
    Klymko, Sullivan and Humble. source: https://arxiv.org/pdf/1210.8395.pdf

    :param m: Number of rows and columns of bi-cliques in the chimera hardware G (i.e. G_{m,m,l})
    :param l: The number of nodes in one half of the bi-clique (i.e. k_{l, l})

    :return: an Embedding.
    """
    m, l = chimera_graph.params
    V = np.zeros((2 * m + 1, l * m + 2))
    for i in range(1, l * m + 2):
        if i < l:
            r, s = 1, i
        elif i > l + 1:
            r = math.ceil((i - 1) / l)
            s = (i - 1) % l
            if s == 0:
                s = l
        else:
            continue

        for j in range(1, m + 1):
            V[j, i] = 2 * l * m * (r - 1) + 2 * l * (j - 1) + l + s
        for j in range(1, m + 1):
            V[j + m, i] = 2 * l * (r - 1) + 2 * l * m * (j - 1) + s

    for j in range(1, m + 1):
        V[j, l] = l + (j - 1) * 2 * l * m
        V[j, l + 1] = j * 2 * l

    # Original algorithm used 1-indexing, so the first row & column are all zero and can be deleted
    V = V[1:, 1:]
    V = V.T

    # Generate hardware and embedding (and fix 1-indexing)
    embed = {i: [x - 1 for x in V[i] if x != 0] for i in range(l * m + 1)}

    return embed
