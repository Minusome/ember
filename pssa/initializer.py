import math
import random
from typing import List, Dict

import dwave_networkx as dnx
import numpy as np


def divide_guiding_pattern(guiding_pattern: Dict[int, List[int]],
                           vertex_count: int, strategy: str = "balanced") -> Dict[int, List[int]]:
    """
    Randomly splits the vertex sets in the guiding pattern to create an initial embedding of the
    input graph into the hardware graph.

    :param guiding_pattern: A known clique (or near clique) embedding !NOTE: The vertex sets are
    represented as lists but the ordering matters. The nodes should be ordered according to the
    chain formed on the hardware graph.
    :param vertex_count: The number of vertices in the input graph (i.e. The number of vertex
    sets in the final embedding)
    :param strategy: Either 'random' (all vertex sets have equal chance of being divided) or
    'balanced' (large vertex sets get divided first)
    :return: an initial embedding with vertex_count equal to number of vertex sets
    """
    if vertex_count <= len(guiding_pattern):
        return {i: guiding_pattern[i] for i in range(vertex_count)}

    vertex_sets = list(guiding_pattern.values())
    selection_window = len(vertex_sets)

    while len(vertex_sets) < vertex_count:
        if strategy == "balanced":
            rand_idx = random.randrange(0, selection_window)
            selection_window -= 1
            if selection_window == 0:
                selection_window = len(vertex_sets)
        elif strategy == "random":
            rand_idx = random.randrange(0, len(vertex_sets))
        else:
            raise Exception("Unsupported strategy: {}".format(strategy))

        # Divide a selected vertex set into two vertex sets
        # Is this slow? idk
        vertex_sets = \
            vertex_sets[:rand_idx] + \
            vertex_sets[rand_idx + 1:] + \
            [vertex_sets[rand_idx][:len(vertex_sets[rand_idx]) // 2]] + \
            [vertex_sets[rand_idx][len(vertex_sets[rand_idx]) // 2:]]

    assert len(vertex_sets) == vertex_count
    return {i: vertex_sets[i] for i in range(vertex_count)}


def triangle_semi_clique_embed(m: int, l: int) -> Dict[int, List[int]]:
    """
    Performs a double-sided triangle embedding in a similar fashion as described by the PSSA paper.
    'Graph Minors from Simulated Annealing for Annealing Machines with Sparse Connectivity' by 
    Sugie et al.

    :param m: Number of rows and columns of bi-cliques in the chimera graph G (i.e. G_{m,m,l})
    :param l: The number of nodes in one half of the bi-clique (i.e. k_{l, l})
    
    :return: an Embedding. !NOTE: The vertex sets are represented as lists but the ordering
    matters. The nodes should be ordered according to the chain formed on the hardware graph.
    """
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


def max_clique_embed(m: int, l: int) -> Dict[int, List[int]]:
    """
    Algorithm adapted from 'Adiabatic Quantum Computing: Minor Embedding with Hard Faults' by
    Klymko, Sullivan and Humble. source: https://arxiv.org/pdf/1210.8395.pdf

    :param m: Number of rows and columns of bi-cliques in the chimera graph G (i.e. G_{m,m,l})
    :param l: The number of nodes in one half of the bi-clique (i.e. k_{l, l})

    :return: an Embedding.
    """
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

    # Generate graph and embedding (and fix 1-indexing)
    embed = {i: [x - 1 for x in V[i] if x != 0] for i in range(l * m + 1)}

    return embed
