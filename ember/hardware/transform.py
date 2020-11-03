import math
from typing import Dict, List

import dwave_networkx as dnx
import numpy as np

from ember.hardware.chimera import ChimeraGraph


def quadripartite_with_faults(chimera_graph: ChimeraGraph):
    """
    Create a quadripartite template embedding which allows for faulty nodes and edges.

    Args:
        chimera_graph: Chimera host to transform.

    Returns: Tuple (U1, U2, U3, U4) for embedding in each respective partition.
    """

    def append_nonempty(super, sub):
        if sub:
            super.append(sub)

    m, l = chimera_graph.params
    faulty = chimera_graph.faulty_nodes
    to_linear = dnx.chimera_coordinates(m, t=l).chimera_to_linear

    U1, U4 = [], []
    for i in range(m * l):
        chain1, chain4 = [], []
        cell, unit = i // l, i % l
        for j in range(m):
            ln = to_linear((cell, j, 1, unit))
            if ln in faulty:
                if i < m * l / 2:
                    append_nonempty(U1, chain1)
                    chain1 = []
                else:
                    append_nonempty(U4, chain4)
                    chain4 = []
            else:
                if i < m * l / 2:
                    chain1.append(ln)
                else:
                    chain4.append(ln)
        append_nonempty(U1, chain1)
        append_nonempty(U4, chain4)

    U2, U3 = [], []
    for i in range(m * l):
        chain2, chain3 = [], []
        cell, unit = i // l, i % l
        for j in range(m):
            ln = to_linear((j, cell, 0, unit))
            if ln in faulty:
                if j < m / 2:
                    append_nonempty(U2, chain2)
                    chain2 = []
                else:
                    append_nonempty(U3, chain3)
                    chain3 = []
            else:
                if j < m / 2:
                    chain2.append(ln)
                else:
                    chain3.append(ln)
        append_nonempty(U2, chain2)
        append_nonempty(U3, chain3)

    return U1, U2, U3, U4


def bipartite_with_faults(chimera_graph: ChimeraGraph):
    """
    Create a bipartite template embedding which allows for faulty nodes and edges.

    Args:
        chimera_graph: Chimera host to transform.

    Returns: Tuple (left, right) for embedding in each respective partition.
    """

    def append_nonempty(super, sub):
        if sub:
            super.append(sub)

    m, l = chimera_graph.params
    faulty = chimera_graph.faulty_nodes
    to_linear = dnx.chimera_coordinates(m, t=l).chimera_to_linear

    h_embed = []
    for i in range(m * l):
        chain = []
        cell, unit = i // l, i % l
        for j in range(m):
            ln = to_linear((cell, j, 1, unit))
            if ln in faulty:
                append_nonempty(h_embed, chain)
                chain = []
            else:
                chain.append(ln)
        append_nonempty(h_embed, chain)

    v_embed = []
    for i in range(m * l):
        chain = []
        cell, unit = i // l, i % l
        for j in range(m):
            ln = to_linear((j, cell, 0, unit))
            if ln in faulty:
                append_nonempty(v_embed, chain)
                chain = []
            else:
                chain.append(ln)
        append_nonempty(v_embed, chain)

    return h_embed, v_embed


def overlap_clique(chimera_graph: ChimeraGraph):
    """
    Returns a clique overlap template embedding as described in 'Template-based minor embedding
    for adiabatic quantum optimization'.

    Reference:
        T. Serra, T. Huang, A. Raghunathan, and D. Bergman, Template-based minor embedding for
        adiabatic quantum optimization, 2019. arXiv:1910.02179 [cs.DS].

    Args:
        chimera_graph: Chimera host to transform.

    Returns: an embedding.
    """
    m, l = chimera_graph.params
    to_linear = dnx.chimera_coordinates(m, t=l).chimera_to_linear

    # Embed the clique major
    top_embed = [[] for _ in range(m * l)]
    for i in range(m * l):
        cell, unit = i // l, i % l
        # Add the nodes above diagonal cell
        for j in range(cell):
            top_embed[i].append(to_linear((j, cell, 0, unit)))
        # Add the two nodes in the diagonal cell
        top_embed[i].extend((to_linear(
            (cell, cell, 0, unit)), to_linear((cell, cell, 1, unit))))
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
    'Graph Minors from Simulated Annealing for Annealing Machines with Sparse Connectivity'

    Reference:
        Y. Sugie, Y. Yoshida, N. Mertig, T. Takemoto, H. Teramoto, A. Nakamura, I. Takigawa, S.-i.
        Minato, M.Yamaoka, and T. Komatsuzaki, Minor-embedding heuristics for large-scale
        annealing processors with sparse hardware graphs of up to 102,400 nodes,
        2020. arXiv:2004.03819 [quant-ph]:

    Args:
        chimera_graph: Chimera host to transform.

    Returns: an embedding.
        !NOTE: The vertex sets are represented as lists but the ordering matters.
        The nodes should be ordered according to the chain formed on the hardware hardware.
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
        top_embed[i].extend((to_linear(
            (cell, cell, 0, unit)), to_linear((cell, cell, 1, unit))))
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
        bot_embed[i].extend((to_linear(
            (cell + 1, cell, 1, unit)), to_linear((cell + 1, cell, 0, unit))))
        # Add the nodes below diagonal cell
        for j in range(cell + 1, m - 1):
            bot_embed[i].append(to_linear((j + 1, cell, 0, unit)))

    combined = top_embed + bot_embed

    return {i: combined[i] for i in range(len(combined))}


def klymko_max_clique(chimera_graph: ChimeraGraph) -> Dict[int, List[int]]:
    """
    Algorithm adapted from 'Adiabatic Quantum Computing: Minor Embedding with Hard Faults'

    References:
        Klymko C, Sullivan BD, Humble TS (2014) Adiabatic quantum programming: minor embedding with
        hard faults. Quantum Information Processing 13(3):709–729

    Args:
        chimera_graph: Chimera host to transform.

    Returns: an embedding.
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
