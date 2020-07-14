from collections import defaultdict

from pssa.context import OptimizationContext
from pssa.graph import FastMutableGraph
from pssa.types import *


def run_simulated_annealing(context: OptimizationContext, initial_embed: Embedding):
    # Current solution + data helpers
    forward_embed = initial_embed
    reverse_embed = {l: k for k, ll in initial_embed.items() for l in ll}

    # Need to initialize the contact graph
    contact_graph = FastMutableGraph()

    # Best solution found
    cost_best = 0
    forward_embed_best = initial_embed


#
#     for step in range(T_MAX):
#         temperature, shift, any_dir = move_params(step)
#         # swap
#         if not shift:
#             i,k = random.choice(context.input_edge_list)
#             j = random.choice(tuple(contact_graph.nodes[k].neighbours)).val
#
#         # shift
#         else:


def delta_swap(input_graph: FastMutableGraph, contact_graph: FastMutableGraph, n1: int, n2: int):
    delta = 0
    for n1_nb in contact_graph.nodes[n1].neighbours:
        if n2 == n1_nb.val:
            continue
        if input_graph.has_edge(n1, n1_nb.val):
            delta -= 1
        if input_graph.has_edge(n2, n1_nb.val):
            delta += 1
    for n2_nb in contact_graph.nodes[n2].neighbours:
        if n1 == n2_nb.val:
            continue
        if input_graph.has_edge(n2, n2_nb.val):
            delta -= 1
        if input_graph.has_edge(n1, n2_nb.val):
            delta += 1
    return delta


def swap(contact_graph: FastMutableGraph, forward_embed: Embedding,
         inverse_embed: Inverse_Embedding, n1: int, n2: int):
    for g1 in forward_embed[n1]:
        inverse_embed[g1] = n2
    for g2 in forward_embed[n2]:
        inverse_embed[g2] = n1
    forward_embed[n1], forward_embed[n2] = forward_embed[n2], forward_embed[n1]
    contact_graph.swap_node(n1, n2)


def delta_shift(input_graph, contact_graph, chimera_graph, inverse_embed, g_from, g_to):
    n_from = inverse_embed[g_from]
    n_to = inverse_embed[g_to]
    n_nb_count = defaultdict(int)
    delta = 0
    for g_to_nb in iter(chimera_graph[g_to]):
        n_to_nb = inverse_embed[g_to_nb]
        if n_to_nb == n_from or n_to_nb == n_to or n_to_nb in n_nb_count:
            continue
        if not contact_graph.has_edge(n_from, n_to_nb) \
                and input_graph.has_edge(n_from, n_to_nb):
            delta += 1
        n_nb_count[n_to_nb] += 1
    for n_to_nb, count in n_nb_count.items():
        assert contact_graph.edge_weight(n_to_nb, n_to) >= count  # Debug
        if contact_graph.edge_weight(n_to_nb, n_to) == count \
                and input_graph.has_edge(n_to_nb, n_to):
            delta -= 1
    return delta


def shift(input_graph, contact_graph, chimera_graph, forward_embed, inverse_embed, g_from, g_to):
    n_from = inverse_embed[g_from]
    n_to = inverse_embed[g_to]

    # Update forward embed
    if forward_embed[n_from][-1] == g_from:
        forward_embed[n_from].append(g_to)
    else:
        forward_embed[n_from].append_left(g_to)
    if forward_embed[n_to][-1] == g_to:
        forward_embed[n_to].pop()
    else:
        forward_embed[n_to].popleft()

    # Update inverse embed
    inverse_embed[g_to] = n_from

    # Update contact graph
