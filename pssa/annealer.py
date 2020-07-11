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
        if input_graph.contains_edge(n1, n1_nb.val):
            delta -= 1
        if input_graph.contains_edge(n2, n1_nb.val):
            delta += 1
    for n2_nb in contact_graph.nodes[n2].neighbours:
        if n1 == n2_nb.val:
            continue
        if input_graph.contains_edge(n2, n2_nb.val):
            delta -= 1
        if input_graph.contains_edge(n1, n2_nb.val):
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
    delta = 0
    n_nb_set = set()
    for g_nb in iter(chimera_graph[g]):
        n_nb = inverse_embed[g_nb]
        if n_nb in n_nb_set or n_nb == n_from or n_nb == n_to:
            continue
        if input_graph.contains_edge(n_from, n_nb) \
                and not contact_graph.contains_edge(n_from, n_nb):
            delta += 1
        n_nb_set.add(n_nb)

# def perimeter_bfs():
