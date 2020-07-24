import math
import random
import copy
from collections import defaultdict, deque
from typing import List, Dict

from pssa.context import OptimizationContext
from pssa.graph import FastMutableGraph
from pssa.schedule import T_MAX, move_params


def run_simulated_annealing(context: OptimizationContext, initial_embed: Dict[int, List[int]]):
    forward_embed = [deque(initial_embed[i]) for i in range(len(initial_embed))]
    inverse_embed = {l: k for k, ll in initial_embed.items() for l in ll}

    # Fill in -1 for unmapped G nodes
    for i in range(len(context.chimera_graph)):
        if i not in inverse_embed:
            inverse_embed[i] = -1

    # Initialize contact graph and cost
    contact_graph, cost = context.create_contact_graph(initial_embed)

    # Best solution found
    cost_best = 0
    forward_embed_best = forward_embed.copy()

    for step in range(T_MAX):
        temperature, shift_mode, any_dir = move_params(step)
        # print("temp: {}\tshift?: {}\tany_dir?: {}".format(temperature, shift_mode, any_dir))
        if not shift_mode:  # swap
            n1, n1_nb = random.choice(context.input_edge_list)
            n2 = random.choice(tuple(contact_graph.nodes[n1_nb].neighbours)).val
            delta = delta_swap(context.input_graph, contact_graph, n1, n2)
        else:  # shift
            n_to = random.randrange(context.input_graph.num_nodes)
            if len(forward_embed[n_to]) < 2:
                continue
            g_to = forward_embed[n_to][0] if random.getrandbits(1) == 0 else forward_embed[n_to][-1]
            cand = []
            for g_to_nb in iter(context.chimera_graph[g_to]):
                n_to_nb = inverse_embed[g_to_nb]
                if n_to_nb == -1:
                    continue
                if inverse_embed[g_to] == n_to_nb:
                    continue
                if forward_embed[n_to_nb][0] != g_to_nb and forward_embed[n_to_nb][-1] != g_to_nb:
                    continue
                if any_dir:
                    cand.append(g_to_nb)
                elif context.guiding_pattern_dict[g_to_nb] == context.guiding_pattern_dict[g_to]:
                    cand.append(g_to_nb)
            if len(cand) == 0:
                continue
            g_from = random.choice(cand)
            delta = delta_shift(context.input_graph, contact_graph, context.chimera_graph,
                                inverse_embed, g_from, g_to)
        print("\tStep: {}\tCost: {}\tBest Cost: {}\tShift?: {}\tDelta: {}"
              .format(step, cost, cost_best, shift_mode, delta))
        if math.exp(delta / temperature) > random.random():
            if shift_mode:
                print("Accepted shift")
                # noinspection PyUnboundLocalVariable
                shift(contact_graph, context.chimera_graph, forward_embed, inverse_embed, g_from,
                      g_to)
            else:
                print("Accepted swap")
                # noinspection PyUnboundLocalVariable
                swap(contact_graph, forward_embed, inverse_embed, n1, n2)
            cost += delta
            if cost_best < cost:
                cost_best = cost
                forward_embed_best = copy.deepcopy(forward_embed)
                print("Updated best cost: {}".format(cost_best))
                if cost_best == context._input_graph_nx.number_of_edges():
                    print("Solution found")
                    return forward_embed_best

    return forward_embed_best


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


def swap(contact_graph: FastMutableGraph, forward_embed, inverse_embed, n1: int, n2: int):
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

    # Consider neighbours of g_to, increment delta for new segments added to n_from
    for g_to_nb in iter(chimera_graph[g_to]):
        n_to_nb = inverse_embed[g_to_nb]
        if n_to_nb == -1:
            continue
        if n_to_nb == n_from or n_to_nb == n_to:
            continue
        if n_to_nb not in n_nb_count and not contact_graph.has_edge(n_from, n_to_nb) \
                and input_graph.has_edge(n_from, n_to_nb):
            delta += 1
        n_nb_count[n_to_nb] += 1

    # If g_to is in all edges connecting n_to to n_to_nb then decrement delta
    for n_to_nb, count in n_nb_count.items():
        assert contact_graph.edge_weight(n_to_nb, n_to) >= count  # Debug
        if contact_graph.edge_weight(n_to_nb, n_to) == count \
                and input_graph.has_edge(n_to_nb, n_to):
            delta -= 1
    return delta


def shift(contact_graph, chimera_graph, forward_embed, inverse_embed, g_from, g_to):
    n_from = inverse_embed[g_from]
    n_to = inverse_embed[g_to]

    # Update forward embed
    if forward_embed[n_from][-1] == g_from:
        forward_embed[n_from].append(g_to)
    else:
        forward_embed[n_from].appendleft(g_to)
    if forward_embed[n_to][-1] == g_to:
        forward_embed[n_to].pop()
    else:
        forward_embed[n_to].popleft()

    # Update inverse embed
    inverse_embed[g_to] = n_from

    # Update contact graph
    for g_to_nb in iter(chimera_graph[g_to]):
        n_to_nb = inverse_embed[g_to_nb]
        if n_to_nb == -1:
            continue
        if n_to_nb == n_from:
            contact_graph.decrement_edge_weight(n_from, n_to)
        elif n_to_nb == n_to:
            contact_graph.increment_edge_weight(n_from, n_to)
        else:
            contact_graph.increment_edge_weight(n_from, n_to_nb)
            contact_graph.decrement_edge_weight(n_to, n_to_nb)
