import math
import random
import copy
import sys
from collections import defaultdict, deque
from typing import List, Dict
import dwave_networkx as dnx

from ember.pssa.context import OptimizationContext
from ember.pssa.graph import FastMutableGraph
from ember.pssa2.schedule import T_MAX, move_params


def run_simulated_annealing(context: OptimizationContext, initial_embed: Dict[int, List[int]]):
    forward_embed = [set(initial_embed[i]) for i in range(len(initial_embed))]
    # inverse_embed = {l: k for k, ll in initial_embed.items() for l in ll}
    inverse_embed = {}
    for i in range(len(forward_embed)):
        for n in forward_embed[i]:
            inverse_embed[n] = i

    # Fill in -1 for unmapped G nodes
    # for i in range(len(context.chimera_graph)):
    #     if i not in inverse_embed:
    #         inverse_embed[i] = -1

    # Initialize contact graph and cost
    contact_graph, cost = context.create_contact_graph(initial_embed)

    print(f"initial cost: {cost}")

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
            z_idx = random.randint(0,  len(context._input_graph_nx) - 64 - 1)
            delta = delta_shift(context.input_graph, inverse_embed, z_idx)
        print("\tStep: {}\tCost: {}\tBest Cost: {}\tShift?: {}\tDelta: {}"
              .format(step, cost, cost_best, shift_mode, delta))
        if math.exp(delta / temperature) > random.random():
            if shift_mode:
                print("Accepted shift")
                # noinspection PyUnboundLocalVariable
                shift(contact_graph, forward_embed, inverse_embed, z_idx)
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


to_linear = dnx.chimera_coordinates(16, t=4).chimera_to_linear
def get_n_minors(z_idx, inverse_embed):
    cell = z_idx // 4
    try:
        for c in range(cell + 1):
            for u in range(4):
                yield inverse_embed[to_linear((1 + cell, c, 0, u))]
    except KeyError:
        pass

def get_g_minors(z_idx):
    cell, unit = z_idx // 4, z_idx % 4
    try:
        for c in range(cell + 1):
            yield to_linear((1 + cell, c, 1, unit))
    except KeyError:
        pass

def get_overlap_state(inverse_embed, z_idx):
    cell, unit = z_idx // 4, z_idx % 4
    # minor, major, overlap
    return inverse_embed[to_linear((1 + cell, cell, 0, unit))], \
           inverse_embed[to_linear((1 + cell, 15, 1, unit))], \
           inverse_embed[to_linear((1 + cell, cell, 1, unit))]


def delta_shift(input_graph, inverse_embed, z_idx):
    delta = 0
    n_minor, n_major, n_overlap = get_overlap_state(inverse_embed, z_idx)

    # check ownership status
    if n_overlap == n_major: # major loss, minor gain
        for n_nb in get_n_minors(z_idx, inverse_embed):
            if n_nb == n_minor:
                continue
            if input_graph.has_edge(n_major, n_nb):
                delta -= 1
            if input_graph.has_edge(n_minor, n_nb):
                delta += 1
    elif n_overlap == n_minor: # major gain, minor loss
        for n_nb in get_n_minors(z_idx, inverse_embed):
            if n_nb == n_minor:
                continue
            if input_graph.has_edge(n_minor, n_nb):
                delta -= 1
            if input_graph.has_edge(n_major, n_nb):
                delta += 1
    else:
        raise Exception("Bad state")

    return delta


def shift(contact_graph, forward_embed, inverse_embed, z_idx):
    n_minor, n_major, n_overlap = get_overlap_state(inverse_embed, z_idx)

    if n_overlap == n_major:  # major loss, minor gain
        for g_nb in get_g_minors(z_idx):
            forward_embed[n_major].remove(g_nb)
            forward_embed[n_minor].add(g_nb)
            inverse_embed[g_nb] = n_minor
        for n_nb in get_n_minors(z_idx, inverse_embed):
            if n_nb == n_minor:
                continue
            contact_graph.remove_edge(n_major, n_nb)
            contact_graph.add_edge(n_minor, n_nb)
    elif n_overlap == n_minor:  # major gain, minor loss
        for g_nb in get_g_minors(z_idx):
            forward_embed[n_major].add(g_nb)
            forward_embed[n_minor].remove(g_nb)
            inverse_embed[g_nb] = n_major
        for n_nb in get_n_minors(z_idx, inverse_embed):
            if n_nb == n_minor:
                continue
            contact_graph.add_edge(n_major, n_nb)
            contact_graph.remove_edge(n_minor, n_nb)
    else:
        raise Exception("Bad state")

