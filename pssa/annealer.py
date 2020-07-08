import random

from networkx import Graph

from pssa.initializer import divide_guiding_pattern
from pssa.types import Dwave_Embedding
from pssa.schedule import T_MAX,move_params
from pssa.context import OptimizationContext
from pssa.graph import FastMutableGraph


def run_simulated_annealing(context: OptimizationContext , initial_embed: Dwave_Embedding):

    # Current solution + data helpers
    forward_embed = initial_embed
    reverse_embed = {l: k for k, ll in initial_embed.items() for l in ll}

    # Need to initialize the contact graph
    contact_graph = FastMutableGraph()

    # Best solution found
    cost_best = 0
    forward_embed_best = initial_embed

    for step in range(T_MAX):
        temperature, shift, any_dir = move_params(step)
        # swap
        if not shift:
            i,k = random.choice(context.input_edge_list)
            j = random.choice(list(contact_graph.nodes[k].neighbours)).val

        # shift
        else:



def delta_swap(contact_graph, n1, n2):
    delta = 0




def delta_shift(contact_graph, n1, n2):
    delta = 0





def perimeter_bfs():

