from networkx import Graph

from pssa.initializer import divide_guiding_pattern
from pssa.types import Dwave_Embedding
from pssa.schedule import T_MAX,move_params
from pssa.context import OptimizationContext


def run_simulated_annealing(context: OptimizationContext , initial_embed: Dwave_Embedding):

    # Current solution + data helpers
    forward_embed = initial_embed
    reverse_embed = {l: k for k, ll in initial_embed.items() for l in ll}
    edges_fulfilled =

    # Best solution found
    cost_best = 0
    forward_embed_best = initial_embed

    for step in range(T_MAX):
        temperature, shift, any_dir = move_params(step)
        if not shift:


        else:





def perimeter_bfs():

