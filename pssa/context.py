import dwave_networkx as dnx
from networkx import Graph

from pssa.types import *


class OptimizationContext:
    """
    Data class that stores immutable constants during the optimization process
    """

    def __init__(self, m: int, l: int, input_graph: Graph, guiding_pattern: Dwave_Embedding):
        # Not sure about complexity of networkx.edge_view, so converting to python set
        self.input_edge_set = {e for e in input_graph.edges}

        # Subscriptable for random selection
        self.input_edge_list = list(self.input_edge_set)

        # Target chimera graph
        self.chimera_graph = dnx.chimera_graph(m, m, l)

        # Dict from chimera node to super vertex id
        self.guiding_pattern_dict = {l: k for k, ll in guiding_pattern.items() for l in ll}
