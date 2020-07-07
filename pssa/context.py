import dwave_networkx as dnx
from networkx import Graph

from pssa.types import *


class OptimizationContext:
    """
    Stores data that does not mutate (i.e. constants) during the optimization process
    """

    def __init__(self, m: int, l: int, input_graph: Graph, guiding_pattern: Dwave_Embedding):
        # Not sure about complexity of networkx.edge_view, so converting to python set
        self.input_edge_set = {e for e in input_graph.edges}

        # Target chimera graph
        self.chimera_graph = dnx.chimera_graph(m, m, l)

        # Dict from target node to super vertex id
        self.guiding_pattern_dict = {l: k for k, ll in guiding_pattern.items() for l in ll}
