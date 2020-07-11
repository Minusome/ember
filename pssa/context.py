import dwave_networkx as dnx
from networkx import Graph

from pssa.graph import FastMutableGraph
from pssa.types import *


class OptimizationContext:
    """
    Data class that stores constants prior to optimization
    """

    def __init__(self, m: int, l: int, input_graph: Graph, guiding_pattern: Dwave_Embedding):
        self.input_graph = FastMutableGraph(input_graph)

        # Subscriptable for random selection
        self.input_edge_list = list(self.input_graph.edges)

        # Target chimera graph
        self.chimera_graph = dnx.chimera_graph(m, m, l)

        # Dict from chimera node to super vertex id
        self.guiding_pattern_dict = {l: k for k, ll in guiding_pattern.items() for l in ll}
