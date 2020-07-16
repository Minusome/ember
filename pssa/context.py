import dwave_networkx as dnx
import networkx as nx
import numpy as np

from pssa.graph import FastMutableGraph
from pssa.types import *


class OptimizationContext:
    """
    Primarily stores constants prior to optimization
    """

    def __init__(self, m: int, l: int, input_graph: nx.Graph, guiding_pattern: Dwave_Embedding):
        self._m = m
        self._l = l
        self._input_graph_nx = input_graph

        self.input_graph = FastMutableGraph(input_graph)

        # Subscriptable for random selection
        self.input_edge_list = list(self.input_graph.edges)

        # Target chimera graph
        self.chimera_graph = dnx.chimera_graph(m, m, l)
        self.chimera_distance = np.empty((len(self.chimera_graph), len(self.chimera_graph)))

        assert len(self.chimera_graph) == m * m * l * 2

        # Nodes in chimera graph must be labelled 0...N
        # (Which they are for dnx library)
        for g1 in range(len(self.chimera_graph)):
            for g2 in range(len(self.chimera_graph)):
                if g2 < g1:
                    self.chimera_distance[g1][g2] = self.chimera_distance[g2][g1]
                else:
                    self.chimera_distance[g1][g2] = self._chimera_distance(g1, g2)

        # Dict from chimera node to super vertex id
        self.guiding_pattern_dict = {l: k for k, ll in guiding_pattern.items() for l in ll}

    def _chimera_distance(self, g1: int, g2: int):
        if g1 == g2:
            return 0
        lc = dnx.chimera_coordinates(10, t=4).linear_to_chimera
        (i1, j1, u1, k1) = lc(g1)
        (i2, j2, u2, k2) = lc(g2)
        dist = abs(i1 - i2) + abs(j1 - j2)
        dist += 2 if u1 == u2 else 1
        if u1 == 0 and u2 == 0 and (j1 - j2) == 0 and k1 == k2:
            dist -= 2
        if u1 == 1 and u2 == 1 and (i1 - i2) == 0 and k1 == k2:
            dist -= 2
        return dist

    def create_contact_graph(self, embed):
        # Perimeter BFS might be a bit more efficient
        contact_graph = FastMutableGraph(self._input_graph_nx, include_edges=False)
        num_contact = 0

        for n1 in range(len(embed)):
            e1 = embed[n1]
            for n2 in range(e1):
                e2 = embed[n2]
                weight = 0
                for g1 in e1:
                    for g2 in e2:
                        weight += 1 if self.chimera_distance[g1][g2] == 1 else 0
                if weight > 0:
                    contact_graph.add_edge(n1, n2, weight)
                    num_contact += 1

        return contact_graph, num_contact
