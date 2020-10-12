from random import sample

import dwave_networkx as dnx
from networkx import Graph


class ChimeraGraph(Graph):

    def __init__(self, m, l, node_fault_rate=0.0, edge_fault_rate=0.0, **attr):
        super().__init__(**attr)

        chimera = dnx.chimera_graph(m=m, t=l)
        prev_edge_count = len(chimera.edges)

        nodes = set(chimera.nodes)
        faulty_nodes = sample(nodes, round(node_fault_rate * len(nodes)))
        chimera.remove_nodes_from(faulty_nodes)

        edges = set(chimera.edges)
        faulty_edges = sample(edges, max(0, round(edge_fault_rate * len(edges)) - (
                prev_edge_count - len(edges))))
        chimera.remove_edges_from(faulty_edges)

        self.__dict__.update(chimera.__dict__)
        self.params = (m, l)
        self.faulty_nodes = faulty_nodes
        self.faulty_edges = faulty_edges
        self.internal = chimera

    def subgraph(self, nodes):
        return self.internal.subgraph(nodes)


def D_WAVE_2000Q(**kwargs):
    return ChimeraGraph(16, 4, **kwargs)


def D_WAVE_2X(**kwargs):
    return ChimeraGraph(12, 4, **kwargs)
