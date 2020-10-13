from random import sample
from typing import List

import dwave_networkx as dnx
from networkx import Graph


class ChimeraGraph(Graph):

    def __init__(self, m, l,
                 node_fault_rate=0.0,
                 edge_fault_rate=0.0,
                 node_faults: List = None,
                 **attr):
        super().__init__(**attr)

        chimera = dnx.chimera_graph(m=m, t=l)
        original_edge_set = set(chimera.edges)

        nodes = set(chimera.nodes)
        if node_faults:
            faulty_nodes = node_faults
        else:
            faulty_nodes = sample(nodes,
                                  round(node_fault_rate * len(nodes)))
        chimera.remove_nodes_from(faulty_nodes)

        edges = set(chimera.edges)
        faulty_edges = sample(edges,
                              max(0, round(edge_fault_rate * len(edges)) -
                                  (len(original_edge_set) - len(edges))))
        chimera.remove_edges_from(faulty_edges)

        self.__dict__.update(chimera.__dict__)
        self.params = (m, l)
        self.faulty_nodes = set(faulty_nodes)
        self.faulty_edges = original_edge_set - set(chimera.edges)
        self.internal = chimera

    def subgraph(self, nodes):
        return self.internal.subgraph(nodes)


def D_WAVE_2000Q(**kwargs):
    return ChimeraGraph(16, 4, **kwargs)


def D_WAVE_2000Q_2_1():
    return ChimeraGraph(16, 4,
                        node_faults=[215, 336, 577, 691, 1012,
                                     1105, 1276, 1730, 1776, 1858])


def D_WAVE_2000Q_5():
    return ChimeraGraph(16, 4,
                        node_faults=[49, 123, 264, 1676, 1804,
                                     1805, 1806, 1812, 1920, 1921,
                                     1932, 1934, 1935, 1940, 1941,
                                     1942, 1943, 1977])


def D_WAVE_2000Q_6():
    return ChimeraGraph(16, 4,
                        node_faults=[43, 46, 524, 548, 1723, 1735,
                                     1804])


def D_WAVE_2000Q_QuAIL():
    return ChimeraGraph(16, 4,
                        node_faults=[15, 59, 329, 335, 363, 367,
                                     706, 783, 810, 815, 834, 970,
                                     975, 992, 999, 1833, 1881])


def D_WAVE_2X(**kwargs):
    return ChimeraGraph(12, 4, **kwargs)


def D_WAVE_2X_LANL():
    return ChimeraGraph(12, 4,
                        node_faults=[582, 590, 596, 801, 807, 810,
                                     814, 815, 817, 897, 939])
