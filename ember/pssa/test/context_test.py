import networkx as nx
import pytest

from ember.hardware.chimera import ChimeraGraph
from ember.pssa.model import BaseModel


def test_chimera_distance_dict():
    guest = nx.empty_graph()
    host = ChimeraGraph(16, 4)

    model = BaseModel(guest, host)

    chimera_dist1 = {}

    for i in range(2048):
        chimera_dist1[i] = {}
        for j in range(2048):
            chimera_dist1[i][j] = model._chimera_distance(i, j)

    chimera_dist2 = dict(
        nx.algorithms.all_pairs_shortest_path_length(host))

    assert chimera_dist1 == chimera_dist2


def test_contact_graph():
    input = nx.empty_graph(4)
    input.add_edge(0, 1)
    input.add_edge(0, 3)
    host = ChimeraGraph(2, 4)
    model = BaseModel(input, host)
    embed = {0: [4, 5], 1: [0, 1], 2: [12], 3: [8]}
    model._create_contact_graph(embed)
    cost = model.initial_cost
    contact = model.contact_graph

    assert cost == 1
    assert contact.edge_weight(0, 1) == 4
    assert contact.edge_weight(0, 2) == 1
    assert contact.edge_weight(2, 3) == 1
    assert not contact.has_edge(0, 3)
    assert not contact.has_edge(1, 3)


if __name__ == '__main__':
    pytest.main()
