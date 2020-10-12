import networkx as nx
import pytest

from ember.pssa.context import OptimizationContext


def test_chimera_distance_dict():
    context = OptimizationContext(10, 4, nx.empty_graph(), {})
    chimera_dist1 = {}

    for i in range(len(context.chimera_distance)):
        chimera_dist1[i] = {}
        for j in range(len(context.chimera_distance)):
            chimera_dist1[i][j] = context.chimera_distance[i][j]

    chimera_dist2 = dict(
        nx.algorithms.all_pairs_shortest_path_length(context.chimera_graph))

    assert chimera_dist1 == chimera_dist2


def test_contact_graph():
    input = nx.empty_graph(4)
    input.add_edge(0, 1)
    input.add_edge(0, 3)

    context = OptimizationContext(2, 4, input, {})
    embed = {0: [4, 5], 1: [0, 1], 2: [12], 3: [8]}
    contact, cost = context.create_contact_graph(embed)

    assert cost == 1
    assert contact.edge_weight(0, 1) == 4
    assert contact.edge_weight(0, 2) == 1
    assert contact.edge_weight(2, 3) == 1
    assert not contact.has_edge(0, 3)
    assert not contact.has_edge(1, 3)


if __name__ == '__main__':
    pytest.main()
