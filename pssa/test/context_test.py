import networkx as nx
import pytest

from pssa.context import OptimizationContext


def test_chimera_distance_dict():
    context = OptimizationContext(10, 4, nx.empty_graph(), {})
    chimera_dist1 = {}

    for i in range(len(context.chimera_distance)):
        chimera_dist1[i] = {}
        for j in range(len(context.chimera_distance)):
            chimera_dist1[i][j] = context.chimera_distance[i][j]

    chimera_dist2 = dict(nx.algorithms.all_pairs_shortest_path_length(context.chimera_graph))

    assert chimera_dist1 == chimera_dist2


if __name__ == '__main__':
    pytest.main()
