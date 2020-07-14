import networkx as nx
import pytest

from pssa.annealer import delta_swap
from pssa.graph import FastMutableGraph

# Setup input graphs for test
input_builder = nx.empty_graph(4)
input_builder.add_edge(0, 1)
input_builder.add_edge(0, 2)
input_builder.add_edge(1, 3)
input_builder.add_edge(2, 3)
input_graph_1 = FastMutableGraph(input_builder)
input_builder.add_edge(1, 2)
input_graph_2 = FastMutableGraph(input_builder)

# Setup contact graphs for test
contact_builder = nx.empty_graph(4)
contact_builder.add_edge(0, 1)
contact_builder.add_edge(0, 3)
contact_builder.add_edge(1, 2)
contact_builder.add_edge(2, 3)
contact_graph_1 = FastMutableGraph(contact_builder)
contact_builder.add_edge(1, 3)
contact_graph_2 = FastMutableGraph(contact_builder)


def delta_swap_naive(input_graph: FastMutableGraph, contact_graph: FastMutableGraph, n1: int,
                     n2: int):
    delta = 0
    for n1_nb in contact_graph.nodes[n1].neighbours:
        if input_graph.has_edge(n1, n1_nb.val):
            delta -= 1
    for n2_nb in contact_graph.nodes[n2].neighbours:
        if input_graph.has_edge(n2, n2_nb.val):
            delta -= 1
    for n1_nb in contact_graph.nodes[n1].neighbours:
        to = n1_nb.val if n1_nb.val != n2 else n1
        if input_graph.has_edge(n2, to):
            delta += 1
    for n2_nb in contact_graph.nodes[n2].neighbours:
        to = n2_nb.val if n2_nb.val != n1 else n2
        if input_graph.has_edge(n1, to):
            delta += 1
    return delta


def test_delta_swap_1():
    assert delta_swap(input_graph_1, contact_graph_1, 2, 3) == 2
    assert delta_swap(input_graph_1, contact_graph_1, 3, 2) == 2
    assert delta_swap_naive(input_graph_1, contact_graph_1, 2, 3) == 2
    assert delta_swap_naive(input_graph_1, contact_graph_1, 3, 2) == 2


def test_delta_swap_2():
    assert delta_swap(input_graph_2, contact_graph_1, 2, 3) == 1
    assert delta_swap(input_graph_2, contact_graph_1, 3, 2) == 1
    assert delta_swap_naive(input_graph_2, contact_graph_1, 2, 3) == 1
    assert delta_swap_naive(input_graph_2, contact_graph_1, 3, 2) == 1


def test_delta_swap_3():
    assert delta_swap(input_graph_1, contact_graph_2, 2, 3) == 1
    assert delta_swap(input_graph_1, contact_graph_2, 3, 2) == 1
    assert delta_swap_naive(input_graph_1, contact_graph_2, 2, 3) == 1
    assert delta_swap_naive(input_graph_1, contact_graph_2, 3, 2) == 1


def test_delta_swap_4():
    assert delta_swap(input_graph_2, contact_graph_2, 2, 3) == 1
    assert delta_swap(input_graph_2, contact_graph_2, 3, 2) == 1
    assert delta_swap_naive(input_graph_2, contact_graph_2, 2, 3) == 1
    assert delta_swap_naive(input_graph_2, contact_graph_2, 3, 2) == 1


if __name__ == '__main__':
    pytest.main()
