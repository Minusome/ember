import networkx as nx
import pytest

from pssa.annealer import delta_shift
from pssa.graph import FastMutableGraph

# Small grid graph for testing
target = nx.empty_graph(6)
target.add_edge(0, 1)
target.add_edge(0, 2)
target.add_edge(1, 3)
target.add_edge(2, 3)
target.add_edge(2, 4)
target.add_edge(3, 5)
target.add_edge(4, 5)


def test_delta_shift_simple_gain():
    input = FastMutableGraph(nx.empty_graph(3))
    input.add_edge(0, 2)

    contact = FastMutableGraph(nx.empty_graph(3))
    contact.add_edge(0, 1, weight=2)
    contact.add_edge(1, 2, weight=2)

    inverse = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    assert delta_shift(input, contact, target, inverse, 1, 3) == 1
    assert delta_shift(input, contact, target, inverse, 0, 2) == 1
    assert delta_shift(input, contact, target, inverse, 4, 2) == 1
    assert delta_shift(input, contact, target, inverse, 5, 3) == 1


def test_delta_shift_simple_gain_2():
    input = FastMutableGraph(nx.empty_graph(5))
    input.add_edge(0, 3)
    input.add_edge(0, 4)

    contact = FastMutableGraph(nx.empty_graph(5))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 3, weight=1)
    contact.add_edge(2, 3, weight=1)
    contact.add_edge(2, 4, weight=1)
    contact.add_edge(3, 4, weight=1)

    inverse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4}

    assert delta_shift(input, contact, target, inverse, 0, 2) == 2


def test_delta_shift_simple_gain_no_duplicate():
    input = FastMutableGraph(nx.empty_graph(4))
    input.add_edge(0, 3)

    contact = FastMutableGraph(nx.empty_graph(4))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 3, weight=1)
    contact.add_edge(2, 3, weight=2)

    inverse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}

    assert delta_shift(input, contact, target, inverse, 0, 2) == 1


def test_delta_shift_no_gain_already_connected():
    input = FastMutableGraph(nx.empty_graph(3))
    input.add_edge(0, 2)

    contact = FastMutableGraph(nx.empty_graph(3))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 2, weight=2)

    inverse = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}

    assert delta_shift(input, contact, target, inverse, 0, 2) == 0
    assert delta_shift(input, contact, target, inverse, 1, 3) == 0


def test_delta_shift_simple_loss():
    input = FastMutableGraph(nx.empty_graph(3))
    input.add_edge(0, 2)

    contact = FastMutableGraph(nx.empty_graph(3))
    contact.add_edge(0, 1, weight=2)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 2, weight=1)

    inverse = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

    assert delta_shift(input, contact, target, inverse, 3, 2) == -1


def test_delta_shift_multi_loss():
    input = FastMutableGraph(nx.empty_graph(4))
    input.add_edge(0, 1)
    input.add_edge(2, 1)
    input.add_edge(3, 1)

    contact = FastMutableGraph(nx.empty_graph(4))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 2, weight=1)
    contact.add_edge(1, 3, weight=1)
    contact.add_edge(2, 3, weight=1)

    inverse = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}

    # Assume that 3 and 1 still connected....
    assert delta_shift(input, contact, target, inverse, 4, 2) == -2


def test_delta_shift_no_loss_still_connected():
    input = FastMutableGraph(nx.empty_graph(3))
    input.add_edge(1, 2)

    contact = FastMutableGraph(nx.empty_graph(3))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 2, weight=2)

    inverse = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}

    assert delta_shift(input, contact, target, inverse, 1, 3) == 0


def test_delta_shift_gain_loss():
    input = FastMutableGraph(nx.empty_graph(6))
    input.add_edge(1, 2)
    input.add_edge(1, 3)
    input.add_edge(2, 5)
    input.add_edge(3, 5)

    contact = FastMutableGraph(nx.empty_graph(6))
    contact.add_edge(0, 1, weight=1)
    contact.add_edge(0, 2, weight=1)
    contact.add_edge(1, 3, weight=1)
    contact.add_edge(2, 3, weight=1)
    contact.add_edge(2, 4, weight=1)
    contact.add_edge(3, 5, weight=1)
    contact.add_edge(4, 5, weight=1)

    inverse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    assert delta_shift(input, contact, target, inverse, 2, 3) == 0


if __name__ == '__main__':
    pytest.main()
