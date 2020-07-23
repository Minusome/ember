from typing import Dict

from networkx import Graph


class FastMutableGraph:
    """
    This graph supports fast swapping of nodes by relabelling them.
    Networkx does not provide an efficient, in-place method of relabelling nodes.
    """

    def __init__(self, input_graph: Graph, include_edges: bool = True):
        """
        Converts networkx graph into a (better?) format

        :param input_graph: Assume nodes are labelled 0...n, the graph is
        undirected and there are no self-linked edges
        """
        self.num_nodes = len(input_graph)
        self.nodes = [_Node(i) for i in range(self.num_nodes)]
        if include_edges:
            for node_val, adj_dict in input_graph.adjacency():
                for adj_val in [*adj_dict]:
                    self.nodes[node_val].neighbours[self.nodes[adj_val]] = 0
        self._dirty = True
        self._edges = self.edges

    @property
    def edges(self):
        if self._dirty:
            self._edges = set()
            for n1 in self.nodes:
                for n2 in n1.neighbours:
                    if n1.val < n2.val:
                        self._edges.add((n1.val, n2.val))
                    else:
                        self._edges.add((n2.val, n1.val))
            self._dirty = False
            return self._edges
        else:
            return self._edges

    def swap_node(self, n1: int, n2: int):
        self.nodes[n1].val, self.nodes[n2].val = self.nodes[n2].val, self.nodes[n1].val
        self.nodes[n1], self.nodes[n2] = self.nodes[n2], self.nodes[n1]
        self._dirty = True

    def has_edge(self, n1: int, n2: int) -> bool:
        e1 = self.nodes[n1] in self.nodes[n2].neighbours
        e2 = self.nodes[n2] in self.nodes[n1].neighbours
        assert e1 == e2
        return e1

    def edge_weight(self, n1: int, n2: int) -> int:
        w1 = self.nodes[n1].neighbours[self.nodes[n2]]
        w2 = self.nodes[n2].neighbours[self.nodes[n1]]
        assert w1 == w2
        return w1

    def increment_edge_weight(self, n1: int, n2: int):
        e1 = self.nodes[n1] in self.nodes[n2].neighbours
        e2 = self.nodes[n2] in self.nodes[n1].neighbours
        assert e1 == e2
        if e1:
            self.nodes[n1].neighbours[self.nodes[n2]] += 1
            self.nodes[n2].neighbours[self.nodes[n1]] += 1
        else:
            self.nodes[n1].neighbours[self.nodes[n2]] = 1
            self.nodes[n2].neighbours[self.nodes[n1]] = 1
            self._dirty = True

    def decrement_edge_weight(self, n1: int, n2: int):
        w1 = self.nodes[n1].neighbours[self.nodes[n2]]
        w2 = self.nodes[n2].neighbours[self.nodes[n1]]
        assert w1 == w2
        if w1 == 1:
            del self.nodes[n1].neighbours[self.nodes[n2]]
            del self.nodes[n2].neighbours[self.nodes[n1]]
            self._dirty = True
        else:
            self.nodes[n1].neighbours[self.nodes[n2]] = w1 - 1
            self.nodes[n2].neighbours[self.nodes[n1]] = w1 - 1

    def add_edge(self, n1: int, n2: int, weight: int = 1):
        assert n1 != n2
        self.nodes[n1].neighbours[self.nodes[n2]] = weight
        self.nodes[n2].neighbours[self.nodes[n1]] = weight
        self._dirty = True

    def remove_edge(self, n1: int, n2: int):
        del self.nodes[n1].neighbours[self.nodes[n2]]
        del self.nodes[n2].neighbours[self.nodes[n1]]
        self._dirty = True

    def __str__(self):
        ret = ""
        for n in self.nodes:
            ret += str(n) + "\n"
        return ret


class _Node:
    """
    Internal node representation
    """

    def __init__(self, val: int, neighbours: Dict["_Node", int] = None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else {}

    def __str__(self):
        return "Val: {}, Neighbours: {}".format(self.val, [(nk.val, nv) for nk, nv in
                                                           self.neighbours.items()])
