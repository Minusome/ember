from typing import Set

from networkx import Graph


class FastMutableGraph:
    """
    This graph supports fast swapping of nodes by relabelling them.
    Networkx does not provide an efficient, in-place method of relabelling nodes.
    """

    def __init__(self, input_graph: Graph):
        """
        Converts networkx graph into a new format

        :param input_graph: Assume nodes are labelled 0...n, the graph is
        undirected and there are no self-linked edges
        """
        self.nodes = [_Node(i) for i in range(len(input_graph))]
        for node_val, adj_dict in input_graph.adjacency():
            for adj_val in [*adj_dict]:
                self.nodes[node_val].neighbours.add(self.nodes[adj_val])
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

    def contains_edge(self, n1: int, n2: int) -> bool:
        e1 = self.nodes[n1] in self.nodes[n2].neighbours
        e2 = self.nodes[n2] in self.nodes[n1].neighbours
        assert e1 == e2
        return e1

    def add_edge(self, n1: int, n2: int):
        assert n1 != n2
        self.nodes[n1].neighbours.add(self.nodes[n2])
        self.nodes[n2].neighbours.add(self.nodes[n1])
        self._dirty = True

    def remove_edge(self, n1: int, n2: int):
        self.nodes[n1].neighbours.remove(self.nodes[n2])
        self.nodes[n2].neighbours.remove(self.nodes[n1])
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

    def __init__(self, val: int, neighbours: Set["_Node"] = None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else set()

    def __str__(self):
        return "Val: {}, Neighbours: {}".format(self.val, [n.val for n in self.neighbours])
