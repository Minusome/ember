from random import choices

from dwave import embedding as de
from matplotlib import pyplot as plt
import dwave_networkx as dnx


class Chimera:

    def __init__(self, M, L, faulty=None):
        self.M = M
        self.L = L
        self.graph = dnx.chimera_graph(m=M, t=L)
        self.faulty = set(faulty) if faulty is not None else {}
        self.graph.remove_nodes_from(self.faulty)

    def random_faulty(self, k):
        self.faulty = choices(list(range((self.M ** 2) * self.L * 2)), k=100)
        self.graph = dnx.chimera_graph(m=self.M, t=self.L)
        self.graph.remove_nodes_from(self.faulty)
        return self


def check_embedding(em, G, C):
    is_valid = True

    diagnosis = de.diagnose_embedding(em, G, C.graph)

    for prob in diagnosis:
        is_valid = False
        print(prob)

    if is_valid:
        print("Embedding found")
    else:
        print("Embedding not found")

    return is_valid


def plot_chimera_embedding(em, C):
    plt.ion()
    plt.figure(figsize=(20, 20))
    dnx.draw_chimera_embedding(C.graph, em, with_labels=True, unused_color=(1.0, 1.0, 1.0, 1.0))
