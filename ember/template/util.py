from random import choices

from dwave import embedding as de
from matplotlib import pyplot as plt
import dwave_networkx as dnx

from networkx import Graph


class Chimera(Graph):

    def __init__(self, m, l, faulty=None, k_rand_faulty=0, **attr):
        super().__init__(**attr)
        chimera = dnx.chimera_graph(m=m, t=l)
        faulty = set(faulty) if faulty is not None else set()
        if k_rand_faulty > 0:
            faulty = set(choices(range((m ** 2) * l * 2), k=k_rand_faulty))
        if faulty:
            chimera.remove_nodes_from(faulty)
        self.__dict__.update(chimera.__dict__)
        self.M, self.L = m, l
        self.faulty = faulty
        self.internal = chimera

    def subgraph(self, nodes):
        return self.internal.subgraph(nodes)


def D_WAVE_2000Q(**kwargs):
    return Chimera(16, 4, **kwargs)


def D_WAVE_2X(**kwargs):
    return Chimera(12, 4, **kwargs)


def check_embedding(em, G, C):
    is_valid = True

    diagnosis = de.diagnose_embedding(em, G, C)

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
    dnx.draw_chimera_embedding(C, em, with_labels=True, unused_color=(1.0, 1.0, 1.0, 1.0))
