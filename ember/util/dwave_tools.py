import dwave_networkx as dnx
from dwave import embedding as de
from matplotlib import pyplot as plt


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
    dnx.draw_chimera_embedding(C,
                               em,
                               with_labels=True,
                               unused_color=(1.0, 1.0, 1.0, 1.0))
