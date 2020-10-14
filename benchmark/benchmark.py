# Generate the input grpahs
from ember.sample import *
import pickle

edge_densities = ["low", "medium", "high"]

guest_generators = {
    "barabasi_albert": barabasi_albert_graph,
    "d_regular": d_regular_graph,
    "erdos_reyni": erdos_reyni_graph,
    "noisy_bipartite": noisy_bipartite_graph
}
seeds = [10, 20, 30, 40, 50]

all_guests = {}

for graph_type, generator in guest_generators.items():
    print(f"Graph type: {graph_type}")
    all_guests[graph_type] = {}
    for num_nodes in range(65, 96):
        print(f"Nodes: {num_nodes}")
        all_guests[graph_type][num_nodes] = {}
        for edge_density in edge_densities:
            print(f"Density: {edge_density}")
            all_guests[graph_type][num_nodes][edge_density] = {}
            for i, seed in enumerate(seeds):
                all_guests[graph_type][num_nodes][edge_density][i] = generator(
                    num_nodes, edge_density, seed)

    with open(f'{graph_type}.pickle', 'wb') as handle:
        pickle.dump(all_guests[graph_type], handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store them into files
for graph_type, _ in guest_generators.items():
     with open(f'./benchmark/guests/{graph_type}.pickle', 'rb') as handle:
            all_guests[graph_type] = pickle.load( handle)




# Run the non faulty benchmarks

# Run the faulty benchmarks
