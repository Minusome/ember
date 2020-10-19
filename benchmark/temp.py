import pickle5 as pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_all_benchmarks(directories):
    benchmarks = {directory: [] for directory in directories}
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".pickle"):
                benchmarks[directory].append(file)
    return benchmarks


def return_times(filename, typename, guest_type, edge_density):
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    times = []
    for n, dict in result[guest_type][edge_density].items():
        if 'walltime' in dict:
            times.append([int(n), dict['walltime']])

    if times:
        print(times)
        df = pd.DataFrame(np.array(times), columns=['n', 'times'])
        df['type'] = typename
        return df
    else:
        return pd.DataFrame()

def return_time_performance(guest_type, edge_density, fault_condition, algorithms):
    times = []
    for path in benchmark_directories:
        for file in benchmark_results[path]:
            for alg in algorithms:
                if alg in file and fault_condition in file:
                    times.append(return_times(os.path.join(path, file), alg, guest_type, edge_density))
    df = pd.concat(times)
    df = df.rename(columns={"n": "number of guest graph nodes", "times": "time to embed"})
    return df


def plot_performance(df, title):
    sns.set_style("darkgrid")
    plt.title(title)
    sns.lineplot(data=df, x="number of guest graph nodes", y="time to embed", hue='type')
    plt.show()


fault_conditions = ['nonfaulty']

benchmark_directories = ['results']
benchmark_results = load_all_benchmarks(benchmark_directories)

guest_types = ["barabasi_albert", "d_regular", "erdos_reyni", "noisy_bipartite"]
edge_densities = ["low", "medium", "high"]

dfs = []

for guest in guest_types:
    for density in edge_densities:
        for fault in fault_conditions:
            df = return_time_performance(guest, density, fault, ['pssa', 'coa',  'minorminer'])
            if not df.empty:
                dfs.append(df)

for df in dfs:
    print(df)
    sns.set_style("darkgrid")
    # plt.title(title)
    sns.lineplot(data=df, x="number of guest graph nodes", y="time to embed", hue='type')
    plt.show()