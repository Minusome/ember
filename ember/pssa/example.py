import random

import networkx as nx

from ember.hardware.chimera import ChimeraGraph
from ember.pssa.model import ProbabilisticSwapShiftModel
from ember.pssa.optimize import run_simulated_annealing

if __name__ == '__main__':

    T_MAX = 10**6

    def schedule(iter_count):
        progress_ratio = iter_count / T_MAX
        if 0 <= iter_count < (T_MAX // 2):
            temperature = 0.603 * (1 - 2 * progress_ratio)
        else:
            temperature = 0.334 * 2 * (1 - progress_ratio)
        shift = random.random() < 0.3
        any_dir = True
        return temperature, shift, any_dir

    def pssa_schedule(iter_count):
        progress_ratio = iter_count / T_MAX
        if 0 <= iter_count < (T_MAX // 2):
            temperature = 0.603 * (1 - 2 * progress_ratio)
        else:
            temperature = 0.334 * 2 * (1 - progress_ratio)

        shift = random.random() < min(1.2 * progress_ratio, 0.8)
        if shift:
            if progress_ratio < 0.5:
                any_dir = random.random() < progress_ratio * 0.8
            else:
                any_dir = True
        else:
            any_dir = False

        return temperature, shift, any_dir

    input = nx.generators.fast_gnp_random_graph(69, 0.2, seed=1)
    hardware = ChimeraGraph(16, 4)
    model = ProbabilisticSwapShiftModel(input, hardware)
    run_simulated_annealing(model, pssa_schedule, T_MAX)
