from random import random
from typing import Tuple

T_MAX = 10 ** 8


def move_params(iter_count: int) -> Tuple[float, bool, bool]:
    """
    Annealing schedule as defined in the original PSSA paper.
    Originally optimized for King's graphs, probably doesn't work well for Chimera.

    :param iter_count: The current iteration index
    :return: Tuple[annealing temperature, True if shift else swap, True if any direction shift
    else along guiding pattern only]
    """
    shift = False
    any_dir = False
    progress_ratio = iter_count / T_MAX
    if 0 <= iter_count < (T_MAX // 2):
        temperature = 0.603 * (1 - 2 * progress_ratio)
        if random() < (1 - progress_ratio):
            shift = True
    else:
        temperature = 0.334 * 2 * (1 - progress_ratio)
        if random() < (1 - progress_ratio):
            shift = True
            if random() < (progress_ratio * 0.392 + 0.095):
                any_dir = True

    return temperature, shift, any_dir
