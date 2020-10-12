import random
from typing import Dict, List


def divide_guiding_pattern(guiding_pattern: Dict[int, List[int]],
                           vertex_count: int,
                           strategy: str = "balanced") -> Dict[int, List[int]]:
    """
    Randomly splits the vertex sets in the guiding pattern to create an initial embedding of the
    input hardware into the hardware hardware.

    :param guiding_pattern: A known clique (or near clique) embedding !NOTE: The vertex sets are
    represented as lists but the ordering matters. The nodes should be ordered according to the
    chain formed on the hardware hardware.
    :param vertex_count: The number of vertices in the input hardware (i.e. The number of vertex
    sets in the final embedding)
    :param strategy: Either 'random' (all vertex sets have equal chance of being divided) or
    'balanced' (large vertex sets get divided first)
    :return: an initial embedding with vertex_count equal to number of vertex sets
    """
    if vertex_count <= len(guiding_pattern):
        return {i: guiding_pattern[i] for i in range(vertex_count)}

    vertex_sets = list(guiding_pattern.values())
    selection_window = len(vertex_sets)

    while len(vertex_sets) < vertex_count:
        if strategy == "balanced":
            rand_idx = random.randrange(0, selection_window)
            selection_window -= 1
            if selection_window == 0:
                selection_window = len(vertex_sets)
        elif strategy == "random":
            rand_idx = random.randrange(0, len(vertex_sets))
        else:
            raise Exception("Unsupported strategy: {}".format(strategy))

        vertex_sets = \
            vertex_sets[:rand_idx] + \
            vertex_sets[rand_idx + 1:] + \
            [vertex_sets[rand_idx][:len(vertex_sets[rand_idx]) // 2]] + \
            [vertex_sets[rand_idx][len(vertex_sets[rand_idx]) // 2:]]

    assert len(vertex_sets) == vertex_count
    return {i: vertex_sets[i] for i in range(vertex_count)}
