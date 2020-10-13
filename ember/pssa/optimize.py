import copy
import math
import random
from itertools import cycle
from typing import Callable, Tuple

from ember.pssa.model import BaseModel


def run_simulated_annealing(model: BaseModel,
                            schedule: Callable[[int], Tuple[float, bool, bool]],
                            max_iterations: int):
    print(f"Optimal: {len(model.guest.edges)}")
    cost_best = cost = model.initial_cost
    forward_embed_best = copy.deepcopy(model.forward_embed)

    for step in range(max_iterations):
        temperature, shift_mode, any_dir = schedule(step)
        if not shift_mode:  # swap
            swap_move = model.random_swap_move()
            delta = model.delta_swap(swap_move)
        else:  # shift
            shift_move = model.random_shift_move(any_dir)
            if not shift_move:
                continue
            delta = model.delta_shift(shift_move)

        print("\tStep: {}\tCost: {}\tBest Cost: {}\tShift?: {}\tDelta: {}".format(
              step, cost, cost_best, shift_mode, delta))

        if math.exp(delta / temperature) > random.random():
            if shift_mode:
                print("Accepted shift")
                # noinspection PyUnboundLocalVariable
                model.shift(shift_move)
            else:
                print("Accepted swap")
                # noinspection PyUnboundLocalVariable
                model.swap(swap_move)
            cost += delta
            if cost_best < cost:
                cost_best = cost
                forward_embed_best = copy.deepcopy(model.forward_embed)
                print("Updated best cost: {}".format(cost_best))
                if cost_best == len(model.guest.edges):
                    print("Solution found")
                    return forward_embed_best

    return forward_embed_best


def run_steepest_descent_with_kicks(model: BaseModel, kicks: int,
                                    max_iterations: int):
    print(f"Optimal: {len(model.guest.edges)}")

    cost_best = cost = model.initial_cost
    forward_embed_best = copy.deepcopy(model.forward_embed)

    swap_moves, shift_moves = model.all_moves()
    moves = [("swap", move) for move in swap_moves]
    moves += [("shift", move) for move in shift_moves]

    for step in range(max_iterations):
        best_delta, best_move, type = 0, None, None
        for move in swap_moves:
            delta = model.delta_swap(move)
            if delta > best_delta:
                best_delta, best_move, type = delta, move, "swap"
        for move in shift_moves:
            delta = model.delta_shift(move)
            if delta > best_delta:
                best_delta, best_move, type = delta, move, "shift"
        if best_delta > 0:
            print( f"\tStep: {step}\tCost: {cost}\tBest Cost: {cost_best}\tDelta: {best_delta}")
            if type == "swap":
                model.swap(best_move)
            elif type == "shift":
                model.shift(best_move)
            cost += best_delta
            if cost_best < cost:
                cost_best = cost
                forward_embed_best = copy.deepcopy(model.forward_embed)
                print(f"Updated best cost: {cost_best}")
                if cost_best == len(model.guest.edges):
                    print("Solution found")
                    return forward_embed_best

        else:
            for _ in range(kicks):
                type, move = random.choice(moves)
                if type == "swap":
                    delta = model.delta_swap(move)
                    model.swap(move)
                elif type == "shift":
                    delta = model.delta_shift(move)
                    model.shift(move)
                # noinspection PyUnboundLocalVariable
                cost += delta
            print(f"Performed random restart with new cost: {cost}")


def run_next_descent_with_random_restarts(model: BaseModel,
                                          max_iterations: int):
    print(f"Optimal: {len(model.guest.edges)}")

    cost_best = cost = model.initial_cost
    forward_embed_best = copy.deepcopy(model.forward_embed)

    swap_moves, shift_moves = model.all_moves()
    moves = [("swap", move) for move in swap_moves]
    moves += [("shift", move) for move in shift_moves]

    iter = 0
    for step, (type, move) in zip(range(max_iterations), cycle(moves)):
        if type == "swap":
            delta = model.delta_swap(move)
        elif type == "shift":
            delta = model.delta_shift(move)
        # noinspection PyUnboundLocalVariable
        if delta > 0:
            print(f"\tStep: {step}\tCost: {cost}\tBest Cost: {cost_best}\tDelta: {delta}")
            if type == "swap":
                model.swap(move)
            elif type == "shift":
                model.shift(move)
            cost += delta
            iter = 0
            if cost_best < cost:
                cost_best = cost
                forward_embed_best = copy.deepcopy(model.forward_embed)
                print(f"Updated best cost: {cost_best}")
                if cost_best == len(model.guest.edges):
                    print("Solution found")
                    return forward_embed_best
        else:
            iter += 1
        if iter == len(moves):
            model.randomize()
            cost = model.initial_cost
            print(f"Random restart with new cost: {cost}")
