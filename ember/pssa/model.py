import random
from collections import deque, defaultdict
from functools import lru_cache
from itertools import combinations
import dwave_networkx as dnx
from networkx import Graph

from ember.hardware.chimera import ChimeraGraph
from ember.hardware.transform import overlap_clique, double_triangle_clique
from ember.hardware.transform_helper import divide_guiding_pattern
from ember.pssa.graph import MutableGraph

__all__ = ["ProbabilisticSwapShiftModel", "CliqueOverlapModel"]

from ember.template.util import plot_chimera_embedding


class BaseModel:
    def __init__(self, guest: Graph, host: ChimeraGraph):
        if host.faulty_nodes or host.faulty_edges:
            raise NotImplementedError("Chimera graphs with faults are not supported by these algorithms")

        self.guest = guest
        self.host = host
        m, l = host.params
        dnx_coords = dnx.chimera_coordinates(m, t=l)
        self.linear_to_chimera = dnx_coords.linear_to_chimera
        self.chimera_to_linear = dnx_coords.chimera_to_linear

    def _chimera_distance(self, g1: int, g2: int):
        if g1 == g2:
            return 0
        (i1, j1, u1, k1) = self.linear_to_chimera(g1)
        (i2, j2, u2, k2) = self.linear_to_chimera(g2)
        dist = abs(i1 - i2) + abs(j1 - j2)
        dist += 2 if u1 == u2 else 1
        if u1 == 0 and u2 == 0 and (j1 - j2) == 0 and k1 == k2:
            dist -= 2
        if u1 == 1 and u2 == 1 and (i1 - i2) == 0 and k1 == k2:
            dist -= 2
        return dist

    def _create_contact_graph(self, embed):
        self.contact_graph = MutableGraph(self.guest, include_edges=False)
        self.initial_cost = 0

        for n1 in range(len(embed)):
            e1 = embed[n1]
            for n2 in range(n1):
                e2 = embed[n2]
                weight = 0
                for g1 in e1:
                    for g2 in e2:
                        weight += 1 if self._chimera_distance(g1, g2) == 1 else 0
                if weight > 0:
                    self.contact_graph.add_edge(n1, n2, weight)
                    self.initial_cost += 1 if self.guest.has_edge(n1, n2) else 0


class ProbabilisticSwapShiftModel(BaseModel):
    def __init__(self, guest, host: ChimeraGraph):
        super().__init__(guest, host)

        guiding_pattern = double_triangle_clique(host)
        initial_emb = divide_guiding_pattern(guiding_pattern, len(guest))

        self.forward_embed = [deque(initial_emb[i]) for i in range(len(initial_emb))]
        self.inverse_embed = {l: k for k, ll in initial_emb.items() for l in ll}
        self.inverse_guiding_pattern = {l: k for k, ll in guiding_pattern.items() for l in ll}

        for i in range(len(host)):
            if i not in self.inverse_embed:
                self.inverse_embed[i] = -1

        self._create_contact_graph(initial_emb)

    def all_moves(self):
        raise NotImplementedError()

    def random_swap_move(self):
        n1, n1_nb = random.choice(tuple(self.guest.edges))
        n2 = random.choice(tuple(self.contact_graph.nodes[n1_nb].neighbours)).val
        return n1, n2

    def random_shift_move(self, any_dir=False):
        n_to = random.randrange(len(self.guest))
        if len(self.forward_embed[n_to]) < 2:
            return None
        g_to = self.forward_embed[n_to][0] if random.getrandbits(1) == 0 else self.forward_embed[
            n_to][-1]
        cand = []
        for g_to_nb in iter(self.host[g_to]):
            n_to_nb = self.inverse_embed[g_to_nb]
            if n_to_nb == -1:
                continue
            if self.inverse_embed[g_to] == n_to_nb:
                continue
            if self.forward_embed[n_to_nb][0] != g_to_nb and self.forward_embed[n_to_nb][-1] != \
                    g_to_nb:
                continue
            if any_dir:
                cand.append(g_to_nb)
            elif self.inverse_guiding_pattern[g_to_nb] == self.inverse_guiding_pattern[g_to]:
                cand.append(g_to_nb)
        if len(cand) == 0:
            return None
        g_from = random.choice(cand)
        return g_from, g_to

    def delta_swap(self, swap_move):
        n1, n2 = swap_move
        delta = 0
        for n1_nb in self.contact_graph.nodes[n1].neighbours:
            if n2 == n1_nb.val:
                continue
            if self.guest.has_edge(n1, n1_nb.val):
                delta -= 1
            if self.guest.has_edge(n2, n1_nb.val):
                delta += 1
        for n2_nb in self.contact_graph.nodes[n2].neighbours:
            if n1 == n2_nb.val:
                continue
            if self.guest.has_edge(n2, n2_nb.val):
                delta -= 1
            if self.guest.has_edge(n1, n2_nb.val):
                delta += 1
        return delta

    def swap(self, swap_move):
        n1, n2 = swap_move
        for g1 in self.forward_embed[n1]:
            self.inverse_embed[g1] = n2
        for g2 in self.forward_embed[n2]:
            self.inverse_embed[g2] = n1
        self.forward_embed[n1], self.forward_embed[n2] = self.forward_embed[n2], self.forward_embed[ n1]
        self.contact_graph.swap_node(n1, n2)

    def delta_shift(self, shift_move):
        g_from, g_to = shift_move
        n_from = self.inverse_embed[g_from]
        n_to = self.inverse_embed[g_to]
        n_nb_count = defaultdict(int)
        delta = 0

        # Consider neighbours of g_to, increment delta for new segments added to n_from
        for g_to_nb in iter(self.host[g_to]):
            n_to_nb = self.inverse_embed[g_to_nb]
            if n_to_nb == -1:
                continue
            if n_to_nb == n_from or n_to_nb == n_to:
                continue
            if n_to_nb not in n_nb_count and not self.contact_graph.has_edge(n_from, n_to_nb) \
                    and self.guest.has_edge(n_from, n_to_nb):
                delta += 1
            n_nb_count[n_to_nb] += 1

        # If g_to is in all edges connecting n_to to n_to_nb then decrement delta
        for n_to_nb, count in n_nb_count.items():
            assert self.contact_graph.edge_weight(n_to_nb, n_to) >= count  # Debug
            if self.contact_graph.edge_weight(n_to_nb, n_to) == count \
                    and self.guest.has_edge(n_to_nb, n_to):
                delta -= 1
        return delta

    def shift(self, shift_move):
        g_from, g_to = shift_move
        n_from = self.inverse_embed[g_from]
        n_to = self.inverse_embed[g_to]

        # Update forward embed
        if self.forward_embed[n_from][-1] == g_from:
            self.forward_embed[n_from].append(g_to)
        else:
            self.forward_embed[n_from].appendleft(g_to)
        if self.forward_embed[n_to][-1] == g_to:
            self.forward_embed[n_to].pop()
        else:
            self.forward_embed[n_to].popleft()

        # Update inverse embed
        self.inverse_embed[g_to] = n_from

        # Update contact hardware
        for g_to_nb in iter(self.host[g_to]):
            n_to_nb = self.inverse_embed[g_to_nb]
            if n_to_nb == -1:
                continue
            if n_to_nb == n_from:
                self.contact_graph.decrement_edge_weight(n_from, n_to)
            elif n_to_nb == n_to:
                self.contact_graph.increment_edge_weight(n_from, n_to)
            else:
                self.contact_graph.increment_edge_weight(n_from, n_to_nb)
                self.contact_graph.decrement_edge_weight(n_to, n_to_nb)


class CliqueOverlapModel(BaseModel):
    def __init__(self, guest, host: ChimeraGraph):
        super().__init__(guest, host)
        initial_embed = overlap_clique(host)

        self.forward_embed = [set(initial_embed[i]) for i in range(len(guest))]
        self.inverse_embed = {n: i for i in range(len(guest)) for n in initial_embed[i]}

        self._create_contact_graph(self.forward_embed)

    def randomize(self):
        random.shuffle(self.forward_embed)
        self.inverse_embed = {n: i for i in range(len(self.guest)) for n in self.forward_embed[i]}
        self._create_contact_graph(self.forward_embed)

    @lru_cache
    def all_moves(self):
        m, l = self.host.params
        swaps = list(combinations(range(len(self.guest)), 2))
        shifts = list(range(len(self.guest) - m * l))
        return swaps, shifts

    def random_swap_move(self):
        n1, n1_nb = random.choice(tuple(self.guest.edges))
        n2 = random.choice(tuple(self.contact_graph.nodes[n1_nb].neighbours)).val
        return n1, n2

    def random_shift_move(self, *args):
        m, l = self.host.params
        z_idx = random.randint(0, len(self.guest) - m * l - 1)
        return z_idx

    def delta_swap(self, swap_move):
        n1, n2 = swap_move
        delta = 0
        for n1_nb in self.contact_graph.nodes[n1].neighbours:
            if n2 == n1_nb.val:
                continue
            if self.guest.has_edge(n1, n1_nb.val):
                delta -= 1
            if self.guest.has_edge(n2, n1_nb.val):
                delta += 1
        for n2_nb in self.contact_graph.nodes[n2].neighbours:
            if n1 == n2_nb.val:
                continue
            if self.guest.has_edge(n2, n2_nb.val):
                delta -= 1
            if self.guest.has_edge(n1, n2_nb.val):
                delta += 1
        return delta

    def swap(self, swap_move):
        n1, n2 = swap_move
        for g1 in self.forward_embed[n1]:
            self.inverse_embed[g1] = n2
        for g2 in self.forward_embed[n2]:
            self.inverse_embed[g2] = n1
        self.forward_embed[n1], self.forward_embed[n2] = self.forward_embed[n2], self.forward_embed[ n1]
        self.contact_graph.swap_node(n1, n2)

    def delta_shift(self, shift_move):
        delta = 0
        n_minor, n_major, n_overlap = self._get_overlap_state(shift_move)

        # check ownership status
        if n_overlap == n_major: # major loss, minor gain
            for n_nb in self._get_n_minors(shift_move):
                if n_nb == n_minor:
                    continue
                if self.guest.has_edge(n_major, n_nb):
                    delta -= 1
                if self.guest.has_edge(n_minor, n_nb) and not self.contact_graph.has_edge(n_minor, n_nb):
                    delta += 1
        elif n_overlap == n_minor: # major gain, minor loss
            for n_nb in self._get_n_minors(shift_move):
                if n_nb == n_minor:
                    continue
                if self.guest.has_edge(n_minor, n_nb) \
                        and self.contact_graph.edge_weight(n_minor, n_nb) == 1:
                    delta -= 1
                if self.guest.has_edge(n_major, n_nb):
                    delta += 1
        else:
            raise Exception("Bad state")

        return delta


    def shift(self, shift_move):
        n_minor, n_major, n_overlap = self._get_overlap_state(shift_move)
        if n_overlap == n_major:  # major loss, minor gain
            for g_nb in self._get_g_minors(shift_move):
                self.forward_embed[n_major].remove(g_nb)
                self.forward_embed[n_minor].add(g_nb)
                self.inverse_embed[g_nb] = n_minor
            for n_nb in self._get_n_minors(shift_move):
                if n_nb == n_minor:
                    continue
                self.contact_graph.decrement_edge_weight(n_major, n_nb)
                self.contact_graph.increment_edge_weight(n_minor, n_nb)
        elif n_overlap == n_minor:  # major gain, minor loss
            for g_nb in self._get_g_minors(shift_move):
                self.forward_embed[n_major].add(g_nb)
                self.forward_embed[n_minor].remove(g_nb)
                self.inverse_embed[g_nb] = n_major
            for n_nb in self._get_n_minors(shift_move):
                if n_nb == n_minor:
                    continue
                self.contact_graph.increment_edge_weight(n_major, n_nb)
                self.contact_graph.decrement_edge_weight(n_minor, n_nb)
        else:
            raise Exception("Bad state")

    def _get_g_minors(self, z_idx):
        cell, unit = z_idx // 4, z_idx % 4
        try:
            for c in range(cell + 1):
                yield self.chimera_to_linear((1 + cell, c, 1, unit))
        except KeyError:
            pass

    def _get_n_minors(self, z_idx):
        cell = z_idx // 4
        _, l = self.host.params
        try:
            for c in range(cell + 1):
                for u in range(l):
                    yield self.inverse_embed[self.chimera_to_linear((1 + cell, c, 0, u))]
        except KeyError:
            pass

    def _get_overlap_state(self, z_idx):
        cell, unit = z_idx // 4, z_idx % 4
        m, _ = self.host.params
        # minor, major, overlap
        return self.inverse_embed[self.chimera_to_linear((1 + cell, cell, 0, unit))], \
               self.inverse_embed[self.chimera_to_linear((1 + cell, m - 1, 1, unit))], \
               self.inverse_embed[self.chimera_to_linear((1 + cell, cell, 1, unit))]
