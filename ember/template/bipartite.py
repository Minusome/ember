from collections import defaultdict

import numpy as np
from networkx import Graph

from ember.hardware.chimera import ChimeraGraph
from ember.hardware.transform import bipartite_with_faults

__all__ = ["BipartiteSat"]


def _run_bipartite_sat(I, input_edges, adj, h_count, v_count, verbose, timeout,
                       return_walltime):
    from ortools.sat.python import cp_model

    NL, NR = adj.shape
    assert NL == len(h_count)
    assert NR == len(v_count)

    model = cp_model.CpModel()

    yl = np.array([
        [model.NewBoolVar(f"yl_{i}_{j}") for j in range(NL)] for i in range(I)
    ])
    yr = np.array([
        [model.NewBoolVar(f"yr_{i}_{j}") for j in range(NR)] for i in range(I)
    ])

    valid_edge = [(l, r) for l in range(NL) for r in range(NR) if adj[l, r] == 1]

    if verbose:
        print(f"(NL: {NL} NR: {NR} Valid edges: {len(valid_edge)})")
        print("Building constraints...")

    for u, v in input_edges:
        or_terms = []
        for l, r in valid_edge:
            uv = model.NewBoolVar(f"lr_({u},{v})_({l},{r})")
            model.AddImplication(uv, yl[u, l])
            model.AddImplication(uv, yr[v, r])

            vu = model.NewBoolVar(f"lr_({v},{u})_({l},{r})")
            model.AddImplication(vu, yl[v, l])
            model.AddImplication(vu, yr[u, r])

            or_terms.extend((uv, vu))
        model.AddBoolOr(or_terms)

    for i in range(I):
        for l in range(NL):
            for r in range(NR):
                model.Add(yl[i, l] + yr[i, r] <= int(1 + adj[l, r]))

    for i in range(I):
        model.Add(sum(yl[i, :]) <= 1)

    for i in range(I):
        model.Add(sum(yr[i, :]) <= 1)

    for l in range(NL):
        model.Add(sum(yl[:, l]) <= h_count[l])

    for r in range(NR):
        model.Add(sum(yr[:, r]) <= v_count[r])

    if verbose:
        print("Finished building constraints")

    solver = cp_model.CpSolver()
    solver.parameters.use_pb_resolution = True
    solver.parameters.log_search_progress = verbose
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(model)

    if status == cp_model.UNKNOWN:
        return None

    result = np.full((I, 2), -1)
    for i in range(I):
        for l in range(NL):
            if solver.BooleanValue(yl[i, l]):
                result[i, 0] = l
                break
        for r in range(NR):
            if solver.BooleanValue(yr[i, r]):
                result[i, 1] = r
                break

    if return_walltime:
        return result, solver.WallTime()
    else:
        return result


class BipartiteSat:

    def __init__(self, guest: Graph, host: ChimeraGraph):
        self.guest = guest
        self.host = host
        self.h_embed, self.v_embed = bipartite_with_faults(host)
        self.adj = self._construct_adj_matrix()
        self.h_embed, self.v_embed, self.adj = self._compress_to_unique()

    def _construct_adj_matrix(self):

        def neighbours(graph, chain):
            chain = set(chain)
            nb_nodes = {node for c in chain for node in graph[c]}
            nb_nodes.difference_update(chain)
            return nb_nodes

        v_inverse = {
            v: i for i in range(len(self.v_embed)) for v in self.v_embed[i]
        }

        adj = np.zeros((len(self.h_embed), len(self.v_embed)))
        for i, h_chain in enumerate(self.h_embed):
            for nb in neighbours(self.host, h_chain):
                adj[i][v_inverse[nb]] = 1

        return adj

    def _compress_to_unique(self):
        adj, h_inverse = np.unique(self.adj, return_inverse=True, axis=0)
        adj, v_inverse = np.unique(adj, return_inverse=True, axis=1)
        h_group, v_group = defaultdict(list), defaultdict(list)
        for idx, chain in zip(h_inverse, self.h_embed):
            h_group[idx].append(chain)
        for idx, chain in zip(v_inverse, self.v_embed):
            v_group[idx].append(chain)
        return h_group, v_group, adj

    def solve(self, verbose=True, timeout=500, return_walltime=False):
        h_count = np.array(
            [len(self.h_embed[l]) for l in range(len(self.h_embed))])
        v_count = np.array(
            [len(self.v_embed[r]) for r in range(len(self.v_embed))])
        I = len(self.guest)
        try:
            import ember.template._native.embed as embed
            run_bipartite_sat = embed.run_bipartite_sat
        except ImportError:
            run_bipartite_sat = _run_bipartite_sat

        result = run_bipartite_sat(I, np.array(self.guest.edges), self.adj,
                                   h_count, v_count, verbose, timeout,
                                   return_walltime)

        emb = {i: [] for i in range(I)}

        if result is None:
            return emb

        if return_walltime:
            result, walltime = result

        for i in range(I):
            l, r = result[i]
            if l != -1:
                emb[i].extend(self.h_embed[l].pop())
            if r != -1:
                emb[i].extend(self.v_embed[r].pop())

        if return_walltime:
            return emb, walltime
        else:
            return emb
