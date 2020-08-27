from collections import defaultdict

import dwave_networkx as dnx
import numpy as np
from ortools.sat.python import cp_model
from networkx import Graph

from ember.template.util import Chimera


class BipartiteSAT:

    def __init__(self, guest: Graph, target: Chimera):
        self.G = guest
        self.C = target
        self.h_embed, self.v_embed = self._bipartite_embed()
        self.adj = self._construct_adj_matrix()
        self.h_embed, self.v_embed, self.adj = self._compress_to_unique()

    def _bipartite_embed(self):

        def append_nonempty(super, sub):
            if sub:
                super.append(sub)

        M, L, faulty = self.C.M, self.C.L, self.C.faulty
        to_linear = dnx.chimera_coordinates(M, t=L).chimera_to_linear

        h_embed = []
        for i in range(M * L):
            chain = []
            cell, unit = i // L, i % L
            for j in range(M):
                ln = to_linear((cell, j, 1, unit))
                if ln in faulty:
                    append_nonempty(h_embed, chain)
                    chain = []
                else:
                    chain.append(ln)
            append_nonempty(h_embed, chain)

        v_embed = []
        for i in range(M * L):
            chain = []
            cell, unit = i // L, i % L
            for j in range(M):
                ln = to_linear((j, cell, 0, unit))
                if ln in faulty:
                    append_nonempty(v_embed, chain)
                    chain = []
                else:
                    chain.append(ln)
            append_nonempty(v_embed, chain)

        return h_embed, v_embed

    def _construct_adj_matrix(self):

        def neighbours(graph, chain):
            chain = set(chain)
            nb_nodes = {node for c in chain for node in graph[c]}
            nb_nodes.difference_update(chain)
            return nb_nodes

        v_inverse = {v: i for i in range(len(self.v_embed)) for v in self.v_embed[i]}

        adj = np.zeros((len(self.h_embed), len(self.v_embed)))
        for i, h_chain in enumerate(self.h_embed):
            for nb in neighbours(self.C, h_chain):
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
        NL, NR = self.adj.shape
        assert NL == len(self.h_embed)
        assert NR == len(self.v_embed)
        I = len(self.G)

        model = cp_model.CpModel()

        yl = np.array([[model.NewBoolVar(f"yl_{i}_{j}") for j in range(NL)] for i in range(I)])
        yr = np.array([[model.NewBoolVar(f"yr_{i}_{j}") for j in range(NR)] for i in range(I)])

        valid_edge = [(l, r) for l in range(NL) for r in range(NR) if self.adj[l, r] == 1]

        if verbose:
            print(f"(NL: {NL} NR: {NR} Valid edges: {len(valid_edge)})")
            print("Building constraints...")

        for u, v in self.G.edges:
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
                    model.Add(yl[i, l] + yr[i, r] <= int(1 + self.adj[l, r]))

        for i in range(I):
            model.Add(sum(yl[i, :]) <= 1)

        for i in range(I):
            model.Add(sum(yr[i, :]) <= 1)

        for l in range(NL):
            model.Add(sum(yl[:, l]) <= len(self.h_embed[l]))

        for r in range(NR):
            model.Add(sum(yr[:, r]) <= len(self.v_embed[r]))

        if verbose:
            print("Finished building constraints")

        solver = cp_model.CpSolver()
        solver.parameters.use_pb_resolution = True
        solver.parameters.log_search_progress = verbose
        solver.parameters.max_time_in_seconds = timeout
        # solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        # solver.parameters.binary_minimization_algorithm = 2
        status = solver.Solve(model)

        emb = {i: [] for i in range(I)}

        if status == cp_model.UNKNOWN:
            return emb

        for i in range(I):
            for l in range(NL):
                if solver.BooleanValue(yl[i, l]):
                    emb[i].extend(self.h_embed[l].pop())
                    break
            for r in range(NR):
                if solver.BooleanValue(yr[i, r]):
                    emb[i].extend(self.v_embed[r].pop())
                    break

        if return_walltime:
            return emb, solver.WallTime()
        else:
            return emb
