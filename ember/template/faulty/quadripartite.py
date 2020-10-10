from collections import defaultdict
from random import seed

import networkx as nx
import dwave_networkx as dnx
import numpy as np
from ortools.sat.python import cp_model
from ember.template.util import *

from ember.template.util import Chimera, check_embedding

__all__ = ["Quadripartite"]


def _run_quadripartite(I, G, U1, U2, U3, U4, adj12, adj23, adj34, verbose, timeout, return_walltime):
    N1, N2 = adj12.shape
    N2_, N3_ = adj23.shape
    N3, N4 = adj34.shape
    assert N1 == len(U1)
    assert N2_ == N2 == len(U2)
    assert N3_ == N3 == len(U3)
    assert N4 == len(U4)

    model = cp_model.CpModel()

    y1 = np.array([[model.NewBoolVar(f"y1_{i}_{j}") for j in range(N1)] for i in range(I)])
    y2 = np.array([[model.NewBoolVar(f"y2_{i}_{j}") for j in range(N2)] for i in range(I)])
    y3 = np.array([[model.NewBoolVar(f"y3_{i}_{j}") for j in range(N3)] for i in range(I)])
    y4 = np.array([[model.NewBoolVar(f"y4_{i}_{j}") for j in range(N4)] for i in range(I)])

    valid_edge12 = [(n1, n2) for n2 in range(N2) for n1 in range(N1) if adj12[n1, n2] == 1]
    valid_edge34 = [(n3, n4) for n4 in range(N4) for n3 in range(N3) if adj34[n3, n4] == 1]

    # decision variable for every valid mapping of guest edge to node edge
    for u, v in G:
        or_terms = []
        for n1, n2 in valid_edge12:
            uv = model.NewBoolVar(f"12_({u},{v})_({n1},{n2})")
            model.AddImplication(uv, y1[u, n1])
            model.AddImplication(uv, y2[v, n2])

            vu = model.NewBoolVar(f"12_({v},{u})_({n1},{n2})")
            model.AddImplication(vu, y1[v, n1])
            model.AddImplication(vu, y2[u, n2])

            or_terms.extend((uv, vu))

        for n3, n4 in valid_edge34:
            uv = model.NewBoolVar(f"34_({u},{v})_({n3},{n4})")
            model.AddImplication(uv, y3[u, n3])
            model.AddImplication(uv, y4[v, n4])

            vu = model.NewBoolVar(f"34_({v},{u})_({n3},{n4})")
            model.AddImplication(vu, y3[v, n3])
            model.AddImplication(vu, y4[u, n4])

            or_terms.extend((uv, vu))

        model.AddBoolOr(or_terms)

    for i in range(I):
        for n1 in range(N1):
            for n2 in range(N2):
                model.Add(y2[i, n2] + y1[i, n1] <= int(1 + adj12[n1, n2]))

        for n2 in range(N2):
            for n3 in range(N3):
                model.Add(y3[i, n3] + y2[i, n2] <= int(1 + adj23[n2, n3]))

        for n3 in range(N3):
            for n4 in range(N4):
                model.Add(y4[i, n4] + y3[i, n3] <= int(1 + adj34[n3, n4]))

        model.Add(sum(y1[i, :]) + sum(y3[i, :]) - sum(y2[i, :]) <= 1)
        model.Add(sum(y2[i, :]) + sum(y4[i, :]) - sum(y3[i, :]) <= 1)
        model.Add(sum(y1[i, :]) + sum(y4[i, :]) - sum(y3[i, :]) - sum(y2[i, :]) < 1)

    # guest node should only be assigned once per partite
    for i in range(I):
        model.Add(sum(y1[i, :]) <= 1)
        model.Add(sum(y2[i, :]) <= 1)
        model.Add(sum(y3[i, :]) <= 1)
        model.Add(sum(y4[i, :]) <= 1)

    # number of nodes embedded per partite node not exceed number of duplicates
    for i in range(N1):
        model.Add(sum(y1[:, i]) <= U1[i])
    for i in range(N2):
        model.Add(sum(y2[:, i]) <= U2[i])
    for i in range(N3):
        model.Add(sum(y3[:, i]) <= U3[i])
    for i in range(N4):
        model.Add(sum(y4[:, i]) <= U4[i])

    solver = cp_model.CpSolver()
    solver.parameters.use_pb_resolution = True
    solver.parameters.log_search_progress = verbose
    solver.parameters.max_time_in_seconds = timeout
    # solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    # solver.parameters.binary_minimization_algorithm = 2
    status = solver.Solve(model)

    result = np.full((I, 4), -1)

    if status != cp_model.OPTIMAL:
        return result

    for i in range(I):
        for p1 in range(N1):
            if solver.BooleanValue(y1[i, p1]):
                result[i, 0] = p1
                break
        for p2 in range(N2):
            if solver.BooleanValue(y2[i, p2]):
                result[i, 1] = p2
                break
        for p3 in range(N3):
            if solver.BooleanValue(y3[i, p3]):
                result[i, 2] = p3
                break
        for p4 in range(N4):
            if solver.BooleanValue(y4[i, p4]):
                result[i, 3] = p4
                break

    if return_walltime:
        return result, solver.WallTime()
    else:
        return result


class Quadripartite:
    def __init__(self, guest: nx.Graph, host: Chimera):
        self.G = guest
        self.C = host
        self.U1, self.U2, self.U3, self.U4 = self._quadripartite_embed()
        self.adj12 = self._construct_adj_matrix(self.U1, self.U2)
        self.adj23 = self._construct_adj_matrix(self.U2, self.U3)
        self.adj34 = self._construct_adj_matrix(self.U3, self.U4)
        self.U1, self.U2, self.U3, self.U4, self.adj12, self.adj23, self.adj34 = self._compress_to_unique()
        self.U23 = self._create_U2_U3_pairs()

    def _quadripartite_embed(self):

        def append_nonempty(super, sub):
            if sub:
                super.append(sub)

        M, L, faulty = self.C.M, self.C.L, self.C.faulty
        to_linear = dnx.chimera_coordinates(M, t=L).chimera_to_linear

        U1 = []
        U4 = []
        for i in range(M * L):
            chain1 = []
            chain4 = []
            cell, unit = i // L, i % L
            for j in range(M):
                ln = to_linear((cell, j, 1, unit))
                if ln in faulty:
                    if i < M * L / 2:
                        append_nonempty(U1, chain1)
                        chain1 = []
                    else:
                        append_nonempty(U4, chain4)
                        chain4 = []
                else:
                    if i < M * L / 2:
                        chain1.append(ln)
                    else:
                        chain4.append(ln)
            append_nonempty(U1, chain1)
            append_nonempty(U4, chain4)

        U2 = []
        U3 = []
        for i in range(M * L):
            chain2 = []
            chain3 = []
            cell, unit = i // L, i % L
            for j in range(M):
                ln = to_linear((j, cell, 0, unit))
                if ln in faulty:
                    if j < M / 2:
                        append_nonempty(U2, chain2)
                        chain2 = []
                    else:
                        append_nonempty(U3, chain3)
                        chain3 = []
                else:
                    if j < M / 2:
                        chain2.append(ln)
                    else:
                        chain3.append(ln)
            append_nonempty(U2, chain2)
            append_nonempty(U3, chain3)

        return U1, U2, U3, U4

    def _neighbours(self, graph, chain):
        chain = set(chain)
        nb_nodes = {node for c in chain for node in graph[c]}
        nb_nodes.difference_update(chain)
        return nb_nodes

    def _construct_adj_matrix(self, p1, p2):

        v_inverse = {v: i for i in range(len(p2)) for v in p2[i]}
        chimera = self.C.internal

        adj = np.zeros((len(p1), len(p2)))
        for i, h_chain in enumerate(p1):
            for nb in self._neighbours(chimera, h_chain):
                try:
                    adj[i][v_inverse[nb]] = 1
                except:
                    pass
        return adj

    def _compress_to_unique(self):
        u1_group, u2_group, u3_group, u4_group = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(
            list)

        u2_ind, u3_ind = np.where(self.adj23 == 1)
        adj1234 = np.concatenate([self.adj12[:, u2_ind], np.transpose(self.adj34[u3_ind, :])])
        adj1234, adj1234_ind = np.unique(adj1234, return_inverse=True, axis=1)

        adj12_ind = np.array(adj1234_ind)
        for unconnected in sorted(set(range(len(self.U2))) - set(u2_ind)):
            adj12_ind = np.insert(adj12_ind, unconnected, max(adj12_ind) + 1)

        adj34_ind = np.array(adj1234_ind)
        for unconnected in sorted(set(range(len(self.U3))) - set(u3_ind)):
            adj34_ind = np.insert(adj34_ind, unconnected, max(adj34_ind) + 1)

        for idx, chain in zip(adj12_ind, self.U2):
            u2_group[idx].append(chain)

        for idx, chain in zip(adj34_ind, self.U3):
            u3_group[idx].append(chain)

        new_u2 = [u2_group[key][0] for key in sorted(list(u2_group.keys()))]
        new_u3 = [u3_group[key][0] for key in sorted(list(u3_group.keys()))]

        adj12 = self._construct_adj_matrix(self.U1, new_u2)
        adj23 = self._construct_adj_matrix(new_u2, new_u3)
        adj34 = self._construct_adj_matrix(new_u3, self.U4)

        adj12, u1_inv = np.unique(adj12, return_inverse=True, axis=0)
        adj34, u4_inv = np.unique(adj34, return_inverse=True, axis=1)

        for idx, chain in zip(u1_inv, self.U1):
            u1_group[idx].append(chain)

        for idx, chain in zip(u4_inv, self.U4):
            u4_group[idx].append(chain)

        return u1_group, u2_group, u3_group, u4_group, adj12, adj23, adj34

    def _create_U2_U3_pairs(self):
        index2, index3 = np.where(self.adj23 == 1)
        U23 = defaultdict(list)
        chimera = self.C.internal

        for i in range(len(index2)):
            u2 = self.U2[index2[i]]
            u3 = self.U3[index3[i]]
            for chain2 in u2:
                for nb in self._neighbours(chimera, chain2):
                    for chain3 in u3:
                        if nb in chain3:
                            U23[(index2[i], index3[i])].append((chain2, chain3))
        return U23

    def solve(self, verbose=True, timeout=500, return_walltime=False):
        U1_count = np.array([len(self.U1[u1]) for u1 in range(len(self.U1))])
        U2_count = np.array([len(self.U2[u2]) for u2 in range(len(self.U2))])
        U3_count = np.array([len(self.U3[u3]) for u3 in range(len(self.U3))])
        U4_count = np.array([len(self.U4[u4]) for u4 in range(len(self.U4))])
        I = len(self.G)

        #try:
        #    import ember.template._native.embed as embed
        #    print("Running C++")
        #    run_quadripartite = embed.run_quadripartite
        #except ImportError:
        #    print("Running Python")
        #    run_quadripartite = _run_quadripartite
        print("trying to import CPP")
        import ember.template._native.embed as embed
	print("imported CPP")
        run_quadripartite = embed.run_quadripartite

        result = run_quadripartite(I, np.array(self.G.edges), U1_count, U2_count, U3_count, U4_count,
                                   self.adj12, self.adj23, self.adj34, verbose, timeout, return_walltime)

        emb = {i: [] for i in range(I)}

        if return_walltime:
            result, walltime = result
        print(result)
        for i in range(I):
            p1, p2, p3, p4 = result[i]
            if p1 != -1:
                emb[i].extend(self.U1[p1].pop())
            if (p2 != -1) & (p3 != -1):
                nodes = self.U23[(p2, p3)].pop()
                emb[i].extend(nodes[0])
                emb[i].extend(nodes[1])
                self.U2[p2].remove(nodes[0])
                self.U3[p3].remove(nodes[1])
            if p4 != -1:
                emb[i].extend(self.U4[p4].pop())

        for i in range(I):
            p1, p2, p3, p4 = result[i]
            if p2 != -1 & p3 == -1:
                emb[i].extend(self.U2[p2].pop())
            elif p3 != -1 & p2 == -1:
                emb[i].extend(self.U3[p3].pop())

        if return_walltime:
            return emb, walltime
        else:
            return emb


seed(12)
G = nx.generators.gnp_random_graph(10, 0.2, seed=2)
C = D_WAVE_2000Q(k_rand_faulty=10)
em, walltime = Quadripartite(G, C).solve(return_walltime=True)
print(em)

if check_embedding(em, G, C):
    print("found embedding")
    plot_chimera_embedding(em, C)
