from collections import defaultdict
from random import seed

import networkx as nx
import dwave_networkx as dnx
import numpy as np
from ortools.sat.python import cp_model

from template.util import Chimera, check_embedding


class Quadripartite:
    def __init__(self, G, C: Chimera):
        self.G = G
        self.C = C
        self.U1, self.U2, self.U3, self.U4 = self._quadripartite_embed()
        self.adj12 = self._construct_adj_matrix(self.U1, self.U2)
        self.adj23 = self._construct_adj_matrix(self.U2, self.U3)
        self.adj34 = self._construct_adj_matrix(self.U3, self.U4)
        self.U1, self.U2, self.U3, self.U4, self.adj12, self.adj23, self.adj34 = self._compress_to_unique()
        self.U23 = self._create_U2_U3_pairs()

    def _create_U2_U3_pairs(self):
        index3, index2 = np.where(self.adj23 == 1)
        U23 = defaultdict(list)
        chimera = self.C.graph

        for i in range(len(index2)):
            u2 = self.U2[index2[i]]
            u3 = self.U3[index3[i]]
            for chain2 in u2:
                for nb in self._neighbours(chimera, chain2):
                    for chain3 in u3:
                        if nb in chain3:
                            U23[(index2[i], index3[i])].append((chain2, chain3))
        return U23

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

        v_inverse = {v: i for i in range(len(p1)) for v in p1[i]}
        chimera = self.C.graph

        adj = np.zeros((len(p2), len(p1)))
        for i, h_chain in enumerate(p2):
            for nb in self._neighbours(chimera, h_chain):
                try:
                    adj[i][v_inverse[nb]] = 1
                except:
                    pass
        return adj

    def _compress_to_unique(self):
        u1_group, u2_group, u3_group, u4_group = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(
            list)

        adj12, u1_inv = np.unique(self.adj12, return_inverse=True, axis=1)
        adj34, u4_inv = np.unique(self.adj34, return_inverse=True, axis=0)

        for idx, chain in zip(u1_inv, self.U1):
            u1_group[idx].append(chain)

        for idx, chain in zip(u4_inv, self.U4):
            u4_group[idx].append(chain)

        [u3_ind, u2_ind] = np.where(self.adj23 == 1)
        unconnected_u2 = [i for i in range(self.adj23.shape[1]) if i not in u2_ind]
        unconnected_u3 = [i for i in range(self.adj23.shape[0]) if i not in u3_ind]

        adj1234 = np.concatenate([np.transpose(adj12[u2_ind, :]), adj34[:, u3_ind]])
        adj1234, adj1234_ind = np.unique(adj1234, return_inverse=True, axis=1)

        adj12_ind = np.array(adj1234_ind)
        for unconnected in unconnected_u2:
            adj12_ind = np.insert(adj12_ind, unconnected, max(adj12_ind) + 1)

        adj34_ind = np.array(adj1234_ind)
        for unconnected in unconnected_u3:
            adj34_ind = np.insert(adj34_ind, unconnected, max(adj34_ind) + 1)

        for idx, chain in zip(adj12_ind, self.U2):
            u2_group[idx].append(chain)

        for idx, chain in zip(adj34_ind, self.U3):
            u3_group[idx].append(chain)

        new_u1 = [item[0] for item in list(u1_group.values())]
        new_u2 = [item[0] for item in list(u2_group.values())]
        new_u3 = [item[0] for item in list(u3_group.values())]
        new_u4 = [item[0] for item in list(u4_group.values())]
        adj12 = self._construct_adj_matrix(new_u1, new_u2)
        adj23 = self._construct_adj_matrix(new_u2, new_u3)
        adj34 = self._construct_adj_matrix(new_u3, new_u4)
        final_u1 = {list(u1_group.values()).index(item): item for item in list(u1_group.values())}
        final_u2 = {list(u2_group.values()).index(item): item for item in list(u2_group.values())}
        final_u3 = {list(u3_group.values()).index(item): item for item in list(u3_group.values())}
        final_u4 = {list(u4_group.values()).index(item): item for item in list(u4_group.values())}

        return final_u1, final_u2, final_u3, final_u4, adj12, adj23, adj34

    def solve(self, verbose=True, timeout=500, return_walltime=False):
        N2, N1 = self.adj12.shape
        N3_, N2_ = self.adj23.shape
        N4, N3 = self.adj34.shape
        assert N1 == len(self.U1)
        assert N2_ == N2 == len(self.U2)
        assert N3_ == N3 == len(self.U3)
        assert N4 == len(self.U4)

        I = len(self.G)

        model = cp_model.CpModel()

        y1 = np.array([[model.NewBoolVar(f"y1_{i}_{j}") for j in range(N1)] for i in range(I)])
        y2 = np.array([[model.NewBoolVar(f"y2_{i}_{j}") for j in range(N2)] for i in range(I)])
        y3 = np.array([[model.NewBoolVar(f"y3_{i}_{j}") for j in range(N3)] for i in range(I)])
        y4 = np.array([[model.NewBoolVar(f"y4_{i}_{j}") for j in range(N4)] for i in range(I)])

        valid_edge12 = [(l, r) for l in range(N2) for r in range(N1) if self.adj12[l, r] == 1]
        valid_edge34 = [(l, r) for l in range(N4) for r in range(N3) if self.adj34[l, r] == 1]

        # decision variable for every valid mapping of guest edge to node edge
        for u, v in self.G.edges:
            or_terms = []
            for l, r in valid_edge12:
                uv = model.NewBoolVar(f"12_({u},{v})_({l},{r})")
                model.AddImplication(uv, y2[u, l])
                model.AddImplication(uv, y1[v, r])

                vu = model.NewBoolVar(f"12_({v},{u})_({l},{r})")
                model.AddImplication(vu, y2[v, l])
                model.AddImplication(vu, y1[u, r])

                or_terms.extend((uv, vu))

            for l, r in valid_edge34:
                uv = model.NewBoolVar(f"34_({u},{v})_({l},{r})")
                model.AddImplication(uv, y4[u, l])
                model.AddImplication(uv, y3[v, r])

                vu = model.NewBoolVar(f"34_({v},{u})_({l},{r})")
                model.AddImplication(vu, y4[v, l])
                model.AddImplication(vu, y3[u, r])

                or_terms.extend((uv, vu))

            model.AddBoolOr(or_terms)

        for i in range(I):
            for n1 in range(N1):
                for n2 in range(N2):
                    model.Add(y2[i, n2] + y1[i, n1] <= int(1 + self.adj12[n2, n1]))

            for n2 in range(N2):
                for n3 in range(N3):
                    model.Add(y3[i, n3] + y2[i, n2] <= int(1 + self.adj23[n3, n2]))

            for n3 in range(N3):
                for n4 in range(N4):
                    model.Add(y4[i, n4] + y3[i, n3] <= int(1 + self.adj34[n4, n3]))

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
            model.Add(sum(y1[:, i]) <= len(self.U1[i]))
        for i in range(N2):
            model.Add(sum(y2[:, i]) <= len(self.U2[i]))
        for i in range(N3):
            model.Add(sum(y3[:, i]) <= len(self.U3[i]))
        for i in range(N4):
            model.Add(sum(y4[:, i]) <= len(self.U4[i]))

        solver = cp_model.CpSolver()
        solver.parameters.use_pb_resolution = True
        solver.parameters.log_search_progress = verbose
        solver.parameters.max_time_in_seconds = timeout
        # solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        # solver.parameters.binary_minimization_algorithm = 2
        status = solver.Solve(model)

        emb = {i: [] for i in range(I)}

        if status != cp_model.OPTIMAL:
            return emb

        pairs = defaultdict(tuple)
        u2_emb = defaultdict(int)
        u3_emb = defaultdict(int)

        for i in range(I):
            for p1 in range(N1):
                if solver.BooleanValue(y1[i, p1]):
                    emb[i].extend(self.U1[p1].pop())
                    break

            for p2 in range(N2):
                for p3 in range(N3):
                    if solver.BooleanValue(y2[i, p2]) and solver.BooleanValue(y3[i, p3]):
                        pairs[i] = (p2, p3)
                        break

            if i not in pairs.keys():
                for p2 in range(N2):
                    if solver.BooleanValue(y2[i, p2]):
                        u2_emb[i] = p2
                        break
                for p3 in range(N3):
                    if solver.BooleanValue(y3[i, p3]):
                        u3_emb[i] = p3
                        break

            for p4 in range(N4):
                if solver.BooleanValue(y4[i, p4]):
                    emb[i].extend(self.U4[p4].pop())
                    break

        for i, pair in pairs.items():
            nodes = self.U23[pair].pop()
            emb[i].extend(nodes[0])
            emb[i].extend(nodes[1])
            self.U2[pair[0]].remove(nodes[0])
            self.U3[pair[1]].remove(nodes[1])
        for i, u2 in u2_emb.items():
            emb[i].extend(self.U2[u2].pop())
        for i, u3 in u3_emb.items():
            emb[i].extend(self.U3[u3].pop())

        if return_walltime:
            return emb, solver.WallTime()
        else:
            return emb


seed(0)
G = nx.generators.complete_graph(10)
C = Chimera(16, 4).random_faulty(20)

q = Quadripartite(G, C)
em = q.solve()
print(em)
if check_embedding(em, G, C):
    print("true")
