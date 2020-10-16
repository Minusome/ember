#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ortools/sat/cp_model.h>
#include <ortools/sat/model.h>
#include <ortools/sat/sat_parameters.pb.h>

namespace py = pybind11;
using namespace py::literals;
using namespace operations_research::sat;

namespace template_embed_faulty {

py::object RunQuadripartite(
    int num_input_nodes,
    py::array_t<int> input_edges,
    py::array_t<int> partite_u1,
    py::array_t<int> partite_u2,
    py::array_t<int> partite_u3,
    py::array_t<int> partite_u4,
    py::array_t<int> adjacency_matrix_12,
    py::array_t<int> adjacency_matrix_23,
    py::array_t<int> adjacency_matrix_34,
    bool verbose,
    int timeout,
    bool return_walltime)
{
    int N1 = adjacency_matrix_12.shape()[0];
    int N2 = adjacency_matrix_12.shape()[1];
    int N2_ = adjacency_matrix_23.shape()[0];
    int N3_ = adjacency_matrix_23.shape()[1];
    int N3 = adjacency_matrix_34.shape()[0];
    int N4 = adjacency_matrix_34.shape()[1];

    assert(N1 == partite_u1.size());
    assert(N2 == partite_u2.size());
    assert(N2_ == partite_u2.size());
    assert(N3 == partite_u3.size());
    assert(N3_ == partite_u3.size());
    assert(N4 == partite_u4.size());

    auto input_edges_proxy = input_edges.unchecked<2>();

    auto partite_u1_proxy = partite_u1.unchecked<1>();
    auto partite_u2_proxy = partite_u2.unchecked<1>();
    auto partite_u3_proxy = partite_u3.unchecked<1>();
    auto partite_u4_proxy = partite_u4.unchecked<1>();

    auto adjacency_matrix_12_proxy = adjacency_matrix_12.unchecked<2>();
    auto adjacency_matrix_23_proxy = adjacency_matrix_23.unchecked<2>();
    auto adjacency_matrix_34_proxy = adjacency_matrix_34.unchecked<2>();

    CpModelBuilder cp_model;

    std::vector<std::vector<BoolVar>> y1(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int n1 = 0; n1 < N1; n1++) {
            y1[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::vector<BoolVar>> y2(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int n2 = 0; n2 < N2; n2++) {
            y2[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::vector<BoolVar>> y3(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int n3 = 0; n3 < N3; n3++) {
            y3[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::vector<BoolVar>> y4(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int n4 = 0; n4 < N4; n4++) {
            y4[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::pair<int, int>> valid_edges12;
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            if (adjacency_matrix_12_proxy(n1, n2) == 1) {
                valid_edges12.push_back({n1, n2});
            }
        }
    }

    std::vector<std::pair<int, int>> valid_edges34;
    for (int n3 = 0; n3 < N3; n3++) {
        for (int n4 = 0; n4 < N4; n4++) {
            if (adjacency_matrix_34_proxy(n3, n4) == 1) {
                valid_edges34.push_back({n3, n4});
            }
        }
    }

    for (int i = 0; i < input_edges.shape()[0]; i++) {
        const int& u = input_edges_proxy(i, 0);
        const int& v = input_edges_proxy(i, 1);
        std::vector<BoolVar> or_terms;
        for (auto& pair : valid_edges12) {
            const int& n1 = pair.first;
            const int& n2 = pair.second;

            BoolVar uv = cp_model.NewBoolVar();
            cp_model.AddImplication(uv, y1[u][n1]);
            cp_model.AddImplication(uv, y2[v][n2]);
            or_terms.push_back(uv);

            BoolVar vu = cp_model.NewBoolVar();
            cp_model.AddImplication(vu, y1[v][n1]);
            cp_model.AddImplication(vu, y2[u][n2]);
            or_terms.push_back(vu);
        }

        for (auto& pair : valid_edges34) {
            const int& n3 = pair.first;
            const int& n4 = pair.second;

            BoolVar uv = cp_model.NewBoolVar();
            cp_model.AddImplication(uv, y3[u][n3]);
            cp_model.AddImplication(uv, y4[v][n4]);
            or_terms.push_back(uv);

            BoolVar vu = cp_model.NewBoolVar();
            cp_model.AddImplication(vu, y3[v][n3]);
            cp_model.AddImplication(vu, y4[u][n4]);
            or_terms.push_back(vu);
        }
        cp_model.AddBoolOr(or_terms);
    }

    for (int i = 0; i < num_input_nodes; i++) {
        for (int n1 = 0; n1 < N1; n1++) {
            for (int n2 = 0; n2 < N2; n2++) {
                cp_model.AddLessOrEqual(
                    LinearExpr::BooleanSum({y1[i][n1], y2[i][n2]}),
                    1 + adjacency_matrix_12_proxy(n1 ,n2));
            }
        }

        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                cp_model.AddLessOrEqual(
                    LinearExpr::BooleanSum({y2[i][n2], y3[i][n3]}),
                    1 + adjacency_matrix_23_proxy(n2 ,n3));
            }
        }

        for (int n3 = 0; n3 < N3; n3++) {
            for (int n4 = 0; n4 < N4; n4++) {
                cp_model.AddLessOrEqual(
                    LinearExpr::BooleanSum({y3[i][n3], y4[i][n4]}),
                    1 + adjacency_matrix_34_proxy(n3 ,n4));
            }
        }

        std::vector<BoolVar> vars1(N1 + N2 + N3);
        std::vector<int64> coeffs1(N1 + N2 + N3);
        for (int n1 = 0; n1 < N1; n1++){
            vars1[n1] = y1[i][n1];
            coeffs1[n1] = 1;
        }
        for (int n3 = 0; n3 < N3; n3++){
            vars1[N1+n3] = y3[i][n3];
            coeffs1[N1+n3] = 1;
        }
        for (int n2 = 0; n2 < N2; n2++){
            vars1[N1+N3+n2] = y2[i][n2];
            coeffs1[N1+N3+n2] = -1;
        }

        std::vector<BoolVar> vars2(N2 + N3 + N4);
        std::vector<int64> coeffs2(N2 + N3 + N4);
        for (int n2 = 0; n2 < N2; n2++){
            vars2[n2] = y2[i][n2];
            coeffs2[n2] = 1;
        }
        for (int n4 = 0; n4 < N4; n4++){
            vars2[N2+n4] = y4[i][n4];
            coeffs2[N2+n4] = 1;
        }
        for (int n3 = 0; n3 < N3; n3++){
            vars2[N2+N4+n3] = y3[i][n3];
            coeffs2[N2+N4+n3] = -1;
        }

        std::vector<BoolVar> vars3(N1 + N2 + N3 + N4);
        std::vector<int64> coeffs3(N1 + N2 + N3 + N4);
        for (int n1 = 0; n1 < N1; n1++){
            vars3[n1] = y1[i][n1];
            coeffs3[n1] = 1;
        }
        for (int n4 = 0; n4 < N4; n4++){
            vars3[N1+n4] = y4[i][n4];
            coeffs3[N1+n4] = 1;
        }
        for (int n3 = 0; n3 < N3; n3++){
            vars3[N1+N4+n3] = y3[i][n3];
            coeffs3[N1+N4+n3] = -1;
        }
        for (int n2 = 0; n2 < N2; n2++){
            vars3[N1+N4+N3+n2] = y2[i][n2];
            coeffs3[N1+N4+N3+n2] = -1;
        }

        cp_model.AddLessOrEqual(LinearExpr::BooleanScalProd(vars1, coeffs1), 1);
        cp_model.AddLessOrEqual(LinearExpr::BooleanScalProd(vars2, coeffs2), 1);
        cp_model.AddLessOrEqual(LinearExpr::BooleanScalProd(vars3, coeffs3), 1);
    }

    for (int i = 0; i < num_input_nodes; i++) {
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(y1[i]), 1);
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(y2[i]), 1);
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(y3[i]), 1);
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(y4[i]), 1);
    }

    for (int n1 = 0; n1 < N1; n1++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = y1[i][n1];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), partite_u1_proxy(n1));
    }

    for (int n2 = 0; n2 < N2; n2++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = y2[i][n2];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), partite_u2_proxy(n2));
    }

    for (int n3 = 0; n3 < N3; n3++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = y3[i][n3];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), partite_u3_proxy(n3));
    }

    for (int n4 = 0; n4 < N4; n4++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = y4[i][n4];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), partite_u4_proxy(n4));
    }

    SatParameters parameters;
    parameters.set_use_pb_resolution(true);
    parameters.set_log_search_progress(verbose);
    parameters.set_max_time_in_seconds(timeout);

    CpSolverResponse response = SolveWithParameters(cp_model.Build(), parameters);

    if (response.status() != CpSolverStatus::OPTIMAL) {
        return std::move(py::none());
    }

    py::array_t<int> result({num_input_nodes,4});
    std::memset(result.mutable_data(), -1, sizeof(int) * num_input_nodes * 4);
    auto result_mut_proxy = result.mutable_unchecked<2>();

    for (int i = 0; i < num_input_nodes; i++) {
        for (int n1 = 0; n1 < N1; n1++) {
            if (SolutionBooleanValue(response, y1[i][n1])) {
                result_mut_proxy(i, 0) = n1;
                break;
            }
        }
        for (int n2 = 0; n2 < N2; n2++) {
            if (SolutionBooleanValue(response, y2[i][n2])) {
                result_mut_proxy(i, 1) = n2;
                break;
            }
        }
        for (int n3 = 0; n3 < N3; n3++) {
            if (SolutionBooleanValue(response, y3[i][n3])) {
                result_mut_proxy(i, 2) = n3;
                break;
            }
        }
        for (int n4 = 0; n4 < N4; n4++) {
            if (SolutionBooleanValue(response, y4[i][n4])) {
                result_mut_proxy(i, 3) = n4;
                break;
            }
        }
    }

    if (return_walltime) {
        return std::move(py::make_tuple(result, response.wall_time()));
    }

    return std::move(result);
}

py::object RunBipartite(
    int num_input_nodes,
    py::array_t<int> input_edges,
    py::array_t<int> biadjacency_matrix,
    py::array_t<int> horizontal,
    py::array_t<int> vertical,
    bool verbose,
    int timeout,
    bool return_walltime) 
{
    int num_left = horizontal.size();
    int num_right = vertical.size();
    
    assert(num_left == biadjacency_matrix.shape()[0]);
    assert(num_right == biadjacency_matrix.shape()[1]);

    auto biadjacency_matrix_proxy = biadjacency_matrix.unchecked<2>();
    auto input_edges_proxy = input_edges.unchecked<2>();
    auto horizontal_proxy = horizontal.unchecked<1>();
    auto vertical_proxy = vertical.unchecked<1>();

    CpModelBuilder cp_model;

    std::vector<std::vector<BoolVar>> yl(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int l = 0; l < num_left; l++) {
            yl[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::vector<BoolVar>> yr(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int r = 0; r < num_right; r++) {
            yr[i].push_back(cp_model.NewBoolVar());
        }
    }

    std::vector<std::pair<int, int>> valid_edges;
    for (int l = 0; l < num_left; l++) {
        for (int r = 0; r < num_right; r++) {
            if (biadjacency_matrix_proxy(l, r) == 1) {
                valid_edges.push_back({l, r});
            }
        }
    }

    if (verbose) {
        std::cout << "(NL: " << num_left 
                  << " NR: " << num_right 
                  << " Valid edges: " << valid_edges.size() << ")\n"
                  << "Building constraints..." << "\n";
    }

    for (int i = 0; i < input_edges.shape()[0]; i++) {
        const int& u = input_edges_proxy(i, 0);
        const int& v = input_edges_proxy(i, 1);
        std::vector<BoolVar> or_terms;
        for (auto& pair : valid_edges) {
            const int& l = pair.first;
            const int& r = pair.second;

            BoolVar uv = cp_model.NewBoolVar();
            cp_model.AddImplication(uv, yl[u][l]);
            cp_model.AddImplication(uv, yr[v][r]);
            or_terms.push_back(uv);

            BoolVar vu = cp_model.NewBoolVar();
            cp_model.AddImplication(vu, yl[v][l]);
            cp_model.AddImplication(vu, yr[u][r]);
            or_terms.push_back(vu);
        }
        cp_model.AddBoolOr(or_terms);
    }

    for (int i = 0; i < num_input_nodes; i++) {
        for (int l = 0; l < num_left; l++) {
            for (int r = 0; r < num_right; r++) {
                cp_model.AddLessOrEqual(LinearExpr::BooleanSum({yl[i][l], yr[i][r]}),
                                        1 + biadjacency_matrix_proxy(l ,r));
            }
        }
    }

    for (int i = 0; i < num_input_nodes; i++) {
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(yl[i]), 1);
    }

    for (int i = 0; i < num_input_nodes; i++) {
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(yr[i]), 1);
    }

    for (int l = 0; l < num_left; l++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = yl[i][l];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), horizontal_proxy(l));
    }

    for (int r = 0; r < num_right; r++) {
        std::vector<BoolVar> col(num_input_nodes);
        for (int i = 0; i < num_input_nodes; i++) {
            col[i] = yr[i][r];
        }
        cp_model.AddLessOrEqual(LinearExpr::BooleanSum(col), vertical_proxy(r));
    }

    SatParameters parameters;
    parameters.set_use_pb_resolution(true);
    parameters.set_log_search_progress(verbose);
    parameters.set_max_time_in_seconds(timeout);

    CpSolverResponse response = SolveWithParameters(cp_model.Build(), parameters);

    if (response.status() != CpSolverStatus::OPTIMAL) {
        return std::move(py::none());
    }

    py::array_t<int> result({num_input_nodes,2});
    std::memset(result.mutable_data(), -1, sizeof(int) * num_input_nodes * 2);
    auto result_mut_proxy = result.mutable_unchecked<2>();

    for (int i = 0; i < num_input_nodes; i++) {
        for (int l = 0; l < num_left; l++) {
            if (SolutionBooleanValue(response, yl[i][l])) {
                result_mut_proxy(i, 0) = l;
                break;
            }
        }
        for (int r = 0; r < num_right; r++) {
            if (SolutionBooleanValue(response, yr[i][r])) {
                result_mut_proxy(i, 1) = r;
                break;
            }
        }
    }

    if (return_walltime) {
        return std::move(py::make_tuple(result, response.wall_time()));
    }

    return std::move(result);
}
}  // namepsace template_embed_faulty

PYBIND11_MODULE(embed, m) 
{
    m.doc() = "Loading template embedding with faulty qubits using C++"; 

    m.def("run_bipartite_sat", 
          &template_embed_faulty::RunBipartite,
          "Load the bipartite constraint model using C++");

    m.def("run_quadripartite",
          &template_embed_faulty::RunQuadripartite,
          "Load the quadripartite constraint model using C++");
}