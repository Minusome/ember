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

    if (response.status() == CpSolverStatus::UNKNOWN) {
        return std::move(py::none());
    }

    py::array_t<int> result({num_input_nodes,2});
    std::memset(result.mutable_data(), -1, sizeof(int) * num_input_nodes * 2);
    auto result_mut_proxy = result.mutable_unchecked<2>();

    for (int i = 0; i < num_input_nodes; i++) {
        // result_mut_proxy(i, 0) = -1;
        // result_mut_proxy(i, 1) = -1;

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
}