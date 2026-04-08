// solution_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;

std::tuple<double, std::unordered_map<int, double>> calculate_total_distance(
    const std::vector<std::vector<int>> &routes,
    const py::array_t<double> &distance_matrix)
{
    auto dist = distance_matrix.unchecked<2>();

    double total_distance = 0.0;
    std::unordered_map<int, double> route_lengths;

    for (size_t idx = 0; idx < routes.size(); ++idx)
    {
        const auto &route = routes[idx];

        if (route.size() < 2)
        {
            route_lengths[static_cast<int>(idx)] = 0.0;
            continue;
        }

        double route_distance = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i)
        {
            int from_node = route[i];
            int to_node = route[i + 1];
            route_distance += dist(from_node, to_node);
        }

        route_lengths[static_cast<int>(idx)] = route_distance;
        total_distance += route_distance;
    }

    return std::make_tuple(total_distance, route_lengths);
}

std::unordered_map<int, int> build_node_to_route(
    const std::vector<std::vector<int>> &routes,
    const std::vector<int> &pickup_nodes,
    const std::vector<int> &delivery_nodes)
{
    std::unordered_set<int> pickups(pickup_nodes.begin(), pickup_nodes.end());
    std::unordered_set<int> deliveries(delivery_nodes.begin(), delivery_nodes.end());

    std::unordered_map<int, int> mapping;

    for (size_t route_idx = 0; route_idx < routes.size(); ++route_idx)
    {
        const auto &route = routes[route_idx];
        for (int node : route)
        {
            if (pickups.find(node) != pickups.end() || deliveries.find(node) != deliveries.end())
            {
                mapping[node] = static_cast<int>(route_idx);
            }
        }
    }

    return mapping;
}

PYBIND11_MODULE(solution_core, m)
{
    m.doc() = "Optimized solution properties (C++ / pybind11)";

    m.def("calculate_total_distance", &calculate_total_distance,
          py::arg("routes"),
          py::arg("distance_matrix"),
          "Calculate total distance and route lengths for all routes");

    m.def("build_node_to_route", &build_node_to_route,
          py::arg("routes"),
          py::arg("pickup_nodes"),
          py::arg("delivery_nodes"),
          "Build mapping from node indices to route indices");
}
