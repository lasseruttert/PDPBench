"""Prompt construction for PDPBench tasks.

Converts PDPTWProblem and PDPTWSolution objects into JSON-based prompts
for LLM evaluation. Supports multiple distance representation modes.
"""

import json
from enum import Enum

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution


class DistanceMode(Enum):
    MATRIX = "matrix"
    COORDINATES = "coordinates"
    TOOL_USE = "tool_use"


# ---------------------------------------------------------------------------
# Shared PDPTW rules header
# ---------------------------------------------------------------------------

PDPTW_RULES = """You are solving a Pickup and Delivery Problem with Time Windows (PDPTW).

## Rules
- There are exactly {num_vehicles} vehicles, each with capacity {capacity}. The number of vehicles is fixed.
- Each vehicle starts and ends at the depot (node 0).
- Each request consists of a pickup node (positive demand) and a delivery node (negative demand).
- The pickup must be visited before its corresponding delivery, and both must be on the same vehicle.
- Vehicle load must never exceed capacity or drop below 0 at any point along the route.
- Each node must be visited within its time window [earliest, latest]. If a vehicle arrives early it waits; if it arrives after the latest time, the solution is infeasible.
- Service time is spent at each node after arrival before departing to the next node.
- Travel time between nodes equals the distance between them.
- Every pickup and delivery node must be served exactly once across all routes.
- The primary goal is feasibility. The secondary goal is minimizing total travel distance."""


# ---------------------------------------------------------------------------
# Problem & solution JSON builders
# ---------------------------------------------------------------------------

def build_problem_json(problem: PDPTWProblem, distance_mode: DistanceMode) -> dict:
    """Convert a PDPTWProblem into a JSON-serializable dict for the prompt."""
    if distance_mode == DistanceMode.COORDINATES and problem.dataset == "Mendeley":
        raise ValueError(
            "COORDINATES mode cannot be used with Mendeley instances "
            "(they use OSRM road-network distances, not Euclidean)."
        )

    nodes = []
    for node in problem.nodes:
        node_dict = {
            "index": node.index,
            "demand": node.demand,
            "time_window": list(node.time_window),
            "service_time": node.service_time,
        }

        # Classify node type
        if node.index == 0:
            node_dict["type"] = "depot"
        elif problem.is_pickup(node.index):
            node_dict["type"] = "pickup"
            node_dict["paired_delivery"] = node.delivery_index
        elif problem.is_delivery(node.index):
            node_dict["type"] = "delivery"
            node_dict["paired_pickup"] = node.pickup_index

        # Add coordinates in COORDINATES or TOOL_USE mode
        if distance_mode in (DistanceMode.COORDINATES, DistanceMode.TOOL_USE):
            node_dict["x"] = round(float(node.x), 2)
            node_dict["y"] = round(float(node.y), 2)

        nodes.append(node_dict)

    result = {
        "num_vehicles": problem.num_vehicles,
        "vehicle_capacity": problem.vehicle_capacity,
        "nodes": nodes,
    }

    if distance_mode == DistanceMode.MATRIX:
        # Round to int — Li & Lim are near-integer Euclidean, Mendeley are already int
        matrix = problem.distance_matrix
        result["distance_matrix"] = [
            [int(round(float(matrix[i, j]))) for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]

    if distance_mode == DistanceMode.COORDINATES:
        result["distance_note"] = (
            "Distances are Euclidean: dist(i,j) = sqrt((xi-xj)^2 + (yi-yj)^2). "
            "Travel time equals distance."
        )

    if distance_mode == DistanceMode.TOOL_USE:
        result["distance_note"] = (
            "Use the get_distance(i, j) tool to query the travel distance/time "
            "between any two nodes."
        )

    return result


def build_solution_json(solution: PDPTWSolution, include_distance: bool = True) -> dict:
    """Convert a PDPTWSolution into a JSON-serializable dict."""
    result = {"routes": solution.routes}
    if include_distance:
        result["total_distance"] = round(float(solution.total_distance), 2)
    return result


def _format_prompt(problem: PDPTWProblem, distance_mode: DistanceMode,
                   task_instructions: str, extra_data: dict | None = None) -> str:
    """Assemble a complete prompt from rules, problem data, and task instructions."""
    rules = PDPTW_RULES.format(
        num_vehicles=problem.num_vehicles,
        capacity=problem.vehicle_capacity,
    )

    problem_json = build_problem_json(problem, distance_mode)
    sections = [
        rules,
        "## Problem Data",
        f"```json\n{json.dumps(problem_json, separators=(',', ':'))}\n```",
    ]

    if extra_data:
        for title, data in extra_data.items():
            sections.append(f"## {title}")
            sections.append(f"```json\n{json.dumps(data, separators=(',', ':'))}\n```")

    sections.append("## Task")
    sections.append(task_instructions)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Task-specific prompt builders
# ---------------------------------------------------------------------------

def build_masked_node_prompt(
    problem: PDPTWProblem,
    solution: PDPTWSolution,
    route_idx: int,
    position_idx: int,
    distance_mode: DistanceMode,
) -> str:
    """Build prompt for Task 1: Masked Node Prediction."""
    # Create masked solution
    masked_routes = [route[:] for route in solution.routes]
    correct_node = masked_routes[route_idx][position_idx]
    masked_routes[route_idx][position_idx] = "?"

    sol_data = {"routes": masked_routes}

    instructions = (
        f"One node in Route {route_idx} at position {position_idx} has been masked with '?'.\n\n"
        "Identify the correct node index that should replace '?'. "
        "The original solution is feasible — your predicted node must maintain feasibility.\n\n"
        "Respond with the predicted node index and your reasoning."
    )

    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Solution (with masked node)": sol_data})


def build_request_insertion_prompt(
    problem: PDPTWProblem,
    partial_solution: PDPTWSolution,
    removed_requests: list[tuple[int, int]],
    distance_mode: DistanceMode,
) -> str:
    """Build prompt for Task 2: Request Insertion."""
    sol_data = build_solution_json(partial_solution, include_distance=False)

    # Build info about removed requests
    requests_info = []
    for pickup_idx, delivery_idx in removed_requests:
        pickup_node = problem.nodes_dict[pickup_idx]
        delivery_node = problem.nodes_dict[delivery_idx]
        requests_info.append({
            "pickup": {
                "index": pickup_idx,
                "demand": pickup_node.demand,
                "time_window": list(pickup_node.time_window),
                "service_time": pickup_node.service_time,
            },
            "delivery": {
                "index": delivery_idx,
                "demand": delivery_node.demand,
                "time_window": list(delivery_node.time_window),
                "service_time": delivery_node.service_time,
            },
        })

    instructions = (
        f"{len(removed_requests)} request(s) have been removed from the solution. "
        "Insert the removed pickup and delivery nodes back into the routes.\n\n"
        "Requirements:\n"
        "- Each pickup must appear before its paired delivery on the same route.\n"
        "- All capacity and time window constraints must be satisfied.\n"
        "- Minimize the total travel distance.\n"
        "- You may insert nodes into any existing route.\n"
        "- Return ALL routes (including unchanged ones) with depot (0) at start and end.\n\n"
        "Respond with the complete set of routes and your reasoning."
    )

    return _format_prompt(
        problem, distance_mode, instructions,
        extra_data={
            "Current Partial Solution": sol_data,
            "Removed Requests (to insert)": requests_info,
        },
    )


def build_distance_prediction_prompt(
    problem: PDPTWProblem,
    solution: PDPTWSolution,
    distance_mode: DistanceMode,
) -> str:
    """Build prompt for Task 3: Distance Prediction."""
    sol_data = build_solution_json(solution, include_distance=False)

    instructions = (
        "Calculate the total travel distance for the given solution.\n\n"
        "The total distance is the sum of distances traveled across all routes. "
        "For each route, sum the distances between consecutive nodes "
        "(including depot at start and end).\n\n"
        "Respond with the predicted total distance and your reasoning."
    )

    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Solution": sol_data})


def build_route_completion_prompt(
    problem: PDPTWProblem,
    partial_solution: PDPTWSolution,
    incomplete_route_idx: int,
    remaining_nodes: list[int],
    distance_mode: DistanceMode,
) -> str:
    """Build prompt for Task 4: Route Completion."""
    sol_data = build_solution_json(partial_solution, include_distance=False)

    instructions = (
        f"Route {incomplete_route_idx} is incomplete — it has been truncated and does not "
        "return to the depot yet.\n\n"
        f"The following nodes still need to be visited on this route: {remaining_nodes}\n\n"
        "Complete the route by determining the correct order for the remaining nodes, "
        "then return to depot (0). The completed route must satisfy all PDPTW constraints.\n\n"
        "Respond with the full completed route (from depot to depot, including the "
        "already-visited portion) and your reasoning."
    )

    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Partial Solution": sol_data})


def build_full_solution_prompt(
    problem: PDPTWProblem,
    distance_mode: DistanceMode,
) -> str:
    """Build prompt for Task 5: Full Solution Generation."""
    instructions = (
        f"Generate a complete feasible solution using exactly {problem.num_vehicles} vehicle routes.\n\n"
        "Requirements:\n"
        "- Every pickup and delivery node must be served exactly once.\n"
        "- Each route starts and ends at depot (0).\n"
        "- Pickup before delivery on the same vehicle.\n"
        "- Capacity and time window constraints must be satisfied.\n"
        "- Minimize total travel distance.\n\n"
        "Respond with all routes (each as a list of node indices starting and ending with 0) "
        "and your reasoning."
    )

    return _format_prompt(problem, distance_mode, instructions)
