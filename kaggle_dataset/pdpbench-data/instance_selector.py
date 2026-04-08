"""Deterministic instance selection and task-specific data preparation for PDPBench.

Selects 8 benchmark instances (5 Li & Lim + 3 Mendeley) covering different
categories and cities, then prepares task-specific parameters for each.
"""

from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution


# Fixed instance selection for reproducibility
LI_LIM_INSTANCES = ["lc101", "lc201", "lr101", "lr201", "lrc101"]
MENDELEY_INSTANCES = ["bar-n100-1", "ber-n100-1", "nyc-n100-1"]


def get_benchmark_instances(
    data_dir: str = "data",
    bks_dir: str = "bks",
) -> list[tuple[PDPTWProblem, PDPTWSolution]]:
    """Load all 8 benchmark instances with their best known solutions.

    Returns:
        List of (problem, bks_solution) tuples.
    """
    li_lim_mgr = LiLimInstanceManager(base_dir=data_dir)
    mendeley_mgr = MendeleyInstanceManager(base_dir=data_dir)
    bks = BestKnownSolutions(bks_path=bks_dir)

    instances = []

    for name in LI_LIM_INSTANCES:
        problem = li_lim_mgr.load(name, size=100)
        solution = bks.get_bks_as_solution(problem)
        instances.append((problem, solution))

    for name in MENDELEY_INSTANCES:
        problem = mendeley_mgr.load(name, size=100)
        solution = bks.get_bks_as_solution(problem)
        instances.append((problem, solution))

    return instances


# ---------------------------------------------------------------------------
# Task-specific data preparation
# ---------------------------------------------------------------------------

def prepare_masked_node_data(
    instances: list[tuple[PDPTWProblem, PDPTWSolution]],
) -> list[dict]:
    """Prepare data for Task 1: Masked Node Prediction.

    Picks the 3rd customer node in the longest route (deterministic).
    """
    rows = []
    for problem, bks in instances:
        # Find longest route by customer count
        longest_idx = max(
            range(len(bks.routes)),
            key=lambda i: len(bks.routes[i]),
        )
        route = bks.routes[longest_idx]

        # Pick 3rd customer node (position 3, since 0 is depot)
        position_idx = min(3, len(route) - 2)  # clamp to valid range
        correct_node = route[position_idx]

        rows.append({
            "problem": problem,
            "solution": bks,
            "route_idx": longest_idx,
            "position_idx": position_idx,
            "correct_node": correct_node,
            "instance_name": problem.name,
        })

    return rows


def prepare_request_insertion_data(
    instances: list[tuple[PDPTWProblem, PDPTWSolution]],
    num_requests_to_remove: int = 2,
) -> list[dict]:
    """Prepare data for Task 2: Request Insertion.

    Removes the first k requests (by index) from the BKS.
    """
    rows = []
    for problem, bks in instances:
        partial = bks.clone()
        removed = []

        # Remove first k requests
        pairs = problem.pickups_deliveries[:num_requests_to_remove]
        for pickup_idx, delivery_idx in pairs:
            partial.remove_request(problem, pickup_idx)
            removed.append((pickup_idx, delivery_idx))

        rows.append({
            "problem": problem,
            "partial_solution": partial,
            "removed_requests": removed,
            "bks_distance": bks.total_distance,
            "instance_name": problem.name,
        })

    return rows


def prepare_distance_prediction_data(
    instances: list[tuple[PDPTWProblem, PDPTWSolution]],
) -> list[dict]:
    """Prepare data for Task 3: Distance Prediction."""
    rows = []
    for problem, bks in instances:
        rows.append({
            "problem": problem,
            "solution": bks,
            "actual_distance": bks.total_distance,
            "instance_name": problem.name,
        })

    return rows


def prepare_route_completion_data(
    instances: list[tuple[PDPTWProblem, PDPTWSolution]],
) -> list[dict]:
    """Prepare data for Task 4: Route Completion.

    Cuts the longest route at its midpoint. Remaining nodes are given
    as an unordered list so the LLM must determine sequencing.
    """
    rows = []
    for problem, bks in instances:
        # Find longest route
        longest_idx = max(
            range(len(bks.routes)),
            key=lambda i: len(bks.routes[i]),
        )
        route = bks.routes[longest_idx]

        # Cut at midpoint (excluding depot bookends)
        customer_nodes = route[1:-1]
        midpoint = len(customer_nodes) // 2
        kept_nodes = customer_nodes[:midpoint]
        remaining_nodes = customer_nodes[midpoint:]

        # Build partial solution with truncated route (no closing depot)
        partial_routes = [r[:] for r in bks.routes]
        partial_routes[longest_idx] = [0] + kept_nodes  # open-ended

        partial = PDPTWSolution(problem=problem, routes=partial_routes)

        rows.append({
            "problem": problem,
            "partial_solution": partial,
            "incomplete_route_idx": longest_idx,
            "remaining_nodes": remaining_nodes,
            "bks_distance": bks.total_distance,
            "instance_name": problem.name,
        })

    return rows


def prepare_full_solution_data(
    instances: list[tuple[PDPTWProblem, PDPTWSolution]],
) -> list[dict]:
    """Prepare data for Task 5: Full Solution Generation."""
    rows = []
    for problem, bks in instances:
        rows.append({
            "problem": problem,
            "bks_distance": bks.total_distance,
            "instance_name": problem.name,
        })

    return rows
