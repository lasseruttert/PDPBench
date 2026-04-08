"""Scoring functions for PDPBench tasks.

Handles route normalization from LLM output, solution reconstruction,
and scoring metrics (feasibility, distance gap, exact match).
"""

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.feasibility import is_feasible


def normalize_routes(routes_raw: list) -> list[list[int]] | None:
    """Normalize LLM-returned routes into the expected format.

    Handles: missing depot bookends, string node IDs, empty routes,
    non-list inputs. Returns None if completely malformed.
    """
    if not isinstance(routes_raw, list):
        return None

    normalized = []
    for route in routes_raw:
        if not isinstance(route, list) or len(route) == 0:
            continue

        # Cast all elements to int
        try:
            route = [int(node) for node in route]
        except (ValueError, TypeError):
            return None

        # Add depot bookends if missing
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)

        # Skip depot-only routes
        if len(route) > 2:
            normalized.append(route)

    return normalized if normalized else None


def build_solution_from_llm_output(
    problem: PDPTWProblem, routes_raw: list
) -> PDPTWSolution | None:
    """Construct a PDPTWSolution from raw LLM output routes.

    Returns None if routes are malformed or cannot be normalized.
    """
    routes = normalize_routes(routes_raw)
    if routes is None:
        return None

    try:
        return PDPTWSolution(problem=problem, routes=routes)
    except Exception:
        return None


def score_feasibility(problem: PDPTWProblem, solution: PDPTWSolution) -> float:
    """Returns 1.0 if feasible, 0.0 otherwise."""
    return 1.0 if is_feasible(problem, solution) else 0.0


def score_distance_gap(actual_distance: float, bks_distance: float) -> float:
    """Score based on gap to best known solution distance.

    Returns max(0, 1 - (actual - bks) / bks).
    A perfect match scores 1.0, 100%+ worse scores 0.0.
    Only meaningful when the solution is feasible.
    """
    if bks_distance <= 0:
        return 0.0
    gap = (actual_distance - bks_distance) / bks_distance
    return max(0.0, 1.0 - gap)


def score_distance_prediction(predicted: float, actual: float) -> float:
    """Score based on percentage error of distance prediction.

    Returns max(0, 1 - |predicted - actual| / actual).
    """
    if actual <= 0:
        return 0.0
    pct_error = abs(predicted - actual) / actual
    return max(0.0, 1.0 - pct_error)


def score_exact_match(predicted: int, correct: int) -> float:
    """Returns 1.0 if predicted equals correct, 0.0 otherwise."""
    return 1.0 if predicted == correct else 0.0
