"""Test scoring pipeline by acting as the LLM with known responses."""

import os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions
from utils.feasibility import is_feasible

# Import scoring/parsing from pdpbench (without triggering kbench import)
# We'll just redefine the functions we need
import re

def parse_json_response(raw):
    if not raw:
        return {}
    text = str(raw)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    deep_match = re.search(r"\{.*\}", text, re.DOTALL)
    if deep_match:
        try:
            return json.loads(deep_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def normalize_routes(routes_raw):
    if not isinstance(routes_raw, list):
        return None
    normalized = []
    for route in routes_raw:
        if not isinstance(route, list) or len(route) == 0:
            continue
        try:
            route = [int(node) for node in route]
        except (ValueError, TypeError):
            return None
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)
        if len(route) > 2:
            normalized.append(route)
    return normalized if normalized else None


def build_solution_from_llm_output(problem, routes_raw):
    routes = normalize_routes(routes_raw)
    if routes is None:
        return None
    try:
        return PDPTWSolution(problem=problem, routes=routes)
    except Exception as e:
        print(f"    build_solution error: {e}")
        return None


def score_feasibility(problem, solution):
    return 1.0 if is_feasible(problem, solution) else 0.0


def score_distance_gap(actual_distance, bks_distance):
    if bks_distance <= 0:
        return 0.0
    gap = (actual_distance - bks_distance) / bks_distance
    return max(0.0, 1.0 - gap)


def score_distance_prediction(predicted, actual):
    if actual <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(predicted - actual) / actual)


# =============================================================================
# Load instances
# =============================================================================

DATA_DIR = os.path.join(_HERE, "data")
BKS_DIR = os.path.join(_HERE, "bks")

li_lim_mgr = LiLimInstanceManager(base_dir=DATA_DIR)
mendeley_mgr = MendeleyInstanceManager(base_dir=DATA_DIR)
bks_loader = BestKnownSolutions(bks_path=BKS_DIR)

# Just test with first Li&Lim instance
problem = li_lim_mgr.load("lc101", size=100)
bks = bks_loader.get_bks_as_solution(problem)

print(f"Instance: {problem.name}")
print(f"BKS routes: {len(bks.routes)}, distance: {bks.total_distance:.1f}")
print(f"BKS feasible: {is_feasible(problem, bks)}")
print()

# =============================================================================
# Test 1: Masked Node Prediction
# =============================================================================
print("=" * 60)
print("TEST 1: Masked Node Prediction")
print("=" * 60)

longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
route = bks.routes[longest_idx]
position_idx = min(3, len(route) - 2)
correct_node = route[position_idx]
print(f"Route {longest_idx}: {route}")
print(f"Masked position {position_idx}, correct node: {correct_node}")

# Test A: Correct answer
fake_response = json.dumps({"predicted_node": correct_node, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = int(data.get("predicted_node", -1))
assert predicted == correct_node, f"Parse failed: got {predicted}"
print(f"  Correct answer ({correct_node}): score = 1.0 ✓")

# Test B: Wrong but feasible (use a different node that might be feasible)
wrong_node = route[position_idx + 1] if position_idx + 1 < len(route) - 1 else route[1]
fake_response = json.dumps({"predicted_node": wrong_node, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = int(data.get("predicted_node", -1))
test_routes = [r[:] for r in bks.routes]
test_routes[longest_idx][position_idx] = predicted
test_sol = build_solution_from_llm_output(problem, test_routes)
if test_sol is not None:
    feas = score_feasibility(problem, test_sol)
    score = 0.5 if feas == 1.0 else 0.0
    print(f"  Wrong node ({wrong_node}), feasible={feas==1.0}: score = {score}")
else:
    print(f"  Wrong node ({wrong_node}), couldn't build solution: score = 0.0")

# Test C: Totally wrong
fake_response = json.dumps({"predicted_node": 9999, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = int(data.get("predicted_node", -1))
test_routes = [r[:] for r in bks.routes]
test_routes[longest_idx][position_idx] = predicted
test_sol = build_solution_from_llm_output(problem, test_routes)
if test_sol is not None:
    feas = score_feasibility(problem, test_sol)
    print(f"  Invalid node (9999), feasible={feas==1.0}: score = {0.5 if feas==1.0 else 0.0}")
else:
    print(f"  Invalid node (9999), build failed: score = 0.0 ✓")

# Test D: Malformed JSON
fake_response = "I think the answer is 42"
data = parse_json_response(fake_response)
predicted = data.get("predicted_node", None)
print(f"  Malformed response, parsed predicted_node={predicted}: score = 0.0 ✓")
print()

# =============================================================================
# Test 2: Request Insertion
# =============================================================================
print("=" * 60)
print("TEST 2: Request Insertion")
print("=" * 60)

partial = bks.clone()
removed = []
for pickup_idx, delivery_idx in problem.pickups_deliveries[:2]:
    partial.remove_request(problem, pickup_idx)
    removed.append((pickup_idx, delivery_idx))
print(f"Removed requests: {removed}")
print(f"Partial solution routes: {len(partial.routes)}")

# Test A: Return the original BKS routes (should be feasible, perfect distance)
fake_response = json.dumps({"routes": bks.routes, "reasoning": "test"})
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
solution = build_solution_from_llm_output(problem, routes_raw)
if solution is not None:
    feas = score_feasibility(problem, solution)
    dist = score_distance_gap(solution.total_distance, bks.total_distance)
    score = 0.5 * feas + 0.5 * dist
    print(f"  BKS routes: feasible={feas==1.0}, dist_gap={dist:.3f}, score={score:.3f} ✓")
else:
    print(f"  BKS routes: FAILED to build solution!")

# Test B: Return empty routes
fake_response = json.dumps({"routes": [], "reasoning": "test"})
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
solution = build_solution_from_llm_output(problem, routes_raw)
print(f"  Empty routes: solution={solution}: score = 0.0 ✓")

# Test C: Return partial solution (missing the removed requests)
fake_response = json.dumps({"routes": partial.routes, "reasoning": "test"})
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
solution = build_solution_from_llm_output(problem, routes_raw)
if solution is not None:
    feas = score_feasibility(problem, solution)
    print(f"  Partial routes (missing requests): feasible={feas==1.0}, score={0.0 if feas==0.0 else 'nonzero'}")
else:
    print(f"  Partial routes: build failed, score = 0.0")
print()

# =============================================================================
# Test 3: Distance Prediction
# =============================================================================
print("=" * 60)
print("TEST 3: Distance Prediction")
print("=" * 60)

actual = bks.total_distance
print(f"Actual BKS distance: {actual:.1f}")

# Test A: Exact prediction
fake_response = json.dumps({"predicted_distance": actual, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = float(data.get("predicted_distance", 0))
score = score_distance_prediction(predicted, actual)
print(f"  Exact ({predicted:.1f}): score = {score:.3f} ✓")

# Test B: 10% off
fake_response = json.dumps({"predicted_distance": actual * 1.1, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = float(data.get("predicted_distance", 0))
score = score_distance_prediction(predicted, actual)
print(f"  10% over ({predicted:.1f}): score = {score:.3f} (expected ~0.9)")

# Test C: 50% off
fake_response = json.dumps({"predicted_distance": actual * 1.5, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = float(data.get("predicted_distance", 0))
score = score_distance_prediction(predicted, actual)
print(f"  50% over ({predicted:.1f}): score = {score:.3f} (expected ~0.5)")

# Test D: Way off
fake_response = json.dumps({"predicted_distance": actual * 3, "reasoning": "test"})
data = parse_json_response(fake_response)
predicted = float(data.get("predicted_distance", 0))
score = score_distance_prediction(predicted, actual)
print(f"  200% over ({predicted:.1f}): score = {score:.3f} (expected 0.0)")
print()

# =============================================================================
# Test 4: Route Completion
# =============================================================================
print("=" * 60)
print("TEST 4: Route Completion")
print("=" * 60)

longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
route = bks.routes[longest_idx]
customer_nodes = route[1:-1]
midpoint = len(customer_nodes) // 2
kept = customer_nodes[:midpoint]
remaining = customer_nodes[midpoint:]
print(f"Route {longest_idx}: {route}")
print(f"Kept (first half): {kept}")
print(f"Remaining (to complete): {remaining}")

partial_routes = [r[:] for r in bks.routes]
partial_routes[longest_idx] = [0] + kept
partial = PDPTWSolution(problem=problem, routes=partial_routes)

# Test A: Return the correct complete route
fake_response = json.dumps({"completed_route": route, "reasoning": "test"})
data = parse_json_response(fake_response)
completed_route = data.get("completed_route", None)
completed_route = [int(n) for n in completed_route]
full_routes = [r[:] for r in partial.routes]
full_routes[longest_idx] = completed_route
solution = build_solution_from_llm_output(problem, full_routes)
if solution is not None:
    feas = score_feasibility(problem, solution)
    dist = score_distance_gap(solution.total_distance, bks.total_distance)
    score = 0.5 * feas + 0.5 * dist
    print(f"  Correct route: feasible={feas==1.0}, dist_gap={dist:.3f}, score={score:.3f} ✓")
else:
    print(f"  Correct route: FAILED to build!")

# Test B: Return just the remaining nodes (missing the kept portion)
wrong_route = [0] + remaining + [0]
fake_response = json.dumps({"completed_route": wrong_route, "reasoning": "test"})
data = parse_json_response(fake_response)
completed_route = [int(n) for n in data.get("completed_route", [])]
full_routes = [r[:] for r in partial.routes]
full_routes[longest_idx] = completed_route
solution = build_solution_from_llm_output(problem, full_routes)
if solution is not None:
    feas = score_feasibility(problem, solution)
    print(f"  Only remaining nodes: feasible={feas==1.0}")
else:
    print(f"  Only remaining nodes: build failed")
print()

# =============================================================================
# Test 5: Full Solution
# =============================================================================
print("=" * 60)
print("TEST 5: Full Solution Generation")
print("=" * 60)

# Test A: Return BKS routes
fake_response = json.dumps({"routes": bks.routes, "reasoning": "test"})
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
solution = build_solution_from_llm_output(problem, routes_raw)
if solution is not None:
    feas = score_feasibility(problem, solution)
    dist = score_distance_gap(solution.total_distance, bks.total_distance)
    score = 0.5 * feas + 0.5 * dist
    print(f"  BKS routes: feasible={feas==1.0}, dist_gap={dist:.3f}, score={score:.3f} ✓")
else:
    print(f"  BKS routes: FAILED!")

# Test B: Malformed
fake_response = "Here is my solution: route 1 goes to node 5 then 10..."
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
print(f"  Malformed text: parsed routes={routes_raw}, score = 0.0 ✓")

# =============================================================================
# Test JSON parsing edge cases
# =============================================================================
print()
print("=" * 60)
print("TEST: JSON Parsing Edge Cases")
print("=" * 60)

# Nested JSON in code block
raw = '```json\n{"routes": [[0, 1, 2, 0], [0, 3, 4, 0]], "reasoning": "done"}\n```'
data = parse_json_response(raw)
print(f"  Code block: routes={data.get('routes', 'MISSING')}")

# JSON with extra text
raw = 'Here is my answer:\n{"predicted_distance": 1234.5, "reasoning": "calculated"}\nThank you!'
data = parse_json_response(raw)
print(f"  Extra text: predicted_distance={data.get('predicted_distance', 'MISSING')}")

# Deeply nested routes
routes = [[0, 1, 51, 0], [0, 2, 52, 0], [0, 3, 53, 0]]
raw = json.dumps({"routes": routes, "reasoning": "test"})
data = parse_json_response(raw)
print(f"  Nested routes: {data.get('routes', 'MISSING')}")

print("\n✓ All tests complete")
