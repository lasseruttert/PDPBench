"""
PDPBench: LLM Benchmark for Pickup and Delivery Problems with Time Windows
===========================================================================
Tests LLM understanding of PDPTW across five tasks of increasing difficulty:
1. Masked Node Prediction
2. Request Insertion
3. Distance Prediction
4. Route Completion
5. Full Solution Generation

Tracks: Executive Functions & Attention (Combinatorial Optimization)
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "kaggle-benchmarks", "pyparsing", "-q"], check=True)

import os
import json
import re
import time
import numpy as np
from enum import Enum

try:
    import kaggle_benchmarks as kbench
    HAS_KBENCH = hasattr(kbench, "llm")
except (RuntimeError, ImportError):
    kbench = None
    HAS_KBENCH = False

print(f"kbench available: {HAS_KBENCH}")

# =============================================================================
# Path setup & logging
# =============================================================================

KAGGLE = os.path.exists("/kaggle")
if KAGGLE:
    _CANDIDATES = [
        "/kaggle/input/pdpbench-data",
        "/kaggle/input/datasets/lasseruttert/pdpbench-data",
    ]
    BASE_DIR = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
BKS_DIR = os.path.join(BASE_DIR, "bks")

# Tee stdout/stderr to a log file
LOG_PATH = "/kaggle/working/pdpbench_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_log.txt")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

if KAGGLE:
    print(f"Kaggle dataset path: {BASE_DIR}")


# =============================================================================
# Imports from utils (in dataset)
# =============================================================================

from utils.pdptw_problem import PDPTWProblem, Node, Request
from utils.pdptw_solution import PDPTWSolution
from utils.pdptw_reader import pdptw_reader
from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions
from utils.feasibility import is_feasible


# =============================================================================
# Scoring functions
# =============================================================================

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
    except Exception:
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
# LLM call with retry
# =============================================================================

def llm_prompt(llm, prompt, retries=3, delay=10):
    """Call llm.prompt with retry on transient errors (503, etc.)."""
    for attempt in range(retries):
        try:
            return str(llm.prompt(prompt))
        except Exception as e:
            if attempt < retries - 1 and ("503" in str(e) or "could not reach" in str(e).lower()):
                print(f"    Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(delay)
            else:
                raise
    return ""


# =============================================================================
# JSON response parsing (with regex fallback)
# =============================================================================

def parse_json_response(raw):
    """Parse JSON from LLM response. Tries code block extraction, then raw parse, then regex."""
    if not raw:
        return {}
    text = str(raw)
    # Try extracting from code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding a JSON object in the text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    # Try finding a large JSON object (with nested braces for routes)
    deep_match = re.search(r"\{.*\}", text, re.DOTALL)
    if deep_match:
        try:
            return json.loads(deep_match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: extract values from plain text responses (e.g. LaTeX \boxed{}, "the answer is X")
    # Try to extract a single integer (for predicted_node)
    boxed = re.search(r"\\boxed\{\\text\{(\d+)\}\}", text) or re.search(r"\\boxed\{(\d+)\}", text)
    if boxed:
        return {"_fallback_number": int(boxed.group(1))}
    # "the answer is 42", "answer: 42", "predicted node is 42"
    answer_match = re.search(r"(?:answer|predicted[_ ]node|node)\s*(?:is|:|=)\s*(\d+)", text, re.IGNORECASE)
    if answer_match:
        return {"_fallback_number": int(answer_match.group(1))}
    # Try to extract routes from plain text like [[0,1,2,0],[0,3,4,0]]
    routes_match = re.search(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", text)
    if routes_match:
        try:
            routes = json.loads(routes_match.group(0))
            if isinstance(routes, list) and all(isinstance(r, list) for r in routes):
                return {"routes": routes}
        except json.JSONDecodeError:
            pass
    # Try to extract a float (for predicted_distance)
    dist_match = re.search(r"(?:distance|total)\s*(?:is|:|=)\s*([\d.]+)", text, re.IGNORECASE)
    if dist_match:
        return {"_fallback_number": float(dist_match.group(1))}
    return {}


# =============================================================================
# Prompt builder
# =============================================================================

class DistanceMode(Enum):
    MATRIX = "matrix"
    COORDINATES = "coordinates"
    TOOL_USE = "tool_use"


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


def build_problem_json(problem, distance_mode):
    if distance_mode == DistanceMode.COORDINATES and problem.dataset == "Mendeley":
        raise ValueError("COORDINATES mode cannot be used with Mendeley instances.")
    nodes = []
    for node in problem.nodes:
        d = {"index": node.index, "demand": node.demand,
             "time_window": list(node.time_window), "service_time": node.service_time}
        if node.index == 0:
            d["type"] = "depot"
        elif problem.is_pickup(node.index):
            d["type"] = "pickup"
            d["paired_delivery"] = node.delivery_index
        elif problem.is_delivery(node.index):
            d["type"] = "delivery"
            d["paired_pickup"] = node.pickup_index
        if distance_mode in (DistanceMode.COORDINATES, DistanceMode.TOOL_USE):
            d["x"] = round(float(node.x), 2)
            d["y"] = round(float(node.y), 2)
        nodes.append(d)
    result = {"num_vehicles": problem.num_vehicles, "vehicle_capacity": problem.vehicle_capacity, "nodes": nodes}
    if distance_mode == DistanceMode.MATRIX:
        m = problem.distance_matrix
        result["distance_matrix"] = [[int(round(float(m[i, j]))) for j in range(m.shape[1])] for i in range(m.shape[0])]
    if distance_mode == DistanceMode.COORDINATES:
        result["distance_note"] = "Distances are Euclidean: dist(i,j) = sqrt((xi-xj)^2 + (yi-yj)^2). Travel time equals distance."
    return result


def build_solution_json(solution, include_distance=True):
    result = {"routes": solution.routes}
    if include_distance:
        result["total_distance"] = round(float(solution.total_distance), 2)
    return result


def _format_prompt(problem, distance_mode, task_instructions, extra_data=None):
    rules = PDPTW_RULES.format(num_vehicles=problem.num_vehicles, capacity=problem.vehicle_capacity)
    pjson = build_problem_json(problem, distance_mode)
    sections = [rules, "## Problem Data", f"```json\n{json.dumps(pjson, separators=(',', ':'))}\n```"]
    if extra_data:
        for title, data in extra_data.items():
            sections.append(f"## {title}")
            sections.append(f"```json\n{json.dumps(data, separators=(',', ':'))}\n```")
    sections.append("## Task")
    sections.append(task_instructions)
    return "\n\n".join(sections)


def build_masked_node_prompt(problem, solution, route_idx, position_idx, distance_mode):
    masked_routes = [route[:] for route in solution.routes]
    masked_routes[route_idx][position_idx] = "?"
    sol_data = {"routes": masked_routes}
    instructions = (
        f"One node in Route {route_idx} at position {position_idx} has been masked with '?'.\n\n"
        "Identify the correct node index that should replace '?'. "
        "The original solution is feasible — your predicted node must maintain feasibility.\n\n"
        'Respond with EXACTLY this JSON:\n{"predicted_node": <integer>, "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions, extra_data={"Solution (with masked node)": sol_data})


def build_request_insertion_prompt(problem, partial_solution, removed_requests, distance_mode):
    sol_data = build_solution_json(partial_solution, include_distance=False)
    requests_info = []
    for pickup_idx, delivery_idx in removed_requests:
        pn = problem.nodes_dict[pickup_idx]
        dn = problem.nodes_dict[delivery_idx]
        requests_info.append({
            "pickup": {"index": pickup_idx, "demand": pn.demand, "time_window": list(pn.time_window), "service_time": pn.service_time},
            "delivery": {"index": delivery_idx, "demand": dn.demand, "time_window": list(dn.time_window), "service_time": dn.service_time},
        })
    instructions = (
        f"{len(removed_requests)} request(s) have been removed from the solution. "
        "Insert the removed pickup and delivery nodes back into the routes.\n\n"
        "Requirements:\n- Each pickup must appear before its paired delivery on the same route.\n"
        "- All capacity and time window constraints must be satisfied.\n- Minimize the total travel distance.\n"
        "- You may insert nodes into any existing route.\n"
        "- Return ALL routes (including unchanged ones) with depot (0) at start and end.\n\n"
        'Respond with EXACTLY this JSON:\n{"routes": [[0, ...node ids..., 0], ...], "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Current Partial Solution": sol_data, "Removed Requests (to insert)": requests_info})


def build_distance_prediction_prompt(problem, solution, distance_mode):
    sol_data = build_solution_json(solution, include_distance=False)
    instructions = (
        "Calculate the total travel distance for the given solution.\n\n"
        "The total distance is the sum of distances traveled across all routes. "
        "For each route, sum the distances between consecutive nodes (including depot at start and end).\n\n"
        'Respond with EXACTLY this JSON:\n{"predicted_distance": <number>, "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions, extra_data={"Solution": sol_data})


def build_route_completion_prompt(problem, partial_solution, incomplete_route_idx, remaining_nodes, distance_mode):
    sol_data = build_solution_json(partial_solution, include_distance=False)
    instructions = (
        f"Route {incomplete_route_idx} is incomplete — it has been truncated and does not return to the depot yet.\n\n"
        f"The following nodes still need to be visited on this route: {remaining_nodes}\n\n"
        "Complete the route by determining the correct order for the remaining nodes, "
        "then return to depot (0). The completed route must satisfy all PDPTW constraints.\n\n"
        'Respond with EXACTLY this JSON:\n{"completed_route": [0, ...node ids..., 0], "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions, extra_data={"Partial Solution": sol_data})


def build_full_solution_prompt(problem, distance_mode):
    instructions = (
        f"Generate a complete feasible solution using exactly {problem.num_vehicles} vehicle routes.\n\n"
        "Requirements:\n- Every pickup and delivery node must be served exactly once.\n"
        "- Each route starts and ends at depot (0).\n- Pickup before delivery on the same vehicle.\n"
        "- Capacity and time window constraints must be satisfied.\n- Minimize total travel distance.\n\n"
        'Respond with EXACTLY this JSON:\n{"routes": [[0, ...node ids..., 0], ...], "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions)


# =============================================================================
# Instance selection & data preparation
# =============================================================================

LI_LIM_INSTANCES = ["lc101", "lc201", "lr101", "lr201", "lrc101"]
MENDELEY_INSTANCES = ["bar-n100-1", "ber-n100-1", "nyc-n100-1"]
DISTANCE_MODE = DistanceMode.MATRIX


def get_benchmark_instances():
    li_lim_mgr = LiLimInstanceManager(base_dir=DATA_DIR)
    mendeley_mgr = MendeleyInstanceManager(base_dir=DATA_DIR)
    bks = BestKnownSolutions(bks_path=BKS_DIR)
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


INSTANCES = get_benchmark_instances()
print(f"Loaded {len(INSTANCES)} instances")


# =============================================================================
# Identify model + print metric legend
# =============================================================================

RESULTS_PATH = "/kaggle/working/results.json" if KAGGLE else os.path.join(BASE_DIR, "results.json")
RESULTS = {"model": "unknown", "distance_mode": DISTANCE_MODE.value, "tasks": {}}

def save_results():
    with open(RESULTS_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

if HAS_KBENCH:
    _model_name = getattr(kbench.llm, "model", "unknown")
    if _model_name == "unknown":
        try:
            _model_name = str(kbench.llm.prompt("What model are you? Reply with ONLY your model name, nothing else."))
        except Exception:
            _model_name = "unknown"
    RESULTS["model"] = _model_name
    print(f"\n{'='*60}")
    print(f"  MODEL: {_model_name}")
    print(f"  INSTANCES: {len(INSTANCES)} ({', '.join(p.name for p, _ in INSTANCES)})")
    print(f"  DISTANCE MODE: {DISTANCE_MODE.value}")
    print(f"{'='*60}")
    print(f"""
  METRIC LEGEND:
  Task 1 (Masked Node):      predicted vs correct node
    EXACT=1.0 | FEASIBLE=0.5 (wrong but valid swap) | WRONG=0.0
    json_ok/json_fail = whether LLM returned parseable JSON

  Task 2 (Request Insertion): re-insert removed pickup/delivery pairs
    FEASIBLE/INFEASIBLE + distance gap vs BKS
    score = 0.5*feasibility + 0.5*distance_gap

  Task 3 (Distance Prediction): predict total route distance
    err% = |predicted - actual| / actual
    score = max(0, 1 - err%)

  Task 4 (Route Completion):  complete a truncated route
    FEASIBLE/INFEASIBLE + distance gap vs BKS
    score = 0.5*feasibility + 0.5*distance_gap

  Task 5 (Full Solution):    generate complete solution from scratch
    FEASIBLE/INFEASIBLE + distance gap vs BKS
    served = nodes visited vs expected
    score = 0.5*feasibility + 0.5*distance_gap
""")


# =============================================================================
# Task definitions (define all, only run the chosen one)
# =============================================================================

if HAS_KBENCH:

    @kbench.task(name="pdptw_masked_node")
    def pdptw_masked_node(llm) -> float:
        """Predict a masked node in a PDPTW solution route."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
            route = bks.routes[longest_idx]
            position_idx = min(3, len(route) - 2)
            correct_node = route[position_idx]
            prompt = build_masked_node_prompt(problem, bks, longest_idx, position_idx, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)

            try:
                predicted = int(data.get("predicted_node", data.get("_fallback_number", -1)))
            except (ValueError, TypeError):
                predicted = -1

            if predicted == correct_node:
                score = 1.0
            else:
                test_routes = [r[:] for r in bks.routes]
                test_routes[longest_idx][position_idx] = predicted
                test_sol = build_solution_from_llm_output(problem, test_routes)
                if test_sol is not None and score_feasibility(problem, test_sol) == 1.0:
                    score = 0.5
                else:
                    score = 0.0

            scores.append(score)
            ok = "EXACT" if score == 1.0 else ("FEASIBLE" if score == 0.5 else "WRONG")
            if "_fallback_number" in data:
                parse_ok = "boxed_fallback"
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | JSON parse failed, extracted from boxed/text: {data['_fallback_number']}")
            elif not data:
                parse_ok = "json_fail"
                raw_str = str(raw)
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | JSON PARSE FAILED | first 500 chars:\n    {raw_str[:500]}\n    ...last 500 chars:\n    {raw_str[-500:]}")
            else:
                parse_ok = "json_ok"
            print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | predicted={predicted} correct={correct_node} | {ok} | {parse_ok} | score={score}")
            details.append({"instance": problem.name, "predicted": predicted, "correct": correct_node, "result": ok, "parse": parse_ok, "score": score})

        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Task 1 MASKED NODE: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
        print(f"  Breakdown: {', '.join(f'{s:.1f}' for s in scores)}")
        print(f"{'='*60}")
        RESULTS["tasks"]["masked_node"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results()
        return avg


    @kbench.task(name="pdptw_request_insertion")
    def pdptw_request_insertion(llm) -> float:
        """Insert removed requests back into a PDPTW solution."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            partial = bks.clone()
            removed = []
            for pickup_idx, delivery_idx in problem.pickups_deliveries[:2]:
                partial.remove_request(problem, pickup_idx)
                removed.append((pickup_idx, delivery_idx))
            prompt = build_request_insertion_prompt(problem, partial, removed, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)

            routes_raw = data.get("routes", None)
            solution = build_solution_from_llm_output(problem, routes_raw) if routes_raw else None
            d = {"instance": problem.name, "removed": removed, "bks_dist": round(bks.total_distance, 1)}

            if solution is None:
                score = 0.0
                d.update(result="PARSE_FAIL", score=0.0)
                raw_str = str(raw)
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | PARSE_FAIL | first 500 chars:\n    {raw_str[:500]}\n    ...last 500 chars:\n    {raw_str[-500:]}")
            else:
                feasible = score_feasibility(problem, solution)
                d["llm_dist"] = round(solution.total_distance, 1)
                d["feasible"] = feasible == 1.0
                if feasible == 0.0:
                    score = 0.0
                    d.update(result="INFEASIBLE", score=0.0)
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} | score=0.0")
                else:
                    dist_score = score_distance_gap(solution.total_distance, bks.total_distance)
                    score = 0.5 * feasible + 0.5 * dist_score
                    d.update(result="FEASIBLE", dist_gap=round(dist_score, 3), score=round(score, 3))
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | FEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={dist_score:.3f} | score={score:.3f}")

            scores.append(score)
            details.append(d)

        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Task 2 REQUEST INSERTION: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
        print(f"  Breakdown: {', '.join(f'{s:.2f}' for s in scores)}")
        print(f"{'='*60}")
        RESULTS["tasks"]["request_insertion"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results()
        return avg


    @kbench.task(name="pdptw_distance_prediction")
    def pdptw_distance_prediction(llm) -> float:
        """Predict the total distance of a PDPTW solution."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            prompt = build_distance_prediction_prompt(problem, bks, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)

            try:
                predicted = float(data.get("predicted_distance", data.get("_fallback_number", 0)))
            except (ValueError, TypeError):
                predicted = 0.0

            if "_fallback_number" in data:
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | JSON parse failed, extracted from boxed/text: {data['_fallback_number']}")
            elif not data:
                raw_str = str(raw)
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | JSON PARSE FAILED | first 500 chars:\n    {raw_str[:500]}\n    ...last 500 chars:\n    {raw_str[-500:]}")

            actual = bks.total_distance
            score = score_distance_prediction(predicted, actual)
            scores.append(score)
            pct_err = abs(predicted - actual) / actual * 100 if actual > 0 else 0
            print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | predicted={predicted:.1f} actual={actual:.1f} err={pct_err:.1f}% | score={score:.3f}")
            details.append({"instance": problem.name, "predicted": round(predicted, 1), "actual": round(actual, 1), "error_pct": round(pct_err, 1), "score": round(score, 3)})

        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Task 3 DISTANCE PREDICTION: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
        print(f"  Breakdown: {', '.join(f'{s:.3f}' for s in scores)}")
        print(f"{'='*60}")
        RESULTS["tasks"]["distance_prediction"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results()
        return avg


    @kbench.task(name="pdptw_route_completion")
    def pdptw_route_completion(llm) -> float:
        """Complete a truncated route in a PDPTW solution."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
            route = bks.routes[longest_idx]
            customer_nodes = route[1:-1]
            midpoint = len(customer_nodes) // 2
            kept = customer_nodes[:midpoint]
            remaining = customer_nodes[midpoint:]
            partial_routes = [r[:] for r in bks.routes]
            partial_routes[longest_idx] = [0] + kept
            partial = PDPTWSolution(problem=problem, routes=partial_routes)
            prompt = build_route_completion_prompt(problem, partial, longest_idx, remaining, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)
            d = {"instance": problem.name, "kept": len(kept), "remaining": len(remaining), "bks_dist": round(bks.total_distance, 1)}

            completed_route = data.get("completed_route", None)
            # Fallback: if routes were extracted but not completed_route, use the first route
            if completed_route is None and "routes" in data and isinstance(data["routes"], list) and data["routes"]:
                completed_route = data["routes"][0] if len(data["routes"]) == 1 else data["routes"][longest_idx] if longest_idx < len(data["routes"]) else data["routes"][0]
            if not isinstance(completed_route, list):
                scores.append(0.0)
                details.append({**d, "result": "PARSE_FAIL", "score": 0.0})
                raw_str = str(raw)
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | kept={len(kept)} remaining={len(remaining)} | PARSE_FAIL | first 500 chars:\n    {raw_str[:500]}\n    ...last 500 chars:\n    {raw_str[-500:]}")
                continue

            try:
                completed_route = [int(n) for n in completed_route]
            except (ValueError, TypeError):
                scores.append(0.0)
                details.append({**d, "result": "PARSE_FAIL", "score": 0.0})
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | kept={len(kept)} remaining={len(remaining)} | PARSE_FAIL | score=0.0")
                continue

            if completed_route and completed_route[0] != 0:
                completed_route.insert(0, 0)
            if completed_route and completed_route[-1] != 0:
                completed_route.append(0)

            full_routes = [r[:] for r in partial.routes]
            full_routes[longest_idx] = completed_route
            solution = build_solution_from_llm_output(problem, full_routes)

            if solution is None:
                score = 0.0
                details.append({**d, "result": "BUILD_FAIL", "score": 0.0})
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | kept={len(kept)} remaining={len(remaining)} | BUILD_FAIL | score=0.0")
            else:
                feasible = score_feasibility(problem, solution)
                d["llm_dist"] = round(solution.total_distance, 1)
                d["feasible"] = feasible == 1.0
                if feasible == 0.0:
                    score = 0.0
                    details.append({**d, "result": "INFEASIBLE", "score": 0.0})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | kept={len(kept)} remaining={len(remaining)} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} | score=0.0")
                else:
                    dist_score = score_distance_gap(solution.total_distance, bks.total_distance)
                    score = 0.5 * feasible + 0.5 * dist_score
                    details.append({**d, "result": "FEASIBLE", "dist_gap": round(dist_score, 3), "score": round(score, 3)})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | kept={len(kept)} remaining={len(remaining)} | FEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={dist_score:.3f} | score={score:.3f}")

            scores.append(score)

        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Task 4 ROUTE COMPLETION: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
        print(f"  Breakdown: {', '.join(f'{s:.2f}' for s in scores)}")
        print(f"{'='*60}")
        RESULTS["tasks"]["route_completion"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results()
        return avg


    @kbench.task(name="pdptw_full_solution")
    def pdptw_full_solution(llm) -> float:
        """Generate a complete feasible PDPTW solution from scratch."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            prompt = build_full_solution_prompt(problem, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)

            routes_raw = data.get("routes", None)
            solution = build_solution_from_llm_output(problem, routes_raw) if routes_raw else None
            d = {"instance": problem.name, "nodes": len(problem.nodes), "vehicles": problem.num_vehicles, "bks_dist": round(bks.total_distance, 1)}

            if solution is None:
                score = 0.0
                details.append({**d, "result": "PARSE_FAIL", "score": 0.0})
                raw_str = str(raw)
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} vehicles={problem.num_vehicles} | PARSE_FAIL | first 500 chars:\n    {raw_str[:500]}\n    ...last 500 chars:\n    {raw_str[-500:]}")
            else:
                feasible = score_feasibility(problem, solution)
                n_served = sum(len(r) - 2 for r in solution.routes if len(r) > 2)
                n_expected = len(problem.nodes) - 1
                d.update(llm_dist=round(solution.total_distance, 1), feasible=feasible == 1.0, served=n_served, expected=n_expected, routes=len(solution.routes))
                if feasible == 0.0:
                    score = 0.0
                    details.append({**d, "result": "INFEASIBLE", "score": 0.0})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} served={n_served}/{n_expected} routes={len(solution.routes)} | score=0.0")
                else:
                    dist_score = score_distance_gap(solution.total_distance, bks.total_distance)
                    score = 0.5 * feasible + 0.5 * dist_score
                    details.append({**d, "result": "FEASIBLE", "dist_gap": round(dist_score, 3), "score": round(score, 3)})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} | FEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={dist_score:.3f} served={n_served}/{n_expected} | score={score:.3f}")

            scores.append(score)

        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Task 5 FULL SOLUTION: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
        print(f"  Breakdown: {', '.join(f'{s:.2f}' for s in scores)}")
        print(f"{'='*60}")
        RESULTS["tasks"]["full_solution"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results()
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")


# =============================================================================
# RUN TASKS
# On Kaggle: each task.run() + %choose should be in its OWN CELL.
# The %choose comment registers the task score with the Benchmarks platform.
# You can run all 5 cells to evaluate all tasks in one session.
# =============================================================================

# --- CELL: Task 1 ---
if HAS_KBENCH:
    pdptw_masked_node.run(llm=kbench.llm)
# %choose pdptw_masked_node

# --- CELL: Task 2 ---
if HAS_KBENCH:
    pdptw_request_insertion.run(llm=kbench.llm)
# %choose pdptw_request_insertion

# --- CELL: Task 3 ---
if HAS_KBENCH:
    pdptw_distance_prediction.run(llm=kbench.llm)
# %choose pdptw_distance_prediction

# --- CELL: Task 4 ---
if HAS_KBENCH:
    pdptw_route_completion.run(llm=kbench.llm)
# %choose pdptw_route_completion

# --- CELL: Task 5 ---
if HAS_KBENCH:
    pdptw_full_solution.run(llm=kbench.llm)
# %choose pdptw_full_solution
