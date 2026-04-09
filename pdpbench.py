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


def count_violations(problem, solution):
    """Walk a solution without short-circuiting and count each type of constraint violation.

    Mirrors the checks in utils.feasibility.is_feasible but keeps going so we can see
    what the LLM did wrong. Returns a dict mapping violation type -> count.
    """
    counts = {
        "not_start_at_depot": 0,
        "not_end_at_depot": 0,
        "depot_in_middle": 0,
        "duplicate_node": 0,
        "delivery_before_pickup": 0,
        "invalid_node": 0,
        "capacity": 0,
        "time_window": 0,
        "missing_nodes": 0,
        "extra_vehicles": 0,
    }
    pickup_to_delivery = problem.pickup_to_delivery
    delivery_to_pickup = problem.delivery_to_pickup
    demands = problem.demands
    vehicle_capacity = problem.vehicle_capacity
    distance_matrix = problem.distance_matrix
    time_windows = problem.time_windows
    service_times = problem.service_times
    n_nodes = len(problem.nodes)

    seen_total = set()
    non_empty_routes = 0
    for route in solution.routes:
        if not route or len(route) < 2:
            continue
        non_empty_routes += 1

        if route[0] != 0:
            counts["not_start_at_depot"] += 1
        if route[-1] != 0:
            counts["not_end_at_depot"] += 1

        load = 0
        current_time = 0
        seen = set()
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            if 0 < i + 1 < len(route) - 1 and to_node == 0:
                counts["depot_in_middle"] += 1

            if to_node in seen:
                counts["duplicate_node"] += 1

            is_valid_node = to_node == 0 or to_node in pickup_to_delivery or to_node in delivery_to_pickup
            if not is_valid_node:
                counts["invalid_node"] += 1
            elif to_node in delivery_to_pickup and delivery_to_pickup[to_node] not in seen:
                counts["delivery_before_pickup"] += 1

            if 0 <= to_node < n_nodes:
                load += int(demands[to_node])
                if load < 0 or load > vehicle_capacity:
                    counts["capacity"] += 1

                if 0 <= from_node < n_nodes:
                    current_time += float(distance_matrix[from_node, to_node])
                tw_start, tw_end = time_windows[to_node]
                if current_time < tw_start:
                    current_time = float(tw_start)
                if current_time > tw_end:
                    counts["time_window"] += 1
                current_time += float(service_times[to_node])

            seen.add(to_node)

        seen_total.update(seen)

    all_idx = set(node.index for node in problem.nodes)
    counts["missing_nodes"] = len(all_idx - seen_total)

    if non_empty_routes > problem.num_vehicles:
        counts["extra_vehicles"] = non_empty_routes - problem.num_vehicles

    return counts


def format_violations(counts):
    """Format non-zero violations as a compact string, or 'none' if all zero."""
    active = [(k, v) for k, v in counts.items() if v > 0]
    if not active:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in active)


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

def _extract_balanced_json_object(text):
    """Scan text for the first balanced {...} block, respecting string literals.

    Returns the substring or None. Handles nested braces and ignores braces inside
    double-quoted strings (with backslash escapes).
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _tolerant_json_loads(s):
    """Try json.loads with light cleanup: trailing commas, // comments, single quotes."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip // line comments
    cleaned = re.sub(r"//[^\n]*", "", s)
    # Strip trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Last resort: swap single quotes for double quotes (only if no double quotes used for keys)
    if '"' not in cleaned:
        swapped = cleaned.replace("'", '"')
        try:
            return json.loads(swapped)
        except json.JSONDecodeError:
            pass
    return None


def parse_json_response(raw):
    """Parse JSON from LLM response.

    Strategy (each layer returns on success):
      1. Strip <think>/<thinking> blocks
      2. Fenced code block ```json ... ```
      3. Whole-text json.loads
      4. Balanced-brace scan + tolerant cleanup
      5. Routes-only regex
      6. \\boxed{} / "answer is X" / "distance is X"
      7. Ultimate fallback: last number in text -> _fallback_number
    """
    if not raw:
        return {}
    text = str(raw)

    # 1. Strip thinking/reasoning tags
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Try extracting from fenced code block
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
        block = m.group(1).strip()
        parsed = _tolerant_json_loads(block)
        if parsed is not None:
            return parsed
        # The block itself may contain prose around a JSON object
        inner = _extract_balanced_json_object(block)
        if inner:
            parsed = _tolerant_json_loads(inner)
            if parsed is not None:
                return parsed

    # 3. Try parsing the whole text as JSON
    parsed = _tolerant_json_loads(text.strip())
    if parsed is not None:
        return parsed

    # 4. Balanced-brace scan for the first full JSON object in the text
    candidate = _extract_balanced_json_object(text)
    if candidate:
        parsed = _tolerant_json_loads(candidate)
        if parsed is not None:
            return parsed

    # 5. Try to extract routes from plain text like [[0,1,2,0],[0,3,4,0]]
    routes_match = re.search(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", text)
    if routes_match:
        try:
            routes = json.loads(routes_match.group(0))
            if isinstance(routes, list) and all(isinstance(r, list) for r in routes):
                return {"routes": routes}
        except json.JSONDecodeError:
            pass

    # 6. LaTeX \boxed{} / "the answer is X" / "distance is X"
    boxed = re.search(r"\\boxed\{\\text\{(-?\d+\.?\d*)\}\}", text) or re.search(r"\\boxed\{(-?\d+\.?\d*)\}", text)
    if boxed:
        val = boxed.group(1)
        return {"_fallback_number": float(val) if "." in val else int(val)}
    answer_match = re.search(r"(?:answer|predicted[_ ]node|node)\s*(?:is|:|=)\s*(-?\d+)", text, re.IGNORECASE)
    if answer_match:
        return {"_fallback_number": int(answer_match.group(1))}
    dist_match = re.search(r"(?:distance|total)\s*(?:is|:|=)\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if dist_match:
        return {"_fallback_number": float(dist_match.group(1))}

    # 7. Ultimate fallback: take the last number anywhere in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    numbers = [n for n in numbers if n not in ("", "-", ".", "-.")]
    if numbers:
        last = numbers[-1]
        try:
            return {"_fallback_number": float(last) if "." in last else int(last)}
        except ValueError:
            pass

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


def build_route_completion_prompt(problem, partial_solution, removed_requests, distance_mode):
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
        "One route has been removed from the solution. The partial solution below contains every other route unchanged, "
        f"and the {len(removed_requests)} requests listed below were all served by the removed route.\n\n"
        "Construct a single NEW route (one vehicle) that serves ALL of the listed requests. "
        "The other routes must stay exactly as shown.\n\n"
        "Requirements:\n"
        "- The new route must start and end at the depot (0).\n"
        "- Every pickup and its paired delivery from the request list must appear exactly once in the new route.\n"
        "- Each pickup must appear before its paired delivery.\n"
        "- Capacity and time window constraints must be satisfied for this single vehicle.\n"
        "- Minimize the travel distance of the new route.\n\n"
        'Respond with EXACTLY this JSON:\n{"new_route": [0, ...node ids..., 0], "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Partial Solution (one route removed)": sol_data,
                                      "Requests served by the removed route": requests_info})


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

  Task 4 (Route Completion):  reconstruct a missing route from its request list
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
            elif not data:
                parse_ok = "json_fail"
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
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | PARSE_FAIL | score=0.0")
            else:
                feasible = score_feasibility(problem, solution)
                d["llm_dist"] = round(solution.total_distance, 1)
                d["feasible"] = feasible == 1.0
                if feasible == 0.0:
                    score = 0.0
                    violations = count_violations(problem, solution)
                    d.update(result="INFEASIBLE", score=0.0, violations=violations)
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} | violations: {format_violations(violations)} | score=0.0")
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
        """Reconstruct a missing route: BKS with one full route removed + its request list."""
        scores = []
        details = []
        start = time.time()
        for i, (problem, bks) in enumerate(INSTANCES):
            # Remove the longest BKS route entirely and extract its requests
            longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
            removed_route = bks.routes[longest_idx]
            removed_customers = set(removed_route[1:-1])
            removed_requests = [(p, d) for p, d in problem.pickups_deliveries if p in removed_customers]
            partial_routes = [r[:] for idx, r in enumerate(bks.routes) if idx != longest_idx]
            partial = PDPTWSolution(problem=problem, routes=partial_routes)

            prompt = build_route_completion_prompt(problem, partial, removed_requests, DISTANCE_MODE)
            raw = llm_prompt(llm, prompt)
            data = parse_json_response(raw)
            d = {"instance": problem.name, "removed_route_len": len(removed_route), "n_requests": len(removed_requests), "bks_dist": round(bks.total_distance, 1)}

            new_route = data.get("new_route", data.get("completed_route", None))
            # Fallback: LLM returned a 'routes' list — take the first
            if new_route is None and "routes" in data and isinstance(data["routes"], list) and data["routes"]:
                new_route = data["routes"][0]
            if not isinstance(new_route, list):
                scores.append(0.0)
                details.append({**d, "result": "PARSE_FAIL", "score": 0.0})
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | PARSE_FAIL | score=0.0")
                continue

            try:
                new_route = [int(n) for n in new_route]
            except (ValueError, TypeError):
                scores.append(0.0)
                details.append({**d, "result": "PARSE_FAIL", "score": 0.0})
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | PARSE_FAIL | score=0.0")
                continue

            if new_route and new_route[0] != 0:
                new_route.insert(0, 0)
            if new_route and new_route[-1] != 0:
                new_route.append(0)

            full_routes = [r[:] for r in partial.routes] + [new_route]
            solution = build_solution_from_llm_output(problem, full_routes)

            if solution is None:
                score = 0.0
                details.append({**d, "result": "BUILD_FAIL", "score": 0.0})
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | BUILD_FAIL | score=0.0")
            else:
                feasible = score_feasibility(problem, solution)
                d["llm_dist"] = round(solution.total_distance, 1)
                d["feasible"] = feasible == 1.0
                if feasible == 0.0:
                    score = 0.0
                    violations = count_violations(problem, solution)
                    details.append({**d, "result": "INFEASIBLE", "score": 0.0, "violations": violations})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} | violations: {format_violations(violations)} | score=0.0")
                else:
                    dist_score = score_distance_gap(solution.total_distance, bks.total_distance)
                    score = 0.5 * feasible + 0.5 * dist_score
                    details.append({**d, "result": "FEASIBLE", "dist_gap": round(dist_score, 3), "score": round(score, 3)})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | FEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={dist_score:.3f} | score={score:.3f}")

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
                print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} vehicles={problem.num_vehicles} | PARSE_FAIL | score=0.0")
            else:
                feasible = score_feasibility(problem, solution)
                n_served = sum(len(r) - 2 for r in solution.routes if len(r) > 2)
                n_expected = len(problem.nodes) - 1
                d.update(llm_dist=round(solution.total_distance, 1), feasible=feasible == 1.0, served=n_served, expected=n_expected, routes=len(solution.routes))
                if feasible == 0.0:
                    score = 0.0
                    violations = count_violations(problem, solution)
                    details.append({**d, "result": "INFEASIBLE", "score": 0.0, "violations": violations})
                    print(f"  [{i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} | INFEASIBLE | llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} served={n_served}/{n_expected} routes={len(solution.routes)} | violations: {format_violations(violations)} | score=0.0")
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
