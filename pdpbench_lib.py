"""PDPBench shared library — imported by all per-task files.

Contains all scoring functions, the JSON parser, prompt builders, and
instance loading utilities. Each per-task file does:

    from pdpbench_lib import *

This lib is self-locating: BASE_DIR resolves to the directory containing
this file (the dataset root on Kaggle, the project root locally), which
must be co-located with utils/ and the data/ and bks/ directories.
"""

import os
import sys
import ast
import json
import re
import time
import numpy as np
from enum import Enum


# =============================================================================
# Paths (self-locating — always points to dataset root)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BKS_DIR = os.path.join(BASE_DIR, "bks")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# =============================================================================
# Utils imports (from dataset root/utils/)
# =============================================================================

from utils.pdptw_problem import PDPTWProblem, Node, Request
from utils.pdptw_solution import PDPTWSolution
from utils.pdptw_reader import pdptw_reader
from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions
from utils.feasibility import is_feasible


# =============================================================================
# Logging helper (Tee — no leading underscore so import * exports it)
# =============================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


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


def apply_insertions(partial_routes, insertions, required_nodes=None):
    """Apply a list of 'insert X before/after Y' ops to a copy of partial_routes.

    Each op is a dict with keys:
      - 'insert': int — node index to add
      - 'before' or 'after': int — anchor node index (non-depot); the new node is placed
        immediately before or immediately after the anchor in whichever route contains it

    Returns (new_routes, applied_count, skipped_reasons).
    - new_routes: deep-copied list of routes with as many insertions applied as possible
    - applied_count: number of successful insertions
    - skipped_reasons: list of strings describing what went wrong per skipped op

    Skipped ops are silently dropped rather than aborting — this lets coverage scoring
    give partial credit for whatever the LLM got right. If required_nodes is given,
    only those node indices are allowed as 'insert' targets.
    """
    routes = [list(r) for r in partial_routes]
    present = set()
    for r in routes:
        for n in r:
            if n != 0:
                present.add(n)
    applied = 0
    skipped = []
    if not isinstance(insertions, list):
        return routes, 0, [f"insertions_not_list:{type(insertions).__name__}"]
    for op_idx, op in enumerate(insertions):
        if not isinstance(op, dict):
            skipped.append(f"op{op_idx}:not_dict")
            continue
        x = op.get("insert")
        try:
            x = int(x)
        except (TypeError, ValueError):
            skipped.append(f"op{op_idx}:bad_insert")
            continue
        if required_nodes is not None and x not in required_nodes:
            skipped.append(f"op{op_idx}:not_required:{x}")
            continue
        if x in present:
            skipped.append(f"op{op_idx}:duplicate:{x}")
            continue
        # Accept 'before' or 'after' (prefer 'before' if both given)
        anchor = op.get("before")
        mode = "before"
        if anchor is None:
            anchor = op.get("after")
            mode = "after"
        try:
            anchor = int(anchor)
        except (TypeError, ValueError):
            skipped.append(f"op{op_idx}:bad_anchor")
            continue
        if anchor == 0:
            skipped.append(f"op{op_idx}:anchor_is_depot")
            continue
        # Locate anchor
        found = False
        for r in routes:
            for i, node in enumerate(r):
                if node == anchor:
                    insert_pos = i if mode == "before" else i + 1
                    r.insert(insert_pos, x)
                    found = True
                    break
            if found:
                break
        if not found:
            skipped.append(f"op{op_idx}:anchor_not_in_routes:{anchor}")
            continue
        present.add(x)
        applied += 1
    return routes, applied, skipped


def score_feasibility_scaled(problem, solution):
    """1/(1+v)^2 where v = total constraint violations. 1.0 if feasible, 0.0 if None.

    Quadratic decay gives a large penalty on the first violation (1.0→0.25)
    with progressively smaller reductions for each additional violation.
    """
    if solution is None:
        return 0.0
    if is_feasible(problem, solution):
        return 1.0
    total_v = sum(count_violations(problem, solution).values())
    return 1.0 / (1.0 + total_v) ** 2


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


def print_task_header(task_id, title, n_instances, distance_mode_str="auto"):
    print(f"\n{'='*60}")
    print(f"  TASK {task_id} : {title}")
    print(f"  Running {n_instances} instances in {distance_mode_str} distance mode")
    print(f"{'='*60}")


def print_task_footer(task_id, title, avg, scores, elapsed):
    print(f"\n{'='*60}")
    print(f"  TASK {task_id} : {title}")
    print(f"  Score: {avg:.3f} ({len(scores)} instances, {elapsed:.1f}s)")
    print(f"  Breakdown: {', '.join(f'{s:.2f}' for s in scores)}")
    print(f"{'='*60}")


def instance_prefix(task_id, i, n):
    return f"  [{task_id} {i+1}/{n}]"


def score_distance_gap(actual_distance, bks_distance):
    if bks_distance <= 0:
        return 0.0
    gap = (actual_distance - bks_distance) / bks_distance
    return max(0.0, 1.0 - gap)


def score_completion_t1(routes, removed_pairs):
    """T1/T4: fraction of removed pairs correctly re-inserted.

    A pair is correct if both pickup and delivery appear exactly once,
    in the same route, with pickup before delivery.
    """
    if not routes or not removed_pairs:
        return 0.0
    positions = {}
    for r_idx, route in enumerate(routes):
        for p_idx, node in enumerate(route):
            if node == 0:
                continue
            if node in positions:
                positions[node] = "DUP"
            else:
                positions[node] = (r_idx, p_idx)
    correct = 0
    for p, d in removed_pairs:
        pp = positions.get(p)
        dd = positions.get(d)
        if pp is None or dd is None or pp == "DUP" or dd == "DUP":
            continue
        pr, pi = pp
        dr, di = dd
        if pr == dr and pi < di:
            correct += 1
    return correct / len(removed_pairs)


def score_completion_t2(new_route, required_nodes):
    """T2/T5: fraction of required customer nodes present in the new route."""
    if not new_route or not required_nodes:
        return 0.0
    present = set(n for n in new_route if n != 0)
    return len(present & required_nodes) / len(required_nodes)


def score_completion_t3(routes, pickups_deliveries):
    """T3/T6: fraction of requests where both pickup and delivery are present anywhere."""
    if not routes or not pickups_deliveries:
        return 0.0
    present = set()
    for route in routes:
        for node in route:
            if node != 0:
                present.add(node)
    correct = sum(1 for p, d in pickups_deliveries if p in present and d in present)
    return correct / len(pickups_deliveries)


def compute_score(problem, solution, bks_distance, completion):
    """Unified score: (1/3)*completion + (1/3)*feasibility_scaled + (1/3)*distance_gap.

    - completion: task-specific, computed before this call from raw routes
    - feasibility_scaled: 1.0 if feasible, 1/(1+violations) otherwise
    - distance_gap: vs BKS distance, always awarded (not gated on feasibility)

    Returns a dict with all four components.
    """
    feasibility = score_feasibility_scaled(problem, solution)
    distance_gap = score_distance_gap(solution.total_distance, bks_distance) if solution is not None else 0.0
    score = (completion + feasibility + distance_gap) / 3.0
    return {"completion": completion, "feasibility": feasibility, "distance_gap": distance_gap, "score": score}


# =============================================================================
# LLM call with retry
# =============================================================================

def llm_prompt(llm, prompt, retries=4, delay=30):
    """Call llm.prompt with retry on transient errors (503, etc.).

    Backoff: 30s, 60s, 90s, 120s — the Kaggle model proxy needs minutes
    to recover from overload; aggressive sub-second sleep was futile.
    """
    for attempt in range(retries):
        try:
            return str(llm.prompt(prompt))
        except Exception as e:
            if attempt < retries - 1 and ("503" in str(e) or "could not reach" in str(e).lower()):
                print(f"    Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                raise
    return ""


# =============================================================================
# JSON response parsing
# =============================================================================
#
# The parser is deliberately aggressive about extracting structured data from
# whatever the LLM emits. It handles:
#
#   Wrappings:
#     - <think>, <thinking>, <reasoning>, <scratchpad> reasoning blocks (stripped)
#     - <answer>, <solution>, <final>, <output> answer tags (extracted)
#     - \boxed{X} and \boxed{\text{X}} LaTeX wrappers (unwrapped)
#     - $...$ and $$...$$ math delimiters (stripped)
#     - ```json / ```python / ``` fenced code blocks
#     - Markdown prose surrounding a JSON blob
#     - Smart quotes normalized to ASCII
#
#   JSON flavors:
#     - Trailing commas
#     - // line comments and /* block comments */
#     - Python literals (True/False/None -> true/false/null)
#     - Unquoted object keys ({key: "x"} -> {"key": "x"})
#     - Single-quoted strings (Python-style dict repr)
#     - ast.literal_eval fallback for pure Python dict/list repr
#
#   Schema aliases (normalized to canonical keys):
#     - insertions: inserts, operations, ops, actions, moves
#     - insertion op node: insert, node, item, new_node, pickup, target
#     - insertion op anchor: before/after aliases (anchor_before, insert_before, prev; ...)
#     - routes: solution, plan, vehicle_routes, vehicles, tours
#     - new_route: completed_route, route, added_route, reconstructed_route
#     - top-level bare list auto-wrapped by inferred shape
#     - nested answer/result/output/response promoted to top level
#     - dict-of-routes: {"vehicle_0": [0,...,0], "vehicle_1": [0,...,0]} pattern
#
#   Prose fallbacks (when nothing structured can be extracted):
#     - "Insert X before/after Y" natural language for insertion ops
#     - \boxed{42} or "answer is X" for numeric tasks
# =============================================================================


_REASONING_TAG_RE = re.compile(
    r"<(think(?:ing)?|reasoning|scratchpad)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(
    r"<(answer|solution|final|output|final_answer)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_BLOCK_RE = re.compile(
    r"```(?:json|python|javascript|js|txt)?\s*(.*?)\s*```",
    re.DOTALL,
)
_INSERTION_PROSE_RE = re.compile(
    r"(?:insert|place|add|put)\s+(?:node\s+)?(\d+)\s+(before|after)\s+(?:node\s+)?(\d+)",
    re.IGNORECASE,
)

_INSERTIONS_KEY_ALIASES = ("insertions", "inserts", "operations", "ops", "actions", "moves")
_INSERTION_NODE_ALIASES = ("insert", "node", "item", "new_node", "pickup", "target", "value")
_INSERTION_BEFORE_ALIASES = ("before", "anchor_before", "insert_before", "prev", "predecessor")
_INSERTION_AFTER_ALIASES = ("after", "anchor_after", "insert_after", "next", "successor")
_ROUTES_KEY_ALIASES = ("routes", "solution", "plan", "vehicle_routes", "vehicles", "tours", "trips")
_NEW_ROUTE_KEY_ALIASES = ("new_route", "completed_route", "route", "added_route", "reconstructed_route", "missing_route")
_NESTED_WRAPPER_KEYS = ("answer", "result", "output", "response", "final_answer", "data")


def _strip_reasoning_tags(text):
    return _REASONING_TAG_RE.sub("", text)


def _normalize_quotes(text):
    return (
        text
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u00ab", '"').replace("\u00bb", '"')
    )


def _strip_math_delimiters(text):
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", r"\1", text)
    return text


def _strip_boxed_wrappers(text):
    """Replace \\boxed{X} and \\boxed{\\text{X}} with X, respecting balanced braces."""
    pattern = re.compile(r"\\boxed\s*\{")
    out = text
    safety = 0
    while safety < 50:
        safety += 1
        m = pattern.search(out)
        if not m:
            break
        brace_open = m.end() - 1
        depth = 0
        close = -1
        for j in range(brace_open, len(out)):
            ch = out[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close == -1:
            break
        inner = out[brace_open + 1:close]
        tm = re.match(r"\s*\\text\s*\{", inner)
        if tm:
            inner_brace_open = tm.end() - 1
            d = 0
            inner_close = -1
            for k in range(inner_brace_open, len(inner)):
                c = inner[k]
                if c == "{":
                    d += 1
                elif c == "}":
                    d -= 1
                    if d == 0:
                        inner_close = k
                        break
            if inner_close != -1:
                inner = inner[inner_brace_open + 1:inner_close]
        out = out[:m.start()] + inner + out[close + 1:]
    return out


def _extract_answer_tag(text):
    m = _ANSWER_TAG_RE.search(text)
    return m.group(2) if m else None


def _iter_balanced_blocks(text, open_ch, close_ch):
    """Yield each top-level balanced block delimited by open_ch/close_ch."""
    i = 0
    n = len(text)
    while i < n:
        start = text.find(open_ch, i)
        if start == -1:
            return
        depth = 0
        in_str = False
        escape = False
        end = -1
        for j in range(start, n):
            ch = text[j]
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
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
        if end == -1:
            return
        yield text[start:end + 1]
        i = end + 1


def _extract_balanced_json_object(text):
    """Return the first balanced {...} substring, or None. (Back-compat helper.)"""
    for block in _iter_balanced_blocks(text, "{", "}"):
        return block
    return None


def _to_json_compatible(obj):
    """Recursively convert Python-native values to JSON-compatible ones."""
    if isinstance(obj, tuple):
        return [_to_json_compatible(x) for x in obj]
    if isinstance(obj, set):
        return [_to_json_compatible(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_compatible(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_compatible(v) for k, v in obj.items()}
    return obj


def _tolerant_json_loads(s):
    """Progressive cleanup parser: json -> cleanups -> unquoted keys -> ast.literal_eval."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    cleaned = s
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    unquoted = re.sub(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:", r'\1"\2":', cleaned)
    try:
        return json.loads(unquoted)
    except json.JSONDecodeError:
        pass

    try:
        return _to_json_compatible(ast.literal_eval(s))
    except (ValueError, SyntaxError, MemoryError, TypeError):
        pass

    if '"' not in cleaned:
        swapped = cleaned.replace("'", '"')
        try:
            return json.loads(swapped)
        except json.JSONDecodeError:
            pass

    # YAML fallback — covers `insertions:\n  - insert: 1\n    before: 2` style.
    # Lazy import so missing PyYAML on Kaggle doesn't break the parser.
    try:
        import yaml as _yaml
        parsed = _yaml.safe_load(s)
        if isinstance(parsed, (dict, list)):
            return _to_json_compatible(parsed)
    except Exception:
        pass

    return None


def _find_alias(d, aliases):
    """Case-insensitive lookup for any of `aliases` in dict `d`. Returns the key found."""
    lower = {k.lower(): k for k in d.keys() if isinstance(k, str)}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _normalize_insertion_op(op):
    if not isinstance(op, dict):
        return op
    result = dict(op)
    if "insert" not in result:
        k = _find_alias(result, _INSERTION_NODE_ALIASES)
        if k is not None:
            result["insert"] = result[k]
    if "before" not in result and "after" not in result:
        k = _find_alias(result, _INSERTION_BEFORE_ALIASES)
        if k is not None:
            result["before"] = result[k]
        else:
            k = _find_alias(result, _INSERTION_AFTER_ALIASES)
            if k is not None:
                result["after"] = result[k]
    return result


def _dict_from_routes_list(lst):
    """If a list looks like routes (list of lists of ints, or list of dicts wrapping routes),
    return a normalized list-of-lists. Otherwise None."""
    if not isinstance(lst, list) or not lst:
        return None
    if all(isinstance(x, list) for x in lst):
        return lst
    if all(isinstance(x, dict) for x in lst):
        routes = []
        for d in lst:
            r = None
            for key in ("route", "path", "nodes", "sequence", "visits"):
                k = _find_alias(d, (key,))
                if k is not None and isinstance(d[k], list):
                    r = d[k]
                    break
            if r is not None:
                routes.append(r)
        if routes:
            return routes
    return None


def _looks_like_insertion_ops(lst):
    if not isinstance(lst, list) or not lst:
        return False
    for op in lst:
        if not isinstance(op, dict):
            return False
        if _find_alias(op, _INSERTION_NODE_ALIASES) is None:
            return False
    return True


def _normalize_parsed(data):
    """Apply schema aliases so downstream tasks can use canonical keys."""
    if isinstance(data, list):
        if _looks_like_insertion_ops(data):
            return {"insertions": [_normalize_insertion_op(op) for op in data]}
        routes = _dict_from_routes_list(data)
        if routes is not None:
            if len(routes) == 1 and all(isinstance(n, (int, float, str)) for n in routes[0]):
                return {"routes": routes, "new_route": routes[0]}
            return {"routes": routes}
        if all(isinstance(x, (int, float, str)) for x in data):
            return {"new_route": data}
        return {}

    if not isinstance(data, dict):
        return {}

    result = dict(data)

    for wrapper in _NESTED_WRAPPER_KEYS:
        k = _find_alias(result, (wrapper,))
        if k is not None:
            inner = result[k]
            if isinstance(inner, (dict, list)):
                normalized_inner = _normalize_parsed(inner)
                if isinstance(normalized_inner, dict):
                    for nk, nv in normalized_inner.items():
                        if nk not in result:
                            result[nk] = nv
                    if isinstance(inner, dict):
                        result[k] = normalized_inner

    if "insertions" not in result:
        k = _find_alias(result, _INSERTIONS_KEY_ALIASES)
        if k is not None and isinstance(result[k], list):
            result["insertions"] = result[k]
    if isinstance(result.get("insertions"), list):
        result["insertions"] = [_normalize_insertion_op(op) for op in result["insertions"]]
    if "insertions" not in result:
        single = _normalize_insertion_op(result) if _find_alias(result, _INSERTION_NODE_ALIASES) else None
        if single and "insert" in single and ("before" in single or "after" in single):
            result["insertions"] = [single]

    if "routes" not in result:
        k = _find_alias(result, _ROUTES_KEY_ALIASES)
        if k is not None:
            v = result[k]
            routes = _dict_from_routes_list(v) if isinstance(v, list) else None
            if routes is not None:
                result["routes"] = routes

    if "new_route" not in result:
        k = _find_alias(result, _NEW_ROUTE_KEY_ALIASES)
        if k is not None and isinstance(result[k], list):
            result["new_route"] = result[k]
    if "new_route" not in result and isinstance(result.get("routes"), list) and len(result["routes"]) == 1:
        only = result["routes"][0]
        if isinstance(only, list):
            result["new_route"] = only

    for canonical in ("done", "tool", "args", "answer"):
        if canonical not in result:
            k = _find_alias(result, (canonical,))
            if k is not None and k != canonical:
                result[canonical] = result[k]

    # Dict-of-routes fallback: {"vehicle_0": [0,1,2,0], "vehicle_1": [0,3,4,0]}
    # or {"route 1": [0,...,0], "route 2": [0,...,0]}.
    # Fires only when no canonical key was resolved AND every dict value is a
    # route-shaped list (list of at least 2 numbers). The guard
    # `len(candidate_routes) == len(result)` prevents false positives when the
    # dict mixes route-lists with scalars (e.g. "total_distance": 123.4).
    if "routes" not in result and "insertions" not in result and "new_route" not in result:
        def _is_route_list(v):
            return isinstance(v, list) and len(v) >= 2 and all(isinstance(n, (int, float)) for n in v)
        candidate_routes = [v for v in result.values() if _is_route_list(v)]
        if candidate_routes and len(candidate_routes) == len(result):
            result["routes"] = candidate_routes

    return result


_KNOWN_SCHEMA_KEYS = ("insertions", "routes", "new_route", "tool", "answer", "done")


def _iter_parse_candidates(text):
    """Yield parsed-but-not-normalized candidates from `text` in discovery order."""
    if not text:
        return

    for m in _FENCED_BLOCK_RE.finditer(text):
        block = m.group(1).strip()
        parsed = _tolerant_json_loads(block)
        if parsed is not None:
            yield parsed
        for candidate in _iter_balanced_blocks(block, "{", "}"):
            parsed = _tolerant_json_loads(candidate)
            if parsed is not None:
                yield parsed
        for candidate in _iter_balanced_blocks(block, "[", "]"):
            parsed = _tolerant_json_loads(candidate)
            if parsed is not None:
                yield parsed

    parsed = _tolerant_json_loads(text)
    if parsed is not None:
        yield parsed

    for candidate in _iter_balanced_blocks(text, "{", "}"):
        parsed = _tolerant_json_loads(candidate)
        if parsed is not None:
            yield parsed

    for candidate in _iter_balanced_blocks(text, "[", "]"):
        parsed = _tolerant_json_loads(candidate)
        if parsed is not None:
            yield parsed


def _try_parse_structured(text):
    """Return the best parsed candidate from `text`, normalized.

    Preference order:
      1. first candidate whose normalized form has a known schema key
      2. first parseable candidate (normalized, even if unknown shape)
      3. None
    """
    first_fallback = None
    for parsed in _iter_parse_candidates(text):
        normalized = _normalize_parsed(parsed)
        if isinstance(normalized, dict) and any(k in normalized for k in _KNOWN_SCHEMA_KEYS):
            return normalized
        if first_fallback is None:
            first_fallback = normalized if normalized is not None else parsed
    return first_fallback


def _extract_insertions_from_prose(text):
    """Regex for 'insert X before/after Y' natural language. Returns {"insertions": [...]} or None."""
    matches = _INSERTION_PROSE_RE.findall(text)
    if not matches:
        return None
    ops = []
    for n, direction, anchor in matches:
        ops.append({"insert": int(n), direction.lower(): int(anchor)})
    return {"insertions": ops}


def parse_json_response(raw):
    """Parse structured output from an LLM response, robust to many wrapping/schema variants.

    Returns a dict with canonical keys (insertions / routes / new_route / done / tool / args /
    answer / _fallback_number). Always returns a dict; an empty dict means nothing was extracted.
    """
    if not raw:
        return {}
    text = str(raw)

    text = _strip_reasoning_tags(text)
    text = _normalize_quotes(text)
    text = _strip_boxed_wrappers(text)
    text = _strip_math_delimiters(text)

    candidate_texts = []
    answer_block = _extract_answer_tag(text)
    if answer_block:
        candidate_texts.append(answer_block)
    candidate_texts.append(text)

    for cand in candidate_texts:
        parsed = _try_parse_structured(cand)
        if parsed is not None:
            normalized = _normalize_parsed(parsed)
            if normalized:
                return normalized

    prose = _extract_insertions_from_prose(text)
    if prose:
        return prose

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
    n_ops = 2 * len(removed_requests)
    instructions = (
        f"{len(removed_requests)} request(s) have been removed from the solution. "
        "Your job is to put the removed pickup and delivery nodes BACK into the routes.\n\n"
        "You will NOT rewrite the routes. Instead you only tell us WHERE each removed node goes, "
        "using simple insert-before-or-after instructions. This makes it impossible for you to "
        "accidentally drop, duplicate, or reorder any of the existing nodes.\n\n"
        "Instruction format: each entry inserts one node X into the solution, placed immediately\n"
        "before or immediately after an anchor node Y that currently exists in the routes:\n"
        '  {"insert": X, "before": Y}   -> put X immediately before Y\n'
        '  {"insert": X, "after": Y}    -> put X immediately after Y\n'
        "Y must be a non-depot node (not 0). Any customer node currently in the routes may be used "
        "as an anchor; we will find which route contains it.\n\n"
        f"You must provide EXACTLY {n_ops} insertions — one for each removed node (pickup AND delivery). "
        "Insertions are applied in the order you list them, so a later insertion may reference a node "
        "you just inserted. A common pattern is: insert the pickup after some existing node, then "
        "insert the delivery after the pickup you just placed.\n\n"
        "Requirements:\n"
        "- Each pickup must end up before its paired delivery on the same route.\n"
        "- Capacity and time window constraints must hold after all insertions are applied.\n"
        "- Minimize the added travel distance.\n\n"
        "Example with one removed request (pickup=41, delivery=42):\n"
        '  {"insertions": [\n'
        '    {"insert": 41, "after": 7},\n'
        '    {"insert": 42, "after": 41}\n'
        "  ]}\n\n"
        'Respond with EXACTLY this JSON:\n'
        '{"insertions": [{"insert": ..., "before"|"after": ...}, ...], "reasoning": "your reasoning"}')
    return _format_prompt(problem, distance_mode, instructions,
                          extra_data={"Current Partial Solution": sol_data, "Removed Requests (to insert)": requests_info})


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
# Iterative infrastructure
# =============================================================================

def format_conversation_history(turns):
    """Format prior assistant responses + system notes as a compact summary.

    Deliberately excludes the previous user prompts: they contain the full problem
    description (and sometimes the distance matrix), which is already re-included
    on every turn by the state_builder. Re-embedding them here produced O(N^2)
    context growth that was blowing out token budgets on Kaggle.
    """
    if not turns:
        return ""
    lines = ["## Prior turns (your responses only)", ""]
    for i, turn in enumerate(turns, 1):
        lines.append(f"### Turn {i} — you responded:")
        lines.append(turn.get("assistant", ""))
        if turn.get("system"):
            lines.append(f"[System note: {turn['system']}]")
        lines.append("")
    return "\n".join(lines)


def run_iterative_steps(llm, problem, steps, state_builder, response_extractor,
                        initial_state=None, max_parse_failures=2):
    """Drive a multi-turn iterative task with a fixed step list.

    state_builder(step, state, history_text) -> prompt string for this turn
    response_extractor(data, state, step) -> new state, or None on parse failure

    If response_extractor returns None, the step is retried (up to
    max_parse_failures extra attempts) with a system nudge appended to history.
    Only after exhausting retries on the same step does the driver abort.

    Returns (final_state, turns_used, abort_reason_or_None).
    """
    history = []
    state = initial_state
    turns_used = 0
    for turn_idx, step in enumerate(steps):
        progressed = False
        for attempt in range(max_parse_failures + 1):
            history_text = format_conversation_history(history)
            prompt = state_builder(step, state, history_text)
            raw = llm_prompt(llm, prompt)
            turns_used += 1
            data = parse_json_response(raw)
            new_state = response_extractor(data, state, step)
            if new_state is not None:
                state = new_state
                history.append({"assistant": raw})
                progressed = True
                break
            history.append({
                "assistant": raw,
                "system": "Your last response could not be parsed. "
                          "Respond with EXACTLY the JSON schema shown in the task instructions.",
            })
        if not progressed:
            return state, turns_used, f"parse_fail_turn_{turn_idx + 1}"
    return state, turns_used, None


# --- Iterative prompt builders ---

def build_iterative_insertion_step_prompt(problem, current_routes, request_to_insert, distance_mode, history_text):
    pickup_idx, delivery_idx = request_to_insert
    pn = problem.nodes_dict[pickup_idx]
    dn = problem.nodes_dict[delivery_idx]
    req_info = {
        "pickup": {"index": pickup_idx, "demand": pn.demand, "time_window": list(pn.time_window), "service_time": pn.service_time},
        "delivery": {"index": delivery_idx, "demand": dn.demand, "time_window": list(dn.time_window), "service_time": dn.service_time},
    }
    sol_data = {"routes": current_routes}
    instructions = (
        "This is a MULTI-TURN task. On each turn you insert ONE request (a pickup + its delivery) "
        "into the current solution using simple insert-before-or-after instructions.\n\n"
        f"This turn: insert pickup {pickup_idx} and delivery {delivery_idx}.\n\n"
        "You do NOT rewrite routes. You only say where each of the two nodes goes, by naming an "
        "anchor node Y that currently exists in the routes:\n"
        '  {"insert": X, "before": Y}   -> put X immediately before Y\n'
        '  {"insert": X, "after": Y}    -> put X immediately after Y\n'
        "Y must be a non-depot node (not 0). Any customer in the current routes is a valid anchor.\n"
        "Insertions are applied in order, so the delivery can reference the pickup you just placed.\n\n"
        "Requirements:\n"
        f"- The pickup {pickup_idx} must end up before its delivery {delivery_idx} on the same route.\n"
        "- Capacity and time windows must hold after this turn (and at the end).\n\n"
        'Respond with EXACTLY this JSON (2 entries — one for the pickup, one for the delivery):\n'
        '{"insertions": [{"insert": ..., "before"|"after": ...}, {"insert": ..., "before"|"after": ...}], "reasoning": "your reasoning"}'
    )
    extra = {"Current Solution": sol_data, "Request to insert on this turn": req_info}
    prompt = _format_prompt(problem, distance_mode, instructions, extra_data=extra)
    if history_text:
        prompt = prompt + "\n\n" + history_text
    return prompt


def build_iterative_route_build_step_prompt(problem, partial_routes, new_route_so_far, request_to_insert, distance_mode, history_text):
    pickup_idx, delivery_idx = request_to_insert
    pn = problem.nodes_dict[pickup_idx]
    dn = problem.nodes_dict[delivery_idx]
    req_info = {
        "pickup": {"index": pickup_idx, "demand": pn.demand, "time_window": list(pn.time_window), "service_time": pn.service_time},
        "delivery": {"index": delivery_idx, "demand": dn.demand, "time_window": list(dn.time_window), "service_time": dn.service_time},
    }
    instructions = (
        "This is a MULTI-TURN route-reconstruction task. One vehicle's route is missing from the BKS solution.\n"
        "On each turn you insert ONE request into the new route you are building. The other routes stay fixed.\n\n"
        f"Insert pickup {pickup_idx} and its paired delivery {delivery_idx} into the new route below.\n"
        "- Pickup must appear before its paired delivery.\n"
        "- Capacity and time windows must hold for this single vehicle.\n"
        "- The new route must start and end at depot (0).\n\n"
        'Respond with EXACTLY this JSON:\n{"new_route": [0, ...node ids..., 0], "reasoning": "your reasoning"}'
    )
    extra = {
        "Other Routes (fixed)": {"routes": partial_routes},
        "New Route So Far": {"new_route": new_route_so_far},
        "Request to insert on this turn": req_info,
    }
    prompt = _format_prompt(problem, distance_mode, instructions, extra_data=extra)
    if history_text:
        prompt = prompt + "\n\n" + history_text
    return prompt


def build_iterative_full_route_step_prompt(problem, completed_routes, unserved_requests, distance_mode, history_text):
    requests_info = []
    for pickup_idx, delivery_idx in unserved_requests:
        pn = problem.nodes_dict[pickup_idx]
        dn = problem.nodes_dict[delivery_idx]
        requests_info.append({
            "pickup": {"index": pickup_idx, "demand": pn.demand, "time_window": list(pn.time_window), "service_time": pn.service_time},
            "delivery": {"index": delivery_idx, "demand": dn.demand, "time_window": list(dn.time_window), "service_time": dn.service_time},
        })
    remaining_vehicles = problem.num_vehicles - len(completed_routes)
    instructions = (
        "This is a MULTI-TURN full-solution task. You build one route per turn.\n\n"
        f"You have already committed {len(completed_routes)} route(s). {remaining_vehicles} vehicle(s) remain.\n"
        f"{len(unserved_requests)} request(s) still need to be served.\n\n"
        "Respond with ONE new route (one vehicle) that serves some of the remaining requests.\n"
        "- The new route must start and end at depot (0).\n"
        "- Pickup before delivery on the same vehicle.\n"
        "- Capacity and time windows must hold for this single vehicle.\n"
        "- If all requests are served and no more routes are needed, respond with {\"done\": true}.\n\n"
        'Respond with EXACTLY ONE of:\n'
        '{"route": [0, ...node ids..., 0], "reasoning": "your reasoning"}\n'
        '{"done": true, "reasoning": "your reasoning"}'
    )
    extra = {
        "Completed Routes": {"routes": completed_routes},
        "Remaining Requests": requests_info,
    }
    prompt = _format_prompt(problem, distance_mode, instructions, extra_data=extra)
    if history_text:
        prompt = prompt + "\n\n" + history_text
    return prompt


# =============================================================================
# Instance selection & data preparation
# =============================================================================

LI_LIM_INSTANCES = ["lc101", "lc201", "lr101", "lr201", "lrc101"]
MENDELEY_INSTANCES = ["bar-n100-1", "ber-n100-1", "nyc-n100-1"]
DISTANCE_MODE = DistanceMode.MATRIX

LI_LIM_INSTANCES_20 = [
    "lc101", "lc102",
    "lc201", "lc202",
    "lr101", "lr102",
    "lr201", "lr202",
    "lrc101", "lrc102",
]
MENDELEY_INSTANCES_20 = [
    "bar-n100-1", "bar-n100-2", "bar-n100-3",
    "ber-n100-1", "ber-n100-2", "ber-n100-3",
    "nyc-n100-1", "nyc-n100-2",
    "poa-n100-1", "poa-n100-2",
]


def iterative_distance_mode(problem):
    """Pick a distance mode for iterative tasks.

    Iterative tasks re-send the problem data every turn, so the 100x100 matrix
    becomes expensive. For Li & Lim instances we switch to COORDINATES (~200
    numbers vs 10000). Mendeley instances have no usable coordinates in the
    reader, so they keep MATRIX.
    """
    if problem.dataset == "Li & Lim":
        return DistanceMode.COORDINATES
    return DistanceMode.MATRIX


def get_benchmark_instances():
    """Load all benchmark instances. Uses DATA_DIR and BKS_DIR from this lib's BASE_DIR."""
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


def get_benchmark_instances_20():
    """Load 20 benchmark instances for T1–T3 (10 Li & Lim + 10 Mendeley)."""
    li_lim_mgr = LiLimInstanceManager(base_dir=DATA_DIR)
    mendeley_mgr = MendeleyInstanceManager(base_dir=DATA_DIR)
    bks = BestKnownSolutions(bks_path=BKS_DIR)
    instances = []
    for name in LI_LIM_INSTANCES_20:
        problem = li_lim_mgr.load(name, size=100)
        solution = bks.get_bks_as_solution(problem)
        instances.append((problem, solution))
    for name in MENDELEY_INSTANCES_20:
        problem = mendeley_mgr.load(name, size=100)
        solution = bks.get_bks_as_solution(problem)
        instances.append((problem, solution))
    return instances


def save_results(results, path):
    """Write results dict to JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")
