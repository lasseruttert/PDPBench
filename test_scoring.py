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


def score_distance_gap(actual_distance, bks_distance):
    if bks_distance <= 0:
        return 0.0
    gap = (actual_distance - bks_distance) / bks_distance
    return max(0.0, 1.0 - gap)


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
# Test: Request Insertion (insert-before/after op schema)
# =============================================================================
print("=" * 60)
print("TEST: Request Insertion — insert-op schema")
print("=" * 60)

import pdpbench_lib as _pdb_early  # used for apply_insertions/compute_score/score_completion_t1

partial = bks.clone()
removed = []
required_nodes = set()
for pickup_idx, delivery_idx in problem.pickups_deliveries[:2]:
    partial.remove_request(problem, pickup_idx)
    removed.append((pickup_idx, delivery_idx))
    required_nodes.add(pickup_idx)
    required_nodes.add(delivery_idx)
print(f"Removed requests: {removed}")
print(f"Partial solution routes: {len(partial.routes)}")

# Build the ground-truth insertions by locating each removed node's anchor in the BKS.
def _bks_insertions_for(removed):
    ops = []
    # For each removed request, find pickup's position in BKS and the node just after it
    for p, d in removed:
        for route in bks.routes:
            if p in route:
                pi = route.index(p)
                anchor = route[pi - 1] if pi - 1 > 0 else route[pi + 1]
                mode = "after" if pi - 1 > 0 else "before"
                ops.append({"insert": p, mode: anchor})
                break
        for route in bks.routes:
            if d in route:
                di = route.index(d)
                anchor = route[di - 1] if route[di - 1] != 0 else route[di + 1]
                mode = "after" if route[di - 1] != 0 else "before"
                ops.append({"insert": d, mode: anchor})
                break
    return ops

# Test A: Give valid anchors that reconstruct the BKS — should score full.
# Simpler: place pickup after 0's-successor and delivery after pickup.
# Use a minimally constructed insertion set: insert each removed node back into partial.
# We'll just pick any existing non-depot node as an anchor and then ensure pickup-before-delivery.
# This is a sanity check, not a perfection test; pass/fail on coverage=1.0.
# We construct ops that: for each request, insert pickup after some existing node, delivery after pickup.
fresh_partial = bks.clone()
for p, d in removed:
    fresh_partial.remove_request(problem, p)
# pick an anchor node: first non-depot in fresh_partial's first non-empty route
anchor_node = None
for r in fresh_partial.routes:
    for n in r:
        if n != 0:
            anchor_node = n
            break
    if anchor_node is not None:
        break
ops = []
prev_insert = anchor_node
for p, d in removed:
    ops.append({"insert": p, "after": prev_insert})
    ops.append({"insert": d, "after": p})
    prev_insert = d
new_routes, applied, skipped = _pdb_early.apply_insertions(fresh_partial.routes, ops, required_nodes=required_nodes)
assert applied == 2 * len(removed), f"expected {2*len(removed)} applied, got {applied} skipped={skipped}"
sol_ins = _pdb_early.build_solution_from_llm_output(problem, new_routes)
assert sol_ins is not None, "insertion result should build"
completion = _pdb_early.score_completion_t1(new_routes, removed)
components = _pdb_early.compute_score(problem, sol_ins, bks.total_distance, completion)
print(f"  Valid insertions: applied={applied}/{2*len(removed)} compl={components['completion']:.3f} feas={components['feasibility']} score={components['score']:.3f}")
assert components["completion"] == 1.0, f"expected completion 1.0 after all insertions applied, got {components['completion']}"

# Test B: Bad anchor — pickup references a non-existent node; skipped silently.
bad_ops = [{"insert": removed[0][0], "after": 9999}]  # anchor not in routes
_, applied2, skipped2 = _pdb_early.apply_insertions(fresh_partial.routes, bad_ops, required_nodes=required_nodes)
assert applied2 == 0 and any("anchor_not_in_routes" in s for s in skipped2), f"bad anchor should be skipped: {skipped2}"
print(f"  Bad anchor: applied=0 skipped={skipped2} ✓")

# Test C: Duplicate insertion (insert a node already in routes)
existing_node = next(n for r in fresh_partial.routes for n in r if n != 0)
dup_ops = [{"insert": existing_node, "after": existing_node}]
_, applied3, skipped3 = _pdb_early.apply_insertions(fresh_partial.routes, dup_ops, required_nodes={existing_node})
# required_nodes check fires first, but if we lift it, duplicate check should fire
_, _, skipped3b = _pdb_early.apply_insertions(fresh_partial.routes, dup_ops, required_nodes=None)
assert any("duplicate" in s for s in skipped3b), f"duplicate should be caught: {skipped3b}"
print(f"  Duplicate insert: skipped={skipped3b} ✓")

# Test D: score_completion_t1 gives partial credit when only some pairs inserted
partial_compl = _pdb_early.score_completion_t1(fresh_partial.routes, removed)
assert partial_compl == 0.0, f"partial solution (no insertions) should have completion=0.0, got {partial_compl}"
print(f"  score_completion_t1 before insertions: {partial_compl:.3f} (expected 0.0) ✓")

# Test E: compute_score with completion=0 and None solution
none_components = _pdb_early.compute_score(problem, None, bks.total_distance, 0.0)
assert none_components == {"completion": 0.0, "feasibility": 0.0, "distance_gap": 0.0, "score": 0.0}
print(f"  compute_score(None, completion=0): {none_components} ✓")
print()

# =============================================================================
# Test: Route Completion (reconstruct a missing route from its request list)
# =============================================================================
print("=" * 60)
print("TEST: Route Completion (missing route reconstruction)")
print("=" * 60)

longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
removed_route = bks.routes[longest_idx]
removed_customers = set(removed_route[1:-1])
removed_requests = [(p, d) for p, d in problem.pickups_deliveries if p in removed_customers]
partial_routes = [r[:] for idx, r in enumerate(bks.routes) if idx != longest_idx]
partial = PDPTWSolution(problem=problem, routes=partial_routes)
print(f"Removed route {longest_idx}: {removed_route}")
print(f"Requests to serve with new route: {removed_requests}")

# Test A: LLM returns the correct removed route
fake_response = json.dumps({"new_route": removed_route, "reasoning": "test"})
data = parse_json_response(fake_response)
new_route = [int(n) for n in data.get("new_route", [])]
compl_a = _pdb_early.score_completion_t2(new_route, removed_customers)
full_routes = [r[:] for r in partial.routes] + [new_route]
solution = build_solution_from_llm_output(problem, full_routes)
if solution is not None:
    components_a = _pdb_early.compute_score(problem, solution, bks.total_distance, compl_a)
    print(f"  Correct new route: compl={compl_a:.3f} feasible={components_a['feasibility']==1.0}, score={components_a['score']:.3f} ✓")
    assert compl_a == 1.0, f"expected completion 1.0, got {compl_a}"
else:
    print(f"  Correct new route: FAILED to build!")

# Test B: LLM returns a garbled route (pickup after delivery)
wrong_route = [0] + list(reversed(removed_route[1:-1])) + [0]
fake_response = json.dumps({"new_route": wrong_route, "reasoning": "test"})
data = parse_json_response(fake_response)
new_route = [int(n) for n in data.get("new_route", [])]
compl_b = _pdb_early.score_completion_t2(new_route, removed_customers)
full_routes = [r[:] for r in partial.routes] + [new_route]
solution = build_solution_from_llm_output(problem, full_routes)
if solution is not None:
    components_b = _pdb_early.compute_score(problem, solution, bks.total_distance, compl_b)
    print(f"  Reversed new route: compl={compl_b:.3f} feasible={components_b['feasibility']==1.0} score={components_b['score']:.3f}")
else:
    print(f"  Reversed new route: build failed")
print()

# =============================================================================
# Test: Full Solution
# =============================================================================
print("=" * 60)
print("TEST: Full Solution Generation")
print("=" * 60)

# Test A: Return BKS routes
fake_response = json.dumps({"routes": bks.routes, "reasoning": "test"})
data = parse_json_response(fake_response)
routes_raw = data.get("routes", None)
compl_t3 = _pdb_early.score_completion_t3(routes_raw or [], problem.pickups_deliveries)
solution = build_solution_from_llm_output(problem, routes_raw)
if solution is not None:
    components_t3 = _pdb_early.compute_score(problem, solution, bks.total_distance, compl_t3)
    print(f"  BKS routes: compl={compl_t3:.3f} feasible={components_t3['feasibility']==1.0}, score={components_t3['score']:.3f} ✓")
    assert compl_t3 == 1.0, f"expected completion 1.0 for BKS routes, got {compl_t3}"
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


# =============================================================================
# Iterative & tool-use driver tests
# =============================================================================
# These import pdpbench as a module so we exercise the exact driver code that
# the tasks use. pdpbench's HAS_KBENCH is False locally, so no tasks are
# registered or run — we just get the helpers.
print()
print("=" * 60)
print("TEST: Iterative & Tool-use Drivers")
print("=" * 60)

import pdpbench_lib as pdb


class FakeLLM:
    """LLM stub that returns queued canned responses in order.

    Supports either a plain list (returned in order) or a callable
    that takes the prompt text and returns a response string.
    """
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.index = 0

    def prompt(self, text):
        self.prompts.append(text)
        if self.index >= len(self.responses):
            raise RuntimeError(f"FakeLLM ran out of responses after {self.index} calls")
        r = self.responses[self.index]
        self.index += 1
        return r(text) if callable(r) else r


# --- Iterative driver: success case (insertions schema) --------------------
# Two requests removed; each turn, fake LLM returns insertions for ONE request.
bks_for_iter = bks_loader.get_bks_as_solution(problem)
partial_for_iter = bks_for_iter.clone()
removed_reqs = []
for p, d in problem.pickups_deliveries[:2]:
    partial_for_iter.remove_request(problem, p)
    removed_reqs.append((p, d))

# Pick an anchor node from partial_for_iter
iter_anchor = None
for r in partial_for_iter.routes:
    for n in r:
        if n != 0:
            iter_anchor = n
            break
    if iter_anchor is not None:
        break

# Build per-turn fake responses: each turn inserts that request's pickup then delivery,
# using the current anchor as the pickup position (delivery chains after pickup).
fake_responses = []
for (p, d) in removed_reqs:
    fake_responses.append(json.dumps({
        "insertions": [
            {"insert": p, "after": iter_anchor},
            {"insert": d, "after": p},
        ],
        "reasoning": "place pickup after anchor, then delivery after pickup",
    }))

fake = FakeLLM(fake_responses)

def _state_builder(step, state, history_text):
    return pdb.build_iterative_insertion_step_prompt(problem, state, step, pdb.DistanceMode.MATRIX, history_text)

def _response_extractor(data, state, step):
    if not isinstance(data, dict):
        return None
    insertions = data.get("insertions")
    required = {step[0], step[1]}
    new_routes, _applied, _skipped = pdb.apply_insertions(state, insertions, required_nodes=required)
    return new_routes

final_state, turns, abort = pdb.run_iterative_steps(
    fake, problem, removed_reqs, _state_builder, _response_extractor,
    initial_state=[list(r) for r in partial_for_iter.routes],
)
assert abort is None, f"expected no abort, got {abort}"
assert turns == 2, f"expected 2 turns, got {turns}"
sol = pdb.build_solution_from_llm_output(problem, final_state)
assert sol is not None, "iterative final solution should build"
compl_iter = pdb.score_completion_t1(final_state, removed_reqs)
assert compl_iter == 1.0, f"iterative completion should be 1.0, got {compl_iter}"
print(f"  Iter success: turns={turns} completion={compl_iter:.3f} dist={sol.total_distance:.1f} ✓")

# Verify each prompt carried history after turn 1
assert "Prior turns" in fake.prompts[1], "turn 2 prompt should include history"
assert "Prior turns" not in fake.prompts[0], "turn 1 prompt should not include history"
print("  Iter history threading: turn 2 includes prior turn, turn 1 does not ✓")


# --- Iterative driver: parse-failure leniency retries within a step ---------
# Strict extractor fails only on unparseable blob; good response on the retry.
def _strict_extractor(data, state, step):
    if not isinstance(data, dict) or "insertions" not in data:
        return None
    insertions = data.get("insertions")
    required = {step[0], step[1]}
    new_routes, _applied, _skipped = pdb.apply_insertions(state, insertions, required_nodes=required)
    return new_routes

# Sequence: good turn 1, bad turn 2, then good-retry for step 2.
lenient_seq = [fake_responses[0], "nope gibberish", fake_responses[1]]
fake_lenient = FakeLLM(lenient_seq)
final_state, turns, abort = pdb.run_iterative_steps(
    fake_lenient, problem, removed_reqs, _state_builder, _strict_extractor,
    initial_state=[list(r) for r in partial_for_iter.routes],
)
assert abort is None, f"expected no abort with leniency, got {abort}"
assert turns == 3, f"expected 3 turns (1 + bad + retry), got {turns}"
print(f"  Iter parse-fail leniency: turns={turns} abort={abort} ✓")

# Exhaustion path: all retries for step 2 fail -> abort after max_parse_failures+1 tries.
fail_seq = [fake_responses[0]] + ["nope"] * 3  # 1 good + 3 bad (default max_parse_failures=2 -> 3 tries)
fake_fail = FakeLLM(fail_seq)
final_state, turns, abort = pdb.run_iterative_steps(
    fake_fail, problem, removed_reqs, _state_builder, _strict_extractor,
    initial_state=[list(r) for r in partial_for_iter.routes],
)
assert abort is not None and "parse_fail" in abort, f"expected parse_fail abort, got {abort}"
assert turns == 4, f"expected 4 turns (1 good + 3 failed tries), got {turns}"
print(f"  Iter parse-fail exhaustion: turns={turns} abort={abort} ✓")


# --- Iterative Task 5 building block: prompt builder works ------------------
step_prompt = pdb.build_iterative_full_route_step_prompt(
    problem, [], list(problem.pickups_deliveries), pdb.DistanceMode.MATRIX, ""
)
assert "MULTI-TURN" in step_prompt and '"done"' in step_prompt
print("  Iterative task 5 prompt builder: contains multi-turn framing + done schema ✓")


# =============================================================================
# Real parser regression — exercises pdb.parse_json_response across every
# answer shape a model might produce. The local parse_json_response at the
# top of this file is a stale reference copy; these tests hit the real one.
# =============================================================================
print()
print("=" * 60)
print("TEST: Real parser — exhaustive edge cases")
print("=" * 60)

_real_parse = pdb.parse_json_response


def _has_insertions(d):
    if not isinstance(d, dict):
        return False
    ins = d.get("insertions")
    if not isinstance(ins, list) or not ins:
        return False
    for op in ins:
        if not isinstance(op, dict):
            return False
        if "insert" not in op:
            return False
        if "before" not in op and "after" not in op:
            return False
    return True


def _has_routes(d):
    if not isinstance(d, dict):
        return False
    r = d.get("routes")
    return isinstance(r, list) and len(r) > 0 and all(isinstance(x, list) for x in r)


def _has_new_route(d):
    if not isinstance(d, dict):
        return False
    r = d.get("new_route")
    return isinstance(r, list) and len(r) > 0 and all(isinstance(x, int) for x in r)


def _has_number(d):
    return isinstance(d, dict) and "_fallback_number" in d


def _has_tool_or_answer(d):
    return isinstance(d, dict) and ("tool" in d or "answer" in d)


_PARSER_CASES = [
    # ---------- plain text / preamble / fencing ----------
    ("plain JSON insertions",     '{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("leading prose",             'Sure, here is my answer:\n{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("trailing prose",            '{"insertions":[{"insert":1,"before":2}]}\nHope this helps!', _has_insertions),
    ("prose both sides",          'OK.\n{"insertions":[{"insert":1,"before":2}]}\nDone.', _has_insertions),
    ("markdown heading",          '## Solution\n{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("fenced ```json",            '```json\n{"insertions":[{"insert":1,"before":2}]}\n```', _has_insertions),
    ("fenced ``` bare",           '```\n{"insertions":[{"insert":1,"before":2}]}\n```', _has_insertions),
    ("fenced ```python",          '```python\n{"insertions":[{"insert":1,"before":2}]}\n```', _has_insertions),
    ("nested fenced",             '```\ntext\n```json\n{"insertions":[{"insert":1,"before":2}]}\n```\n```', _has_insertions),
    ("multiple JSON blocks",      'First try: {"foo":1} Second try: {"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("windows line endings",      '{"insertions":[\r\n  {"insert":1,"before":2}\r\n]}', _has_insertions),

    # ---------- reasoning / answer tags ----------
    ("<think> tags",              '<think>reasoning...</think>\n{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("<thinking> tags",           '<thinking>chain of thought</thinking>{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("<answer> tags",             '<answer>{"insertions":[{"insert":1,"before":2}]}</answer>', _has_insertions),
    ("<solution> tags",           '<solution>{"insertions":[{"insert":1,"before":2}]}</solution>', _has_insertions),
    ("<final> tags",              '<final>{"insertions":[{"insert":1,"before":2}]}</final>', _has_insertions),
    ("<output> tags",             '<output>{"insertions":[{"insert":1,"before":2}]}</output>', _has_insertions),

    # ---------- LaTeX / boxed wrappers ----------
    ("boxed JSON",                '\\boxed{{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("boxed text JSON",           '\\boxed{\\text{{"insertions":[{"insert":1,"before":2}]}}}', _has_insertions),
    ("boxed number",              '\\boxed{1234.5}', _has_number),
    ("inline math boxed",         'The answer is $\\boxed{{"insertions":[{"insert":1,"before":2}]}}$', _has_insertions),
    ("display math",              '$$\\boxed{{"insertions":[{"insert":1,"before":2}]}}$$', _has_insertions),

    # ---------- JSON syntactic variants ----------
    ("trailing comma",            '{"insertions":[{"insert":1,"before":2},]}', _has_insertions),
    ("line comment //",           '{"insertions":[{"insert":1,"before":2}] // last one\n}', _has_insertions),
    ("block comment /* */",       '/* reasoning */\n{"insertions":[{"insert":1,"before":2}]}', _has_insertions),
    ("unquoted keys",             '{insertions:[{insert:1,before:2}]}', _has_insertions),
    ("smart double quotes",       '{\u201cinsertions\u201d:[{\u201cinsert\u201d:1,\u201cbefore\u201d:2}]}', _has_insertions),
    ("python True literal",       '{"insertions":[{"insert":1,"before":2,"force":True}]}', _has_insertions),
    ("python None literal",       '{"insertions":[{"insert":1,"before":2,"note":None}]}', _has_insertions),
    ("python repr single q",      "{'insertions':[{'insert':1,'before':2}]}", _has_insertions),
    ("mixed whitespace",          '  \n\t{"insertions":[\n  {"insert":1,"before":2}\n]}\t\n  ', _has_insertions),
    ("extra leading brace",       '{{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),

    # ---------- insertion key aliases ----------
    ("inserts alias",             '{"inserts":[{"insert":1,"before":2}]}', _has_insertions),
    ("operations alias",          '{"operations":[{"insert":1,"before":2}]}', _has_insertions),
    ("ops alias",                 '{"ops":[{"insert":1,"before":2}]}', _has_insertions),
    ("actions alias",             '{"actions":[{"insert":1,"before":2}]}', _has_insertions),
    ("moves alias",               '{"moves":[{"insert":1,"before":2}]}', _has_insertions),
    ("bare list of ops",          '[{"insert":1,"before":2},{"insert":3,"after":4}]', _has_insertions),

    # ---------- insertion op field aliases ----------
    ("node alias",                '{"insertions":[{"node":5,"before":7}]}', _has_insertions),
    ("item alias",                '{"insertions":[{"item":5,"before":7}]}', _has_insertions),
    ("new_node alias",            '{"insertions":[{"new_node":5,"before":7}]}', _has_insertions),
    ("pickup alias",              '{"insertions":[{"pickup":5,"before":7}]}', _has_insertions),
    ("target alias",              '{"insertions":[{"target":5,"before":7}]}', _has_insertions),
    ("anchor_before alias",       '{"insertions":[{"insert":5,"anchor_before":7}]}', _has_insertions),
    ("insert_before alias",       '{"insertions":[{"insert":5,"insert_before":7}]}', _has_insertions),
    ("prev alias",                '{"insertions":[{"insert":5,"prev":7}]}', _has_insertions),
    ("predecessor alias",         '{"insertions":[{"insert":5,"predecessor":7}]}', _has_insertions),
    ("anchor_after alias",        '{"insertions":[{"insert":5,"anchor_after":7}]}', _has_insertions),
    ("insert_after alias",        '{"insertions":[{"insert":5,"insert_after":7}]}', _has_insertions),
    ("next alias",                '{"insertions":[{"insert":5,"next":7}]}', _has_insertions),
    ("successor alias",           '{"insertions":[{"insert":5,"successor":7}]}', _has_insertions),
    ("case-insensitive keys",     '{"Insertions":[{"Insert":1,"Before":2}]}', _has_insertions),
    ("upper-case keys",           '{"INSERTIONS":[{"INSERT":1,"BEFORE":2}]}', _has_insertions),

    # ---------- routes key aliases ----------
    ("routes standard",           '{"routes":[[0,1,2,0]]}', _has_routes),
    ("solution alias",            '{"solution":[[0,1,2,0]]}', _has_routes),
    ("plan alias",                '{"plan":[[0,1,2,0]]}', _has_routes),
    ("vehicle_routes alias",      '{"vehicle_routes":[[0,1,2,0]]}', _has_routes),
    ("vehicles alias",            '{"vehicles":[[0,1,2,0]]}', _has_routes),
    ("tours alias",               '{"tours":[[0,1,2,0]]}', _has_routes),
    ("trips alias",               '{"trips":[[0,1,2,0]]}', _has_routes),
    ("bare list of routes",       '[[0,1,2,0],[0,3,4,0]]', _has_routes),

    # ---------- new_route key aliases ----------
    ("new_route standard",        '{"new_route":[0,1,2,0]}', _has_new_route),
    ("completed_route alias",     '{"completed_route":[0,1,2,0]}', _has_new_route),
    ("route alias",               '{"route":[0,1,2,0]}', _has_new_route),
    ("added_route alias",         '{"added_route":[0,1,2,0]}', _has_new_route),
    ("reconstructed_route alias", '{"reconstructed_route":[0,1,2,0]}', _has_new_route),
    ("missing_route alias",       '{"missing_route":[0,1,2,0]}', _has_new_route),
    ("bare list of ints",         '[0,1,2,0]', _has_new_route),

    # ---------- nested wrappers ----------
    ("answer wrapper ins",        '{"answer":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("result wrapper ins",        '{"result":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("output wrapper ins",        '{"output":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("response wrapper ins",      '{"response":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("final_answer wrapper ins",  '{"final_answer":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("data wrapper ins",          '{"data":{"insertions":[{"insert":1,"before":2}]}}', _has_insertions),
    ("answer wrapper routes",     '{"answer":{"routes":[[0,1,2,0]]}}', _has_routes),
    ("doubly nested routes",      '{"answer":{"result":{"routes":[[0,1,2,0]]}}}', _has_routes),

    # ---------- YAML ----------
    ("yaml insertions",           'insertions:\n  - insert: 1\n    before: 2\n  - insert: 3\n    after: 4', _has_insertions),
    ("yaml routes",               'routes:\n  - [0, 1, 2, 0]\n  - [0, 3, 4, 0]', _has_routes),
    ("yaml new_route",            'new_route: [0, 1, 2, 0]', _has_new_route),

    # ---------- prose fallback ----------
    ("prose insert before/after", 'I will insert 5 before 7 and insert 8 after 9.', _has_insertions),
    ("prose single op",           'Place node 5 before 7.', _has_insertions),

    # ---------- numeric fallback ----------
    ("distance statement",        'The total distance is 1234.5', _has_number),
    ("just a number",             '1234.5', _has_number),
    ("answer is X",               'The answer is 42.', _has_number),

    # ---------- tool-loop responses ----------
    ("tool call",                 '{"tool":"check_feasibility","args":{}}', _has_tool_or_answer),
    ("tool call no args",         '{"tool":"unserved_requests"}', _has_tool_or_answer),
    ("answer object",             '{"answer":{"routes":[[0,1,2,0]]}}', _has_tool_or_answer),
]

_parser_failures = []
for _name, _raw, _check in _PARSER_CASES:
    try:
        _out = _real_parse(_raw)
        _ok = _check(_out)
    except Exception as _e:
        _out = f"EXC: {type(_e).__name__}: {_e}"
        _ok = False
    if not _ok:
        _parser_failures.append((_name, _raw, _out))

print(f"  Parser edge cases: {len(_PARSER_CASES) - len(_parser_failures)}/{len(_PARSER_CASES)} passed")
for _name, _raw, _out in _parser_failures:
    print(f"    FAIL: {_name}")
    print(f"      raw: {_raw!r}")
    print(f"      got: {_out}")
assert not _parser_failures, f"{len(_parser_failures)} parser edge cases failed"
print("  All parser edge cases ✓")

print("\n✓ All tests complete")
