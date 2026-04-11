"""
PDPBench — Task 6: Full Solution (iterative)
============================================
Generate a complete PDPTW solution route-by-route over multiple turns.
Each turn: LLM adds one new vehicle route. Capped at min(num_vehicles, 10) turns.

Kaggle cell layout (# %% = cell boundary):
  Cell 1: setup + task definition
  Cell 2: task.run() + %choose
"""

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "kaggle-benchmarks", "pyparsing", "-q"], check=True)

import os

KAGGLE = os.path.exists("/kaggle")
if KAGGLE:
    _CANDIDATES = [
        "/kaggle/input/pdpbench-data",
        "/kaggle/input/datasets/lasseruttert/pdpbench-data",
    ]
    _base = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
    if _base not in sys.path:
        sys.path.insert(0, _base)

from pdpbench_lib import *

try:
    import kaggle_benchmarks as kbench
    HAS_KBENCH = hasattr(kbench, "llm")
except (RuntimeError, ImportError):
    kbench = None
    HAS_KBENCH = False

print(f"kbench available: {HAS_KBENCH}")

LOG_PATH = "/kaggle/working/pdpbench_T6_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_T6_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

INSTANCES = get_benchmark_instances()
print(f"Loaded {len(INSTANCES)} instances")

RESULTS_PATH = "/kaggle/working/results_T6.json" if KAGGLE else os.path.join(BASE_DIR, "results_T6.json")
RESULTS = {"model": "unknown", "tasks": {}}

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
    print(f"  SCORE: (1/3) * completion + (1/3) * feasibility + (1/3) * distance_gap")
    print(f"{'='*60}")

# %%
if HAS_KBENCH:

    @kbench.task(name="pdptw_full_solution_iterative")
    def pdptw_full_solution_iterative(llm) -> float:
        """Generate a complete solution route-by-route over multiple turns."""
        TASK_ID = "T3 iter"
        TASK_TITLE = "Full Solution (iterative)"
        scores = []
        details = []
        start = time.time()
        print_task_header(TASK_ID, TASK_TITLE, len(INSTANCES), distance_mode_str="auto")
        for i, (problem, bks) in enumerate(INSTANCES):
            try:
                if i > 0:
                    time.sleep(10)
                all_requests = list(problem.pickups_deliveries)
                unserved = list(all_requests)
                completed_routes = []
                # Cap at 10 to keep per-instance wall-clock tractable. 25-vehicle
                # instances like lc101 otherwise compound with retries and the
                # matrix-on-every-turn problem.
                max_turns = min(problem.num_vehicles, 10)
                turns_used = 0
                abort = None
                history = []
                iter_mode = iterative_distance_mode(problem)
                MAX_PARSE_RETRIES = 2

                for turn_idx in range(max_turns):
                    if not unserved:
                        break
                    step_done = False
                    for parse_attempt in range(MAX_PARSE_RETRIES + 1):
                        history_text = format_conversation_history(history)
                        prompt = build_iterative_full_route_step_prompt(
                            problem, completed_routes, unserved, iter_mode, history_text
                        )
                        raw = llm_prompt(llm, prompt)
                        turns_used += 1
                        data = parse_json_response(raw)
                        if not isinstance(data, dict):
                            history.append({"assistant": raw,
                                            "system": "Your last response could not be parsed as JSON. "
                                                      "Respond with EXACTLY the JSON schema shown."})
                            continue
                        if data.get("done") is True:
                            history.append({"assistant": raw})
                            step_done = True
                            abort = "__done__"
                            break
                        route_raw = data.get("route") or data.get("new_route")
                        if route_raw is None:
                            history.append({"assistant": raw,
                                            "system": 'Your last response had no "route" field. '
                                                      'Respond with {"route": [0, ..., 0]} or {"done": true}.'})
                            continue
                        try:
                            route = [int(n) for n in route_raw]
                        except (ValueError, TypeError):
                            history.append({"assistant": raw,
                                            "system": "Your last route contained non-integer values. "
                                                      "Respond with a list of integer node indices."})
                            continue
                        if not route or route[0] != 0:
                            route = [0] + route
                        if route[-1] != 0:
                            route.append(0)
                        completed_routes.append(route)
                        visited = set(route[1:-1])
                        unserved = [(p, d) for (p, d) in unserved if p not in visited and d not in visited]
                        history.append({"assistant": raw})
                        step_done = True
                        break
                    if abort == "__done__":
                        abort = None
                        break
                    if not step_done:
                        abort = f"parse_fail_turn_{turn_idx+1}"
                        break

                d = {"instance": problem.name, "bks_dist": round(bks.total_distance, 1),
                     "turns": turns_used, "max_turns": max_turns, "unserved": len(unserved)}

                completion = score_completion_t3(completed_routes, problem.pickups_deliveries)

                if abort is not None:
                    zero_score = completion / 3.0
                    zero = {"completion": round(completion, 3), "feasibility": 0.0, "distance_gap": 0.0, "score": round(zero_score, 3)}
                    scores.append(zero_score)
                    details.append({**d, "result": "ABORT", "abort": abort, **zero})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | turns={turns_used}/{max_turns} unserved={len(unserved)} | ABORT({abort}) | compl={completion:.2f} | score={zero_score:.3f}")
                    continue

                solution = build_solution_from_llm_output(problem, completed_routes)
                components = compute_score(problem, solution, bks.total_distance, completion)
                score = components["score"]
                if solution is None:
                    scores.append(score)
                    details.append({**d, "result": "BUILD_FAIL", **{k: round(v, 3) for k, v in components.items()}})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | turns={turns_used}/{max_turns} | BUILD_FAIL | compl={completion:.2f} | score={score:.3f}")
                    continue

                d["llm_dist"] = round(solution.total_distance, 1)
                result = "FEASIBLE" if components["feasibility"] == 1.0 else "INFEASIBLE"
                d.update(result=result, **{k: round(v, 3) for k, v in components.items()})
                if result == "INFEASIBLE":
                    d["violations"] = count_violations(problem, solution)
                    details.append(d)
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | turns={turns_used}/{max_turns} | INFEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} | violations: {format_violations(d['violations'])} | score={score:.3f}")
                else:
                    details.append(d)
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | turns={turns_used}/{max_turns} | FEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={components['distance_gap']:.3f} | score={score:.3f}")
                scores.append(score)

            except Exception as _e:
                _err = f"{type(_e).__name__}: {_e}"
                print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | ERROR: {_err}")
                scores.append(0.0)
                details.append({"instance": problem.name, "result": "ERROR", "error": _err})
                try:
                    save_results(RESULTS, RESULTS_PATH)
                except Exception:
                    pass
                continue
        avg = sum(scores) / len(scores)
        elapsed = time.time() - start
        print_task_footer(TASK_ID, TASK_TITLE, avg, scores, elapsed)
        RESULTS["tasks"]["full_solution_iterative"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results(RESULTS, RESULTS_PATH)
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")

# %%
if HAS_KBENCH:
    pdptw_full_solution_iterative.run(llm=kbench.llm)
# %choose pdptw_full_solution_iterative
