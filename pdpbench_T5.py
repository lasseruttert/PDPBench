"""
PDPBench — Task 5: Route Completion (iterative)
===============================================
Reconstruct a missing route by inserting its requests one at a time.
Each turn: LLM receives the current partial new_route and one request to add.

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

LOG_PATH = "/kaggle/working/pdpbench_T5_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_T5_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

INSTANCES = get_benchmark_instances()
print(f"Loaded {len(INSTANCES)} instances")

RESULTS_PATH = "/kaggle/working/results_T5.json" if KAGGLE else os.path.join(BASE_DIR, "results_T5.json")
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

    @kbench.task(name="pdptw_route_completion_iterative")
    def pdptw_route_completion_iterative(llm) -> float:
        """Reconstruct a missing route by inserting its requests one at a time."""
        TASK_ID = "T2 iter"
        TASK_TITLE = "Route Completion (iterative)"
        scores = []
        details = []
        start = time.time()
        print_task_header(TASK_ID, TASK_TITLE, len(INSTANCES), distance_mode_str="auto")
        for i, (problem, bks) in enumerate(INSTANCES):
            try:
                if i > 0:
                    time.sleep(10)
                longest_idx = max(range(len(bks.routes)), key=lambda r: len(bks.routes[r]))
                removed_route = bks.routes[longest_idx]
                removed_customers = set(removed_route[1:-1])
                removed_requests = [(p, d) for p, d in problem.pickups_deliveries if p in removed_customers]
                partial_routes = [r[:] for idx, r in enumerate(bks.routes) if idx != longest_idx]
                partial = PDPTWSolution(problem=problem, routes=partial_routes)

                iter_mode = iterative_distance_mode(problem)
                def state_builder(step, state, history_text, _problem=problem, _partial=partial, _mode=iter_mode):
                    return build_iterative_route_build_step_prompt(
                        _problem, _partial.routes, state, step, _mode, history_text
                    )

                def response_extractor(data, state, step):
                    if not isinstance(data, dict):
                        return None
                    new_route = data.get("new_route") or data.get("route")
                    if new_route is None and "routes" in data and isinstance(data["routes"], list) and data["routes"]:
                        new_route = data["routes"][0]
                    if not isinstance(new_route, list):
                        return None
                    try:
                        nr = [int(n) for n in new_route]
                    except (ValueError, TypeError):
                        return None
                    if not nr or nr[0] != 0:
                        nr = [0] + nr
                    if nr[-1] != 0:
                        nr.append(0)
                    return nr

                initial_new_route = [0, 0]
                final_route, turns_used, abort = run_iterative_steps(
                    llm, problem, removed_requests, state_builder, response_extractor, initial_state=initial_new_route
                )
                d = {"instance": problem.name, "n_requests": len(removed_requests),
                     "bks_dist": round(bks.total_distance, 1), "turns": turns_used, "max_turns": len(removed_requests)}

                completion = score_completion_t2(final_route or [], removed_customers)

                if abort is not None or not final_route:
                    zero_score = completion / 3.0
                    zero = {"completion": round(completion, 3), "feasibility": 0.0, "distance_gap": 0.0, "score": round(zero_score, 3)}
                    scores.append(zero_score)
                    details.append({**d, "result": "ABORT", "abort": abort, **zero})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | turns={turns_used}/{len(removed_requests)} | ABORT({abort}) | compl={completion:.2f} | score={zero_score:.3f}")
                    continue

                full_routes = [r[:] for r in partial.routes] + [final_route]
                solution = build_solution_from_llm_output(problem, full_routes)
                components = compute_score(problem, solution, bks.total_distance, completion)
                score = components["score"]
                if solution is None:
                    scores.append(score)
                    details.append({**d, "result": "BUILD_FAIL", **{k: round(v, 3) for k, v in components.items()}})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | turns={turns_used}/{len(removed_requests)} | BUILD_FAIL | compl={completion:.2f} | score={score:.3f}")
                    continue

                d["llm_dist"] = round(solution.total_distance, 1)
                result = "FEASIBLE" if components["feasibility"] == 1.0 else "INFEASIBLE"
                d.update(result=result, **{k: round(v, 3) for k, v in components.items()})
                if result == "INFEASIBLE":
                    d["violations"] = count_violations(problem, solution)
                    details.append(d)
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | turns={turns_used}/{len(removed_requests)} | INFEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} | violations: {format_violations(d['violations'])} | score={score:.3f}")
                else:
                    details.append(d)
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | requests={len(removed_requests)} | turns={turns_used}/{len(removed_requests)} | FEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={components['distance_gap']:.3f} | score={score:.3f}")
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
        RESULTS["tasks"]["route_completion_iterative"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results(RESULTS, RESULTS_PATH)
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")

# %%
if HAS_KBENCH:
    pdptw_route_completion_iterative.run(llm=kbench.llm)
# %choose pdptw_route_completion_iterative
