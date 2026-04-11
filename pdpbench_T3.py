"""
PDPBench — Task 3: Full Solution (one-shot)
===========================================
Generate a complete PDPTW solution from scratch in a single LLM call.
Most challenging task: LLM must produce all routes covering all requests.

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

LOG_PATH = "/kaggle/working/pdpbench_T3_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_T3_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

INSTANCES = get_benchmark_instances_20()
print(f"Loaded {len(INSTANCES)} instances")

RESULTS_PATH = "/kaggle/working/results_T3.json" if KAGGLE else os.path.join(BASE_DIR, "results_T3.json")
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

if HAS_KBENCH:

    @kbench.task(name="pdptw_full_solution")
    def pdptw_full_solution(llm) -> float:
        """Generate a complete feasible PDPTW solution from scratch."""
        TASK_ID = "T3 1s"
        TASK_TITLE = "Full Solution (one-shot)"
        scores = []
        details = []
        start = time.time()
        print_task_header(TASK_ID, TASK_TITLE, len(INSTANCES), distance_mode_str="auto")
        for i, (problem, bks) in enumerate(INSTANCES):
            try:
                if i > 0:
                    time.sleep(10)
                prompt = build_full_solution_prompt(problem, iterative_distance_mode(problem))
                raw = llm_prompt(llm, prompt)
                data = parse_json_response(raw)

                routes_raw = data.get("routes", None)
                completion = score_completion_t3(routes_raw or [], problem.pickups_deliveries)
                solution = build_solution_from_llm_output(problem, routes_raw) if routes_raw else None
                components = compute_score(problem, solution, bks.total_distance, completion)
                score = components["score"]
                d = {"instance": problem.name, "nodes": len(problem.nodes), "vehicles": problem.num_vehicles, "bks_dist": round(bks.total_distance, 1)}

                if solution is None:
                    d.update(result="PARSE_FAIL", **{k: round(v, 3) for k, v in components.items()})
                    details.append(d)
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} vehicles={problem.num_vehicles} | PARSE_FAIL | compl={completion:.2f} | score={score:.3f}")
                else:
                    n_served = sum(len(r) - 2 for r in solution.routes if len(r) > 2)
                    n_expected = len(problem.nodes) - 1
                    d.update(llm_dist=round(solution.total_distance, 1), served=n_served, expected=n_expected, routes=len(solution.routes))
                    result = "FEASIBLE" if components["feasibility"] == 1.0 else "INFEASIBLE"
                    d.update(result=result, **{k: round(v, 3) for k, v in components.items()})
                    if result == "INFEASIBLE":
                        d["violations"] = count_violations(problem, solution)
                        details.append(d)
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} | INFEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} served={n_served}/{n_expected} routes={len(solution.routes)} | violations: {format_violations(d['violations'])} | score={score:.3f}")
                    else:
                        details.append(d)
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | nodes={len(problem.nodes)} | FEASIBLE | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={components['distance_gap']:.3f} served={n_served}/{n_expected} | score={score:.3f}")

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
        RESULTS["tasks"]["full_solution"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results(RESULTS, RESULTS_PATH)
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")

# %%
if HAS_KBENCH:
    pdptw_full_solution.run(llm=kbench.llm)
# %choose pdptw_full_solution
