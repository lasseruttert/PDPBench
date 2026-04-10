"""
PDPBench — Task 4: Request Insertion (iterative)
================================================
Insert 2 removed requests one at a time over multiple LLM turns.
One request (pickup + delivery) per turn using insert-before/after ops.

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

LOG_PATH = "/kaggle/working/pdpbench_T4_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_T4_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

INSTANCES = get_benchmark_instances()
print(f"Loaded {len(INSTANCES)} instances")

RESULTS_PATH = "/kaggle/working/results_T4.json" if KAGGLE else os.path.join(BASE_DIR, "results_T4.json")
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
    print(f"  SCORE: 0.4 * coverage + 0.3 * feasibility + 0.3 * distance_gap")
    print(f"{'='*60}")

# %%
if HAS_KBENCH:

    @kbench.task(name="pdptw_request_insertion_iterative")
    def pdptw_request_insertion_iterative(llm) -> float:
        """Insert removed requests one at a time over multiple turns (one request per turn)."""
        TASK_ID = "T1 iter"
        TASK_TITLE = "Request Insertion (iterative)"
        scores = []
        details = []
        start = time.time()
        print_task_header(TASK_ID, TASK_TITLE, len(INSTANCES), distance_mode_str="auto")
        for i, (problem, bks) in enumerate(INSTANCES):
            try:
                if i > 0:
                    time.sleep(10)
                partial = bks.clone()
                removed = []
                for pickup_idx, delivery_idx in problem.pickups_deliveries[:2]:
                    partial.remove_request(problem, pickup_idx)
                    removed.append((pickup_idx, delivery_idx))

                total_applied = [0]
                total_skipped = []

                iter_mode = iterative_distance_mode(problem)
                def state_builder(step, state, history_text, _problem=problem, _mode=iter_mode):
                    return build_iterative_insertion_step_prompt(_problem, state, step, _mode, history_text)

                def response_extractor(data, state, step, _totals_applied=total_applied, _totals_skipped=total_skipped):
                    if not isinstance(data, dict):
                        return None
                    insertions = data.get("insertions")
                    pickup_idx, delivery_idx = step
                    required = {pickup_idx, delivery_idx}
                    new_routes, applied, skipped = apply_insertions(state, insertions, required_nodes=required)
                    _totals_applied[0] += applied
                    _totals_skipped.extend(skipped)
                    return new_routes  # may be unchanged if all ops were skipped; driver continues

                final_state, turns_used, abort = run_iterative_steps(
                    llm, problem, removed, state_builder, response_extractor, initial_state=[list(r) for r in partial.routes]
                )
                solution = build_solution_from_llm_output(problem, final_state) if final_state is not None else None
                components = compute_score(problem, solution, bks.total_distance) if abort is None else {"coverage": 0.0, "feasibility": 0.0, "distance_gap": 0.0, "score": 0.0}
                score = components["score"]

                d = {"instance": problem.name, "removed": removed, "bks_dist": round(bks.total_distance, 1),
                     "turns": turns_used, "max_turns": len(removed),
                     "insertions_applied": total_applied[0], "insertions_expected": 2 * len(removed),
                     "skipped": total_skipped[:5]}
                if abort is not None:
                    d.update(result="ABORT", abort=abort, **{k: round(v, 3) for k, v in components.items()})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | turns={turns_used}/{len(removed)} | ABORT({abort}) | score={score:.3f}")
                elif solution is None:
                    d.update(result="BUILD_FAIL", **{k: round(v, 3) for k, v in components.items()})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | turns={turns_used}/{len(removed)} | BUILD_FAIL | score=0.0")
                else:
                    d["llm_dist"] = round(solution.total_distance, 1)
                    result = "FEASIBLE" if components["feasibility"] == 1.0 else "INFEASIBLE"
                    d.update(result=result, **{k: round(v, 3) for k, v in components.items()})
                    if result == "INFEASIBLE":
                        d["violations"] = count_violations(problem, solution)
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | turns={turns_used}/{len(removed)} | INFEASIBLE | cov={components['coverage']:.2f} llm_dist={solution.total_distance:.1f} | violations: {format_violations(d['violations'])} | score={score:.3f}")
                    else:
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | turns={turns_used}/{len(removed)} | FEASIBLE | cov={components['coverage']:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={components['distance_gap']:.3f} | score={score:.3f}")

                scores.append(score)
                details.append(d)

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
        RESULTS["tasks"]["request_insertion_iterative"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results(RESULTS, RESULTS_PATH)
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")

# %%
if HAS_KBENCH:
    pdptw_request_insertion_iterative.run(llm=kbench.llm)
# %choose pdptw_request_insertion_iterative
