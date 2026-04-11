"""
PDPBench — Task 1: Request Insertion (one-shot)
===============================================
Insert 2 removed pickup/delivery pairs back into a PDPTW solution using
insert-before/after operations. Single LLM call per instance.

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

LOG_PATH = "/kaggle/working/pdpbench_T1_log.txt" if KAGGLE else os.path.join(BASE_DIR, "pdpbench_T1_log.txt")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)

INSTANCES = get_benchmark_instances_20()
print(f"Loaded {len(INSTANCES)} instances")

RESULTS_PATH = "/kaggle/working/results_T1.json" if KAGGLE else os.path.join(BASE_DIR, "results_T1.json")
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

    @kbench.task(name="pdptw_request_insertion")
    def pdptw_request_insertion(llm) -> float:
        """Insert removed requests back into a PDPTW solution via minimal insert-before/after ops."""
        TASK_ID = "T1 1s"
        TASK_TITLE = "Request Insertion (one-shot)"
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
                required = set()
                for pickup_idx, delivery_idx in problem.pickups_deliveries[:2]:
                    partial.remove_request(problem, pickup_idx)
                    removed.append((pickup_idx, delivery_idx))
                    required.add(pickup_idx)
                    required.add(delivery_idx)
                prompt = build_request_insertion_prompt(problem, partial, removed, iterative_distance_mode(problem))
                raw = llm_prompt(llm, prompt)
                data = parse_json_response(raw)

                parse_note = ""
                if not isinstance(data, dict):
                    parse_note = f"PARSE_FAIL(not_dict:{type(data).__name__})"
                    insertions = None
                elif not isinstance(data.get("insertions"), list):
                    keys = list(data.keys())[:5] if isinstance(data, dict) else []
                    parse_note = f"PARSE_FAIL(no_insertions_key,got_keys={keys})"
                    insertions = None
                else:
                    insertions = data["insertions"]

                new_routes, applied, skipped = apply_insertions(partial.routes, insertions, required_nodes=required)
                if not parse_note and applied == 0:
                    first = skipped[0] if skipped else "empty_insertions_list"
                    parse_note = f"APPLY_FAIL({first})"

                completion = score_completion_t1(new_routes, removed)
                solution = build_solution_from_llm_output(problem, new_routes)
                components = compute_score(problem, solution, bks.total_distance, completion)
                score = components["score"]

                d = {"instance": problem.name, "removed": removed, "bks_dist": round(bks.total_distance, 1),
                     "insertions_applied": applied, "insertions_expected": len(required),
                     "skipped": skipped[:5], "parse_note": parse_note,
                     "raw_preview": str(raw)[:300] if parse_note else ""}
                note_suffix = f" | {parse_note}" if parse_note else ""
                if solution is None:
                    d.update(result="BUILD_FAIL", **{k: round(v, 3) for k, v in components.items()})
                    print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | BUILD_FAIL applied={applied}/{len(required)}{note_suffix} | compl={completion:.2f} | score={score:.3f}")
                else:
                    d["llm_dist"] = round(solution.total_distance, 1)
                    result = "FEASIBLE" if components["feasibility"] == 1.0 else "INFEASIBLE"
                    d.update(result=result, **{k: round(v, 3) for k, v in components.items()})
                    if result == "INFEASIBLE":
                        d["violations"] = count_violations(problem, solution)
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | INFEASIBLE applied={applied}/{len(required)} | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} | violations: {format_violations(d['violations'])}{note_suffix} | score={score:.3f}")
                    else:
                        print(f"  [{TASK_ID} {i+1}/{len(INSTANCES)}] {problem.name} | removed={removed} | FEASIBLE applied={applied}/{len(required)} | compl={completion:.2f} llm_dist={solution.total_distance:.1f} bks={bks.total_distance:.1f} gap={components['distance_gap']:.3f}{note_suffix} | score={score:.3f}")

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
        RESULTS["tasks"]["request_insertion"] = {"score": avg, "time_s": round(elapsed, 1), "instances": details}
        save_results(RESULTS, RESULTS_PATH)
        return avg

else:
    print("kbench not available - run this notebook on Kaggle to execute the benchmark")

# %%
if HAS_KBENCH:
    pdptw_request_insertion.run(llm=kbench.llm)
# %choose pdptw_request_insertion
