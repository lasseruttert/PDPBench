# PDPBench — Claude Code Instructions

## What this is

PDPBench is a benchmark for the **Google DeepMind / Kaggle "Benchmarks for AGI" hackathon** (deadline: 2026-04-16). It tests LLM understanding of the Pickup and Delivery Problem with Time Windows (PDPTW) across 5 tasks of increasing difficulty. Track: **Executive Functions & Attention**.

## Repository structure

```
PDPBench/
  pdpbench.py              # Main benchmark (self-contained, runs on Kaggle Benchmarks)
  save.py                  # Packages files into kaggle_dataset/ for upload
  test_scoring.py          # Local scoring tests with fake LLM responses
  environment.yml          # Conda env definition
  data/                    # PDPTW instance files (Li & Lim + Mendeley datasets)
  bks/                     # Best Known Solutions for each instance
  utils/                   # Helper modules (problem reader, solution class, feasibility checker)
  kaggle_dataset/
    pdpbench-data/         # Packaged dataset uploaded to Kaggle as lasseruttert/pdpbench-data
      dataset-metadata.json
      pdpbench.py          # Copy of main benchmark
      data/, bks/, utils/  # Copies of data directories
```

## How pdpbench.py works

The file is fully self-contained (no imports from local modules — only from `utils/` in the Kaggle dataset and `kaggle_benchmarks` SDK).

### Architecture
1. **Imports + path setup**: Auto-detects Kaggle vs local. Tries two possible dataset paths on Kaggle.
2. **Scoring functions**: `normalize_routes()`, `build_solution_from_llm_output()`, `score_feasibility()`, `score_distance_gap()`, `score_distance_prediction()`
3. **LLM call wrapper**: `llm_prompt()` retries on 503 errors (Kaggle model proxy can be flaky)
4. **JSON parser**: `parse_json_response()` extracts JSON from LLM output with regex fallback
5. **Prompt builder**: Constructs PDPTW prompts with rules, problem data as JSON, and task-specific instructions. Each prompt ends with "Respond with EXACTLY this JSON: {...}"
6. **Instance selection**: 8 instances — 5 Li & Lim (`lc101`, `lc201`, `lr101`, `lr201`, `lrc101`) + 3 Mendeley (`bar-n100-1`, `ber-n100-1`, `nyc-n100-1`)
7. **5 task functions**: Each decorated with `@kbench.task()`, loops over all instances, calls LLM, parses response, scores, returns average `float`
8. **Execution**: `task.run(llm=kbench.llm)` + `# %choose task_name` comments to register with platform

### The 5 tasks
| # | Task | What LLM does | Scoring |
|---|------|---------------|---------|
| 1 | Masked Node | Predict a masked node in a BKS route | 1.0 exact, 0.5 feasible alt, 0.0 wrong |
| 2 | Request Insertion | Re-insert 2 removed requests into partial solution | 0.5*feasibility + 0.5*distance_gap |
| 3 | Distance Prediction | Calculate total distance of a given solution | 1 - |predicted - actual| / actual |
| 4 | Route Completion | Complete a truncated route given remaining nodes | 0.5*feasibility + 0.5*distance_gap |
| 5 | Full Solution | Generate a complete feasible solution from scratch | 0.5*feasibility + 0.5*distance_gap |

### Distance mode
Currently `DISTANCE_MODE = DistanceMode.MATRIX` (full distance matrix in prompt). Other modes exist (`COORDINATES`, `TOOL_USE`) but MATRIX is the default.

## Conda environment

```bash
conda env create -f environment.yml
# or
conda create -n PDPBench python=3.11 pip && conda activate PDPBench && pip install numpy pyparsing pandas kaggle-benchmarks kaggle
```

**Always use `conda run -n PDPBench` for Python commands.**

## Key commands

```bash
# Run local test (instances load, HAS_KBENCH=False path, no LLM calls)
conda run --no-capture-output -n PDPBench python -u pdpbench.py

# Run scoring tests with fake LLM responses
conda run --no-capture-output -n PDPBench python -u test_scoring.py

# Package dataset for upload
conda run --no-capture-output -n PDPBench python -u save.py

# Upload dataset to Kaggle
cd kaggle_dataset/pdpbench-data
conda run --no-capture-output -n PDPBench kaggle datasets version -p . -m "description" --dir-mode zip
```

## Kaggle Benchmarks workflow

1. Edit `pdpbench.py` locally, test with `python -u pdpbench.py` and `python -u test_scoring.py`
2. Run `python -u save.py` to copy into `kaggle_dataset/pdpbench-data/`
3. Upload dataset with `kaggle datasets version` command above
4. On Kaggle: go to kaggle.com/benchmarks/tasks/new, create notebook, add `lasseruttert/pdpbench-data` as input
5. Paste pdpbench.py code into notebook cells, run it
6. The `# %choose` comments register tasks with the Benchmarks platform
7. Create a benchmark at kaggle.com/benchmarks, add the tasks to it

## SDK pattern (critical)

The Kaggle Benchmarks SDK works like this — do NOT use `bind_dataframe` or `schema=`:

```python
@kbench.task(name="my_task")
def my_task(llm) -> float:
    scores = []
    for problem in PROBLEMS:
        raw = str(llm.prompt("prompt text"))   # returns raw text
        data = parse_json_response(raw)         # parse JSON manually
        scores.append(compute_score(data))
    return sum(scores) / len(scores)            # single float

my_task.run(llm=kbench.llm)
# %choose my_task
```

## Utils layer (don't modify unless broken)

- `utils/pdptw_problem.py`: `PDPTWProblem`, `Node`, `Request` dataclasses
- `utils/pdptw_solution.py`: `PDPTWSolution` — routes, `total_distance`, `is_feasible`, `clone()`, `remove_request()`
- `utils/pdptw_reader.py`: `pdptw_reader()` — auto-detects Li & Lim vs Mendeley format
- `utils/li_lim_instance_manager.py`: `LiLimInstanceManager` with `load(name, size)`
- `utils/mendeley_instance_manager.py`: `MendeleyInstanceManager` with `load(name, size)`
- `utils/feasibility.py`: `is_feasible(problem, solution)` — validates all PDPTW constraints
- `utils/best_known_solutions.py`: `BestKnownSolutions.get_bks_as_solution(problem)`

## What still needs to be done

1. **Get benchmark running on Kaggle** — the code works but Kaggle's model proxy returns 503 sometimes. Retry logic is in place. Keep trying.
2. **Create the benchmark** — once tasks run successfully, create a benchmark at kaggle.com/benchmarks and add all 5 tasks.
3. **Write the Kaggle Writeup** — max 1500 words, track: Executive Functions. Must include:
   - Title + subtitle
   - Cognitive framing (why PDPTW tests executive functions)
   - Task descriptions and scoring methodology
   - Expected results / discriminatory power across models
   - Attached benchmark link (mandatory)
   - Cover image (mandatory)
4. **Optional improvements**:
   - Run with COORDINATES distance mode for Li & Lim instances (tests spatial reasoning)
   - Add visualizations of results to the writeup notebook
   - Reduce to fewer instances if prompts are too long / hitting token limits

## Hackathon details

- **Deadline**: 2026-04-16 11:59 PM UTC
- **Submission**: Kaggle Writeup + attached Kaggle Benchmark
- **Evaluation**: Dataset quality (50%), Writeup quality (20%), Novelty/discriminatory power (30%)
- **Quota**: $50/day, $500/month for LLM calls on Kaggle Benchmarks
- **Platform provides LLMs** — you don't choose which model. The platform runs your tasks against multiple models.
- Do NOT use `kaggle kernels push` — use the Benchmarks platform UI at kaggle.com/benchmarks
