"""Package all benchmark data and code into kaggle_dataset/ for upload."""

import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(_HERE, "kaggle_dataset", "pdpbench-data")

# Directories to copy
DIRS_TO_COPY = ["data", "bks", "utils"]

# Python modules to copy (the benchmark code)
FILES_TO_COPY = [
    "pdpbench.py",
    "pdpbench_kaggle.py",   # monolith kept for backward compat
    "pdpbench_lib.py",      # shared library imported by all per-task files
    "pdpbench_T1.py",       # Task 1: Request Insertion (one-shot)
    "pdpbench_T2.py",       # Task 2: Route Completion (one-shot)
    "pdpbench_T3.py",       # Task 3: Full Solution (one-shot)
    "pdpbench_T4.py",       # Task 4: Request Insertion (iterative)
    "pdpbench_T5.py",       # Task 5: Route Completion (iterative)
    "pdpbench_T6.py",       # Task 6: Full Solution (iterative)
]


def main():
    for dirname in DIRS_TO_COPY:
        src = os.path.join(_HERE, dirname)
        dst = os.path.join(DEST, dirname)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copied {dirname}/ -> {dst}")

    for filename in FILES_TO_COPY:
        src = os.path.join(_HERE, filename)
        dst = os.path.join(DEST, filename)
        shutil.copy2(src, dst)
        print(f"  Copied {filename} -> {dst}")

    print("Done. Dataset ready for upload.")


if __name__ == "__main__":
    main()
