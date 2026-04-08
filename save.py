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
