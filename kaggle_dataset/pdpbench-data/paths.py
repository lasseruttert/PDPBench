"""Path configuration — auto-detects Kaggle vs local environment."""

import os
import sys

KAGGLE = os.path.exists("/kaggle")

if KAGGLE:
    BASE_DIR = "/kaggle/input/pdpbench-data"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
BKS_DIR = os.path.join(BASE_DIR, "bks")

# Ensure utils is importable from the dataset directory on Kaggle
if KAGGLE and BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
