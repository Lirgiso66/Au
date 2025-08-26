
import os, time, json, pathlib, numpy as np, pandas as pd
from contextlib import contextmanager

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

@contextmanager
def timer():
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[timer] {t1 - t0:.2f}s")

def save_result(name: str, payload: dict):
    f = RESULTS_DIR / f"{name}.json"
    with open(f, "w") as w:
        json.dump(payload, w, indent=2)

def append_csv(path, row: dict, header=True):
    import csv
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header and not exists:
            writer.writeheader()
        writer.writerow(row)
