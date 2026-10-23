"""Time HiGHS (dual simplex and IPM + crossover, one thread, default tolerances) on the benchmark instances.

Industrial reference for the C# timings: sparse revised simplex with presolve. Only Highs.run() is timed;
building the model is excluded. Output columns match lpbench run where possible.

Usage (from the repository root):
    uv run --project python lphighs --out results/<label>/highs.csv [--classes S,M] [--repeats 3]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .analyze import read_lpb

SOLVERS = {"highs-ds": {"solver": "simplex", "simplex_strategy": 1}, "highs-ipm": {"solver": "ipm", "run_crossover": "on"}}


def build(a: np.ndarray, b: np.ndarray, c: np.ndarray, options: dict) -> highspy.Highs:
    m, n = a.shape
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    for k, v in options.items():
        h.setOptionValue(k, v)
    csc = sp.csc_matrix(a)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.col_cost_ = c
    lp.col_lower_, lp.col_upper_ = np.zeros(n), np.full(n, highspy.kHighsInf)
    lp.row_lower_, lp.row_upper_ = np.full(m, -highspy.kHighsInf), b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = csc.indptr, csc.indices, csc.data
    h.passModel(lp)
    return h


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--manifest", default=str(root / "experiments" / "manifest.csv"))
    ap.add_argument("--set", default="main")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ns = ap.parse_args(argv)

    man = pd.read_csv(ns.manifest)
    man = man[man["set"] == ns.set]
    if ns.classes:
        man = man[man["class"].isin(ns.classes.split(","))]
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task_id", "set", "class", "structure", "m", "n", "method", "repeat", "time_ms", "iterations", "objective",
            "fstar", "gap", "status"]
    t0 = time.perf_counter()
    with out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for k, (_, e) in enumerate(man.iterrows(), 1):
            a, b, c = read_lpb(Path(ns.data) / e["file"])
            for method, options in SOLVERS.items():
                for r in range(ns.warmup + ns.repeats):
                    h = build(a, b, c, options)
                    s = time.perf_counter_ns()
                    h.run()
                    ms = (time.perf_counter_ns() - s) / 1e6
                    info = h.getInfo()
                    obj = float(info.objective_function_value)
                    it = int(info.simplex_iteration_count) + int(info.ipm_iteration_count)
                    status = "Solved" if h.getModelStatus() == highspy.HighsModelStatus.kOptimal else str(h.getModelStatus())
                    fstar = float(e["fstar"])
                    w.writerow([e["id"], e["set"], e["class"], e["structure"], e["m"], e["n"], method, r - ns.warmup + 1,
                                repr(ms), it, repr(obj), repr(fstar), repr(abs(obj - fstar) / max(1.0, abs(fstar))), status])
            print(f"[{k}/{len(man)}] {e["id"]} done, elapsed {time.perf_counter() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
