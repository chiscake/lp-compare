"""Deterministic instance generator and HiGHS reference values.

Every instance is defined by (set, class, structure, index) -> seed. Files are written in the LPB1 binary format
(see src/LpBench.Core/ProblemFile.cs). experiments/manifest.csv stores seed, SHA-256 of the file and the reference
optimum f* computed by HiGHS: both dual simplex and IPM with crossover are run, and f* is taken from the solution
with the tighter primal-dual optimality certificate (stored in fstar_cert).

Usage (from the repository root):
    uv run --project python lpgen                      # all sets, reuse f* from the manifest when hashes match
    uv run --project python lpgen --sets illustrative,train --classes ILL,S,M
    uv run --project python lpgen --verify             # regenerate and compare hashes only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

SIZES = {"S": (10, 20), "M": (100, 200), "L": (500, 1000), "XL": (1000, 2000)}
COUNTS = {
    "main": {"S": 100, "M": 50, "L": 30, "XL": 10},
    "train": {"S": 20, "M": 10},
}
STRUCTURES = ("dense", "sparse", "degenerate")
SET_CODE = {"main": 1, "train": 2}
CLASS_CODE = {"S": 1, "M": 2, "L": 3, "XL": 4}
STRUCT_CODE = {"dense": 1, "sparse": 2, "degenerate": 3}
SPARSE_DENSITY = 0.08
REFERENCE_FIELDS = ["fstar", "fstar_source", "fstar_cert", "fstar_ds", "fstar_ipm", "rel_diff", "highs_time_s",
                    "x_star_nnz"]
FIELDS = ["id", "set", "class", "structure", "m", "n", "seed", "file", "sha256", *REFERENCE_FIELDS]

ILLUSTRATIVE = (
    np.array([[7, 2, 2, 6], [5, 8, 4, 3], [2, 4, 1, 8]], dtype=np.float64),
    np.array([80, 480, 130], dtype=np.float64),
    np.array([3, 4, 3, 1], dtype=np.float64),
)


@dataclass(frozen=True)
class Spec:
    set: str
    cls: str
    structure: str
    index: int

    @property
    def id(self) -> str:
        if self.set == "illustrative":
            return "illustrative"
        return f"{self.set}_{self.cls}_{self.structure}_{self.index:03d}"

    @property
    def seed(self) -> int:
        if self.set == "illustrative":
            return 0
        return SET_CODE[self.set] * 10_000_000 + CLASS_CODE[self.cls] * 100_000 + STRUCT_CODE[self.structure] * 10_000 + self.index

    @property
    def file(self) -> str:
        if self.set == "illustrative":
            return "illustrative.lpb"
        return f"{self.set}/{self.cls}/{self.structure}/{self.id}.lpb"


def ensure_columns(a: np.ndarray, rng: np.random.Generator, value) -> None:
    """Every column needs a positive entry (bounded variable), every row a nonzero (non-trivial resource)."""
    m, n = a.shape
    for j in np.flatnonzero((a > 0).sum(axis=0) == 0):
        a[rng.integers(m), j] = value() if callable(value) else value
    for i in np.flatnonzero((a > 0).sum(axis=1) == 0):
        a[i, rng.integers(n)] = value() if callable(value) else value


def generate(spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.set == "illustrative":
        return tuple(x.copy() for x in ILLUSTRATIVE)  # type: ignore[return-value]
    m, n = SIZES[spec.cls]
    rng = np.random.default_rng(spec.seed)
    if spec.structure == "dense":
        a = rng.uniform(0.1, 1.0, size=(m, n))
        b = rng.uniform(0.1, 0.5, size=m) * a.sum(axis=1)
        c = rng.uniform(1.0, 10.0, size=n)
    elif spec.structure == "sparse":
        mask = rng.random((m, n)) < SPARSE_DENSITY
        a = np.where(mask, rng.uniform(0.1, 1.0, size=(m, n)), 0.0)
        ensure_columns(a, rng, lambda: rng.uniform(0.1, 1.0))
        b = rng.uniform(0.1, 0.5, size=m) * a.sum(axis=1)
        c = rng.uniform(1.0, 10.0, size=n)
    elif spec.structure == "degenerate":
        # Small integers make ratio-test and reduced-cost ties frequent. Half of the rows are tight at an integer
        # point x̂ with small support, so x̂ is a vertex with many more than n active constraints.
        a = rng.choice(np.array([0.0, 1.0, 2.0]), size=(m, n), p=[0.4, 0.4, 0.2])
        ensure_columns(a, rng, 1.0)
        c = rng.integers(1, 6, size=n).astype(np.float64)
        support = rng.choice(n, size=max(1, m // 4), replace=False)
        x_hat = np.zeros(n)
        x_hat[support] = rng.integers(1, 4, size=support.size)
        ax = a @ x_hat
        base = max(1.0, round(0.3 * float(a.sum(axis=1).mean())))
        choice = rng.choice(np.array([base, 2 * base]), size=m)
        tight = rng.random(m) < 0.5
        b = np.where(tight & (ax > 0), ax, np.maximum(choice, ax))
    else:
        raise ValueError(spec.structure)
    return a, b, c


def to_bytes(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bytes:
    m, n = a.shape
    return (b"LPB1" + struct.pack("<ii", m, n)
            + np.ascontiguousarray(b, dtype="<f8").tobytes()
            + np.ascontiguousarray(c, dtype="<f8").tobytes()
            + np.ascontiguousarray(a, dtype="<f8").tobytes())


# HiGHS defaults (1e-7) leave relative errors of ~2e-9 in f* on degenerate L instances, above our 1e-9 check.
HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                 "ipm_optimality_tolerance": 1e-10}


def certificate(a: np.ndarray, b: np.ndarray, c: np.ndarray, res) -> float:
    """Relative optimality certificate of a HiGHS primal-dual pair for max c'x, Ax <= b, x >= 0.

    max of primal infeasibility, dual infeasibility (c - A'y <= 0, y >= 0) and the duality gap |c'x - b'y|,
    each relative to 1 + norm of the corresponding data; f* is exact up to this value.
    """
    x = res.x
    y = -res.ineqlin.marginals
    scale = 1.0 + max(abs(float(c @ x)), 1.0)
    p = max(0.0, float((a @ x - b).max()), float(-x.min())) / (1.0 + float(np.abs(b).max()))
    d = max(0.0, float((c - a.T @ y).max()), float(-y.min())) / (1.0 + float(np.abs(c).max()))
    return max(p, d, abs(float(c @ x) - float(b @ y)) / scale)


def solve_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict:
    """f* from whichever HiGHS solver (dual simplex or IPM + crossover) has the tighter optimality certificate."""
    t0 = time.perf_counter()
    ds = linprog(-c, A_ub=a, b_ub=b, bounds=(0, None), method="highs-ds", options=HIGHS_OPTIONS)
    ipm = linprog(-c, A_ub=a, b_ub=b, bounds=(0, None), method="highs-ipm", options=HIGHS_OPTIONS)
    elapsed = time.perf_counter() - t0
    if ds.status != 0 or ipm.status != 0:
        raise RuntimeError(f"HiGHS failed: ds={ds.message!r} ipm={ipm.message!r}")
    f_ds, f_ipm = float(c @ ds.x), float(c @ ipm.x)
    cert_ds, cert_ipm = certificate(a, b, c, ds), certificate(a, b, c, ipm)
    best, f, cert = (ipm, f_ipm, cert_ipm) if cert_ipm <= cert_ds else (ds, f_ds, cert_ds)
    return {"fstar": repr(f), "fstar_source": "ipm" if best is ipm else "ds", "fstar_cert": f"{cert:.1e}",
            "fstar_ds": repr(f_ds), "fstar_ipm": repr(f_ipm),
            "rel_diff": f"{abs(f_ds - f_ipm) / max(1.0, abs(f)):.3e}",
            "highs_time_s": f"{elapsed:.3f}", "x_star_nnz": int((best.x > 1e-9).sum())}


def all_specs(sets: list[str], classes: list[str] | None) -> list[Spec]:
    specs: list[Spec] = []
    if "illustrative" in sets and (classes is None or "ILL" in classes):
        specs.append(Spec("illustrative", "ILL", "illustrative", 0))
    for s in ("train", "main"):
        if s not in sets:
            continue
        for cls, count in COUNTS[s].items():
            if classes is not None and cls not in classes:
                continue
            for st in STRUCTURES:
                specs.extend(Spec(s, cls, st, i) for i in range(count))
    return specs


def process(args: tuple[Spec, str, dict | None, bool]) -> dict:
    spec, data_dir, known, verify = args
    a, b, c = generate(spec)
    payload = to_bytes(a, b, c)
    sha = hashlib.sha256(payload).hexdigest()
    path = Path(data_dir) / spec.file
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(payload) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        path.write_bytes(payload)
    row = {"id": spec.id, "set": spec.set, "class": spec.cls, "structure": spec.structure, "m": a.shape[0],
           "n": a.shape[1], "seed": spec.seed, "file": spec.file, "sha256": sha}
    if known is not None and known["sha256"] != sha:
        if verify:
            raise RuntimeError(f"{spec.id}: hash mismatch (manifest {known['sha256'][:12]}…, regenerated {sha[:12]}…)")
        known = None
    if known is not None and all(k in known for k in REFERENCE_FIELDS):
        row.update({k: known[k] for k in REFERENCE_FIELDS})
        return row
    if verify:
        raise RuntimeError(f"{spec.id}: not in manifest")
    row.update(solve_reference(a, b, c))
    return row


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--manifest", default=str(root / "experiments" / "manifest.csv"))
    ap.add_argument("--sets", default="illustrative,train,main")
    ap.add_argument("--classes", default=None, help="comma list, e.g. ILL,S,M (default: all)")
    ap.add_argument("--verify", action="store_true", help="regenerate and compare hashes; do not solve")
    ap.add_argument("--resolve", action="store_true", help="recompute f* even for instances already in the manifest")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ns = ap.parse_args(argv)

    manifest = Path(ns.manifest)
    existing: dict[str, dict] = {}
    if manifest.exists():
        with manifest.open(newline="") as f:
            existing = {r["id"]: r for r in csv.DictReader(f)}

    specs = all_specs(ns.sets.split(","), ns.classes.split(",") if ns.classes else None)
    print(f"numpy {np.__version__}, scipy {scipy.__version__} (HiGHS via scipy.optimize.linprog); "
          f"{len(specs)} instances -> {ns.data}", file=sys.stderr)
    t0 = time.perf_counter()
    rows: dict[str, dict] = dict(existing)
    work = [(s, ns.data, None if ns.resolve else existing.get(s.id), ns.verify) for s in specs]
    with ProcessPoolExecutor(max_workers=ns.jobs) as pool:
        for k, row in enumerate(pool.map(process, work, chunksize=1), 1):
            rows[row["id"]] = row
            if k % 25 == 0 or k == len(work):
                print(f"  {k}/{len(work)} ({time.perf_counter() - t0:.0f} s)", file=sys.stderr)

    bad = [r for r in rows.values() if float(r["fstar_cert"]) > 1e-10]
    if bad:
        print(f"WARNING: f* certificate above 1e-10 on {len(bad)} instances: {[r['id'] for r in bad][:5]}", file=sys.stderr)
    if ns.verify:
        print(f"verified {len(specs)} instances: hashes match the manifest", file=sys.stderr)
        return 0

    order = {"illustrative": 0, "train": 1, "main": 2}
    cls_order = {"ILL": 0, "S": 1, "M": 2, "L": 3, "XL": 4}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (order[r["set"]], cls_order[r["class"]], r["structure"], r["id"])):
            w.writerow({k: r[k] for k in FIELDS})
    print(f"manifest: {manifest} ({len(rows)} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
