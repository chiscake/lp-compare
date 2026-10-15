"""Aggregate benchmark CSVs into tables and figures.

Usage (from the repository root):
    uv run --project python lpanalyze --run results/<machine>

Reads from the run directory (whatever exists):
    raw.csv          timed runs: lpbench run          -> summary.csv, summary.md, validation.md, fig_time.*
    lambda.csv       λ study:    lpbench grid         -> lambda_summary.csv, fig_lambda.*
    illustrative.csv illustrative problem (lpbench run --set illustrative) -> illustrative.md
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_ORDER = ["dantzig", "bland", "ipm", "dln", "dln-log"]
METHOD_LABEL = {"dantzig": "Симплекс (Данциг)", "bland": "Симплекс (Бланд)", "ipm": "IPM (Мехротра)",
                "dln": "DLN", "dln-log": "DLN (log)"}
CLASS_ORDER = ["ILL", "S", "M", "L", "XL"]
STRUCT_ORDER = ["illustrative", "dense", "sparse", "degenerate"]
STRUCT_LABEL = {"dense": "плотные", "sparse": "разреженные", "degenerate": "вырожденные", "illustrative": "пример"}


def q(p: float):
    def f(s: pd.Series) -> float:
        return float(s.quantile(p))
    f.__name__ = f"p{int(p * 100)}"
    return f


def fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "time":
        return f"{v:.3g}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "share":
        return f"{100 * v:.0f}%"
    return f"{v:.1e}"


def per_task(raw: pd.DataFrame) -> pd.DataFrame:
    timed = raw[raw["repeat"] >= 1]
    first = timed[timed["repeat"] == 1].set_index(["task_id", "method"])
    med = timed.groupby(["task_id", "method"])["time_ms"].median().rename("time_ms_median")
    n_rep = timed.groupby(["task_id", "method"])["time_ms"].size().rename("repeats_done")
    t = first.drop(columns=["time_ms", "repeat"]).join(med).join(n_rep).reset_index()
    t["solved"] = t["status"] == "Solved"
    t["degenerate_share"] = np.where(t["iterations"] > 0, t["degenerate_pivots"] / t["iterations"].clip(lower=1), np.nan)
    return t


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["class", "structure", "method"])
    s = g.agg(
        tasks=("task_id", "count"),
        solved=("solved", "mean"),
        time_min=("time_ms_median", "min"), time_median=("time_ms_median", "median"),
        time_p90=("time_ms_median", q(0.9)), time_max=("time_ms_median", "max"),
        iter_min=("iterations", "min"), iter_median=("iterations", "median"),
        iter_p90=("iterations", q(0.9)), iter_max=("iterations", "max"),
        gap_median=("gap", "median"), gap_p90=("gap", q(0.9)), gap_max=("gap", "max"),
        viol_median=("viol", "median"), viol_max=("viol", "max"),
        degenerate_share_median=("degenerate_share", "median"),
    ).reset_index()
    s["class"] = pd.Categorical(s["class"], CLASS_ORDER, ordered=True)
    s["structure"] = pd.Categorical(s["structure"], STRUCT_ORDER, ordered=True)
    s["method"] = pd.Categorical(s["method"], METHOD_ORDER, ordered=True)
    return s.sort_values(["class", "structure", "method"]).reset_index(drop=True)


def summary_markdown(s: pd.DataFrame) -> str:
    lines = ["# Сводка по классам и методам", "",
             "Время — медиана из повторов для каждой задачи, затем min / медиана / p90 / max по задачам (мс). "
             "Итерации между методами не сравниваются. «Решено» — доля задач, где выполнен критерий остановки "
             "метода в пределах лимитов.", ""]
    for (cls, st), x in s.groupby(["class", "structure"], observed=True):
        lines += [f"## {cls}, {STRUCT_LABEL.get(st, st)}", "",
                  "| Метод | Задач | Решено | Время min / med / p90 / max, мс | Итерации med (p90) | Вырожд. пивоты | Зазор med / max | Невязка max |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            deg = fmt(r.degenerate_share_median, "share") if r.method in ("dantzig", "bland") else "—"
            lines.append(
                f"| {METHOD_LABEL.get(r.method, r.method)} | {r.tasks} | {fmt(r.solved, 'share')} | "
                f"{fmt(r.time_min, 'time')} / {fmt(r.time_median, 'time')} / {fmt(r.time_p90, 'time')} / {fmt(r.time_max, 'time')} | "
                f"{fmt(r.iter_median, 'int')} ({fmt(r.iter_p90, 'int')}) | {deg} | "
                f"{fmt(r.gap_median, 'e')} / {fmt(r.gap_max, 'e')} | {fmt(r.viol_max, 'e')} |")
        lines.append("")
    return "\n".join(lines)


def validation_markdown(t: pd.DataFrame) -> str:
    out = ["# Валидация", ""]
    simplex = t[t["method"].isin(["dantzig", "bland"])]
    worst = simplex["gap"].max() if len(simplex) else float("nan")
    not_solved = simplex[~simplex["solved"]]
    out.append(f"- Симплекс (оба правила), задач: {simplex['task_id'].nunique()}; максимальный относительный зазор с HiGHS: "
               f"{worst:.2e} (допуск 1e-9) — {'OK' if worst <= 1e-9 else 'ОШИБКА'}.")
    out.append(f"- Не решены симплексом: {len(not_solved)}" + (f" ({', '.join(not_solved['task_id'].head(10))})" if len(not_solved) else "") + ".")
    piv = simplex.pivot_table(index="task_id", columns="method", values="objective")
    if {"dantzig", "bland"} <= set(piv.columns):
        diff = ((piv["dantzig"] - piv["bland"]).abs() / piv["dantzig"].abs().clip(lower=1)).max()
        out.append(f"- Совпадение оптимумов Данцига и Бланда: максимальное относительное расхождение {diff:.2e}.")
    ipm = t[t["method"] == "ipm"]
    if len(ipm):
        out.append(f"- IPM: решено {ipm['solved'].mean():.0%}, зазор max {ipm['gap'].max():.2e}, невязка max {ipm['viol'].max():.2e}.")
    for m in ("dln", "dln-log"):
        d = t[t["method"] == m]
        if len(d):
            out.append(f"- {METHOD_LABEL[m]}: критерий остановки выполнен в {d['solved'].mean():.0%} задач; зазор: медиана "
                       f"{d['gap'].median():.2e}, max {d['gap'].max():.2e}; невязка max {d['viol'].max():.2e}.")
    return "\n".join(out) + "\n"


def fig_time(t: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = {"dense": "-", "degenerate": "--", "sparse": ":"}
    colors = {"dantzig": "#1f77b4", "bland": "#9467bd", "ipm": "#2ca02c", "dln": "#d62728", "dln-log": "#ff7f0e"}
    t = t[t["class"] != "ILL"].copy()
    t["vars"] = t["n"] + t["m"]
    for (method, st), x in t.groupby(["method", "structure"]):
        if st not in ("dense", "degenerate"):
            continue
        g = x.groupby("vars")["time_ms_median"].median().sort_index()
        ax.plot(g.index, g.values, styles[st], marker="o", ms=4, color=colors.get(method, "k"),
                label=f"{METHOD_LABEL.get(method, method)}, {STRUCT_LABEL[st]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число переменных n + m")
    ax.set_ylabel("Время, мс (медиана)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_time.{ext}", dpi=200)
    plt.close(fig)


def read_lpb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    m, n = struct.unpack("<ii", raw[4:12])
    v = np.frombuffer(raw, dtype="<f8", offset=12)
    return v[m + n:].reshape(m, n), v[:m], v[m:m + n]


def entropy_bound(data: Path, manifest: pd.DataFrame, task_ids: list[str]) -> pd.Series:
    """Worst-case relative gap per unit λ: ln N · max c · M / max(1, f*) (docs/theory-and-plan.md, §2.4)."""
    res = {}
    files = manifest.set_index("id")
    for tid in task_ids:
        a, b, c = read_lpb(data / files.loc[tid, "file"])
        with np.errstate(divide="ignore"):
            u = np.where(a > 0, b[:, None] / np.where(a > 0, a, 1), np.inf).min(axis=0)
        big_m = 1.1 * (u.sum() + b.sum())
        n_vars = a.shape[0] + a.shape[1] + 1
        res[tid] = math.log(n_vars) * c.max() * big_m / max(1.0, abs(files.loc[tid, "fstar"]))
    return pd.Series(res)


def lambda_study(grid: pd.DataFrame, out: Path, data: Path | None, manifest: pd.DataFrame | None) -> None:
    grid = grid.copy()
    grid["solved"] = grid["status"] == "Solved"
    if data is not None and manifest is not None and (data / manifest["file"].iloc[0]).exists():
        bound = entropy_bound(data, manifest, sorted(grid["task_id"].unique()))
        grid["bound"] = grid["lambda"] * grid["task_id"].map(bound)
    else:
        grid["bound"] = np.nan
    s = grid.groupby(["class", "structure", "mode", "kappa", "lambda"]).agg(
        tasks=("task_id", "count"), solved=("solved", "mean"),
        gap_median=("gap", "median"), gap_p10=("gap", q(0.1)), gap_p90=("gap", q(0.9)),
        repaired_gap_median=("repaired_gap", "median"),
        viol_median=("viol", "median"), iter_median=("iterations", "median"),
        time_median=("time_ms", "median"), bound_median=("bound", "median")).reset_index()
    s.to_csv(out / "lambda_summary.csv", index=False)

    classes = [c for c in CLASS_ORDER if c in set(s["class"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {"S": "#1f77b4", "M": "#d62728", "L": "#2ca02c"}
    markers = {"dense": "o", "degenerate": "s", "sparse": "^"}
    for (cls, st, mode, kappa), x in s.groupby(["class", "structure", "mode", "kappa"]):
        x = x.sort_values("lambda")
        lab = f"{cls}, {STRUCT_LABEL.get(st, st)}" + (" (log)" if mode == "log" else "")
        ls = "--" if mode == "log" else "-"
        axes[0].plot(x["lambda"], x["gap_median"].clip(lower=1e-16), ls, marker=markers.get(st, "o"), ms=4,
                     color=colors.get(cls, "k"), label=lab)
        axes[1].plot(x["lambda"], x["iter_median"], ls, marker=markers.get(st, "o"), ms=4, color=colors.get(cls, "k"), label=lab)
    if s["bound_median"].notna().any():
        b = s[(s["structure"] == "dense") & (s["mode"] == "plain")].groupby("lambda")["bound_median"].median()
        axes[0].plot(b.index, b.values, "k:", lw=1, label="оценка λ·ln N·c̄M/f*")
    for ax in axes:
        ax.axvline(1.4e-3, color="grey", lw=0.8, ls="-.")
        ax.axvline(6.7e-4, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("λ")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Относительный зазор (медиана)")
    axes[1].set_ylabel("Итерации (медиана)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"DLN: зазор и число итераций от λ (κ = {', '.join(str(k) for k in sorted(s['kappa'].unique()))}; "
                 f"классы {', '.join(classes)})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_lambda.{ext}", dpi=200)
    plt.close(fig)


def illustrative_markdown(ill: pd.DataFrame) -> str:
    t = per_task(ill)
    lines = ["# Иллюстративная задача", "", "| Метод | Статус | Итерации | Время, мс | Доход | Зазор | Невязка |", "|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values("method", key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)})).iterrows():
        lines.append(f"| {METHOD_LABEL.get(r.method, r.method)} | {r.status} | {r.iterations} | {r.time_ms_median:.3g} | "
                     f"{r.objective:.6g} | {r.gap:.1e} | {r.viol:.1e} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory with raw.csv / lambda.csv / illustrative.csv")
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--manifest", default=str(root / "experiments" / "manifest.csv"))
    ns = ap.parse_args(argv)
    run = Path(ns.run)
    manifest = pd.read_csv(ns.manifest) if Path(ns.manifest).exists() else None
    done = []

    if (run / "raw.csv").exists():
        t = per_task(pd.read_csv(run / "raw.csv"))
        t.to_csv(run / "per_task.csv", index=False)
        s = summarize(t)
        s.to_csv(run / "summary.csv", index=False)
        (run / "summary.md").write_text(summary_markdown(s), encoding="utf-8")
        (run / "validation.md").write_text(validation_markdown(t), encoding="utf-8")
        fig_time(t, run)
        done += ["per_task.csv", "summary.csv", "summary.md", "validation.md", "fig_time.png/svg"]
    if (run / "lambda.csv").exists():
        lambda_study(pd.read_csv(run / "lambda.csv"), run, Path(ns.data), manifest)
        done += ["lambda_summary.csv", "fig_lambda.png/svg"]
    if (run / "illustrative.csv").exists():
        (run / "illustrative.md").write_text(illustrative_markdown(pd.read_csv(run / "illustrative.csv")), encoding="utf-8")
        done.append("illustrative.md")
    if not done:
        print(f"nothing to analyze in {run}", file=sys.stderr)
        return 1
    print(f"{run}: " + ", ".join(done), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
