"""Aggregate benchmark CSVs into tables and figures.

Usage (from the repository root):
    uv run --project python lpanalyze --run results/<machine>

Reads from the run directory (whatever exists):
    raw.csv          timed runs: lpbench run          -> summary.csv, summary.md, validation.md, fig_time.*
    lambda.csv       λ study:    lpbench grid         -> lambda_summary.csv, fig_lambda.*
    lambda_large.csv λ study on L (optional, merged with lambda.csv)
    lambda_long.csv  long log-domain runs below the double window -> lambda_long.md
    kappa.csv        κ study:    lpbench grid --kappas -> kappa.md
    highs.csv        HiGHS timings (lphighs), merged with raw*.csv
    trace.csv        DLN trajectories (lpbench trace) -> time_to_target.csv, quality.md
    (raw*.csv + data) exact λ-regularized optimum per DLN task -> entropic_main.csv, dln_decomposition.md
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

METHOD_ORDER = ["dantzig", "bland", "ipm", "dln", "dln-log", "highs-ds", "highs-ipm"]
METHOD_LABEL = {"dantzig": "Симплекс (Данциг)", "bland": "Симплекс (Бланд)", "ipm": "IPM (Мехротра)",
                "dln": "DLN", "dln-log": "DLN (log)", "highs-ds": "HiGHS (двойств. симплекс)",
                "highs-ipm": "HiGHS (IPM + crossover)"}
GAP_TARGETS = (1e-3, 1e-6, 1e-8)
FEAS_TOL = 1e-6
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
    t_min = timed.groupby(["task_id", "method"])["time_ms"].min().rename("time_ms_min")
    t_max = timed.groupby(["task_id", "method"])["time_ms"].max().rename("time_ms_max")
    t = first.drop(columns=["time_ms", "repeat"]).join(med).join(n_rep).join(t_min).join(t_max).reset_index()
    t["time_spread"] = np.where(t["repeats_done"] > 1, (t["time_ms_max"] - t["time_ms_min"]) / t["time_ms_median"], np.nan)
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
        repeats=("repeats_done", "median"), time_spread_median=("time_spread", "median"),
        time_spread_max=("time_spread", "max"),
    ).reset_index()
    s["class"] = pd.Categorical(s["class"], CLASS_ORDER, ordered=True)
    s["structure"] = pd.Categorical(s["structure"], STRUCT_ORDER, ordered=True)
    s["method"] = pd.Categorical(s["method"], METHOD_ORDER, ordered=True)
    return s.sort_values(["class", "structure", "method"]).reset_index(drop=True)


def summary_markdown(s: pd.DataFrame) -> str:
    lines = ["# Сводка по классам и методам", "",
             "Время — медиана из повторов для каждой задачи, затем min / медиана / p90 / max по задачам (мс). "
             "Итерации между методами не сравниваются. «Решено» — доля задач, где выполнен критерий остановки "
             "метода в пределах лимитов. Разброс — (max − min)/медиана по повторам одной задачи, медиана (max) по задачам.", ""]
    for (cls, st), x in s.groupby(["class", "structure"], observed=True):
        lines += [f"## {cls}, {STRUCT_LABEL.get(st, st)}", "",
                  "| Метод | Задач | Решено | Время min / med / p90 / max, мс | Повторов | Разброс | Итерации med (p90) | Вырожд. пивоты | Зазор med / max | Невязка max |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            deg = fmt(r.degenerate_share_median, "share") if r.method in ("dantzig", "bland") else "—"
            lines.append(
                f"| {METHOD_LABEL.get(r.method, r.method)} | {r.tasks} | {fmt(r.solved, 'share')} | "
                f"{fmt(r.time_min, 'time')} / {fmt(r.time_median, 'time')} / {fmt(r.time_p90, 'time')} / {fmt(r.time_max, 'time')} | "
                f"{fmt(r.repeats, 'int')} | {fmt(r.time_spread_median, 'share')} ({fmt(r.time_spread_max, 'share')}) | "
                f"{fmt(r.iter_median, 'int')} ({fmt(r.iter_p90, 'int')}) | {deg} | "
                f"{fmt(r.gap_median, 'e')} / {fmt(r.gap_max, 'e')} | {fmt(r.viol_max, 'e')} |")
        lines.append("")
    return "\n".join(lines)


def validation_markdown(t: pd.DataFrame) -> str:
    out = ["# Валидация", ""]
    simplex = t[t["method"].isin(["dantzig", "bland"])]
    solved = simplex[simplex["solved"]]
    worst = solved["gap"].max() if len(solved) else float("nan")
    out.append(f"- Симплекс (оба правила), задач: {simplex['task_id'].nunique()}; максимальный относительный зазор с HiGHS "
               f"по решённым: {worst:.2e} (допуск 1e-9) — {'OK' if worst <= 1e-9 else 'ОШИБКА'}.")
    for m in ("dantzig", "bland"):
        ns = simplex[(simplex["method"] == m) & ~simplex["solved"]]
        if len(ns):
            st = ", ".join(f"{k}: {v}" for k, v in ns["status"].value_counts().items())
            out.append(f"- {METHOD_LABEL[m]}: не решено {len(ns)} ({st}); "
                       f"примеры: {', '.join(ns['task_id'].head(5))}.")
        else:
            out.append(f"- {METHOD_LABEL[m]}: решены все задачи.")
    piv = solved.pivot_table(index="task_id", columns="method", values="objective")
    if {"dantzig", "bland"} <= set(piv.columns):
        diff = ((piv["dantzig"] - piv["bland"]).abs() / piv["dantzig"].abs().clip(lower=1)).max()
        out.append(f"- Совпадение оптимумов Данцига и Бланда (где решили оба): максимальное относительное расхождение {diff:.2e}.")
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
        big_m = dln_system(a, b, c)[3]
        n_vars = a.shape[0] + a.shape[1] + 1
        res[tid] = math.log(n_vars) * c.max() * big_m / max(1.0, abs(files.loc[tid, "fstar"]))
    return pd.Series(res)


def dln_system(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """The reduced system DlnSolver builds (src/LpBench.Core/DlnSolver.cs): returns Â, b̂, normalized costs, M."""
    m, n = a.shape
    a_std = np.hstack([a, np.eye(m)])
    c_std = np.concatenate([-c, np.zeros(m)])
    with np.errstate(divide="ignore"):
        u = np.where(a_std > 0, b[:, None] / np.where(a_std > 0, a_std, 1), np.inf).min(axis=0)
    big_m = 1.1 * min(u.sum(), max(1.0, 1.0 / a_std.sum(axis=0).min()) * b.sum())
    n_std = n + m
    a_hat = np.zeros((m + 1, n_std + 1))
    a_hat[:m, :n_std] = a_std
    a_hat[m, :] = 1.0
    b_hat = np.concatenate([b / big_m, [1.0]])
    norms = np.linalg.norm(a_hat, axis=1)
    a_hat /= norms[:, None]
    b_hat /= norms
    cc = np.concatenate([c_std, [0.0]])
    cost = (cc - cc.min()) / (cc.max() - cc.min())
    return a_hat, b_hat, cost, big_m


def entropic_solution(a_hat: np.ndarray, b_hat: np.ndarray, cost: np.ndarray, lam: float,
                      tol: float = 1e-10, max_iter: int = 2000,
                      max_log_change: float = 10.0) -> tuple[np.ndarray, float]:
    """Exact minimizer of ĉᵀx + λΣ(x ln x − x), Âx = b̂ (the λ-limit of DLN, Theorem 2.2 of [1]) by damped Newton on
    the smooth concave dual g(μ) = b̂ᵀμ − λΣexp((Âᵀμ − ĉ)/λ), x(μ) = exp((Âᵀμ − ĉ)/λ). The Hessian is nearly singular
    while most xⱼ are tiny, so each step changes ln xⱼ by at most max_log_change. Returns x and its relative residual."""
    def primal(mu: np.ndarray) -> np.ndarray:
        return np.exp(np.minimum((a_hat.T @ mu - cost) / lam, 700.0))

    nb = np.linalg.norm(b_hat)
    mu = np.zeros(len(b_hat))
    x = primal(mu)
    g_val = b_hat @ mu - lam * x.sum()
    for _ in range(max_iter):
        grad = b_hat - a_hat @ x
        res = np.linalg.norm(grad) / nb
        if res <= tol:
            break
        hess = (a_hat * x) @ a_hat.T / lam
        hess[np.diag_indices_from(hess)] += 1e-12 * max(1e-300, np.trace(hess) / len(hess))
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        change = np.abs(a_hat.T @ step).max() / lam
        if change > max_log_change:
            step *= max_log_change / change
        slope = grad @ step
        t = 1.0
        while t > 1e-12:
            mu_t = mu + t * step
            x_t = primal(mu_t)
            g_t = b_hat @ mu_t - lam * x_t.sum()
            if g_t >= g_val + 1e-4 * t * slope:
                break
            t *= 0.5
        if t <= 1e-12:
            break
        mu, x, g_val = mu_t, x_t, g_t
    return x, float(np.linalg.norm(b_hat - a_hat @ x) / nb)


def entropic_gap(a: np.ndarray, b: np.ndarray, c: np.ndarray, fstar: float, lam: float) -> float:
    """Relative gap of the exact λ-regularized solution; NaN when the dual Newton method does not reach feasibility."""
    a_hat, b_hat, cost, big_m = dln_system(a, b, c)
    x, res = entropic_solution(a_hat, b_hat, cost, lam)
    if res > 1e-8:
        return float("nan")
    n = a.shape[1]
    return abs(c @ (big_m * x[:n]) - fstar) / max(1.0, abs(fstar))


def lambda_study(grid: pd.DataFrame, out: Path, data: Path | None, manifest: pd.DataFrame | None,
                 entropic_tasks: int = 5) -> None:
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

    ent = pd.DataFrame()
    if data is not None and manifest is not None and entropic_tasks > 0:
        files = manifest.set_index("id")
        rows = []
        for (cls, st), x in grid[grid["structure"] == "dense"].groupby(["class", "structure"]):
            for tid in sorted(x["task_id"].unique())[:entropic_tasks]:
                path = data / files.loc[tid, "file"]
                if not path.exists():
                    continue
                a, b, c = read_lpb(path)
                for lam in sorted(grid["lambda"].unique()):
                    rows.append({"class": cls, "structure": st, "task_id": tid, "lambda": lam,
                                 "gap": entropic_gap(a, b, c, float(files.loc[tid, "fstar"]), lam)})
        if rows:
            ent = pd.DataFrame(rows)
            ent.to_csv(out / "lambda_entropic.csv", index=False)

    fig_lambda(s, ent, out)


FIG_W_IN = 16 / 2.54  # ширина полосы набора 16 см


def fig_lambda(s: pd.DataFrame, ent: pd.DataFrame, out: Path, panels: str = "both", name: str = "fig_lambda") -> None:
    """Упрощённый рисунок λ-исследования для печати в 16 см: только хранение u (без log), цвет — класс,
    маркер/тип линии — структура, точный энтропийный предел — крестики, легенда под осями."""
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    s = s[s["mode"] == "plain"]
    colors = {"S": "#0072B2", "M": "#D55E00", "L": "#009E73", "XL": "#CC79A7"}  # различимы и при ч/б печати по маркерам
    styles = {"dense": ("o", "-"), "degenerate": ("s", "--"), "sparse": ("^", ":")}
    both = panels == "both"
    fig, axes = plt.subplots(1, 2 if both else 1, figsize=(FIG_W_IN, 2.7 if both else 2.6), squeeze=False)
    axes = axes[0]
    for (cls, st, kappa), x in s.groupby(["class", "structure", "kappa"]):
        x = x.sort_values("lambda")
        mk, ls = styles.get(st, ("o", "-"))
        kw = dict(ls=ls, marker=mk, ms=3.5, lw=1.1, color=colors.get(cls, "k"), mfc="white", mew=1.0)
        axes[0].plot(x["lambda"], x["gap_median"].clip(lower=1e-16), **kw)
        if both:
            axes[1].plot(x["lambda"], x["iter_median"], **kw)
    if len(ent):
        for cls, x in ent.groupby("class"):
            med = x.groupby("lambda")["gap"].median().dropna()
            axes[0].plot(med.index, med.values.clip(1e-16), "x", ms=6, mew=1.6, color=colors.get(cls, "k"))
    for ax in axes:
        ax.axvline(1.4e-3, color="0.4", lw=0.8, ls="-.")
        ax.axvline(6.7e-4, color="0.4", lw=0.8, ls=(0, (1, 1)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("λ")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Зазор (медиана)")
    if both:
        axes[1].set_ylabel("Итерации (медиана)")
    from matplotlib.lines import Line2D
    classes = [c for c in CLASS_ORDER if c in set(s["class"])]
    h = [Line2D([], [], color=colors[c], lw=2.5) for c in classes]
    lab = list(classes)
    for st in [t for t in STRUCT_ORDER if t in set(s["structure"])]:
        mk, ls = styles[st]
        h.append(Line2D([], [], color="k", ls=ls, marker=mk, ms=3.5, mfc="white", lw=1.1)); lab.append(STRUCT_LABEL.get(st, st))
    if len(ent):
        h.append(Line2D([], [], color="k", ls="", marker="x", ms=6, mew=1.6)); lab.append("точный предел")
    h += [Line2D([], [], color="0.4", lw=0.8, ls="-."), Line2D([], [], color="0.4", lw=0.8, ls=(0, (1, 1)))]
    lab += ["λ = 1,4·10⁻³", "λ = 6,7·10⁻⁴"]
    fig.legend(h, lab, loc="lower center", ncol=5 if both else 4, frameon=False, columnspacing=1.2,
               handlelength=2.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17 if both else 0.15, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def illustrative_markdown(ill: pd.DataFrame) -> str:
    t = per_task(ill)
    lines = ["# Иллюстративная задача", "", "| Метод | Статус | Итерации | Время, мс | Доход | Зазор | Невязка |", "|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values("method", key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)})).iterrows():
        lines.append(f"| {METHOD_LABEL.get(r.method, r.method)} | {r.status} | {r.iterations} | {r.time_ms_median:.3g} | "
                     f"{r.objective:.6g} | {r.gap:.1e} | {r.viol:.1e} |")
    return "\n".join(lines) + "\n"


def long_markdown(grid: pd.DataFrame, data: Path, manifest: pd.DataFrame | None) -> str:
    g = grid.copy()
    g["solved"] = g["status"] == "Solved"
    files = manifest.set_index("id") if manifest is not None else None
    ent = []
    for r in g.itertuples(index=False):
        path = data / files.loc[r.task_id, "file"] if files is not None else None
        if path is None or not path.exists():
            ent.append(np.nan)
            continue
        a, b, c = read_lpb(path)
        ent.append(entropic_gap(a, b, c, float(files.loc[r.task_id, "fstar"]), r[g.columns.get_loc("lambda")]))
    g["entropic_gap"] = ent
    lines = ["# Длинные прогоны DLN ниже окна double", "",
             "Класс M, плотные задачи, κ = рабочее значение; лимит итераций задан `--max-iter`, без ограничения по времени. "
             "Точный предел — зазор точного решения регуляризованной задачи при том же λ.", "",
             "| Режим | λ | Задач | Решено | Зазор: медиана | p90 | Точный предел: медиана | Невязка: медиана | Итерации, медиана | Время, с |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (mode, lam), x in g.groupby(["mode", "lambda"]):
        lines.append(f"| {mode} | {lam:g} | {len(x)} | {fmt(x['solved'].mean(), 'share')} | {fmt(x['gap'].median(), 'e')} | "
                     f"{fmt(x['gap'].quantile(.9), 'e')} | {fmt(x['entropic_gap'].median(), 'e')} | {fmt(x['residual'].median(), 'e')} | "
                     f"{fmt(x['iterations'].median(), 'int')} | {fmt(x['time_ms'].median() / 1000, 'time')} |")
    return "\n".join(lines) + "\n"


def kappa_markdown(grid: pd.DataFrame, ill: pd.DataFrame | None) -> str:
    g = grid.copy()
    g["solved"] = g["status"] == "Solved"
    lines = ["# Исследование по κ (DLN, класс S)", "",
             "| κ | Решено | Зазор, медиана | p75 | p90 | Доля с зазором ≤1e-3 | Итерации, медиана |",
             "|---|---|---|---|---|---|---|"]
    for kappa, x in g.groupby("kappa"):
        lines.append(f"| {kappa:g} | {fmt(x['solved'].mean(), 'share')} | {fmt(x['gap'].median(), 'gap')} | "
                     f"{fmt(x['gap'].quantile(.75), 'gap')} | {fmt(x['gap'].quantile(.9), 'gap')} | "
                     f"{fmt((x['gap'] <= 1e-3).mean(), 'share')} | {fmt(x['iterations'].median(), 'int')} |")
    if ill is not None:
        lines += ["", "Иллюстративная задача", "", "| κ | Итерации | Доход | Зазор |", "|---|---|---|---|"]
        for _, r in ill.sort_values("kappa").iterrows():
            lines.append(f"| {r.kappa:g} | {r.iterations} | {r.objective:.6g} | {r.gap:.1e} |")
    return "\n".join(lines) + "\n"


def dln_accuracy_markdown(t: pd.DataFrame) -> str:
    d = t[(t["method"] == "dln") & (t["class"] == "S")]
    solved = d[d["status"] == "Solved"]
    qs = {f"p{int(p * 100)}": d["gap"].quantile(p) for p in (.5, .75, .9, .95)}
    lines = ["# Точность DLN на классе S", "",
             f"Задач: {len(d)}, статус Solved: {len(solved)}. "
             "Статус Solved означает лишь малую невязку ограничений, не точность по целевой функции.", "",
             "| Квантиль зазора (все задачи S) | Значение |", "|---|---|"]
    lines += [f"| {k} | {v:.1e} |" for k, v in qs.items()]
    lines += ["", f"Доля всех задач с зазором ≤1e-3: {fmt((d['gap'] <= 1e-3).mean(), 'share')}; среди Solved-задач "
              f"с зазором >1e-3: {int((solved['gap'] > 1e-3).sum())} из {len(solved)}."]
    return "\n".join(lines) + "\n"


def read_params(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def entropic_main(t: pd.DataFrame, data: Path, manifest: pd.DataFrame, params: dict[str, str], cache: Path) -> pd.DataFrame:
    """f_λ (exact λ-regularized optimum at the frozen per-class λ) for every task DLN was run on; cached in `cache`."""
    tasks = t[t["method"] == "dln"][["task_id", "class", "structure"]].drop_duplicates()
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["task_id", "lambda", "f_lambda", "residual"])
    files = manifest.set_index("id")
    rows = []
    for r in tasks.itertuples(index=False):
        lam = float(params.get(f"dln.lambda.{r[1]}", params["dln.lambda"]))
        hit = done[(done["task_id"] == r.task_id) & np.isclose(done["lambda"].astype(float), lam)]
        if len(hit):
            rows.append(hit.iloc[0].to_dict())
            continue
        path = data / files.loc[r.task_id, "file"]
        if not path.exists():
            continue
        a, b, c = read_lpb(path)
        a_hat, b_hat, cost, big_m = dln_system(a, b, c)
        x, res = entropic_solution(a_hat, b_hat, cost, lam)
        rows.append({"task_id": r.task_id, "lambda": lam, "f_lambda": float(c @ (big_m * x[:a.shape[1]])), "residual": res})
        pd.DataFrame(rows).to_csv(cache, index=False)
    ent = pd.DataFrame(rows)
    if len(ent):
        ent.to_csv(cache, index=False)
    return ent


def dln_decomposition_markdown(t: pd.DataFrame, ent: pd.DataFrame) -> str:
    d = t[t["method"] == "dln"].merge(ent, on="task_id")
    d = d[d["residual_y"] <= 1e-8] if "residual_y" in d else d
    denom = d["fstar"].abs().clip(lower=1.0)
    d["bias"] = (d["f_lambda"] - d["fstar"]).abs() / denom
    d["opt_err"] = (d["objective"] - d["f_lambda"]).abs() / denom
    d["class"] = pd.Categorical(d["class"], CLASS_ORDER, ordered=True)
    d["structure"] = pd.Categorical(d["structure"], STRUCT_ORDER, ordered=True)
    lines = ["# Разложение ошибки DLN", "",
             "$f_\\lambda$ — точное решение энтропийно регуляризованной задачи при рабочем λ класса (двойственный метод "
             "Ньютона, невязка ≤1e-8). Все величины — относительно $\\max(1,|f^\\star|)$: общий зазор $|f_{DLN}-f^\\star|$, "
             "смещение регуляризации $|f_\\lambda-f^\\star|$, ошибка оптимизации $|f_{DLN}-f_\\lambda|$ (недоитерированность и "
             "смещение шага κ). Медианы по задачам; «доля смещения» — доля задач, где смещение ≥ ошибки оптимизации.", "",
             "| Класс | Структура | λ | Задач | Общий зазор | Смещение | Ошибка оптимизации | Доля смещения |",
             "|---|---|---|---|---|---|---|---|"]
    for (cls, st), x in d.groupby(["class", "structure"], observed=True):
        lines.append(f"| {cls} | {STRUCT_LABEL.get(st, st)} | {x['lambda'].iloc[0]:g} | {len(x)} | {fmt(x['gap'].median(), 'e')} | "
                     f"{fmt(x['bias'].median(), 'e')} | {fmt(x['opt_err'].median(), 'e')} | "
                     f"{fmt((x['bias'] >= x['opt_err']).mean(), 'share')} |")
    for cls, x in d.groupby("class", observed=True):
        lines.append(f"| {cls} | все | {x['lambda'].iloc[0]:g} | {len(x)} | {fmt(x['gap'].median(), 'e')} | "
                     f"{fmt(x['bias'].median(), 'e')} | {fmt(x['opt_err'].median(), 'e')} | "
                     f"{fmt((x['bias'] >= x['opt_err']).mean(), 'share')} |")
    return "\n".join(lines) + "\n"


def time_to_target(t: pd.DataFrame, trace: pd.DataFrame | None) -> pd.DataFrame:
    """Per (task, method, target): time to reach gap ≤ target with violation ≤ FEAS_TOL (NaN = never).
    Finite methods: their solve time if the final point qualifies. DLN: first traced iterate that qualifies."""
    rows = []
    for r in t.itertuples(index=False):
        if r.method.startswith("dln") and trace is not None:
            continue
        feas = not (r.viol > FEAS_TOL) if not pd.isna(r.viol) else r.status == "Solved"
        for tau in GAP_TARGETS:
            ok = feas and r.gap <= tau
            rows.append({"task_id": r.task_id, "class": r[t.columns.get_loc("class")], "structure": r.structure,
                         "method": r.method, "target": tau, "feasible": feas,
                         "time_ms": r.time_ms_median if ok else np.nan})
    if trace is not None:
        tr = trace.copy()
        tr["feas"] = tr["viol"] <= FEAS_TOL
        for (tid, method), x in tr.groupby(["task_id", "method"]):
            last = x.iloc[-1]
            for tau in GAP_TARGETS:
                hit = x[x["feas"] & (x["gap"] <= tau)]
                rows.append({"task_id": tid, "class": last["class"], "structure": last["structure"], "method": method,
                             "target": tau, "feasible": bool(last["feas"]),
                             "time_ms": float(hit["time_ms"].iloc[0]) if len(hit) else np.nan})
    return pd.DataFrame(rows)


def quality_markdown(tt: pd.DataFrame) -> str:
    tt = tt.copy()
    tt["class"] = pd.Categorical(tt["class"], CLASS_ORDER, ordered=True)
    tt["method"] = pd.Categorical(tt["method"], METHOD_ORDER, ordered=True)
    head = " | ".join(f"≤{tau:.0e}: доля / время" for tau in GAP_TARGETS)
    lines = ["# Решено с требуемым качеством", "",
             f"«Допустимо» — относительная невязка ограничений ≤{FEAS_TOL:g} в итоговой точке. Для порога τ задача решена, если "
             f"зазор ≤τ при невязке ≤{FEAS_TOL:g}; время — медиана по задачам, достигшим порога, мс. Для DLN время до первой "
             "точки траектории, удовлетворяющей порогу (`lpbench trace`, проверки на итерациях с шагом ×1,12, один замер), "
             "для остальных — время решения.", "",
             f"| Класс | Метод | Задач | Допустимо | {head} |", "|---|---|---|---|" + "---|" * len(GAP_TARGETS)]
    for (cls, method), x in tt.groupby(["class", "method"], observed=True):
        n_tasks = x["task_id"].nunique()
        feas = x.drop_duplicates("task_id")["feasible"].mean()
        cells = []
        for tau in GAP_TARGETS:
            y = x[x["target"] == tau]
            share = y["time_ms"].notna().mean()
            cells.append(f"{fmt(share, 'share')} / {fmt(y['time_ms'].median(), 'time') if share > 0 else '—'}")
        lines.append(f"| {cls} | {METHOD_LABEL.get(method, method)} | {n_tasks} | {fmt(feas, 'share')} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def with_manifest_fstar(df: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    """Recompute gap against the current manifest, so a refined f* does not require rerunning the solvers."""
    if manifest is None:
        return df
    fstar = df["task_id"].map(manifest.set_index("id")["fstar"])
    known = fstar.notna()
    df = df.copy()
    df.loc[known, "fstar"] = fstar[known]
    df.loc[known, "gap"] = (df["objective"] - df["fstar"]).abs()[known] / df["fstar"].abs().clip(lower=1.0)[known]
    return df


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory with raw.csv / lambda.csv / illustrative.csv")
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--manifest", default=str(root / "experiments" / "manifest.csv"))
    ap.add_argument("--no-entropic", action="store_true", help="skip the exact λ-regularized solutions (minutes on L/XL)")
    ns = ap.parse_args(argv)
    run = Path(ns.run)
    manifest = pd.read_csv(ns.manifest) if Path(ns.manifest).exists() else None
    done = []

    raw_parts = sorted(run.glob("raw*.csv"))
    if raw_parts:
        parts = [pd.read_csv(p) for p in raw_parts]
        if (run / "highs.csv").exists():
            parts.append(pd.read_csv(run / "highs.csv"))
        t = per_task(with_manifest_fstar(pd.concat(parts, ignore_index=True), manifest))
        t.to_csv(run / "per_task.csv", index=False)
        trace = with_manifest_fstar(pd.read_csv(run / "trace.csv"), manifest) if (run / "trace.csv").exists() else None
        tt = time_to_target(t, trace)
        tt.to_csv(run / "time_to_target.csv", index=False)
        (run / "quality.md").write_text(quality_markdown(tt), encoding="utf-8")
        done += ["time_to_target.csv", "quality.md"]
        params_path = root / "experiments" / "params.ini"
        if manifest is not None and (Path(ns.data) / manifest["file"].iloc[0]).exists() and not ns.no_entropic:
            ent = entropic_main(t, Path(ns.data), manifest, read_params(params_path), run / "entropic_main.csv")
            if len(ent):
                (run / "dln_decomposition.md").write_text(dln_decomposition_markdown(t, ent), encoding="utf-8")
                done += ["entropic_main.csv", "dln_decomposition.md"]
        s = summarize(t)
        s.to_csv(run / "summary.csv", index=False)
        (run / "summary.md").write_text(summary_markdown(s), encoding="utf-8")
        (run / "validation.md").write_text(validation_markdown(t), encoding="utf-8")
        fig_time(t, run)
        (run / "dln_accuracy.md").write_text(dln_accuracy_markdown(t), encoding="utf-8")
        done += ["per_task.csv", "summary.csv", "summary.md", "validation.md", "dln_accuracy.md", "fig_time.png/svg"]
    if (run / "lambda.csv").exists():
        grids = [pd.read_csv(run / f) for f in ("lambda.csv", "lambda_large.csv") if (run / f).exists()]
        lambda_study(with_manifest_fstar(pd.concat(grids, ignore_index=True), manifest), run, Path(ns.data), manifest)
        done += ["lambda_summary.csv", "fig_lambda.png/svg"]
    if (run / "lambda_long.csv").exists():
        (run / "lambda_long.md").write_text(long_markdown(
            with_manifest_fstar(pd.read_csv(run / "lambda_long.csv"), manifest), Path(ns.data), manifest), encoding="utf-8")
        done.append("lambda_long.md")
    if (run / "kappa.csv").exists():
        ki = run / "kappa_illustrative.csv"
        (run / "kappa.md").write_text(kappa_markdown(
            with_manifest_fstar(pd.read_csv(run / "kappa.csv"), manifest),
            with_manifest_fstar(pd.read_csv(ki), manifest) if ki.exists() else None), encoding="utf-8")
        done.append("kappa.md")
    if (run / "illustrative.csv").exists():
        ill = with_manifest_fstar(pd.read_csv(run / "illustrative.csv"), manifest)
        (run / "illustrative.md").write_text(illustrative_markdown(ill), encoding="utf-8")
        done.append("illustrative.md")
    if not done:
        print(f"nothing to analyze in {run}", file=sys.stderr)
        return 1
    print(f"{run}: " + ", ".join(done), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
