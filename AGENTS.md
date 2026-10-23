# AGENTS.md

Guide for AI agents working in this repository. Human-oriented docs are in Russian (`README.md`, `docs/`); this file is English.

## What this is

A reproducible benchmark for a conference paper (VII МНПК «Цифровая трансформация – шаг в будущее», ЭФ БГУ, 6 Nov 2026, article <= 3 pages, in Russian). It compares methods for resource-allocation LPs `max c'x, Ax <= b, x >= 0` (`A >= 0`, `b > 0`, `c > 0`):

- tableau simplex (Dantzig and Bland entering rules; same leaving rule),
- Mehrotra primal-dual interior point method (IPM),
- DLN: gradient descent on a diagonal linear network (Wang, Ghosal, Mazumder, arXiv:2310.02535), which implicitly solves an entropy-regularized LP controlled by lambda.

All solvers are written in C# (.NET 10, Native AOT) on shared dense kernels. Reference optima f* come from HiGHS (Python) and are stored in `experiments/manifest.csv`; C# runs never call an external solver.

Headline finding: Dantzig simplex is accurate and fastest; Bland needs many more pivots; IPM takes 8-22 iterations but costs O(m^3) each; DLN is accurate only on class S because entropic bias requires lambda below the double-precision window on M and larger.

## Layout

| Path | Purpose |
|---|---|
| `src/LpBench.Core/` | Solvers (`TableauSimplex`, `MehrotraIpm`, `DlnSolver`), `DenseMath` (SIMD kernels), `Metrics`, `Problem`/`StandardForm`, `ProblemFile` (LPB1 binary format), `SolveResult` |
| `src/LpBench.Cli/` | `lpbench` CLI: `env`, `solve`, `run`, `grid`; `Config.cs` (params.ini, manifest, solver factory) |
| `tests/LpBench.Tests/` | xUnit, 17 tests (Beale cycling, illustrative problem, IPM, DLN bias, I/O, metrics) |
| `python/lpbench_py/` | `generate.py` (`lpgen`: seeded instances + HiGHS reference), `analyze.py` (`lpanalyze`: tables, figures) |
| `experiments/params.ini` | Frozen method parameters, iteration limits, timeouts, repeats, lambda grid |
| `experiments/manifest.csv` | 661 instances: seed, SHA-256, f*, certificate |
| `scripts/run.sh`, `scripts/run.ps1` | End-to-end pipeline (quick/full mode) for Linux/macOS and Windows |
| `docs/article.md` | The paper text (Russian) |
| `docs/theory-and-plan.md` | Theory, DLN reduction, hypotheses and their outcomes (section 11) |
| `docs/decisions.md` | Decision log: versions, parameters, post-run changes |
| `results/cloud-xeon/` | Committed results of the reference run (raw CSVs, summaries, figures, env) |
| `data/` | Generated instances (~0.85 GB, gitignored; regenerate with `lpgen`) |

## Commands

Prerequisites: .NET SDK 10, `uv`; for Native AOT a C/C++ linker (clang + zlib1g-dev on Linux). On Linux set `export DOTNET_ROOT=$HOME/.dotnet` if the SDK is user-installed.

```bash
uv run --project python lpgen                  # generate data, verify SHA-256 against manifest
uv run --project python lpgen --verify         # hashes only
dotnet test -c Release                         # 17 tests
scripts/run.sh [--mode full] [--isa native] [--no-aot] [--label NAME]
scripts\run.ps1 [-Mode full] [-Isa native] [-NoAot] [-Label NAME]   # Windows
uv run --project python lpanalyze --run results/<label>   # merges raw*.csv, writes summary/validation/figures
```

Quick mode takes ~20 min; full ~6-8 h. `lpbench --help` lists CLI options.

## Facts and invariants

- Accuracy outputs (gap, violation, iterations, status) must be CPU-independent: DLN limits are iteration counts, not time. Only timings vary by machine.
- Do not change generator code or seeds without re-verifying hashes; changing instances invalidates the manifest and results.
- The simplex has no phase 1 (requires `b >= 0`); DLN requires `A >= 0`, `b > 0`. Violations return status `Unsupported`.
- DLN uses the Remark 1.1 reduction (sum constraint M, costs mapped to [0,1], rows normalized). Defaults in `params.ini`: lambda 0.003 (0.005 for S), kappa 30. Large kappa biases the limit.
- f* is taken from the HiGHS solution (dual simplex or IPM+crossover) with the best primal-dual certificate (`fstar_cert` <= 1e-12). `lpanalyze` recomputes gaps from the current manifest.
- Native AOT ISA is set via `-p:LpIsa=` (valid: `native`, `x86-64-v2/v3/v4`, `apple-m1`, ...). `native` binaries are not portable.
- Cloud timings are reference only (shared VM). Final timings should come from the author's AMD x64 Windows machine, 12 GB RAM.
- Subnormal doubles are very slow on x86; lambda below ~1.4e-3 hits them, below ~6.7e-4 `u0` underflows to zero.
- Simplex pivot limit is `50 * (2m + n)` (`simplex.max_iter_factor` times rows+columns of the standard form). Pivot counts on degenerate tasks depend on the exact pivot update (reciprocal scaling, exact zeroing of the pivot column): a naive numpy variant gives 129 instead of 102 pivots on `main_M_degenerate_000`; the C# result is deterministic.
- DLN kappa study: `kappa.md` (kappa 10 is more accurate than the frozen kappa 30 on S, but defaults are kept to avoid rerunning M/L/XL).
- Bland hits the pivot limit on some L and most sparse/degenerate XL tasks; `validation.md` checks simplex accuracy on solved tasks only.

## Conventions

- Reply to the user in Russian; code, comments and this file in English.
- Math in chat: `\(...\)` inline, `$...$` for blocks; in `.md` files use standard `$` LaTeX.
- Do not open image files (png/svg figures); inspect the CSV/markdown summaries instead.
- Work on branch `main`; commit and push; no PRs unless asked.
- Do not add narrating comments in code. Prefer editing existing files over creating new ones.
- Article constraints: abstract 100-500 chars then a blank line; 5-7 keywords (<= 3 words each, separated by `;`) then a blank line; do not include the illustrative problem's narrative text; GOST STB 7.208-2008 references.

## Pending / ideas

- Replace cloud timings in `docs/article.md` and the README with the author's local full-run results; add the author name and CPU model.
- Check the article fits 3 pages; trim the methodology section if needed.
