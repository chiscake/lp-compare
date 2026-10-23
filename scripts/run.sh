#!/usr/bin/env bash
# Full pipeline on Linux/macOS: generate instances, test, publish Native AOT, benchmark, analyze.
#
#   scripts/run.sh                      # quick mode (~10–20 min): ILL + S + M, fewer tasks
#   scripts/run.sh --mode full          # everything from experiments/params.ini (hours)
#   scripts/run.sh --isa x86-64-v3     # instruction set for Native AOT (default: native)
#   scripts/run.sh --no-aot             # JIT build if the AOT toolchain (clang/Xcode) is missing
#   scripts/run.sh --label my-laptop    # results/<label>/ (default: host name)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=quick
ISA=native
AOT=1
LABEL="$(hostname -s 2>/dev/null || hostname)"
SKIP_TESTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --isa) ISA="$2"; shift 2 ;;
    --no-aot) AOT=0; shift ;;
    --label) LABEL="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

command -v dotnet >/dev/null || { echo "dotnet SDK 10 not found (https://dotnet.microsoft.com/download)" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/getting-started/installation/)" >&2; exit 1; }

OUT="results/$LABEL"
mkdir -p "$OUT"
case "$MODE" in
  quick) CLASSES="ILL,S,M"; RUN_CLASSES="S,M"; MAX_TASKS=10; REPEATS=3; LAMBDA_TASKS=5 ;;
  full)  CLASSES="";        RUN_CLASSES="";    MAX_TASKS=0;  REPEATS="";  LAMBDA_TASKS=20 ;;
  *) echo "mode must be quick or full" >&2; exit 2 ;;
esac

echo "== 1/6 instances (seeded; hashes checked against experiments/manifest.csv)"
if [[ -n "$CLASSES" ]]; then uv run --project python lpgen --classes "$CLASSES"; else uv run --project python lpgen; fi

if [[ $SKIP_TESTS -eq 0 ]]; then
  echo "== 2/6 unit tests"
  dotnet test -c Release --nologo -v q
fi

echo "== 3/6 build"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) RID=linux-x64 ;; Linux-aarch64) RID=linux-arm64 ;;
  Darwin-x86_64) RID=osx-x64 ;; Darwin-arm64) RID=osx-arm64 ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
if [[ $AOT -eq 1 ]]; then
  ISA_ARG=(); [[ -n "$ISA" ]] && ISA_ARG=(-p:LpIsa="$ISA")
  dotnet publish src/LpBench.Cli -c Release -r "$RID" ${ISA_ARG[@]+"${ISA_ARG[@]}"} -o "artifacts/$RID" --nologo -v q
  BIN="artifacts/$RID/lpbench"
else
  dotnet build src/LpBench.Cli -c Release --nologo -v q
  BIN="src/LpBench.Cli/bin/Release/net10.0/lpbench"
fi

echo "== 4/6 environment -> $OUT/env.txt"
"$BIN" env --out "$OUT/env.txt"

echo "== 5/6 benchmark"
"$BIN" run --set illustrative --out "$OUT/illustrative.csv"
RUN_ARGS=(--set main --out "$OUT/raw.csv")
[[ -n "$RUN_CLASSES" ]] && RUN_ARGS+=(--classes "$RUN_CLASSES")
[[ $MAX_TASKS -gt 0 ]] && RUN_ARGS+=(--max-tasks "$MAX_TASKS")
[[ -n "$REPEATS" ]] && RUN_ARGS+=(--repeats "$REPEATS")
"$BIN" run "${RUN_ARGS[@]}"
"$BIN" grid --set main --classes S,M --structures dense,degenerate --max-tasks "$LAMBDA_TASKS" \
  --lambdas "$(grep -E '^lambda_study.lambdas' experiments/params.ini | cut -d= -f2 | tr -d ' ')" \
  --out "$OUT/lambda.csv"
KAPPAS="$(grep -E '^kappa_study.kappas' experiments/params.ini | cut -d= -f2 | tr -d ' ')"
"$BIN" grid --set main --classes S --lambdas "$(grep -E '^dln.lambda.S' experiments/params.ini | cut -d= -f2 | tr -d ' ')" \
  --kappas "$KAPPAS" --out "$OUT/kappa.csv"
"$BIN" grid --set illustrative --lambdas "$(grep -E '^dln.lambda.S' experiments/params.ini | cut -d= -f2 | tr -d ' ')" \
  --kappas "$KAPPAS" --out "$OUT/kappa_illustrative.csv"

echo "== 6/6 analysis"
uv run --project python lpanalyze --run "$OUT"
echo "done: $OUT"
