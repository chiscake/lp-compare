<#
.SYNOPSIS
  Full pipeline on Windows: generate instances, test, publish Native AOT, benchmark, analyze.

.EXAMPLE
  .\scripts\run.ps1                       # quick mode (~10–20 min): ILL + S + M, fewer tasks
  .\scripts\run.ps1 -Mode full             # everything from experiments\params.ini (hours)
  .\scripts\run.ps1 -Isa x86-64-v3       # instruction set for Native AOT (default: native)
  .\scripts\run.ps1 -NoAot                # JIT build if the C++ build tools for AOT are missing
  .\scripts\run.ps1 -Label my-pc          # results\<label>\ (default: computer name)
#>
param(
    [ValidateSet("quick", "full")] [string] $Mode = "quick",
    [string] $Isa = "native",
    [switch] $NoAot,
    [string] $Label = [Environment]::MachineName,
    [switch] $SkipTests
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Checked {
    param([string] $Exe, [string[]] $Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw "dotnet SDK 10 not found: https://dotnet.microsoft.com/download" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv not found: https://docs.astral.sh/uv/getting-started/installation/" }

$out = Join-Path "results" $Label
New-Item -ItemType Directory -Force $out | Out-Null
if ($Mode -eq "quick") { $classes = "ILL,S,M"; $runClasses = "S,M"; $maxTasks = 10; $repeats = "3"; $lambdaTasks = 5 }
else { $classes = ""; $runClasses = ""; $maxTasks = 0; $repeats = ""; $lambdaTasks = 20 }

Write-Host "== 1/6 instances (seeded; hashes checked against experiments\manifest.csv)"
if ($classes) { Invoke-Checked uv @("run", "--project", "python", "lpgen", "--classes", $classes) }
else { Invoke-Checked uv @("run", "--project", "python", "lpgen") }

if (-not $SkipTests) {
    Write-Host "== 2/6 unit tests"
    Invoke-Checked dotnet @("test", "-c", "Release", "--nologo", "-v", "q")
}

Write-Host "== 3/6 build"
# Windows PowerShell 5.1 has no $IsWindows; PowerShell 7 on Linux/macOS sets $IsLinux / $IsMacOS.
$onWindows = -not (Test-Path variable:IsWindows) -or $IsWindows
$os = if ($onWindows) { "win" } elseif ($IsMacOS) { "osx" } else { "linux" }
$cpu = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { "arm64" } else { "x64" }
$rid = "$os-$cpu"
$exe = if ($onWindows) { ".exe" } else { "" }
if (-not $NoAot) {
    $publishArgs = @("publish", "src/LpBench.Cli", "-c", "Release", "-r", $rid, "-o", "artifacts/$rid", "--nologo", "-v", "q")
    if ($Isa) { $publishArgs += "-p:LpIsa=$Isa" }
    Invoke-Checked dotnet $publishArgs
    $bin = "artifacts/$rid/lpbench$exe"
}
else {
    Invoke-Checked dotnet @("build", "src/LpBench.Cli", "-c", "Release", "--nologo", "-v", "q")
    $bin = "src/LpBench.Cli/bin/Release/net10.0/lpbench$exe"
}

Write-Host "== 4/6 environment -> $out\env.txt"
Invoke-Checked $bin @("env", "--out", "$out/env.txt")

Write-Host "== 5/6 benchmark"
Invoke-Checked $bin @("run", "--set", "illustrative", "--out", "$out/illustrative.csv")
$runArgs = @("run", "--set", "main", "--out", "$out/raw.csv")
if ($runClasses) { $runArgs += @("--classes", $runClasses) }
if ($maxTasks -gt 0) { $runArgs += @("--max-tasks", "$maxTasks") }
if ($repeats) { $runArgs += @("--repeats", $repeats) }
Invoke-Checked $bin $runArgs
function Get-Param([string] $Key) {
    ((Get-Content experiments/params.ini | Where-Object { $_ -match "^$([regex]::Escape($Key))\s*=" }) -split '=')[1].Trim() -replace '\s', ''
}
$traceArgs = @("trace", "--set", "main", "--methods", "dln,dln-log", "--out", "$out/trace.csv")
if ($runClasses) { $traceArgs += @("--classes", $runClasses) }
if ($maxTasks -gt 0) { $traceArgs += @("--max-tasks", "$maxTasks") }
Invoke-Checked $bin $traceArgs
$highsArgs = @("run", "--project", "python", "lphighs", "--out", "$out/highs.csv")
if ($runClasses) { $highsArgs += @("--classes", $runClasses) }
Invoke-Checked uv $highsArgs
$lambdas = Get-Param "lambda_study.lambdas"
Invoke-Checked $bin @("grid", "--set", "main", "--classes", "S,M", "--structures", "dense,degenerate",
    "--max-tasks", "$lambdaTasks", "--lambdas", $lambdas, "--modes", "plain,log", "--out", "$out/lambda.csv")
if ($Mode -eq "full") {
    Invoke-Checked $bin @("grid", "--set", "main", "--classes", "L", "--structures", "dense,degenerate",
        "--max-tasks", "5", "--lambdas", $lambdas, "--modes", "plain,log", "--out", "$out/lambda_large.csv")
    Invoke-Checked $bin @("grid", "--set", "main", "--classes", "M", "--structures", "dense", "--max-tasks", "10",
        "--lambdas", (Get-Param "long_study.lambdas"), "--modes", "plain,log", "--max-iter", (Get-Param "long_study.max_iter"),
        "--out", "$out/lambda_long.csv")
}
$kappas = Get-Param "kappa_study.kappas"
$lamS = Get-Param "dln.lambda.S"
Invoke-Checked $bin @("grid", "--set", "main", "--classes", "S", "--lambdas", $lamS, "--kappas", $kappas, "--out", "$out/kappa.csv")
Invoke-Checked $bin @("grid", "--set", "illustrative", "--lambdas", $lamS, "--kappas", $kappas, "--out", "$out/kappa_illustrative.csv")

Write-Host "== 6/6 analysis"
Invoke-Checked uv @("run", "--project", "python", "lpanalyze", "--run", $out)
Write-Host "done: $out"
