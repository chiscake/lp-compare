using System.Diagnostics;
using System.Globalization;
using System.Text;
using LpBench.Cli;
using LpBench.Core;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

const string Usage = """
lpbench — simplex (Dantzig/Bland), Mehrotra IPM and DLN on resource allocation LPs.

Commands:
  env [--out FILE] [--no-bench]                 Environment report (CPU, ISA, Vector<double> width, subnormal cost).
  solve --file F.lpb [--fstar V] [--params P]   Solve one instance with all methods and print a table.
        [--methods dantzig,bland,ipm,dln] [--lambda L] [--kappa K]
  run   --set NAME --out FILE.csv               Timed benchmark over manifest entries (warm-up + repeats).
        [--manifest experiments/manifest.csv] [--data data] [--params experiments/params.ini]
        [--methods dantzig,bland,ipm,dln] [--classes S,M] [--structures dense,sparse,degenerate]
        [--max-tasks N] [--repeats K] [--warmup W]
  grid  --set NAME --out FILE.csv --lambdas a,b,.. [--kappas a,b,..] [--modes plain,log]
        [--manifest ..] [--data ..] [--params ..] [--classes ..] [--structures ..] [--max-tasks N]
                                                 One run of DLN per (task, λ, κ, mode); for tuning and the λ study.
""";

if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    Console.WriteLine(Usage);
    return 0;
}

var opts = ParseOptions(args.AsSpan(1));
try
{
    return args[0] switch
    {
        "env" => CmdEnv(opts),
        "solve" => CmdSolve(opts),
        "run" => CmdRun(opts),
        "grid" => CmdGrid(opts),
        _ => Fail($"Unknown command '{args[0]}'.\n\n{Usage}"),
    };
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or DirectoryNotFoundException or FormatException or KeyNotFoundException or InvalidDataException)
{
    return Fail(ex.Message);
}

static int Fail(string message)
{
    Console.Error.WriteLine($"error: {message}");
    return 2;
}

static Dictionary<string, string> ParseOptions(ReadOnlySpan<string> a)
{
    var d = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < a.Length; i++)
    {
        if (!a[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument '{a[i]}'.");
        string key = a[i][2..];
        if (i + 1 < a.Length && !a[i + 1].StartsWith("--", StringComparison.Ordinal)) d[key] = a[++i];
        else d[key] = "true";
    }
    return d;
}

static string Opt(Dictionary<string, string> o, string key, string fallback) => o.TryGetValue(key, out var v) ? v : fallback;

static string[] ListOpt(Dictionary<string, string> o, string key, string[] fallback) =>
    o.TryGetValue(key, out var v) ? v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) : fallback;

static double[] Doubles(string[] s) => s.Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray();

static int CmdEnv(Dictionary<string, string> o)
{
    var report = EnvironmentReport.Build();
    if (!o.ContainsKey("no-bench")) report += EnvironmentReport.DenormalBenchmark();
    Console.Write(report);
    if (o.TryGetValue("out", out var path))
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, report);
    }
    return 0;
}

static Params LoadParams(Dictionary<string, string> o) => Params.Load(Opt(o, "params", "experiments/params.ini"));

static int CmdSolve(Dictionary<string, string> o)
{
    var p = LoadParams(o);
    var problem = ProblemFile.Read(o.TryGetValue("file", out var f) ? f : throw new ArgumentException("--file is required."));
    double fStar = o.TryGetValue("fstar", out var fs) ? double.Parse(fs, CultureInfo.InvariantCulture) : double.NaN;
    var sf = StandardForm.FromResourceProblem(problem);
    double? lambda = o.TryGetValue("lambda", out var l) ? double.Parse(l, CultureInfo.InvariantCulture) : null;
    double? kappa = o.TryGetValue("kappa", out var k) ? double.Parse(k, CultureInfo.InvariantCulture) : null;
    string cls = Opt(o, "class", "S");

    Console.WriteLine($"instance {problem.Id}: m={problem.M}, n={problem.N}, f*={fStar:R}");
    Console.WriteLine($"{"method",-16} {"status",-16} {"iter",8} {"time_ms",10} {"objective",18} {"gap",10} {"viol",10}  x");
    foreach (var method in ListOpt(o, "methods", SolverFactory.AllMethods))
    {
        var solver = SolverFactory.Create(method, p, cls, lambda, kappa);
        var limits = SolverFactory.Limits(method, p, cls, sf.Rows, sf.Cols);
        var sw = Stopwatch.StartNew();
        var r = solver.Solve(sf, limits);
        sw.Stop();
        var acc = Metrics.Evaluate(problem, r.Z, double.IsNaN(fStar) ? 0 : fStar);
        string xs = r.Z.Length >= problem.N && problem.N <= 8
            ? string.Join(" ", r.Z.Take(problem.N).Select(v => v.ToString("G6", CultureInfo.InvariantCulture)))
            : "";
        Console.WriteLine($"{method,-16} {r.Status,-16} {r.Iterations,8} {sw.Elapsed.TotalMilliseconds,10:F3} {acc.Objective,18:G12} {acc.Gap,10:E2} {acc.Violation,10:E2}  {xs}");
    }
    return 0;
}

static IEnumerable<ManifestEntry> SelectTasks(Dictionary<string, string> o)
{
    var all = Manifest.Load(Opt(o, "manifest", "experiments/manifest.csv"));
    string set = o.TryGetValue("set", out var s) ? s : throw new ArgumentException("--set is required.");
    var classes = ListOpt(o, "classes", []);
    var structures = ListOpt(o, "structures", []);
    int maxTasks = int.Parse(Opt(o, "max-tasks", "0"), CultureInfo.InvariantCulture);
    var chosen = all.Where(e => e.Set == set
                                && (classes.Length == 0 || classes.Contains(e.Class))
                                && (structures.Length == 0 || structures.Contains(e.Structure)));
    if (maxTasks > 0) chosen = chosen.GroupBy(e => (e.Class, e.Structure)).SelectMany(g => g.Take(maxTasks));
    var list = chosen.ToList();
    if (list.Count == 0) throw new ArgumentException($"No manifest entries match set '{set}' and the given filters.");
    return list;
}

static LpProblem LoadInstance(Dictionary<string, string> o, ManifestEntry e) =>
    ProblemFile.Read(Path.Combine(Opt(o, "data", "data"), e.File), e.Id);

static StreamWriter OpenCsv(string path, string header)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
    var w = new StreamWriter(path, false, new UTF8Encoding(false)) { AutoFlush = true };
    w.WriteLine(header);
    return w;
}

static string G(double v) => v.ToString("R", CultureInfo.InvariantCulture);

static int CmdRun(Dictionary<string, string> o)
{
    var p = LoadParams(o);
    int? repeatsOverride = o.TryGetValue("repeats", out var rep) ? int.Parse(rep, CultureInfo.InvariantCulture) : null;
    int? warmupOverride = o.TryGetValue("warmup", out var wu) ? int.Parse(wu, CultureInfo.InvariantCulture) : null;
    var methods = ListOpt(o, "methods", SolverFactory.AllMethods);
    var tasks = SelectTasks(o).ToList();
    string outPath = o.TryGetValue("out", out var op) ? op : throw new ArgumentException("--out is required.");
    WriteRunInfo(outPath, p, o);

    using var csv = OpenCsv(outPath,
        "task_id,set,class,structure,m,n,seed,method,repeat,time_ms,iterations,degenerate_pivots,numerical_events,objective,fstar,gap,viol,repaired_gap,residual,status");
    var total = Stopwatch.StartNew();
    for (int t = 0; t < tasks.Count; t++)
    {
        var e = tasks[t];
        var problem = LoadInstance(o, e);
        var sf = StandardForm.FromResourceProblem(problem);
        foreach (var method in methods)
        {
            var solver = SolverFactory.Create(method, p, e.Class);
            var limits = SolverFactory.Limits(method, p, e.Class, sf.Rows, sf.Cols);
            int repeats = Math.Min(repeatsOverride ?? int.MaxValue, p.IForMethodClass("repeats", method, e.Class));
            int warmup = Math.Min(warmupOverride ?? int.MaxValue, p.IForMethodClass("warmup", method, e.Class));
            for (int r = 0; r < warmup + repeats; r++)
            {
                var sw = Stopwatch.StartNew();
                var res = solver.Solve(sf, limits);
                sw.Stop();
                var acc = Metrics.Evaluate(problem, res.Z, e.FStar);
                int repeat = r - warmup + 1;
                csv.WriteLine(string.Join(',', e.Id, e.Set, e.Class, e.Structure, e.M, e.N, e.Seed, method, repeat,
                    G(sw.Elapsed.TotalMilliseconds), res.Iterations, res.DegeneratePivots, res.NumericalEvents,
                    G(acc.Objective), G(e.FStar), G(acc.Gap), G(acc.Violation), G(acc.RepairedGap), G(res.Residual), res.Status));
                if (res.Status == SolveStatus.Timeout) break;
            }
        }
        Console.Error.WriteLine($"[{t + 1}/{tasks.Count}] {e.Id} done, elapsed {total.Elapsed:hh\\:mm\\:ss}");
    }
    return 0;
}

static int CmdGrid(Dictionary<string, string> o)
{
    var p = LoadParams(o);
    var lambdas = Doubles(ListOpt(o, "lambdas", []));
    if (lambdas.Length == 0) throw new ArgumentException("--lambdas is required.");
    var kappas = ListOpt(o, "kappas", []);
    var modes = ListOpt(o, "modes", ["plain"]);
    var tasks = SelectTasks(o).ToList();
    string outPath = o.TryGetValue("out", out var op) ? op : throw new ArgumentException("--out is required.");
    WriteRunInfo(outPath, p, o);

    using var csv = OpenCsv(outPath,
        "task_id,set,class,structure,m,n,seed,mode,lambda,kappa,time_ms,iterations,objective,fstar,gap,viol,repaired_gap,residual,status");
    var total = Stopwatch.StartNew();
    for (int t = 0; t < tasks.Count; t++)
    {
        var e = tasks[t];
        var problem = LoadInstance(o, e);
        var sf = StandardForm.FromResourceProblem(problem);
        var kappaValues = kappas.Length > 0 ? Doubles(kappas) : [p.DForClass("dln.kappa", e.Class)];
        foreach (var mode in modes)
        foreach (var kappa in kappaValues)
        foreach (var lambda in lambdas)
        {
            var solver = SolverFactory.Create("dln", p, e.Class, lambda, kappa, mode == "log");
            var limits = SolverFactory.Limits("dln", p, e.Class, sf.Rows, sf.Cols);
            var sw = Stopwatch.StartNew();
            var res = solver.Solve(sf, limits);
            sw.Stop();
            var acc = Metrics.Evaluate(problem, res.Z, e.FStar);
            csv.WriteLine(string.Join(',', e.Id, e.Set, e.Class, e.Structure, e.M, e.N, e.Seed, mode, G(lambda), G(kappa),
                G(sw.Elapsed.TotalMilliseconds), res.Iterations, G(acc.Objective), G(e.FStar), G(acc.Gap), G(acc.Violation),
                G(acc.RepairedGap), G(res.Residual), res.Status));
        }
        Console.Error.WriteLine($"[{t + 1}/{tasks.Count}] {e.Id} done, elapsed {total.Elapsed:hh\\:mm\\:ss}");
    }
    return 0;
}

static void WriteRunInfo(string csvPath, Params p, Dictionary<string, string> o)
{
    var sb = new StringBuilder();
    sb.Append("# command options\n");
    foreach (var kv in o.OrderBy(kv => kv.Key, StringComparer.Ordinal)) sb.Append($"option.{kv.Key}: {kv.Value}\n");
    sb.Append("# effective parameters\n");
    foreach (var kv in p.All) sb.Append($"param.{kv.Key}: {kv.Value}\n");
    sb.Append("# environment\n").Append(EnvironmentReport.Build());
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(csvPath))!);
    File.WriteAllText(Path.ChangeExtension(csvPath, ".runinfo.txt"), sb.ToString());
}
