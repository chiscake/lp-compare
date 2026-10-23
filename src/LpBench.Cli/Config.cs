using System.Globalization;
using LpBench.Core;

namespace LpBench.Cli;

/// <summary>Flat "key = value" parameter file (experiments/params.ini); '#' starts a comment.</summary>
internal sealed class Params
{
    private readonly Dictionary<string, string> _values = new(StringComparer.OrdinalIgnoreCase);

    public static Params Load(string path)
    {
        var p = new Params();
        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw;
            int hash = line.IndexOf('#');
            if (hash >= 0) line = line[..hash];
            line = line.Trim();
            if (line.Length == 0) continue;
            int eq = line.IndexOf('=');
            if (eq < 0) throw new FormatException($"{path}: bad line '{raw}'.");
            p._values[line[..eq].Trim()] = line[(eq + 1)..].Trim();
        }
        return p;
    }

    public void Set(string key, string value) => _values[key] = value;

    public string Get(string key) =>
        _values.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException($"Parameter '{key}' is missing.");

    public double D(string key) => double.Parse(Get(key), CultureInfo.InvariantCulture);
    public int I(string key) => int.Parse(Get(key), CultureInfo.InvariantCulture);
    public bool B(string key) => bool.Parse(Get(key));

    /// <summary>Looks up "key.CLASS" first, then "key".</summary>
    public string ForClass(string key, string cls) =>
        _values.TryGetValue($"{key}.{cls}", out var v) ? v : Get(key);

    /// <summary>Looks up "key.METHOD.CLASS", then "key.CLASS", then "key".</summary>
    public int IForMethodClass(string key, string method, string cls) =>
        _values.TryGetValue($"{key}.{method}.{cls}", out var v)
            ? int.Parse(v, CultureInfo.InvariantCulture)
            : IForClass(key, cls);

    public double DForClass(string key, string cls) => double.Parse(ForClass(key, cls), CultureInfo.InvariantCulture);
    public int IForClass(string key, string cls) => int.Parse(ForClass(key, cls), CultureInfo.InvariantCulture);

    public IEnumerable<KeyValuePair<string, string>> All => _values.OrderBy(kv => kv.Key, StringComparer.Ordinal);
}

internal sealed record ManifestEntry(string Id, string Set, string Class, string Structure, int M, int N, long Seed, string File, double FStar);

internal static class Manifest
{
    public static List<ManifestEntry> Load(string path)
    {
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        int Col(string name) => Array.IndexOf(header, name) is var k and >= 0 ? k : throw new FormatException($"{path}: column '{name}' missing.");
        int id = Col("id"), set = Col("set"), cls = Col("class"), st = Col("structure"), m = Col("m"), n = Col("n"),
            seed = Col("seed"), file = Col("file"), fs = Col("fstar");
        var list = new List<ManifestEntry>();
        foreach (var line in lines.Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var f = line.Split(',');
            list.Add(new ManifestEntry(f[id], f[set], f[cls], f[st], int.Parse(f[m], CultureInfo.InvariantCulture),
                int.Parse(f[n], CultureInfo.InvariantCulture), long.Parse(f[seed], CultureInfo.InvariantCulture), f[file],
                double.Parse(f[fs], CultureInfo.InvariantCulture)));
        }
        return list;
    }
}

internal static class SolverFactory
{
    public static readonly string[] AllMethods = ["dantzig", "bland", "ipm", "dln"];

    public static ILpSolver Create(string method, Params p, string cls, double? lambda = null, double? kappa = null, bool? logDomain = null,
        Action<int, double[]>? observer = null) =>
        method switch
        {
            "dantzig" => new TableauSimplex(PivotRule.Dantzig)
            {
                ReducedCostTolerance = p.D("simplex.tol_d"), PivotTolerance = p.D("simplex.tol_p"),
            },
            "bland" => new TableauSimplex(PivotRule.Bland)
            {
                ReducedCostTolerance = p.D("simplex.tol_d"), PivotTolerance = p.D("simplex.tol_p"),
            },
            "ipm" => new MehrotraIpm { Tolerance = p.D("ipm.tol"), StepFraction = p.D("ipm.step_fraction") },
            "dln" or "dln-log" => new DlnSolver
            {
                Lambda = lambda ?? p.DForClass("dln.lambda", cls),
                Kappa = kappa ?? p.DForClass("dln.kappa", cls),
                Tolerance = p.D("dln.tol"),
                RowScale = p.B("dln.row_scale"),
                LogDomain = logDomain ?? (method == "dln-log" || p.B("dln.log_domain")),
                Observer = observer,
            },
            _ => throw new ArgumentException($"Unknown method '{method}'. Known: {string.Join(", ", AllMethods)}, dln-log."),
        };

    public static SolverLimits Limits(string method, Params p, string cls, int rows, int cols) => new()
    {
        TimeLimit = TimeSpan.FromSeconds(p.DForClass("timeout", cls)),
        MaxIterations = method switch
        {
            "dantzig" or "bland" => (int)Math.Min(int.MaxValue, (long)p.I("simplex.max_iter_factor") * (rows + cols)),
            "ipm" => p.I("ipm.max_iter"),
            _ => p.IForClass("dln.max_iter", cls),
        },
    };
}
