namespace LpBench.Core;

public enum PivotRule
{
    /// <summary>Entering: most negative reduced cost (ties: smallest index).</summary>
    Dantzig,
    /// <summary>Entering: smallest index with negative reduced cost.</summary>
    Bland,
}

/// <summary>
/// Dense tableau primal simplex starting from the unit-column basis (no phase 1).
/// Both rules share everything except the entering-variable choice; the leaving variable is always
/// the one with the minimal ratio, ties broken by the smallest basic-variable index (Bland's leaving rule).
/// </summary>
public sealed class TableauSimplex(PivotRule rule) : ILpSolver
{
    public PivotRule Rule { get; } = rule;
    public double ReducedCostTolerance { get; init; } = 1e-9;
    public double PivotTolerance { get; init; } = 1e-9;
    /// <summary>Relative tolerance for treating two ratios as a tie.</summary>
    public double RatioTieTolerance { get; init; } = 1e-12;

    public string Name => Rule == PivotRule.Dantzig ? "simplex-dantzig" : "simplex-bland";

    public SolveResult Solve(StandardForm sf, SolverLimits limits)
    {
        if (sf.ValidateForSimplex() is { } why) return SolveResult.Fail(SolveStatus.Unsupported, why);
        var deadline = limits.StartDeadline();

        int m = sf.Rows, cols = sf.Cols, w = cols + 1;
        var t = new double[(m + 1) * w];
        for (int i = 0; i < m; i++)
        {
            sf.A.AsSpan(i * cols, cols).CopyTo(t.AsSpan(i * w, cols));
            t[i * w + cols] = sf.B[i];
        }
        var basis = (int[])sf.InitialBasis.Clone();
        var obj = t.AsSpan(m * w, w);
        sf.C.AsSpan().CopyTo(obj);
        for (int i = 0; i < m; i++)
        {
            double cb = sf.C[basis[i]];
            if (cb != 0) DenseMath.Axpy(-cb, t.AsSpan(i * w, w), obj);
        }

        int iterations = 0, degenerate = 0;
        SolveStatus status;
        while (true)
        {
            int k = ChooseEntering(t.AsSpan(m * w, cols));
            if (k < 0) { status = SolveStatus.Solved; break; }
            if (iterations >= limits.MaxIterations) { status = SolveStatus.IterationLimit; break; }
            if (deadline.Expired) { status = SolveStatus.Timeout; break; }

            int r = ChooseLeaving(t, m, w, k, basis, out double theta);
            if (r < 0) { status = SolveStatus.Unbounded; break; }
            if (theta <= RatioTieTolerance) degenerate++;

            Pivot(t, m, w, r, k);
            basis[r] = k;
            iterations++;
        }

        var z = new double[cols];
        for (int i = 0; i < m; i++) z[basis[i]] = t[i * w + cols];
        double objective = DenseMath.Dot(sf.C, z);
        return new SolveResult
        {
            Status = status,
            Z = z,
            Objective = objective,
            Iterations = iterations,
            DegeneratePivots = degenerate,
            Residual = 0,
            Message = status == SolveStatus.Unbounded ? "Entering column has no positive entry." : "",
        };
    }

    private int ChooseEntering(ReadOnlySpan<double> d)
    {
        double tol = -ReducedCostTolerance;
        if (Rule == PivotRule.Bland)
        {
            for (int j = 0; j < d.Length; j++) if (d[j] < tol) return j;
            return -1;
        }
        int best = -1;
        double bestVal = tol;
        for (int j = 0; j < d.Length; j++)
            if (d[j] < bestVal) { bestVal = d[j]; best = j; }
        return best;
    }

    private int ChooseLeaving(double[] t, int m, int w, int k, int[] basis, out double theta)
    {
        int rhs = w - 1;
        theta = double.PositiveInfinity;
        for (int i = 0; i < m; i++)
        {
            double a = t[i * w + k];
            if (a > PivotTolerance)
            {
                double ratio = Math.Max(0.0, t[i * w + rhs]) / a;
                if (ratio < theta) theta = ratio;
            }
        }
        if (double.IsPositiveInfinity(theta)) return -1;

        double limit = theta + RatioTieTolerance * Math.Max(1.0, Math.Abs(theta));
        int best = -1;
        for (int i = 0; i < m; i++)
        {
            double a = t[i * w + k];
            if (a > PivotTolerance && Math.Max(0.0, t[i * w + rhs]) / a <= limit && (best < 0 || basis[i] < basis[best])) best = i;
        }
        return best;
    }

    private static void Pivot(double[] t, int m, int w, int r, int k)
    {
        var rowR = t.AsSpan(r * w, w);
        DenseMath.Scale(1.0 / rowR[k], rowR);
        rowR[k] = 1.0;
        for (int i = 0; i <= m; i++)
        {
            if (i == r) continue;
            double f = t[i * w + k];
            if (f == 0) continue;
            var row = t.AsSpan(i * w, w);
            DenseMath.Axpy(-f, rowR, row);
            row[k] = 0.0;
        }
    }
}
