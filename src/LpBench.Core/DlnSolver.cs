namespace LpBench.Core;

/// <summary>
/// Gradient descent on a diagonal linear network (Wang, Ghosal, Mazumder, arXiv:2310.02535, Algorithm 1):
/// x = u∘u, f(u) = ½‖Âx − b̂‖², u ← u∘(1 − 2η Âᵀ(Âx − b̂)), u⁰ᵢ = α·exp(−ĉᵢ/(2λ)).
///
/// The paper requires ĉ > 0, α ∈ (0, ½) and a strictly feasible full-row-rank system, so the standard form
/// min cᵀz, Az = b, z ≥ 0 (with A ≥ 0, b > 0) is reduced as in its Remark 1.1:
///   * explicit bound zⱼ ≤ Uⱼ = min_{aᵢⱼ>0} bᵢ/aᵢⱼ, M = (1+θ)·ΣUⱼ;
///   * extra variable t and row 1ᵀz + t = M, variables scaled by 1/M (the feasible set lies in a simplex);
///   * costs mapped affinely to [0, 1] (t has cost 0 before the map). Since 1ᵀ(z, t) is fixed, adding a
///     constant to all costs does not change the argmin, and α = AlphaMax·exp(−ĉ/(2λ)) equals the paper's
///     exp(−c'/(2λ)) for c' = ĉ + 2λ·ln(1/AlphaMax) > 0.
/// Step: ηₖ = min{1/(4‖Âᵀr‖∞), κ/(5L‖u‖∞²)}, L = ‖Â‖₂² by power iteration (Lemma 2.1 with κ = 1).
/// </summary>
public sealed class DlnSolver : ILpSolver
{
    public double Lambda { get; init; } = 1e-2;
    public double Kappa { get; init; } = 1.0;
    /// <summary>Stop when ‖Âx − b̂‖₂ / ‖b̂‖₂ ≤ Tolerance.</summary>
    public double Tolerance { get; init; } = 1e-8;
    public double BoundSlack { get; init; } = 0.1;
    /// <summary>NaN = 1/√(columns), capped below ½.</summary>
    public double AlphaMax { get; init; } = double.NaN;
    /// <summary>Divide every row of [Â | b̂] by its 2-norm (equivalent constraints, better-conditioned loss).</summary>
    public bool RowScale { get; init; } = true;
    /// <summary>Store w = ln u so that coordinates below the double range keep evolving.</summary>
    public bool LogDomain { get; init; }
    public int PowerIterations { get; init; } = 50;

    public string Name => LogDomain ? "dln-log" : "dln";

    public SolveResult Solve(StandardForm sf, SolverLimits limits)
    {
        if (sf.ValidateForDln() is { } why) return SolveResult.Fail(SolveStatus.Unsupported, why);
        var deadline = limits.StartDeadline();
        int m = sf.Rows, n = sf.Cols;
        int rows = m + 1, cols = n + 1;

        double bigM = (1 + BoundSlack) * SumBound(sf);

        var a = new double[rows * cols];
        var b = new double[rows];
        for (int i = 0; i < m; i++)
        {
            sf.A.AsSpan(i * n, n).CopyTo(a.AsSpan(i * cols, n));
            b[i] = sf.B[i] / bigM;
        }
        a.AsSpan(m * cols, cols).Fill(1.0);
        b[m] = 1.0;
        if (RowScale)
        {
            for (int i = 0; i < rows; i++)
            {
                var row = a.AsSpan(i * cols, cols);
                double inv = 1.0 / DenseMath.Norm2(row);
                DenseMath.Scale(inv, row);
                b[i] *= inv;
            }
        }

        double cMin = 0, cMax = 0;
        foreach (var v in sf.C) { cMin = Math.Min(cMin, v); cMax = Math.Max(cMax, v); }
        double range = cMax - cMin;
        if (range <= 0) range = 1;
        double alphaMax = double.IsNaN(AlphaMax) ? Math.Min(0.49, 1.0 / Math.Sqrt(cols)) : AlphaMax;
        double logAlphaMax = Math.Log(alphaMax);

        var u0Log = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            double cost = ((j < n ? sf.C[j] : 0.0) - cMin) / range;
            u0Log[j] = logAlphaMax - cost / (2 * Lambda);
        }

        double lipschitz = 1.02 * SpectralNormSquared(a, rows, cols, PowerIterations);
        double normB = DenseMath.Norm2(b);

        var u = new double[cols];
        var w = u0Log;
        var x = new double[cols];
        var r = new double[rows];
        var g = new double[cols];
        for (int j = 0; j < cols; j++) u[j] = Math.Exp(w[j]);

        int iter = 0;
        double residual = double.NaN;
        SolveStatus status;
        while (true)
        {
            if (LogDomain) for (int j = 0; j < cols; j++) x[j] = Math.Exp(2 * w[j]);
            else for (int j = 0; j < cols; j++) x[j] = u[j] * u[j];

            DenseMath.MatVec(a, rows, cols, x, r);
            for (int i = 0; i < rows; i++) r[i] -= b[i];
            residual = DenseMath.Norm2(r) / normB;
            if (!double.IsFinite(residual)) { status = SolveStatus.NumericalFailure; break; }
            if (residual <= Tolerance) { status = SolveStatus.Solved; break; }
            if (iter >= limits.MaxIterations) { status = SolveStatus.IterationLimit; break; }
            if (deadline.Expired) { status = SolveStatus.Timeout; break; }

            DenseMath.MatTVec(a, rows, cols, r, g);
            double gInf = DenseMath.NormInf(g);
            double uInf = LogDomain ? Math.Exp(Max(w)) : DenseMath.NormInf(u);
            double eta = Math.Min(1.0 / (4 * gInf), Kappa / (5 * lipschitz * uInf * uInf));
            if (!(eta > 0) || !double.IsFinite(eta)) { status = SolveStatus.NumericalFailure; break; }

            if (LogDomain) for (int j = 0; j < cols; j++) w[j] += Math.Log(1 - 2 * eta * g[j]);
            else for (int j = 0; j < cols; j++) u[j] *= 1 - 2 * eta * g[j];
            iter++;
        }

        var z = new double[n];
        for (int j = 0; j < n; j++) z[j] = bigM * x[j];
        return new SolveResult
        {
            Status = status,
            Z = z,
            Objective = DenseMath.Dot(sf.C, z),
            Iterations = iter,
            Residual = residual,
        };
    }

    /// <summary>
    /// Upper bound on 1ᵀz over the feasible set, the smaller of two valid bounds:
    /// Σⱼ min_{aᵢⱼ>0} bᵢ/aᵢⱼ (per-variable bounds) and γ·Σbᵢ with γ = max(1, 1/minⱼ Σᵢaᵢⱼ)
    /// (for y = γ·1 every column satisfies Aᵀy ≥ 1, hence 1ᵀz ≤ yᵀAz = yᵀb).
    /// </summary>
    public static double SumBound(StandardForm sf)
    {
        int m = sf.Rows, n = sf.Cols;
        double perVariable = 0, minColumnSum = double.PositiveInfinity;
        for (int j = 0; j < n; j++)
        {
            double bound = double.PositiveInfinity, colSum = 0;
            for (int i = 0; i < m; i++)
            {
                double aij = sf.A[i * n + j];
                colSum += aij;
                if (aij > 0) bound = Math.Min(bound, sf.B[i] / aij);
            }
            perVariable += bound;
            minColumnSum = Math.Min(minColumnSum, colSum);
        }
        double sumB = 0;
        foreach (var v in sf.B) sumB += v;
        double dual = Math.Max(1.0, 1.0 / minColumnSum) * sumB;
        return Math.Min(perVariable, dual);
    }

    private static double SpectralNormSquared(double[] a, int rows, int cols, int iterations)
    {
        var v = new double[cols];
        var av = new double[rows];
        Array.Fill(v, 1.0 / Math.Sqrt(cols));
        double estimate = 0;
        for (int k = 0; k < iterations; k++)
        {
            DenseMath.MatVec(a, rows, cols, v, av);
            DenseMath.MatTVec(a, rows, cols, av, v);
            estimate = DenseMath.Norm2(v);
            if (estimate == 0) return 1.0;
            DenseMath.Scale(1.0 / estimate, v);
        }
        return estimate;
    }

    private static double Max(double[] v)
    {
        double r = double.NegativeInfinity;
        foreach (var e in v) r = Math.Max(r, e);
        return r;
    }
}
