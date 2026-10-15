namespace LpBench.Core;

public readonly record struct Accuracy(double Objective, double Gap, double Violation, double RepairedObjective, double RepairedGap);

public static class Metrics
{
    /// <summary>
    /// Accuracy of z = (x, s) on the original problem max cᵀx, Ax ≤ b, x ≥ 0:
    /// gap = |f − f*| / max(1, |f*|);
    /// viol = max(‖Ax + s − b‖∞, max(0, −min(x, s))) / (1 + ‖b‖∞).
    /// Repaired: x⁺ = max(x, 0) scaled by min(1, minᵢ bᵢ/(Ax⁺)ᵢ) is feasible because A ≥ 0.
    /// </summary>
    public static Accuracy Evaluate(LpProblem p, ReadOnlySpan<double> z, double fStar)
    {
        int m = p.M, n = p.N;
        if (z.Length < n + m) return new Accuracy(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);
        var x = z.Slice(0, n);
        var s = z.Slice(n, m);
        double f = DenseMath.Dot(p.C, x);
        double denom = Math.Max(1.0, Math.Abs(fStar));

        double eq = 0, neg = 0;
        foreach (var v in z.Slice(0, n + m)) neg = Math.Max(neg, -v);
        var xPlus = new double[n];
        for (int j = 0; j < n; j++) xPlus[j] = Math.Max(0.0, x[j]);
        double scale = 1.0;
        for (int i = 0; i < m; i++)
        {
            var row = p.A.AsSpan(i * n, n);
            double ax = DenseMath.Dot(row, x);
            eq = Math.Max(eq, Math.Abs(ax + s[i] - p.B[i]));
            double axPlus = DenseMath.Dot(row, xPlus);
            if (axPlus > p.B[i]) scale = Math.Min(scale, p.B[i] / axPlus);
        }
        double viol = Math.Max(eq, neg) / (1.0 + DenseMath.NormInf(p.B));
        double fRep = scale * DenseMath.Dot(p.C, xPlus);
        return new Accuracy(f, Math.Abs(f - fStar) / denom, viol, fRep, (fStar - fRep) / denom);
    }
}
