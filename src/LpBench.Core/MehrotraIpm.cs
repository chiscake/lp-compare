namespace LpBench.Core;

/// <summary>
/// Primal-dual Mehrotra predictor-corrector for min cᵀx, Ax = b, x ≥ 0 (dense normal equations + Cholesky).
/// Starting point: Mehrotra's heuristic (Nocedal–Wright, §14.2). Stops when
/// ‖Ax−b‖₂/(1+‖b‖₂), ‖Aᵀy+s−c‖₂/(1+‖c‖₂) and |cᵀx−bᵀy|/(1+|cᵀx|) are all ≤ Tolerance.
/// </summary>
public sealed class MehrotraIpm : ILpSolver
{
    public double Tolerance { get; init; } = 1e-8;
    /// <summary>Fraction of the step to the boundary.</summary>
    public double StepFraction { get; init; } = 0.995;
    public double DivergenceBound { get; init; } = 1e30;

    public string Name => "ipm-mehrotra";

    public SolveResult Solve(StandardForm sf, SolverLimits limits)
    {
        var deadline = limits.StartDeadline();
        int m = sf.Rows, n = sf.Cols;
        double[] a = sf.A, b = sf.B, c = sf.C;

        var x = new double[n];
        var s = new double[n];
        var y = new double[m];
        var rb = new double[m];
        var rc = new double[n];
        var d = new double[n];
        var sqrtD = new double[n];
        var scaled = new double[m * n];
        var normal = new double[m * m];
        var rhsM = new double[m];
        var tmpN = new double[n];
        var dxAff = new double[n];
        var dsAff = new double[n];
        var dyAff = new double[m];
        var dx = new double[n];
        var ds = new double[n];
        var dy = new double[m];
        var rxs = new double[n];
        int events = 0;

        // Starting point.
        Array.Fill(sqrtD, 1.0);
        events += FormAndFactor(a, m, n, sqrtD, scaled, normal);
        DenseMath.MatVec(a, m, n, c, rhsM);
        CholeskySolve(normal, m, rhsM, y);
        DenseMath.MatTVec(a, m, n, y, s);
        for (int j = 0; j < n; j++) s[j] = c[j] - s[j];
        b.AsSpan().CopyTo(rhsM);
        CholeskySolve(normal, m, rhsM, dy);
        DenseMath.MatTVec(a, m, n, dy, x);
        double dxShift = Math.Max(-1.5 * Min(x), 0.0);
        double dsShift = Math.Max(-1.5 * Min(s), 0.0);
        for (int j = 0; j < n; j++) { x[j] += dxShift; s[j] += dsShift; }
        double xs = DenseMath.Dot(x, s), sumX = Sum(x), sumS = Sum(s);
        if (xs <= 0 || sumX <= 0 || sumS <= 0)
        {
            Array.Fill(x, 1.0); Array.Fill(s, 1.0);
        }
        else
        {
            double hx = 0.5 * xs / sumS, hs = 0.5 * xs / sumX;
            for (int j = 0; j < n; j++) { x[j] += hx; s[j] += hs; }
        }

        double normB = DenseMath.Norm2(b), normC = DenseMath.Norm2(c);
        int iter = 0;
        double residual = double.NaN;
        SolveStatus status;
        while (true)
        {
            DenseMath.MatVec(a, m, n, x, rb);
            for (int i = 0; i < m; i++) rb[i] -= b[i];
            DenseMath.MatTVec(a, m, n, y, rc);
            for (int j = 0; j < n; j++) rc[j] += s[j] - c[j];
            double pObj = DenseMath.Dot(c, x), dObj = DenseMath.Dot(b, y);
            double pRes = DenseMath.Norm2(rb) / (1 + normB);
            double dRes = DenseMath.Norm2(rc) / (1 + normC);
            double gap = Math.Abs(pObj - dObj) / (1 + Math.Abs(pObj));
            residual = Math.Max(pRes, Math.Max(dRes, gap));
            if (!double.IsFinite(residual) || DenseMath.NormInf(x) > DivergenceBound || DenseMath.NormInf(y) > DivergenceBound)
            { status = SolveStatus.NumericalFailure; break; }
            if (residual <= Tolerance) { status = SolveStatus.Solved; break; }
            if (iter >= limits.MaxIterations) { status = SolveStatus.IterationLimit; break; }
            if (deadline.Expired) { status = SolveStatus.Timeout; break; }

            double mu = DenseMath.Dot(x, s) / n;
            for (int j = 0; j < n; j++) { d[j] = x[j] / s[j]; sqrtD[j] = Math.Sqrt(d[j]); }
            events += FormAndFactor(a, m, n, sqrtD, scaled, normal);

            // Predictor: rxs = XSe, so S⁻¹rxs = x.
            for (int j = 0; j < n; j++) tmpN[j] = x[j] - d[j] * rc[j];
            SolveDirection(a, m, n, normal, rb, rc, d, tmpN, rhsM, dyAff, dsAff);
            for (int j = 0; j < n; j++) dxAff[j] = -x[j] - d[j] * dsAff[j];
            double apAff = MaxStep(x, dxAff), adAff = MaxStep(s, dsAff);
            double muAff = 0;
            for (int j = 0; j < n; j++) muAff += (x[j] + apAff * dxAff[j]) * (s[j] + adAff * dsAff[j]);
            muAff /= n;
            double sigma = Math.Pow(muAff / mu, 3);

            // Corrector.
            for (int j = 0; j < n; j++)
            {
                rxs[j] = x[j] * s[j] + dxAff[j] * dsAff[j] - sigma * mu;
                tmpN[j] = rxs[j] / s[j] - d[j] * rc[j];
            }
            SolveDirection(a, m, n, normal, rb, rc, d, tmpN, rhsM, dy, ds);
            for (int j = 0; j < n; j++) dx[j] = -rxs[j] / s[j] - d[j] * ds[j];

            double ap = Math.Min(1.0, StepFraction * MaxStep(x, dx));
            double ad = Math.Min(1.0, StepFraction * MaxStep(s, ds));
            DenseMath.Axpy(ap, dx, x);
            DenseMath.Axpy(ad, dy, y);
            DenseMath.Axpy(ad, ds, s);
            iter++;
        }

        return new SolveResult
        {
            Status = status,
            Z = x,
            Objective = DenseMath.Dot(c, x),
            Iterations = iter,
            NumericalEvents = events,
            Residual = residual,
        };
    }

    /// <summary>Solves A D Aᵀ Δy = −rb + A·tmpN, then Δs = −rc − AᵀΔy.</summary>
    private static void SolveDirection(double[] a, int m, int n, double[] chol, double[] rb, double[] rc,
        double[] d, double[] tmpN, double[] rhsM, double[] dy, double[] ds)
    {
        DenseMath.MatVec(a, m, n, tmpN, rhsM);
        for (int i = 0; i < m; i++) rhsM[i] -= rb[i];
        CholeskySolve(chol, m, rhsM, dy);
        DenseMath.MatTVec(a, m, n, dy, ds);
        for (int j = 0; j < n; j++) ds[j] = -rc[j] - ds[j];
    }

    /// <summary>normal = (A·√D)(A·√D)ᵀ, factored in place (lower triangle). Returns the number of fixed pivots.</summary>
    private static int FormAndFactor(double[] a, int m, int n, double[] sqrtD, double[] scaled, double[] normal)
    {
        for (int i = 0; i < m; i++)
        {
            var src = a.AsSpan(i * n, n);
            var dst = scaled.AsSpan(i * n, n);
            for (int j = 0; j < n; j++) dst[j] = src[j] * sqrtD[j];
        }
        for (int i = 0; i < m; i++)
        {
            var ri = scaled.AsSpan(i * n, n);
            for (int k = 0; k <= i; k++) normal[i * m + k] = DenseMath.Dot(ri, scaled.AsSpan(k * n, n));
        }
        return CholeskyInPlace(normal, m);
    }

    /// <summary>
    /// Row-oriented Cholesky L·Lᵀ in the lower triangle. Tiny or negative pivots are replaced by 1e64,
    /// which zeroes the corresponding component of the solution (Wright, 1997).
    /// </summary>
    internal static int CholeskyInPlace(double[] l, int m)
    {
        int fixes = 0;
        for (int i = 0; i < m; i++)
        {
            var ri = l.AsSpan(i * m, m);
            for (int j = 0; j <= i; j++)
            {
                var rj = l.AsSpan(j * m, j);
                double sum = ri[j] - DenseMath.Dot(ri.Slice(0, j), rj);
                if (j == i)
                {
                    double original = ri[i] == 0 ? 1.0 : Math.Abs(ri[i]);
                    if (!(sum > 1e-30 * Math.Max(1.0, original))) { ri[i] = 1e64; fixes++; }
                    else ri[i] = Math.Sqrt(sum);
                }
                else ri[j] = sum / l[j * m + j];
            }
        }
        return fixes;
    }

    internal static void CholeskySolve(double[] l, int m, double[] rhs, double[] result)
    {
        for (int i = 0; i < m; i++)
            result[i] = (rhs[i] - DenseMath.Dot(l.AsSpan(i * m, i), result.AsSpan(0, i))) / l[i * m + i];
        for (int i = m - 1; i >= 0; i--)
        {
            double v = result[i];
            for (int k = i + 1; k < m; k++) v -= l[k * m + i] * result[k];
            result[i] = v / l[i * m + i];
        }
    }

    private static double MaxStep(double[] v, double[] dv)
    {
        double alpha = 1.0;
        for (int j = 0; j < v.Length; j++)
            if (dv[j] < 0) alpha = Math.Min(alpha, -v[j] / dv[j]);
        return alpha;
    }

    private static double Min(double[] v)
    {
        double r = double.PositiveInfinity;
        foreach (var e in v) r = Math.Min(r, e);
        return r;
    }

    private static double Sum(double[] v)
    {
        double r = 0;
        foreach (var e in v) r += e;
        return r;
    }
}
