namespace LpBench.Core;

/// <summary>Resource allocation problem: max cᵀx, A x ≤ b, x ≥ 0. A is dense row-major m×n.</summary>
public sealed class LpProblem
{
    public LpProblem(int m, int n, double[] a, double[] b, double[] c, string id = "")
    {
        if (a.Length != m * n || b.Length != m || c.Length != n)
            throw new ArgumentException("Dimension mismatch.");
        M = m; N = n; A = a; B = b; C = c; Id = id;
    }

    public int M { get; }
    public int N { get; }
    public double[] A { get; }
    public double[] B { get; }
    public double[] C { get; }
    public string Id { get; }

    public double this[int i, int j] => A[i * N + j];

    public static LpProblem FromRows(double[][] a, double[] b, double[] c, string id = "")
    {
        int m = a.Length, n = c.Length;
        var flat = new double[m * n];
        for (int i = 0; i < m; i++) a[i].AsSpan().CopyTo(flat.AsSpan(i * n, n));
        return new LpProblem(m, n, flat, b, c, id);
    }
}

/// <summary>
/// Equality standard form shared by all solvers: min cᵀz, A z = b, z ≥ 0.
/// InitialBasis[i] is a column that equals the i-th unit vector (slack basis for resource problems).
/// </summary>
public sealed class StandardForm
{
    public StandardForm(int rows, int cols, double[] a, double[] b, double[] c, int[] initialBasis)
    {
        if (a.Length != rows * cols || b.Length != rows || c.Length != cols || initialBasis.Length != rows)
            throw new ArgumentException("Dimension mismatch.");
        Rows = rows; Cols = cols; A = a; B = b; C = c; InitialBasis = initialBasis;
    }

    public int Rows { get; }
    public int Cols { get; }
    public double[] A { get; }
    public double[] B { get; }
    public double[] C { get; }
    public int[] InitialBasis { get; }

    /// <summary>Adds slacks and negates the objective: z = (x, s), [A I] z = b, min (−c, 0)ᵀ z.</summary>
    public static StandardForm FromResourceProblem(LpProblem p)
    {
        int m = p.M, n = p.N, cols = n + m;
        var a = new double[m * cols];
        for (int i = 0; i < m; i++)
        {
            p.A.AsSpan(i * n, n).CopyTo(a.AsSpan(i * cols, n));
            a[i * cols + n + i] = 1.0;
        }
        var c = new double[cols];
        for (int j = 0; j < n; j++) c[j] = -p.C[j];
        var basis = new int[m];
        for (int i = 0; i < m; i++) basis[i] = n + i;
        return new StandardForm(m, cols, a, (double[])p.B.Clone(), c, basis);
    }

    /// <summary>Returns null when the simplex/IPM preconditions hold, otherwise the reason.</summary>
    public string? ValidateForSimplex()
    {
        for (int i = 0; i < Rows; i++)
            if (!(B[i] >= 0)) return $"b[{i}] = {B[i]} < 0: phase 1 is not implemented.";
        for (int i = 0; i < Rows; i++)
        {
            int col = InitialBasis[i];
            for (int r = 0; r < Rows; r++)
                if (A[r * Cols + col] != (r == i ? 1.0 : 0.0))
                    return $"Initial basis column {col} is not the unit vector e{i}.";
        }
        return null;
    }

    /// <summary>DLN reduction needs A ≥ 0, b > 0 and a positive entry in every column (gives explicit bounds).</summary>
    public string? ValidateForDln()
    {
        for (int i = 0; i < Rows; i++)
            if (!(B[i] > 0)) return $"b[{i}] = {B[i]} must be > 0 for the DLN reduction.";
        for (int j = 0; j < Cols; j++)
        {
            bool positive = false;
            for (int i = 0; i < Rows; i++)
            {
                double v = A[i * Cols + j];
                if (v < 0) return $"A[{i},{j}] < 0: DLN reduction requires A ≥ 0.";
                if (v > 0) positive = true;
            }
            if (!positive) return $"Column {j} has no positive entry: variable is unbounded.";
        }
        return null;
    }
}
