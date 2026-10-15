using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LpBench.Core;

/// <summary>Vectorized dense kernels shared by all solvers, so that no method gets a better BLAS than another.</summary>
public static class DenseMath
{
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        int n = a.Length;
        double sum = 0;
        int i = 0;
        if (Vector.IsHardwareAccelerated && n >= 2 * Vector<double>.Count)
        {
            var va = MemoryMarshal.Cast<double, Vector<double>>(a);
            var vb = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, n));
            var acc0 = Vector<double>.Zero;
            var acc1 = Vector<double>.Zero;
            int k = 0;
            for (; k + 1 < va.Length; k += 2)
            {
                acc0 += va[k] * vb[k];
                acc1 += va[k + 1] * vb[k + 1];
            }
            if (k < va.Length) acc0 += va[k] * vb[k];
            sum = Vector.Sum(acc0 + acc1);
            i = va.Length * Vector<double>.Count;
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>y += alpha·x</summary>
    public static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        int n = x.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            var vx = MemoryMarshal.Cast<double, Vector<double>>(x);
            var vy = MemoryMarshal.Cast<double, Vector<double>>(y.Slice(0, n));
            var va = new Vector<double>(alpha);
            for (int k = 0; k < vx.Length; k++) vy[k] += va * vx[k];
            i = vx.Length * Vector<double>.Count;
        }
        for (; i < n; i++) y[i] += alpha * x[i];
    }

    public static void Scale(double alpha, Span<double> x)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<double>.Count)
        {
            var vx = MemoryMarshal.Cast<double, Vector<double>>(x);
            var va = new Vector<double>(alpha);
            for (int k = 0; k < vx.Length; k++) vx[k] *= va;
            i = vx.Length * Vector<double>.Count;
        }
        for (; i < x.Length; i++) x[i] *= alpha;
    }

    /// <summary>y = A·x, A row-major rows×cols.</summary>
    public static void MatVec(double[] a, int rows, int cols, ReadOnlySpan<double> x, Span<double> y)
    {
        for (int i = 0; i < rows; i++) y[i] = Dot(a.AsSpan(i * cols, cols), x);
    }

    /// <summary>x = Aᵀ·y, A row-major rows×cols.</summary>
    public static void MatTVec(double[] a, int rows, int cols, ReadOnlySpan<double> y, Span<double> x)
    {
        x.Slice(0, cols).Clear();
        for (int i = 0; i < rows; i++)
        {
            double yi = y[i];
            if (yi != 0) Axpy(yi, a.AsSpan(i * cols, cols), x);
        }
    }

    public static double NormInf(ReadOnlySpan<double> x)
    {
        double m = 0;
        foreach (var v in x) { double a = Math.Abs(v); if (a > m || double.IsNaN(a)) m = a; }
        return m;
    }

    public static double Norm2(ReadOnlySpan<double> x) => Math.Sqrt(Dot(x, x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsFinite(double v) => double.IsFinite(v);

    public static bool AllFinite(ReadOnlySpan<double> x)
    {
        foreach (var v in x) if (!double.IsFinite(v)) return false;
        return true;
    }
}
