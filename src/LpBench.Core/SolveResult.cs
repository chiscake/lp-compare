using System.Diagnostics;

namespace LpBench.Core;

public enum SolveStatus
{
    /// <summary>Stopping criterion met (optimal basis for simplex, tolerances for IPM/DLN).</summary>
    Solved,
    Unbounded,
    IterationLimit,
    Timeout,
    NumericalFailure,
    Unsupported,
}

public sealed class SolveResult
{
    public SolveStatus Status { get; init; }
    /// <summary>Standard-form variables z = (x, s).</summary>
    public double[] Z { get; init; } = [];
    /// <summary>Objective of the standard form (minimization).</summary>
    public double Objective { get; init; }
    public int Iterations { get; init; }
    /// <summary>Simplex: pivots with zero step length.</summary>
    public int DegeneratePivots { get; init; }
    /// <summary>IPM: Cholesky pivots replaced by the "huge diagonal" fix.</summary>
    public int NumericalEvents { get; init; }
    /// <summary>Final residual used by the stopping rule (method-specific).</summary>
    public double Residual { get; init; } = double.NaN;
    public string Message { get; init; } = "";

    public static SolveResult Fail(SolveStatus status, string message) => new() { Status = status, Message = message };
}

public sealed class SolverLimits
{
    public int MaxIterations { get; init; } = int.MaxValue;
    public TimeSpan TimeLimit { get; init; } = TimeSpan.MaxValue;

    internal Deadline StartDeadline() => new(TimeLimit);
}

internal readonly struct Deadline
{
    private readonly long _start;
    private readonly long _limitTicks;

    public Deadline(TimeSpan limit)
    {
        _start = Stopwatch.GetTimestamp();
        _limitTicks = limit == TimeSpan.MaxValue ? long.MaxValue : (long)(limit.TotalSeconds * Stopwatch.Frequency);
    }

    public bool Expired => _limitTicks != long.MaxValue && Stopwatch.GetTimestamp() - _start > _limitTicks;
}

public interface ILpSolver
{
    string Name { get; }
    SolveResult Solve(StandardForm problem, SolverLimits limits);
}
