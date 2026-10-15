using LpBench.Core;

namespace LpBench.Tests;

public class SolverTests
{
    private static readonly SolverLimits Default = new() { MaxIterations = 100_000, TimeLimit = TimeSpan.FromSeconds(30) };

    /// <summary>Author's illustrative problem: 4 measuring instruments, 3 resources. HiGHS: x* = (0, 30, 10, 0), f* = 150.</summary>
    internal static LpProblem Illustrative() => LpProblem.FromRows(
        [[7, 2, 2, 6], [5, 8, 4, 3], [2, 4, 1, 8]], [80, 480, 130], [3, 4, 3, 1], "illustrative");

    /// <summary>Hillier–Lieberman "Wyndor": max 3x + 5y, x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18 → (2, 6), 36.</summary>
    private static LpProblem Wyndor() => LpProblem.FromRows([[1, 0], [0, 2], [3, 2]], [4, 12, 18], [3, 5]);

    /// <summary>Degenerate at the optimum (1, 1): three constraints active in R².</summary>
    private static LpProblem DegenerateSquare() => LpProblem.FromRows([[1, 0], [0, 1], [1, 1]], [1, 1, 2], [1, 1]);

    private static ILpSolver[] ExactSolvers() => [new TableauSimplex(PivotRule.Dantzig), new TableauSimplex(PivotRule.Bland)];

    [Theory]
    [InlineData(PivotRule.Dantzig, 2)]
    [InlineData(PivotRule.Bland, 3)]
    public void Simplex_Illustrative_MatchesReferenceAndPivotCount(PivotRule rule, int pivots)
    {
        var p = Illustrative();
        var r = new TableauSimplex(rule).Solve(StandardForm.FromResourceProblem(p), Default);
        Assert.Equal(SolveStatus.Solved, r.Status);
        Assert.Equal(pivots, r.Iterations);
        var acc = Metrics.Evaluate(p, r.Z, 150);
        Assert.Equal(150, acc.Objective, 9);
        Assert.Equal([0, 30, 10, 0], r.Z.Take(4).Select(v => Math.Round(v, 9)));
        Assert.True(acc.Violation < 1e-12);
    }

    [Fact]
    public void Simplex_Wyndor_BothRules()
    {
        var p = Wyndor();
        foreach (var s in ExactSolvers())
        {
            var r = s.Solve(StandardForm.FromResourceProblem(p), Default);
            Assert.Equal(SolveStatus.Solved, r.Status);
            Assert.Equal(36, Metrics.Evaluate(p, r.Z, 36).Objective, 10);
            Assert.Equal(2, r.Z[0], 10);
            Assert.Equal(6, r.Z[1], 10);
        }
    }

    [Fact]
    public void Simplex_DegenerateVertex_BothRulesAgree()
    {
        var p = DegenerateSquare();
        foreach (var s in ExactSolvers())
        {
            var r = s.Solve(StandardForm.FromResourceProblem(p), Default);
            Assert.Equal(SolveStatus.Solved, r.Status);
            Assert.Equal(2, -r.Objective, 12);
        }
    }

    /// <summary>
    /// Beale (1955): min −¾x₄ + 20x₅ − ½x₆ + 6x₇ with basis (x₁, x₂, x₃); optimum −5/4.
    /// Dantzig's entering rule cycles; Bland's rule terminates.
    /// </summary>
    private static StandardForm Beale() => new(3, 7,
        [
            1, 0, 0, 0.25, -8, -1, 9,
            0, 1, 0, 0.5, -12, -0.5, 3,
            0, 0, 1, 0, 0, 1, 0,
        ],
        [0, 0, 1], [0, 0, 0, -0.75, 20, -0.5, 6], [0, 1, 2]);

    [Fact]
    public void Simplex_Beale_DantzigCyclesBlandTerminates()
    {
        var limits = new SolverLimits { MaxIterations = 200 };
        var dantzig = new TableauSimplex(PivotRule.Dantzig).Solve(Beale(), limits);
        Assert.Equal(SolveStatus.IterationLimit, dantzig.Status);
        Assert.Equal(dantzig.Iterations, dantzig.DegeneratePivots);

        var bland = new TableauSimplex(PivotRule.Bland).Solve(Beale(), limits);
        Assert.Equal(SolveStatus.Solved, bland.Status);
        Assert.Equal(-1.25, bland.Objective, 12);
    }

    [Fact]
    public void Simplex_Unbounded_IsReported()
    {
        var p = LpProblem.FromRows([[1, 0], [1, 0]], [1, 2], [1, 1]);
        foreach (var s in ExactSolvers())
            Assert.Equal(SolveStatus.Unbounded, s.Solve(StandardForm.FromResourceProblem(p), Default).Status);
    }

    [Fact]
    public void Simplex_NegativeRhs_IsRejected()
    {
        var p = LpProblem.FromRows([[1, 1]], [-1], [1, 1]);
        var r = new TableauSimplex(PivotRule.Bland).Solve(StandardForm.FromResourceProblem(p), Default);
        Assert.Equal(SolveStatus.Unsupported, r.Status);
    }

    [Fact]
    public void Ipm_Illustrative_MatchesReference()
    {
        var p = Illustrative();
        var r = new MehrotraIpm().Solve(StandardForm.FromResourceProblem(p), Default);
        Assert.Equal(SolveStatus.Solved, r.Status);
        var acc = Metrics.Evaluate(p, r.Z, 150);
        Assert.True(acc.Gap < 1e-7, $"gap {acc.Gap}");
        Assert.True(acc.Violation < 1e-7, $"viol {acc.Violation}");
        Assert.InRange(r.Iterations, 3, 50);
    }

    [Fact]
    public void Ipm_Wyndor_And_Degenerate()
    {
        foreach (var (p, f) in new[] { (Wyndor(), 36.0), (DegenerateSquare(), 2.0) })
        {
            var r = new MehrotraIpm().Solve(StandardForm.FromResourceProblem(p), Default);
            Assert.Equal(SolveStatus.Solved, r.Status);
            Assert.True(Metrics.Evaluate(p, r.Z, f).Gap < 1e-7);
        }
    }

    /// <summary>x₁ + x₂ = −1 with x ≥ 0 is infeasible; the IPM must not report success.</summary>
    [Fact]
    public void Ipm_Infeasible_IsNotSolved()
    {
        var sf = new StandardForm(1, 2, [1, 1], [-1], [1, 1], [0]);
        var r = new MehrotraIpm().Solve(sf, new SolverLimits { MaxIterations = 100 });
        Assert.NotEqual(SolveStatus.Solved, r.Status);
    }

    [Fact]
    public void Dln_Illustrative_ApproachesOptimum()
    {
        var p = Illustrative();
        var sf = StandardForm.FromResourceProblem(p);
        var limits = new SolverLimits { MaxIterations = 2_000_000, TimeLimit = TimeSpan.FromSeconds(60) };
        var coarse = new DlnSolver { Lambda = 0.05, Kappa = 3 }.Solve(sf, limits);
        var fine = new DlnSolver { Lambda = 0.005, Kappa = 3 }.Solve(sf, limits);
        Assert.Equal(SolveStatus.Solved, coarse.Status);
        Assert.Equal(SolveStatus.Solved, fine.Status);
        var gCoarse = Metrics.Evaluate(p, coarse.Z, 150).Gap;
        var gFine = Metrics.Evaluate(p, fine.Z, 150);
        Assert.True(gFine.Gap < gCoarse, $"fine {gFine.Gap} vs coarse {gCoarse}");
        Assert.True(gFine.Gap < 1e-3, $"gap {gFine.Gap}");
        Assert.True(gFine.Violation < 1e-6, $"viol {gFine.Violation}");
        Assert.All(fine.Z, v => Assert.True(v > 0));
    }

    /// <summary>Theorem 3.5: the limit carries a bias term proportional to the step size.</summary>
    [Fact]
    public void Dln_LargeStep_BiasesTheLimit()
    {
        var p = Illustrative();
        var sf = StandardForm.FromResourceProblem(p);
        var limits = new SolverLimits { MaxIterations = 2_000_000 };
        var small = new DlnSolver { Lambda = 0.005, Kappa = 3 }.Solve(sf, limits);
        var large = new DlnSolver { Lambda = 0.005, Kappa = 30 }.Solve(sf, limits);
        Assert.Equal(SolveStatus.Solved, large.Status);
        Assert.True(large.Iterations < small.Iterations);
        Assert.True(Metrics.Evaluate(p, large.Z, 150).Gap > 100 * Metrics.Evaluate(p, small.Z, 150).Gap);
    }

    [Fact]
    public void Dln_SumBound_UsesTheTighterOfTwoBounds()
    {
        var sf = StandardForm.FromResourceProblem(Illustrative());
        Assert.Equal(690, DlnSolver.SumBound(sf), 12);
    }

    [Fact]
    public void Dln_LogDomain_MatchesPlainWhenNoUnderflow()
    {
        var sf = StandardForm.FromResourceProblem(Illustrative());
        var plain = new DlnSolver { Lambda = 0.02, Kappa = 10 }.Solve(sf, Default);
        var log = new DlnSolver { Lambda = 0.02, Kappa = 10, LogDomain = true }.Solve(sf, Default);
        Assert.Equal(plain.Iterations, log.Iterations, 2.0);
        Assert.Equal(plain.Objective, log.Objective, 6);
    }

    [Fact]
    public void Dln_Infeasible_IsNotSolved()
    {
        // x₁ ≤ 1 and x₁ ≥ 2 written as resource rows cannot be encoded with A ≥ 0, b > 0;
        // instead feed an equality system that violates the DLN preconditions.
        var sf = new StandardForm(1, 2, [1, -1], [1], [1, 1], [0]);
        Assert.Equal(SolveStatus.Unsupported, new DlnSolver().Solve(sf, Default).Status);
    }

    [Fact]
    public void ProblemFile_RoundTrip()
    {
        var p = Illustrative();
        var path = Path.GetTempFileName();
        try
        {
            ProblemFile.Write(path, p);
            var q = ProblemFile.Read(path);
            Assert.Equal(p.A, q.A);
            Assert.Equal(p.B, q.B);
            Assert.Equal(p.C, q.C);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Metrics_ViolationCountsNegativeSlacks()
    {
        var p = Wyndor();
        var acc = Metrics.Evaluate(p, [2, 6, 2, -0.5, 0], 36);
        Assert.Equal(0.5 / 19, acc.Violation, 12);
    }
}
