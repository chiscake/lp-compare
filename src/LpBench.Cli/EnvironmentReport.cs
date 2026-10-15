using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace LpBench.Cli;

internal static class EnvironmentReport
{
    public static string Build()
    {
        var sb = new StringBuilder();
        void Line(string k, object v) => sb.Append(k).Append(": ").Append(v).Append('\n');
        Line("timestamp_utc", DateTime.UtcNow.ToString("O"));
        Line("cpu", CpuName());
        Line("logical_cores", Environment.ProcessorCount);
        Line("memory_total_gb", (GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1e9).ToString("F1"));
        Line("os", RuntimeInformation.OSDescription);
        Line("arch", RuntimeInformation.OSArchitecture);
        Line("rid", RuntimeInformation.RuntimeIdentifier);
        Line("runtime", RuntimeInformation.FrameworkDescription);
        Line("native_aot", !RuntimeFeature.IsDynamicCodeSupported);
        Line("vector_hw_accelerated", Vector.IsHardwareAccelerated);
        Line("vector_double_count", Vector<double>.Count);
        Line("x86_avx2", Avx2.IsSupported);
        Line("x86_fma", Fma.IsSupported);
        Line("x86_avx512f", Avx512F.IsSupported);
        Line("arm_advsimd", AdvSimd.IsSupported);
        Line("process_priority", SafePriority());
        return sb.ToString();
    }

    private static string SafePriority()
    {
        try { return Process.GetCurrentProcess().PriorityClass.ToString(); }
        catch { return "unknown"; }
    }

    private static string CpuName()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                if (key?.GetValue("ProcessorNameString") is string s) return s.Trim();
            }
            else if (OperatingSystem.IsLinux())
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                    if (line.StartsWith("model name", StringComparison.Ordinal)) return line[(line.IndexOf(':') + 1)..].Trim();
            }
            else if (OperatingSystem.IsMacOS())
            {
                var psi = new ProcessStartInfo("sysctl", "-n machdep.cpu.brand_string") { RedirectStandardOutput = true };
                using var p = Process.Start(psi)!;
                var s = p.StandardOutput.ReadToEnd().Trim();
                p.WaitForExit();
                if (s.Length > 0) return s;
            }
        }
        catch
        {
            // Fall through to the generic identifier.
        }
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
    }

    /// <summary>Cost of arithmetic on subnormal doubles relative to normal ones (u ← u·(1−2ηg) style loop).</summary>
    public static string DenormalBenchmark()
    {
        const int n = 4096, reps = 20000;
        double Run(double start)
        {
            var u = new double[n];
            Array.Fill(u, start);
            var sw = Stopwatch.StartNew();
            double factor = 1.0 - 1e-9;
            for (int r = 0; r < reps; r++)
                for (int j = 0; j < n; j++) u[j] = u[j] * factor + u[j] * 1e-12;
            sw.Stop();
            GC.KeepAlive(u);
            return sw.Elapsed.TotalMilliseconds;
        }
        Run(1.0);
        double normal = Run(1.0);
        double sub = Run(1e-310);
        return $"denormal_bench_normal_ms: {normal:F1}\ndenormal_bench_subnormal_ms: {sub:F1}\ndenormal_slowdown: {sub / normal:F1}x\n";
    }
}
