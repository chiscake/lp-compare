using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace LpBench.Core;

/// <summary>
/// Binary instance format "LPB1" (little-endian): magic, int32 m, int32 n, float64 b[m], c[n], A[m·n] row-major.
/// Written by python/lpbench_py/generate.py.
/// </summary>
public static class ProblemFile
{
    private static ReadOnlySpan<byte> Magic => "LPB1"u8;

    public static LpProblem Read(string path, string id = "")
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 12 || !bytes.AsSpan(0, 4).SequenceEqual(Magic))
            throw new InvalidDataException($"{path}: not an LPB1 file.");
        int m = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(4));
        int n = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(8));
        long expected = 12L + 8L * (m + n + (long)m * n);
        if (bytes.Length != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {bytes.Length}.");
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Big-endian hosts are not supported.");
        var payload = MemoryMarshal.Cast<byte, double>(bytes.AsSpan(12));
        var b = payload.Slice(0, m).ToArray();
        var c = payload.Slice(m, n).ToArray();
        var a = payload.Slice(m + n, m * n).ToArray();
        return new LpProblem(m, n, a, b, c, id == "" ? Path.GetFileNameWithoutExtension(path) : id);
    }

    public static void Write(string path, LpProblem p)
    {
        using var fs = File.Create(path);
        Span<byte> header = stackalloc byte[12];
        Magic.CopyTo(header);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(4), p.M);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(8), p.N);
        fs.Write(header);
        fs.Write(MemoryMarshal.AsBytes(p.B.AsSpan()));
        fs.Write(MemoryMarshal.AsBytes(p.C.AsSpan()));
        fs.Write(MemoryMarshal.AsBytes(p.A.AsSpan()));
    }
}
