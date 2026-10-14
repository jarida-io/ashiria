using System;

namespace Ashiria.Engine
{
    /// <summary>Minimal 3D vector so the engine compiles without UnityEngine (and in plain .NET for tests).</summary>
    public readonly struct Vec3
    {
        public readonly float X, Y, Z;

        public Vec3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static readonly Vec3 Zero = new Vec3(0f, 0f, 0f);

        public static Vec3 operator +(Vec3 a, Vec3 b) => new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec3 operator -(Vec3 a, Vec3 b) => new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec3 operator -(Vec3 a) => new Vec3(-a.X, -a.Y, -a.Z);
        public static Vec3 operator *(Vec3 a, float s) => new Vec3(a.X * s, a.Y * s, a.Z * s);
        public static Vec3 operator /(Vec3 a, float s) => new Vec3(a.X / s, a.Y / s, a.Z / s);

        public static float Dot(Vec3 a, Vec3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

        public static Vec3 Cross(Vec3 a, Vec3 b) =>
            new Vec3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

        public float Length => MathF.Sqrt(X * X + Y * Y + Z * Z);

        /// <summary>v / max(|v|, ε), as in spec §6.</summary>
        public Vec3 Normalized() => this / MathF.Max(Length, Features.Eps);

        public Vec3 MirrorX() => new Vec3(-X, Y, Z);

        public static Vec3 Lerp(Vec3 a, Vec3 b, float t) => a * (1f - t) + b * t;

        /// <summary>Angle in radians via atan2(|a×b|, a·b): accurate near 0 and π in 32-bit floats.</summary>
        public static float Angle(Vec3 a, Vec3 b) => MathF.Atan2(Cross(a, b).Length, Dot(a, b));

        public override string ToString() => $"({X:0.####}, {Y:0.####}, {Z:0.####})";
    }
}
