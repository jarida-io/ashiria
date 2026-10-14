# Ashiria Engine (Unity package)

The C# implementation of the shared hand-sign engine (`spec/FEATURE_SPEC.md`), ported line for line from the Python reference in `training/`.

| File | What it does |
|---|---|
| `Runtime/QuestCanonical.cs` | Quest joints (26, Unity world space) → one canonical frame |
| `Runtime/Canonical.cs` | Mirror, gap fill, trim, resample |
| `Runtime/Features.cs` | 197 features per frame |
| `Runtime/Matching.cs` | Matching descriptor, DTW, per-finger errors (Verify and Recognise) |

- **No UnityEngine dependency.** The runtime assembly is pure C#. The Unity layer reads `OVRSkeleton` or Interaction SDK joint poses and converts them to `Vec3`.
- **Install:** add it to the Unity project's `Packages/manifest.json` as a local package:
  `"com.jarida.ashiria.engine": "file:../../../sdk/unity/com.jarida.ashiria.engine"`.
- **Tests:** Window → General → Test Runner → EditMode. They compare against `spec/golden/engine_v1.json`, so run `ashiria golden` in `training/` after any spec change.

**Status (14 Oct 2026):** written and reviewed against the Python reference. It hasn't been compiled yet, because there's no Unity install on this Mac.
