# Ashiria Engine (Unity package)

The C# hand-sign engine shared by the Ashiria apps. It is a line-for-line port of the Python reference in `training/`, and the two are kept equal by golden parity tests. Rules: [`spec/FEATURE_SPEC.md`](../../../spec/FEATURE_SPEC.md) (frames and features) and [`spec/RUNTIME_SPEC.md`](../../../spec/RUNTIME_SPEC.md) (segmenter, Verify, Recognise).

## Runtime (`Ashiria.Engine`, pure C#, no UnityEngine)

| File | What it does |
|---|---|
| `TickClock.cs` | Samples tracking at a fixed 30 Hz whatever the display refresh rate |
| `QuestCanonical.cs` | One tick of Quest tracking (26 world-space joints per hand, head pose) → one canonical frame; presence gating |
| `AttemptSegmenter.cs` | Live frames → attempts (hold, drop, timeout; cooldown so a held sign fires once) |
| `Pipeline.cs` | `Prepare`: gap fill → trim → active window → resample (as the references were cut) |
| `Canonical.cs`, `Features.cs` | Canonicalisation helpers; the 197 per-frame features |
| `Matching.cs` | Descriptor, DTW (distance and alignment path), per-finger errors |
| `Signs.cs` | `SignRef` / `Take` / `KeyPose` / `Tolerances`: what the engine knows about each sign |
| `Verifier.cs` | Verify: pass/fail, score, per-component errors, hint codes |
| `Recogniser.cs` | Free Sign: match / ambiguous / not sure |
| `HintText.cs` | Default English for hint codes |
| `Onboarding.cs` | The user's rest height from resting wrist samples |
| `EngineParams.cs` | Every parameter; packs override the defaults |

## Content (`Ashiria.Engine.Content`, uses Newtonsoft JSON)

| File | What it does |
|---|---|
| `Content/ContentPack.cs` | Loads a content pack ([spec](../../../spec/CONTENT_PACK.md)): checks versions and hashes, parses tiers, tolerances and engine parameters, binds signs, runs the self-check |
| `Content/StudioTakeWriter.cs` | Writes Studio takes ([spec](../../../spec/STUDIO_FORMAT.md)) for `ashiria import-studio` |

The package depends on `com.unity.nuget.newtonsoft-json`. The content assembly parses by hand over `JToken`, with no reflection, so IL2CPP stripping cannot break it.

## Use from Unity

Add the package to the project's `Packages/manifest.json` as a local package:
```json
"com.jarida.ashiria.engine": "file:../../../sdk/unity/com.jarida.ashiria.engine"
```

The app layer only has to convert SDK types to `Vec3` and call the engine:

```csharp
int ticks = clock.Advance(Time.unscaledDeltaTime);
for (int i = 0; i < ticks; i++)
{
    QuestCanonical.Frame(headPos, headForward, right, left, leftHanded, shoulderWidth, ref axes, handRel, wrist, present);
    var attempt = segmenter.Push(handRel, wrist, present);
    if (attempt == null) continue;
    var prepared = Pipeline.Prepare(attempt.Segment, pack.Params, restHeight);
    var result = Verifier.Verify(prepared, pack.Signs[target], pack.Params, strictness, pack.LookalikesOf(target));
}
```

## Tests

- **In Unity:** Window → General → Test Runner → EditMode.
- **Without Unity:** run the same tests under .NET. `Tests~` is ignored by Unity. The project compiles at C# 9 with warnings as errors, as Unity 6 does.
  ```bash
  dotnet test sdk/unity/com.jarida.ashiria.engine/Tests~/dotnet
  ```

The tests read `spec/golden/` (engine and runtime vectors, a synthetic mini pack, a Studio take). After any change to the Python reference, regenerate them with `ashiria golden` in `training/`.

**Status (14 Oct 2026):** all 16 tests pass under .NET 10. It has not been opened in Unity yet.
