# Ashiria Learn (Quest): project setup

The hackathon entry. This folder will hold the Unity project; create it here following the steps below. Studio is the same project built with the `ASHIRIA_STUDIO` scripting define, and it's never uploaded to the Competition channel.

## 1. Install (once)

1. **Unity Hub** with **Unity 6.1 or later**. Add the Android Build Support module, which includes the Android SDK, NDK and OpenJDK.
2. **Meta Quest Developer Hub** (macOS), to put the headset in developer mode, install builds and record video.
3. **Meta VR CLI:** run `npx metavr@latest init`. It provides device control, logs, performance tracing and store uploads, and can hook into Claude Code.
4. **Meta XR Simulator** (the Apple Silicon build), for testing logic without the headset. On a Mac it tracks your hands through the webcam.
   - It needs the Unity OpenXR Plugin 1.13.0 or later.
   - Webcam hands estimate depth, so never tune recognition tolerances in it.

## 2. Create the project here (`apps/quest-learn/`)

1. **New project:** Unity Hub → New project → **Universal 3D (URP)** → location `ashiria/apps/`, name `quest-learn`.
2. **Meta packages:** Package Manager → add **Meta XR All-in-One SDK v207** (`com.meta.xr.sdk.all`). It includes the Core SDK, Interaction SDK and MR Utility Kit.
3. **Shared engine:** add this line under `dependencies` in `Packages/manifest.json`:
   ```json
   "com.jarida.ashiria.engine": "file:../../../sdk/unity/com.jarida.ashiria.engine"
   ```
4. **Project Setup Tool:** Meta → Tools → Project Setup Tool. Fix everything it flags for Android.
5. **Player settings (Android):**
   - IL2CPP, ARM64 only;
   - minimum API 32, target API **34** (required for apps created after 1 Mar 2026);
   - package name `io.jarida.ashiria.learn`.
6. **OVR Manager:** Hand Tracking Support **Hands Only**. Leave the hand-tracking version on Default.
7. **Building Blocks** (Meta → Building Blocks): Camera Rig, Passthrough, Hand Tracking, Interaction Rig, then the Interaction SDK UI Set.
8. **Check the engine:** run the engine tests (Window → General → Test Runner → EditMode). They read `spec/golden/` and must all pass (16 tests) before any app code relies on the engine. They already pass under .NET, so a failure here means a Unity-specific difference.

## 3. Folder plan inside the project

```
Assets/Ashiria/
  Scripts/Input/      HandFrameSource: OVRSkeleton bones (Transform.position, world space) or ISDK IHand.GetJointPose
                      → HandInput; centre-eye pose → TickClock → QuestCanonical.Frame
  Scripts/Lessons/    ContentPackLoader (StreamingAssets, or a sideloaded pack in persistentDataPath on development
                      builds), lesson flow, spaced repetition, progress (local JSON)
  Scripts/Practice/   AttemptSegmenter → Pipeline.Prepare → Verifier (target sign) or Recogniser (Free Sign) → HintText
  Scripts/Onboarding/ handedness, rest height (Onboarding.RestHeight from 2 s of resting hands), strictness
  Scripts/Studio/     record → trim → replay → label → StudioTakeWriter   (#if ASHIRIA_STUDIO)
  Scripts/MR/         sign wall, Room Words (MR Utility Kit)
  StreamingAssets/Content/ksl-core-<version>/   manifest.json, catalogue.json, references.json (`ashiria pack`)
  Content/            short sign videos of Entrants only (never dictionary video frames: that signer isn't an Entrant)
  Scenes/             Learn.unity, Studio.unity
```

**Where the engine does the work:** the app layer only converts SDK types (Vector3, Quaternion, bone lists) to the
engine's `Vec3`, `HandInput` and `HandSample`. All decisions live in the engine, which is tested. The example loop
is in the [engine README](../../sdk/unity/com.jarida.ashiria.engine/README.md#use-from-unity).

**Content packs:**
- **Build one:** `make -C training pack`, then copy the folder into `StreamingAssets/Content/`.
- **Development builds:** may also load a pack sideloaded to `persistentDataPath/packs/`, so a retuned pack can be
  tested without a rebuild (there is no Link on a Mac).
- **Competition build:** only a pack built with `--release`, or one made entirely from Studio takes recorded by
  Entrants.

## 4. Rules for every build

- **Hands only:** no controller prompts, ever.
- **Hand data:** recorded hand data stays on the headset. Studio saves locally, and you pull files with `adb pull` or `metavr`. Nothing uploads in the competition build.
- **Performance:** at least 72 fps (the minimum is 60), and head-tracked graphics within 4 s of launch.
- **Rights:** never bundle dictionary-derived content without written permission from the KSL Dictionary owners.
