# Ashiria

Learn, practise and recognise Kenyan Sign Language, built with Deaf KSL signers. Several apps share one base; see [docs/PLATFORM.md](docs/PLATFORM.md). The Quest app is the Meta VR Start Developer Competition 2026 entry; see [docs/HACKATHON_PLAN.md](docs/HACKATHON_PLAN.md).

| Folder | What's in it | Status |
|---|---|---|
| `docs/` | Platform overview, hackathon plan, [privacy policy draft](docs/privacy.md) | Living documents |
| `spec/` | Shared contracts: [features](spec/FEATURE_SPEC.md), [runtime](spec/RUNTIME_SPEC.md), [content packs](spec/CONTENT_PACK.md), [Studio takes](spec/STUDIO_FORMAT.md), golden test vectors (`spec/golden/`) | v1 |
| `training/` | Python: videos → landmarks → references; the engine's reference implementation; calibration, catalogue → content packs; Studio import ([README](training/README.md)) | Working, 45 tests |
| `sdk/web/` | TypeScript engine for the WebXR app, same features ([package](sdk/web/ashiria-engine)) | 11 parity tests pass (`node --test`) |
| `sdk/unity/` | C# engine package, kept as the Unity fallback ([README](sdk/unity/com.jarida.ashiria.engine/README.md)) | Compiles; 16 parity tests pass under .NET |
| `content/` | `catalogue/ksl-core.yaml` (what the app teaches, edited by the team), `rights.yaml`; built packs in `content/packs/` (git-ignored) | v0.1.0 internal |
| `apps/quest-learn/` | Ashiria Learn and Studio for Quest: WebXR with Meta's Immersive Web SDK ([README](apps/quest-learn/README.md)) | Runs in the emulator; untested on a headset |
| `services/`, `sdk/android/`, `apps/android/` | PocketBase service, Android engine and apps | Later (PLATFORM.md §6) |

**Check everything:**
```bash
make -C training test
```
```bash
node --test sdk/web/ashiria-engine/test/
```
```bash
dotnet test sdk/unity/com.jarida.ashiria.engine/Tests~/dotnet
```

**Data rule:** sign videos show real people. They and anything derived from them stay out of git and out of any cloud service unless the signers have agreed to it. `training/data/`, `training/runs/`, `training/exports/` and `content/packs/` are git-ignored. Synthetic test data (`spec/golden/`) contains no real people.
