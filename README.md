# Ashiria

Learn, practise and recognise Kenyan Sign Language, built with Deaf KSL signers. Several apps share one base; see [docs/PLATFORM.md](docs/PLATFORM.md). The Quest app is the Meta VR Start Developer Competition 2026 entry; see [docs/HACKATHON_PLAN.md](docs/HACKATHON_PLAN.md).

| Folder | What's in it | Status |
|---|---|---|
| `docs/` | Platform overview, hackathon plan | Living documents |
| `spec/` | Shared contracts: [feature spec](spec/FEATURE_SPEC.md), joint layout, golden test vectors (`spec/golden/`) | v1 |
| `training/` | Python pipeline: videos → landmarks → references and models → ONNX ([README](training/README.md)) | Working, tested |
| `sdk/unity/` | C# engine package (canonicalisation, features, matching) ([README](sdk/unity/com.jarida.ashiria.engine/README.md)) | Written; compiles once Unity is installed |
| `apps/quest-learn/` | Ashiria Learn and Studio for Quest ([setup](apps/quest-learn/README.md)) | Create the Unity project here |
| `content/`, `services/`, `sdk/android/`, `apps/android/` | Catalogue packs, PocketBase service, Android engine and apps | Later (PLATFORM.md §6) |

**Data rule:** sign videos show real people. They and anything derived from them stay out of git and out of any cloud service unless the signers have agreed to it. `training/data/`, `training/runs/` and `training/exports/` are git-ignored.
