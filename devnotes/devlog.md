# devlog

## 05.06

- Merged newer Isaac Lab tooling direction and aligned the project around Ruff-based pre-commit.
- Added `scripts/setup_repo.sh` for post-clone setup: conda env, source install, hooks, and Forrest USD symlink.
- Cleaned root files by removing obsolete notes, old READMEs, debug output, and local test script.
- Removed tracked `saved_models/` artifacts from Git while keeping them ignored locally.
- Expanded `.gitignore` for local outputs, symlinks, caches, generated assets, and agent/IDE files.
- Fixed current Ruff/pre-commit issues in tendon scripts and constants.

## 04.06

- Added two new reward terms to the rough-terrain environment of forrest
- One function punishes the walking if the feet are crossed
- The second function punishes any contacts with the ground that do not happen parallel to the ground
