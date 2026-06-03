# Tendon visualization refactor

Drop these files into the repository root.

## New layout

```text
scripts/tendons/draw_tendon_actuation.py
source/isaaclab/isaaclab/tendons/models/analytic/visualization/
  __init__.py
  animator.py
  context.py
  data.py
  display.py
  kinematics.py
  paths.py
  states.py
  style.py
  validation.py
```

The script is now only the CLI/playground. The large animation class, tendon path construction, plotting style, state extraction, validation, and JSONL loading live under `models/analytic/visualization/`.

## Usage

```bash
python scripts/tendons/draw_tendon_actuation.py
python scripts/tendons/draw_tendon_actuation.py --save outputs/tendon.mp4
python scripts/tendons/draw_tendon_actuation.py --record
python scripts/tendons/draw_tendon_actuation.py --single-plot
python scripts/tendons/draw_tendon_actuation.py --show-debug-geometry --show-debug-text
```

Useful extras:

```bash
python scripts/tendons/draw_tendon_actuation.py --data outputs/gst_data_left.jsonl
python scripts/tendons/draw_tendon_actuation.py --alpha-2-deg 300
python scripts/tendons/draw_tendon_actuation.py --no-validate
```

Keyboard controls:

- Space: play/pause
- Left/Right: step when paused
- Home/End: jump to first/last frame
- `d`: toggle geometry helper lines
- `i`: toggle detailed debug text

## Notes

- The default behavior no longer shows the helper `x`/`h` geometry lines; use `--show-debug-geometry` or press `d`.
- The side panels keep only the readable summary by default; use `--show-debug-text` or press `i` for additional debug conditions.
- The original geometry comments were kept in the split path/validation modules for future debugging.
