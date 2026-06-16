# YOLO Dataset Creator (Screen Tracker)

Capture **only the game window**, select an object, track it, and save images + YOLO labels. The bot also runs YOLO on the game window only, so training and inference match. Use one workflow for **any** target (mound_dharok, banker, barrows_chest, etc.). Final trained models live in **assets/templates/yolo/**; you can delete the YOLO Trainer folder after you’re done.

## All-in-one workflow (recommended)

One script does everything: track → yaml → train → copy → optionally clean for next target.

```bash
python "YOLO Trainer/yolo_workflow.py" full mound_dharok
```

Track (S = select, ESC = exit) → yaml + train + copy → at end, answer **y** to delete dataset/yaml/runs for a fresh next run. Individual steps: `track`, `yaml`, `train`, `copy`, `clean` (see script help).

## Setup

From the **Barrows Bot** project root:

```bash
pip install -r requirements.txt
```

Requires: `opencv-contrib-python`, `dxcam` (Windows).

---

## Dynamic workflow (any target)

### 1. Set TARGET in the tracker

In **yolo_auto_dataset_tracker.py** at the top:

- **TARGET** = the object name, e.g. `"mound_dharok"`, `"banker"`, `"barrows_chest"`.
- **CLASS_NAME** = optional. Class name in labels (default: CamelCase from TARGET). Use for custom names like `"DharokMound"`.

Data is saved to `YOLO Trainer/dataset_<TARGET>/` (e.g. `dataset_mound_dharok`).

### 2. Run the tracker

Ensure the **game window is open** and its title matches **GAME_WINDOW_TITLE_SUBSTRING** in `config.py` (e.g. `RuneLite`). The tracker captures only that window (same as the bot at inference).

```bash
python "YOLO Trainer/yolo_auto_dataset_tracker.py" mound_dharok
```

You should see `Capture: game window only (matches bot YOLO lookup)`. Press **S**, draw a box around the object, confirm. Move the camera for variety. Press **ESC** when done.

### 3. Generate data.yaml

From project root:

```bash
python "YOLO Trainer/create_data_yaml.py" <target> [--class-name ClassName]
```

Examples:

```bash
python "YOLO Trainer/create_data_yaml.py" mound_dharok --class-name DharokMound
python "YOLO Trainer/create_data_yaml.py" banker
```

This creates `YOLO Trainer/<target>/data.yaml` pointing at `dataset_<target>/`.

### 4. Train

From project root:

```bash
yolo detect train data="YOLO Trainer/<target>/data.yaml" model=yolov8n.pt epochs=50
```

Example:

```bash
yolo detect train data="YOLO Trainer/mound_dharok/data.yaml" model=yolov8n.pt epochs=50
```

### 5. Copy weights to templates (final step)

This puts the model where the bot expects it. After this you can delete the YOLO Trainer folder.

```bash
python "YOLO Trainer/copy_to_templates.py" <target> [--run-dir runs/detect/train]
```

Example:

```bash
python "YOLO Trainer/copy_to_templates.py" mound_dharok
```

This copies `runs/detect/train/weights/best.pt` to **assets/templates/yolo/mound_dharok.pt**. Config already points to `assets/templates/yolo/<target>.pt` for known targets.

### 6. Use in the bot

- In **config.py**, **USE_YOLO_FOR** must include the template name (e.g. `["mound_dharok"]`).
- **YOLO_CAPTURE_GAME_WINDOW = True** so the bot only runs YOLO on the game window (matches training).
- Model path is already set to `assets/templates/yolo/<target>.pt` for mound_dharok and banker.
- Class name must match the one in your data.yaml (e.g. **YOLO_MOUND_DHAROK_CLASS** = `"MoundDharok"`).

No code changes needed; the bot will use YOLO for any template in **USE_YOLO_FOR** that has a model in **assets/templates/yolo/**.

---

## Tracker controls

- **S** = select object (draw box, then Space/Enter).
- **R** = reset tracker (then press **S** again to reselect).
- **P** = pause/resume saving.
- **ESC** = exit.

Single window; labels use class 0 when **SINGLE_CLASS_MODE** is True.

---

## Adding a new YOLO target

1. In **yolo_finder.py**, add the template name to **_YOLO_TARGET_CONFIG** with its config keys (e.g. `"barrows_chest": ("YOLO_CHEST_MODEL", "YOLO_CHEST_CLASS")`).
2. In **config.py**, add `YOLO_<NAME>_MODEL` and `YOLO_<NAME>_CLASS`, and add the template to **USE_YOLO_FOR**.
3. In the tracker set **TARGET** = that name (e.g. `"barrows_chest"`), collect data, run **create_data_yaml.py**, train, then **copy_to_templates.py**.
4. Ensure **YOLO_<NAME>_MODEL** points to `assets/templates/yolo/<target>.pt` (same pattern as mound_dharok/banker).

---

## Summary

| Step | Command / action |
|------|-------------------|
| 1 | Set **TARGET** (and optional **CLASS_NAME**) in `yolo_auto_dataset_tracker.py`. |
| 2 | `python "YOLO Trainer/yolo_auto_dataset_tracker.py"` → collect data. |
| 3 | `python "YOLO Trainer/create_data_yaml.py" <target> [--class-name X]` |
| 4 | `yolo detect train data="YOLO Trainer/<target>/data.yaml" model=yolov8n.pt epochs=50` |
| 5 | `python "YOLO Trainer/copy_to_templates.py" <target>` |
| 6 | Bot uses **assets/templates/yolo/<target>.pt**; you can delete the YOLO Trainer folder. |
