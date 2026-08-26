# YOLO Training Workflow

YOLO Training Workflow is a Windows and Python toolkit for building focused object-detection models from live application-window captures. It supports interactive target selection, OpenCV tracking, YOLO-format dataset generation, Ultralytics training, and export of the resulting weights for downstream inference projects.

The workflow was designed for projects where training capture and runtime inference must use the same window region, scale, and coordinate system.

## Features

- Captures a target application window when a parent project provides `config.py` and `game_io.py`.
- Falls back to full-screen capture when no window helper is available.
- Uses OpenCV CSRT tracking after you select an object.
- Saves clean training frames without drawing the tracker overlay.
- Creates `data.yaml` for single-class YOLO training.
- Trains with Ultralytics YOLO.
- Copies `best.pt` to `assets/templates/yolo/<target>.pt` in the parent project.
- Includes an all-in-one workflow for capture, YAML creation, training, copy, and cleanup.

## Requirements

- Windows.
- Python 3.10 or newer.
- A visible target application window.
- Python packages from `requirements.txt`.
- NVIDIA GPU recommended for training, though Ultralytics can run on CPU with different settings.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
.
|-- yolo_workflow.py              # All-in-one workflow
|-- yolo_auto_dataset_tracker.py  # Manual select-and-track dataset capture
|-- create_data_yaml.py           # Generate data.yaml for a target
|-- copy_to_templates.py          # Copy best.pt to parent assets/templates/yolo
|-- yolo_finder.py                # Runtime detector helper for downstream projects
|-- test_yolo_tracking.py         # Tracking checks
|-- models/                       # Base model files
`-- requirements.txt
```

## Basic Workflow

Run from this repository root:

```bash
python yolo_workflow.py full mound_dharok
```

That command:

1. Opens the tracker.
2. Lets you press `S` and draw a bounding box around the target.
3. Saves images and labels to `dataset_mound_dharok/`.
4. Writes `mound_dharok/data.yaml`.
5. Trains a YOLO model.
6. Copies the trained weights.
7. Offers to clean temporary dataset and run folders.

## Tracker Controls

- `S`: select object.
- `R`: reset tracker and reselect.
- `P`: pause or resume saving frames.
- `Esc`: exit.

Move the camera or scene while tracking so the dataset contains useful variation.

## Individual Commands

Capture only:

```bash
python yolo_workflow.py track mound_dharok
```

Create YAML:

```bash
python yolo_workflow.py yaml mound_dharok
```

Train:

```bash
python yolo_workflow.py train mound_dharok --epochs 50 --device 0 --imgsz 640
```

Copy weights:

```bash
python yolo_workflow.py copy mound_dharok
```

Clean generated files for a target:

```bash
python yolo_workflow.py clean mound_dharok
```

You can also run the capture and YAML scripts directly:

```bash
python yolo_auto_dataset_tracker.py mound_dharok
python create_data_yaml.py mound_dharok --class-name MoundDharok
```

`copy_to_templates.py` is meant for the parent-project layout shown below. When this repository is used standalone, copy the trained `best.pt` manually or run the `copy` step through `yolo_workflow.py`, which knows about this folder's local `runs/` directory.

## Parent Project Expectations

Some scripts assume this folder lives inside a larger runtime-inference project:

```text
ParentProject/
|-- config.py
|-- game_io.py
|-- assets/templates/yolo/
`-- yolo-training-workflow/
```

When `config.py` and `game_io.py` are present in the parent project, the tracker captures only the configured application window and `copy_to_templates.py` writes to `../assets/templates/yolo/`.

If you use this repository standalone, the trainer still works, but you may need to copy the final `.pt` file manually from the training output into the project that will use it.

## Runtime Detector

`yolo_finder.py` provides a `find_yolo(template_name, region=None, confidence=0.5)` helper that returns `(x, y, w, h)` screen coordinates. It expects the parent project's `config.py` to define `YOLO_TARGETS` entries with model path, class, and optional confidence values.

Example shape:

```python
YOLO_TARGETS = {
    "mound_dharok": {
        "model": "assets/templates/yolo/mound_dharok.pt",
        "class": "MoundDharok",
        "conf": 0.5,
    }
}
```

## Tips

- Keep training capture and runtime inference on the same display and application-window scale.
- Save enough frames from different camera angles and lighting states.
- Use one target per dataset unless you intentionally expand the scripts for multi-class training.
- Put base models such as `yolov8n.pt` in `models/` so repeated runs do not download them into the working directory.
- If detection fails, enable debug frame saving in the parent config and inspect what image YOLO actually received.

## License

GPL-3.0. See `LICENSE`.
