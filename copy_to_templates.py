"""
Copy trained YOLO weights (best.pt) into assets/templates/yolo/<target>.pt.
Run after training so the bot can use the model from the templates folder.
Once this is done, you can delete the YOLO Trainer folder if desired.
Run from project root or from YOLO Trainer folder.

  python "YOLO Trainer/copy_to_templates.py" mound_dharok
  python "YOLO Trainer/copy_to_templates.py" banker --run-dir runs/detect/train2
"""
import argparse
import os
import shutil

# Project root = parent of YOLO Trainer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATES_YOLO = os.path.join(PROJECT_ROOT, "assets", "templates", "yolo")


def main():
    ap = argparse.ArgumentParser(description="Copy best.pt from training run to assets/templates/yolo/<target>.pt")
    ap.add_argument("target", help="Target name, e.g. mound_dharok, banker")
    ap.add_argument("--run-dir", default=None,
                    help="Training run dir (relative to project root), e.g. runs/detect/train2. If omitted, auto-finds latest run.")
    args = ap.parse_args()

    target = args.target.strip().lower().replace(" ", "_")
    detects_dir = os.path.join(PROJECT_ROOT, "runs", "detect")
    weights_src = None

    if args.run_dir:
        run_dir = os.path.join(PROJECT_ROOT, args.run_dir.replace("/", os.sep))
        candidate = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(candidate):
            weights_src = candidate
    if not weights_src and os.path.isdir(detects_dir):
        # Find latest run that has weights/best.pt (train, train2, train3, ...)
        for name in sorted(os.listdir(detects_dir), reverse=True):
            candidate = os.path.join(detects_dir, name, "weights", "best.pt")
            if os.path.isfile(candidate):
                weights_src = candidate
                break

    os.makedirs(TEMPLATES_YOLO, exist_ok=True)
    dest_name = f"{target}.pt"
    weights_dest = os.path.join(TEMPLATES_YOLO, dest_name)

    if not weights_src or not os.path.isfile(weights_src):
        print("No best.pt found in runs/detect/")
        print("1) Train first: yolo detect train data=\"YOLO Trainer/mound_dharok/data.yaml\" model=yolov8n.pt epochs=50 device=0")
        print("2) If you already trained, specify the run: python \"YOLO Trainer/copy_to_templates.py\" mound_dharok --run-dir runs/detect/train2")
        return 1

    shutil.copy2(weights_src, weights_dest)
    rel_dest = os.path.relpath(weights_dest, PROJECT_ROOT).replace("\\", "/")
    print(f"Copied to {weights_dest}")
    print(f"In config use: assets/templates/yolo/{target}.pt  (or {rel_dest})")
    return 0


if __name__ == "__main__":
    exit(main())
