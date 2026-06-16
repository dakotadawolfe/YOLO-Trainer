"""
Generate data.yaml for any YOLO target (mound_dharok, banker, barrows_chest, etc.).
Dataset folder must be: YOLO Trainer/dataset_<target>/ (same TARGET as in yolo_auto_dataset_tracker.py).
Output: YOLO Trainer/<target>/data.yaml so you can train with:
  yolo detect train data="YOLO Trainer/<target>/data.yaml" model=yolov8n.pt epochs=50
Run from project root or from YOLO Trainer folder.
"""
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def target_to_class_name(target: str) -> str:
    """mound_dharok -> MoundDharok, banker -> Banker."""
    return "".join(w.capitalize() for w in target.split("_"))


def main():
    ap = argparse.ArgumentParser(description="Create data.yaml for a YOLO target (dynamic).")
    ap.add_argument("target", nargs="?", help="Target name, e.g. mound_dharok, banker, barrows_chest")
    ap.add_argument("--class-name", default=None, help="Class name in data.yaml (default: CamelCase from target)")
    args = ap.parse_args()

    target = (args.target or "").strip().lower().replace(" ", "_")
    if not target:
        # Auto-detect: use sole dataset_* folder if present
        candidates = [
            d.replace("dataset_", "", 1) for d in os.listdir(SCRIPT_DIR)
            if d.startswith("dataset_") and os.path.isdir(os.path.join(SCRIPT_DIR, d))
        ]
        if len(candidates) == 1:
            target = candidates[0]
            print("No target given; using dataset folder: dataset_%s" % target)
        else:
            print("Usage: python create_data_yaml.py <target> [--class-name ClassName]")
            print("Example: python create_data_yaml.py mound_dharok --class-name DharokMound")
            if candidates:
                print("Found dataset folders: %s" % ", ".join("dataset_%s" % c for c in candidates))
            return

    class_name = args.class_name if args.class_name else target_to_class_name(target)

    dataset_dir = os.path.join(SCRIPT_DIR, f"dataset_{target}")
    out_dir = os.path.join(SCRIPT_DIR, target)
    yaml_path = os.path.join(out_dir, "data.yaml")

    if not os.path.isdir(dataset_dir):
        print(f"Dataset folder not found: {dataset_dir}")
        print("Run the tracker first with TARGET = %r in yolo_auto_dataset_tracker.py" % target)
        return

    path_for_yaml = os.path.abspath(dataset_dir).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)

    content = f"""# YOLO dataset: {target}
# Dataset from: YOLO Trainer/dataset_{target}/
path: {path_for_yaml}
train: images
val: images
nc: 1
names: ['{class_name}']
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    rel_yaml = os.path.relpath(yaml_path, os.path.dirname(SCRIPT_DIR)) if os.path.dirname(SCRIPT_DIR) else yaml_path
    print(f"Wrote {yaml_path}")
    print(f"Class name: {class_name}")
    print(f"Train (from project root): yolo detect train data=\"{rel_yaml.replace(chr(92), '/')}\" model=yolov8n.pt epochs=50")
    print(f"Then copy weights to templates: python \"YOLO Trainer/copy_to_templates.py\" {target}")


if __name__ == "__main__":
    main()
