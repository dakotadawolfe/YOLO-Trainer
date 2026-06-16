"""
All-in-one YOLO workflow: track → create yaml → train → copy to templates → clean for next run.
Run from project root. Target = object name (e.g. mound_dharok, banker).

  python "YOLO Trainer/yolo_workflow.py" full mound_dharok     # track, yaml, train, copy, then offer to clean
  python "YOLO Trainer/yolo_workflow.py" track mound_dharok   # run tracker only (S=select, R=reset, ESC=exit)
  python "YOLO Trainer/yolo_workflow.py" yaml mound_dharok    # create data.yaml from dataset
  python "YOLO Trainer/yolo_workflow.py" train mound_dharok   # yolo detect train (uses data.yaml)
  python "YOLO Trainer/yolo_workflow.py" copy mound_dharok    # copy best.pt to assets/templates/yolo/
  python "YOLO Trainer/yolo_workflow.py" clean mound_dharok   # delete dataset, yaml dir, runs/detect (fresh start)
"""
import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")  # base .pt files (yolov8n.pt, etc.) for future runs
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs", "detect")  # training output stays inside YOLO Trainer
# Base model filenames that ultralytics may download to cwd; we move to MODELS_DIR and clean from root
BASE_MODEL_NAMES = ("yolov8n.pt", "yolo26n.pt")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------- Helpers --------
def _target_to_class_name(t: str) -> str:
    return "".join(w.capitalize() for w in t.split("_"))


def _get_target_or_auto(script_dir: str) -> str:
    candidates = [
        d.replace("dataset_", "", 1) for d in os.listdir(script_dir)
        if d.startswith("dataset_") and os.path.isdir(os.path.join(script_dir, d))
    ]
    if len(candidates) == 1:
        return candidates[0]
    return ""


# -------- 1. TRACK --------
def run_tracker(target: str, frame_skip: int = 3) -> bool:
    """Run the screen tracker: select object with S, track, ESC to exit. Saves clean frames + labels to dataset_<target>/."""
    import cv2
    import dxcam

    target = target.strip().lower().replace(" ", "_")
    class_name = _target_to_class_name(target)
    dataset_subfolder = f"dataset_{target}"
    dataset_dir = os.path.join(SCRIPT_DIR, dataset_subfolder)
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)

    GAME_REGION = None
    try:
        import config as _config
        import game_io as _game_io
        GAME_REGION = _game_io.get_game_window_region()
    except Exception:
        pass
    if GAME_REGION is not None:
        print("Capture: game window only (matches bot YOLO lookup)")
    else:
        print("Capture: full screen (game window not found; set GAME_WINDOW_TITLE_SUBSTRING in config)")

    camera = dxcam.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=30, video_mode=True)

    tracker = None
    frame_id = 0
    frame_counter = 0
    paused = False
    class_id = 0
    WINDOW_NAME = "YOLO Dataset Creator"
    CLASSES = {0: class_name}

    print(f"\nTarget: {target}  →  dataset: {dataset_subfolder}  class: {class_name}")
    print("Controls: S = select  R = reset  P = pause  ESC = exit\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            if GAME_REGION is not None:
                x, y, w, h = GAME_REGION
                H, W = frame.shape[:2]
                x1 = max(0, min(x, W - 1))
                y1 = max(0, min(y, H - 1))
                x2 = max(x1 + 1, min(x + w, W))
                y2 = max(y1 + 1, min(y + h, H))
                frame = frame[y1:y2, x1:x2]

            key = cv2.waitKey(1) & 0xFF
            if key in range(ord("0"), ord("9")):
                selected = int(chr(key))
                if selected in CLASSES:
                    class_id = selected
            if key == ord("s") or key == ord("S"):
                bbox = cv2.selectROI(WINDOW_NAME, frame, False)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    print("Selection cancelled or invalid. Try again.")
                    continue
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                except AttributeError:
                    try:
                        tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        raise RuntimeError("Install: pip install opencv-contrib-python") from None
                tracker.init(frame, bbox)
                print("Tracker started. (P=pause, R=reset, ESC=exit)")
            if key == ord("r") or key == ord("R"):
                tracker = None
                print("Tracker reset — press S to reselect.")
            if key == ord("p"):
                paused = not paused
            if key == 27:
                break

            if tracker is not None and not paused:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    frame_counter += 1
                    if frame_counter % frame_skip == 0:
                        img_path = os.path.join(dataset_dir, "images", f"frame_{frame_id:06d}.jpg")
                        label_path = os.path.join(dataset_dir, "labels", f"frame_{frame_id:06d}.txt")
                        cv2.imwrite(img_path, frame)
                        H, W, _ = frame.shape
                        xc = max(0.0, min(1.0, (x + w / 2) / W))
                        yc = max(0.0, min(1.0, (y + h / 2) / H))
                        bw = max(0.0, min(1.0, w / W))
                        bh = max(0.0, min(1.0, h / H))
                        with open(label_path, "w") as f:
                            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        frame_id += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "R = reset", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "Tracker lost - press R to reselect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    n = len([f for f in os.listdir(os.path.join(dataset_dir, "images")) if f.endswith(".jpg")])
    print(f"Saved {n} frames to {dataset_subfolder}/")
    return n > 0


# -------- 2. YAML --------
def run_yaml(target: str, class_name: str = None) -> bool:
    """Create data.yaml for target. Dataset must exist (run track first)."""
    target = target.strip().lower().replace(" ", "_")
    class_name = (class_name or _target_to_class_name(target)).strip()
    dataset_dir = os.path.join(SCRIPT_DIR, f"dataset_{target}")
    out_dir = os.path.join(SCRIPT_DIR, target)
    yaml_path = os.path.join(out_dir, "data.yaml")

    if not os.path.isdir(dataset_dir):
        print(f"Dataset not found: {dataset_dir}. Run 'track {target}' first.")
        return False

    path_for_yaml = os.path.abspath(dataset_dir).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    content = f"""# YOLO dataset: {target}
path: {path_for_yaml}
train: images
val: images
nc: 1
names: ['{class_name}']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}  (class: {class_name})")
    return True


# -------- 3. TRAIN --------
def run_train(target: str, epochs: int = 50, device: str = "0", imgsz: int = 640, base_model: str = "yolov8n.pt") -> bool:
    """Run yolo detect train. data.yaml must exist (run yaml first). Uses YOLO Trainer/models/ for base .pt and saves runs to YOLO Trainer/runs/."""
    target = target.strip().lower().replace(" ", "_")
    yaml_path = os.path.join(SCRIPT_DIR, target, "data.yaml")
    if not os.path.isfile(yaml_path):
        print(f"data.yaml not found: {yaml_path}. Run 'yaml {target}' first.")
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_full = os.path.join(MODELS_DIR, base_model)
    if os.path.isfile(model_full):
        model_path = model_full
    else:
        model_path = base_model
        print(f"Note: put {base_model} in YOLO Trainer/models/ for future runs. Using default for this run.")

    project_dir = os.path.join(SCRIPT_DIR, "runs")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        return False

    print(f"Training: data={yaml_path} model={model_path} epochs={epochs} device={device} project={project_dir}")
    model = YOLO(model_path)
    results = model.train(
        data=os.path.abspath(yaml_path),
        epochs=int(epochs),
        device=device,
        imgsz=int(imgsz),
        project=os.path.abspath(project_dir),
        name="detect",
    )
    # Move any base .pt that was downloaded to project root into YOLO Trainer/models/
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in BASE_MODEL_NAMES:
        in_root = os.path.join(PROJECT_ROOT, name)
        in_models = os.path.join(MODELS_DIR, name)
        if os.path.isfile(in_root) and not os.path.isfile(in_models):
            try:
                shutil.move(in_root, in_models)
                print(f"Moved {name} to YOLO Trainer/models/")
            except Exception as e:
                print(f"Could not move {name} to models/: {e}")
    return results is not None


# -------- 4. COPY --------
def run_copy(target: str, run_dir: str = None) -> bool:
    """Copy best.pt from YOLO Trainer/runs/detect (or project runs/detect) to assets/templates/yolo/<target>.pt."""
    target = target.strip().lower().replace(" ", "_")
    templates_yolo = os.path.join(PROJECT_ROOT, "assets", "templates", "yolo")
    # Prefer runs inside YOLO Trainer, then project root
    detects_dirs = [
        RUNS_DIR,
        os.path.join(PROJECT_ROOT, "runs", "detect"),
    ]
    weights_src = None

    if run_dir:
        for base in (SCRIPT_DIR, PROJECT_ROOT):
            candidate = os.path.join(base, run_dir.replace("/", os.sep), "weights", "best.pt")
            if os.path.isfile(candidate):
                weights_src = candidate
                break
    if not weights_src:
        for detects_dir in detects_dirs:
            if not os.path.isdir(detects_dir):
                continue
            # When project=.../runs, name=detect, weights are directly in detect/weights/best.pt
            direct = os.path.join(detects_dir, "weights", "best.pt")
            if os.path.isfile(direct):
                weights_src = direct
                break
            # Else look for train/train2/.../weights/best.pt
            for name in sorted(os.listdir(detects_dir), reverse=True):
                if name == "weights":
                    continue
                candidate = os.path.join(detects_dir, name, "weights", "best.pt")
                if os.path.isfile(candidate):
                    weights_src = candidate
                    break
            if weights_src:
                break

    if not weights_src or not os.path.isfile(weights_src):
        print("No best.pt in YOLO Trainer/runs/detect/ (or runs/detect/). Run 'train' first.")
        return False

    os.makedirs(templates_yolo, exist_ok=True)
    dest = os.path.join(templates_yolo, f"{target}.pt")
    shutil.copy2(weights_src, dest)
    print(f"Copied to {dest}")
    return True


# -------- 5. CLEAN --------
def run_clean(target: str) -> bool:
    """Delete dataset_<target>, <target>/, YOLO Trainer/runs/, and stray base .pt files from project root."""
    target = target.strip().lower().replace(" ", "_")
    to_remove = [
        os.path.join(SCRIPT_DIR, f"dataset_{target}"),
        os.path.join(SCRIPT_DIR, target),
        os.path.join(SCRIPT_DIR, "runs"),  # training output inside YOLO Trainer
    ]
    for path in to_remove:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"Deleted: {path}")
            except Exception as e:
                print(f"Could not delete {path}: {e}")
                return False
    # Remove stray base .pt files from project root (ultralytics may download there)
    for name in BASE_MODEL_NAMES:
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Deleted from root: {name}")
            except Exception as e:
                print(f"Could not delete {path}: {e}")
    print("Clean done. Ready for next target.")
    return True


# -------- MAIN --------
def main():
    ap = argparse.ArgumentParser(
        description="YOLO workflow: track → yaml → train → copy → clean",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("command", choices=["track", "yaml", "train", "copy", "clean", "full"],
                    help="track=run tracker | yaml=create data.yaml | train=run training | copy=copy to templates | clean=delete dataset+yaml+runs+root .pt | full=track→yaml→train→copy→auto clean")
    ap.add_argument("target", nargs="?", default=None,
                    help="Target name (e.g. mound_dharok, banker). For full/track required; for yaml/train/copy/clean can auto-detect if only one dataset_* exists.")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs (default 50)")
    ap.add_argument("--device", default="0", help="GPU device (default 0)")
    ap.add_argument("--imgsz", type=int, default=640, help="Train image size (default 640)")
    ap.add_argument("--frame-skip", type=int, default=3, help="Save every Nth frame when tracking (default 3)")
    ap.add_argument("--run-dir", default=None, help="For copy: e.g. runs/detect/train2 (default: latest)")
    ap.add_argument("--model", default="yolov8n.pt", help="Base model name in YOLO Trainer/models/ (default yolov8n.pt)")
    ap.add_argument("--no-clean", action="store_true", help="With 'full': skip auto-clean at the end")
    args = ap.parse_args()

    target = (args.target or "").strip().lower().replace(" ", "_")
    if not target and args.command in ("yaml", "train", "copy", "clean"):
        target = _get_target_or_auto(SCRIPT_DIR)
    if not target:
        if args.command in ("track", "full"):
            print("Target required for track/full. Example: python yolo_workflow.py track mound_dharok")
            return 1
        print("No target given and no single dataset_* folder found. Pass target name.")
        return 1

    if args.command == "track":
        ok = run_tracker(target, frame_skip=args.frame_skip)
        return 0 if ok else 1

    if args.command == "yaml":
        return 0 if run_yaml(target) else 1

    if args.command == "train":
        return 0 if run_train(target, epochs=args.epochs, device=args.device, imgsz=args.imgsz, base_model=args.model) else 1

    if args.command == "copy":
        return 0 if run_copy(target, run_dir=args.run_dir) else 1

    if args.command == "clean":
        return 0 if run_clean(target) else 1

    if args.command == "full":
        print("=== 1. TRACK ===")
        if not run_tracker(target, frame_skip=args.frame_skip):
            print("No frames saved. Exiting.")
            return 1
        print("\n=== 2. YAML ===")
        if not run_yaml(target):
            return 1
        print("\n=== 3. TRAIN ===")
        if not run_train(target, epochs=args.epochs, device=args.device, imgsz=args.imgsz, base_model=args.model):
            return 1
        print("\n=== 4. COPY TO TEMPLATES ===")
        if not run_copy(target, run_dir=args.run_dir):
            return 1
        print("\n=== 5. CLEAN ===")
        if not args.no_clean:
            run_clean(target)
        print("\nDone. Add target to config.USE_YOLO_FOR and set model path to assets/templates/yolo/%s.pt" % target)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
