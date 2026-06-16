"""
YOLO dataset creator: capture screen, select an object, track it, and save
images + YOLO-format labels (class_id x_center y_center width height normalized).
Run from project root or YOLO Trainer folder. Pass target as CLI arg.
  python "YOLO Trainer/yolo_auto_dataset_tracker.py" mound_dharok
  python "YOLO Trainer/yolo_auto_dataset_tracker.py" --mound_dharok
  python "YOLO Trainer/yolo_auto_dataset_tracker.py" --target barrows_chest
"""
import argparse
import cv2
import dxcam
import os
import sys

def _parse_target():
    """Parse target from CLI: --target X, or --mound_dharok, or positional mound_dharok."""
    parser = argparse.ArgumentParser(description="YOLO dataset tracker. Pass target name.")
    parser.add_argument("--target", default=None, help="Target name, e.g. mound_dharok, banker")
    args, unknown = parser.parse_known_args()
    if args.target:
        return args.target.strip().lower().replace(" ", "_")
    for u in unknown:
        if u.startswith("--") and "=" not in u:
            return u[2:].replace("-", "_").strip().lower()
    if unknown and not unknown[0].startswith("-"):
        return unknown[0].strip().lower().replace(" ", "_")
    return "mound_dharok"

TARGET = _parse_target()

# ---------------- SETTINGS ----------------
FRAME_SKIP = 3  # Save every Nth frame while tracking (1 = every frame)
SINGLE_CLASS_MODE = True  # When True, always write class 0 in labels

def _target_to_class_name(t):
    return "".join(w.capitalize() for w in t.split("_"))
CLASSES = {0: _target_to_class_name(TARGET)}
DATASET_SUBFOLDER = f"dataset_{TARGET}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(SCRIPT_DIR, DATASET_SUBFOLDER)
os.makedirs(os.path.join(DATASET_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels"), exist_ok=True)

# --------------- GAME WINDOW ONLY (match bot inference) ---------------
GAME_REGION = None
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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

# --------------- SCREEN CAPTURE ---------------
camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(target_fps=30, video_mode=True)

tracker = None
frame_id = 0
frame_counter = 0
paused = False
class_id = 0  # Single class (SINGLE_CLASS_MODE)
WINDOW_NAME = "YOLO Dataset Creator"

print(f"\nTarget: {TARGET}  →  dataset: {DATASET_SUBFOLDER}  class: {CLASSES[0]}")
print("Controls: S = select  R = reset (then S again to reselect if it drifted)  P = pause  ESC = exit\n")

# Single window for everything (no extra "Select Object" window)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    while True:

        frame = camera.get_latest_frame()
        if frame is None:
            continue
        # Crop to game window so training data matches bot inference (game window only).
        if GAME_REGION is not None:
            x, y, w, h = GAME_REGION
            H, W = frame.shape[:2]
            x1 = max(0, min(x, W - 1))
            y1 = max(0, min(y, H - 1))
            x2 = max(x1 + 1, min(x + w, W))
            y2 = max(y1 + 1, min(y + h, H))
            frame = frame[y1:y2, x1:x2]

        key = cv2.waitKey(1) & 0xFF

        # -------- CLASS SELECTION --------
        if key in range(ord('0'), ord('9')):
            selected = int(chr(key))
            if selected in CLASSES:
                class_id = selected
                print("Active class:", CLASSES[class_id])

        # -------- SELECT OBJECT --------
        if key == ord('s') or key == ord('S'):
            # Use same window so we don't get a second popup
            bbox = cv2.selectROI(WINDOW_NAME, frame, False)
            if bbox[2] <= 0 or bbox[3] <= 0:
                print("Selection cancelled or invalid (width/height > 0). Try again.")
                continue
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    raise RuntimeError(
                        "OpenCV tracking module not found. Install: pip install opencv-contrib-python"
                    ) from None
            tracker.init(frame, bbox)
            print("Tracker started. Green box = tracking; move camera for variety. (P=pause, R=reset, ESC=exit)")

        # -------- RESET TRACKER --------
        if key == ord('r') or key == ord('R'):
            tracker = None
            print("Tracker reset — press S to draw a new box if it drifted onto something else.")

        # -------- PAUSE --------
        if key == ord('p'):
            paused = not paused
            print("Paused:", paused)

        # -------- EXIT --------
        if key == 27:
            break

        if tracker is not None and not paused:

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = [int(v) for v in bbox]

                frame_counter += 1

                if frame_counter % FRAME_SKIP == 0:
                    # Save CLEAN frame (no overlay) so training matches inference (bot sees no green box).
                    img_path = os.path.join(DATASET_DIR, "images", f"frame_{frame_id:06d}.jpg")
                    label_path = os.path.join(DATASET_DIR, "labels", f"frame_{frame_id:06d}.txt")
                    cv2.imwrite(img_path, frame)

                    H, W, _ = frame.shape
                    x_center = max(0.0, min(1.0, (x + w / 2) / W))
                    y_center = max(0.0, min(1.0, (y + h / 2) / H))
                    width = max(0.0, min(1.0, w / W))
                    height = max(0.0, min(1.0, h / H))
                    write_class = 0 if SINGLE_CLASS_MODE else class_id
                    with open(label_path, "w") as f:
                        f.write(f"{write_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    frame_id += 1

                # draw bounding box (green = tracking) for display only
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    CLASSES[class_id],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.putText(frame, "R = reset", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            else:
                # Tracker lost – show red box and hint so user knows to reselect
                cv2.putText(
                    frame,
                    "Tracker lost - press R to reselect",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # -------- DISPLAY WINDOW --------
        cv2.imshow(WINDOW_NAME, frame)

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()