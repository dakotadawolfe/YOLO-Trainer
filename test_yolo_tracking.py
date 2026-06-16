"""
Standalone YOLO banker detector and tracker.
Finds the banker on screen, tracks him, and shows a purple dot on the tracked position.
Run separately from the main Barrows loop: python test_yolo_banker.py
Exit with 'q' or ESC.
"""
import cv2
import logging
import os
import sys

import config
import game_io

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Set this once to choose what to track.
# Examples:
# - "banker" (config key)
# - "OldMan" (class name inside a model)
TRACK_TARGET = "prayer_low"

# Purple dot and optional bbox
PURPLE_BGR = (255, 0, 255)
DOT_RADIUS = 10
WINDOW_NAME = f"YOLO Tracker - {TRACK_TARGET}"


def _resolve_class_id(model, class_spec):
    """Return integer class id from config (int or str like 'banker' / 'person')."""
    if isinstance(class_spec, int):
        return class_spec
    names = getattr(model, "names", {})
    if isinstance(names, list):
        for idx, name in enumerate(names):
            if name == class_spec:
                return idx
    else:
        for idx, name in names.items():
            if name == class_spec:
                return idx
    logger.warning("Class %r not in model names %s; using 0", class_spec, names)
    return 0


def _resolve_target_config(target_name: str):
    """
    Resolve (model_path, class_spec, source_desc) from:
    1) exact YOLO_TARGETS key
    2) first YOLO_TARGETS entry whose class == target_name
    3) fallback to assets/templates/yolo/<target_name>.pt with class=target_name
    """
    targets = getattr(config, "YOLO_TARGETS", {}) or {}
    if target_name in targets:
        cfg = targets[target_name] or {}
        return cfg.get("model"), cfg.get("class", target_name), f"key:{target_name}"

    class_matches = []
    for key, cfg in targets.items():
        cfg = cfg or {}
        if cfg.get("class") == target_name:
            class_matches.append((key, cfg))
    if class_matches:
        key, cfg = class_matches[0]
        if len(class_matches) > 1:
            logger.warning(
                "Multiple YOLO targets have class %r; using first key=%s",
                target_name,
                key,
            )
        return cfg.get("model"), cfg.get("class", target_name), f"class:{target_name} (key={key})"

    model_path = os.path.join(config.TEMPLATES_DIR, "yolo", f"{target_name}.pt")
    return model_path, target_name, "fallback"


def run_tracker():
    """Load YOLO model, capture screen, detect/track banker, draw purple dot; exit on 'q' or ESC."""
    model_path, class_spec, source = _resolve_target_config(TRACK_TARGET)
    if not model_path:
        logger.error("No model path resolved for TRACK_TARGET=%r", TRACK_TARGET)
        sys.exit(1)
    use_track = True  # persist identity across frames

    logger.info("Tracking target: %s (source=%s)", TRACK_TARGET, source)
    logger.info("Loading YOLO model: %s", model_path)
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics>=8.0.0")
        sys.exit(1)

    model = YOLO(model_path)
    target_cls_id = _resolve_class_id(model, class_spec)
    logger.info("Target class id: %s", target_cls_id)

    game_io.focus_game_window()
    logger.info("Starting tracker. Show the game window with the banker. Press 'q' or ESC to quit.")

    while True:
        frame = game_io.screenshot(region=None)
        if frame is None or frame.size == 0:
            continue

        if use_track:
            results = model.track(frame, persist=True, verbose=False)
        else:
            results = model.predict(frame, verbose=False)

        display = frame.copy()
        cx, cy = None, None
        bbox = None

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id != target_cls_id:
                        continue
                    xyxy = boxes.xyxy[i]
                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    break  # first matching detection

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), PURPLE_BGR, 2)
        if cx is not None and cy is not None:
            cv2.circle(display, (cx, cy), DOT_RADIUS, PURPLE_BGR, -1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    logger.info("Tracker stopped.")


if __name__ == "__main__":
    run_tracker()
