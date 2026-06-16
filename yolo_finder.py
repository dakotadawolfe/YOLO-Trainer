"""
YOLO-based object finder: same (x, y, w, h) contract as old template matching.
Strict mode: every lookup key must exist in config.YOLO_TARGETS.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import config
import game_io

logger = logging.getLogger(__name__)

Region = Tuple[int, int, int, int]

_models_cache = {}  # model_path -> YOLO model


class YoloRuntimeValidationError(RuntimeError):
    """Raised when a required YOLO target/model config is missing in deferred-validation debug mode."""


def _defer_validation_enabled() -> bool:
    return bool(getattr(config, "YOLO_DEBUG_DEFER_VALIDATION", False))


def _target_config(template_name: str) -> Optional[Dict]:
    targets = getattr(config, "YOLO_TARGETS", None) or {}
    return targets.get(template_name)


def _get_model(model_path: str):
    """Lazy-load YOLO model by path; cache by path."""
    global _models_cache
    if model_path not in _models_cache:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed; YOLO finder disabled.")
            return None
        if not model_path or not os.path.exists(model_path):
            msg = f"YOLO model file not found: {model_path}"
            if _defer_validation_enabled():
                raise YoloRuntimeValidationError(msg)
            logger.warning(msg)
            return None
        _models_cache[model_path] = YOLO(model_path)
        logger.info("YOLO loaded: %s  names=%s", model_path, getattr(_models_cache[model_path], "names", None))
    return _models_cache[model_path]


def _resolve_class_id(model, class_spec) -> Optional[int]:
    """Return integer class id from config; return None if not present in model names (strict)."""
    names = getattr(model, "names", {})
    if isinstance(class_spec, int):
        if isinstance(names, list):
            return class_spec if 0 <= class_spec < len(names) else None
        return class_spec if class_spec in names else None
    if isinstance(names, list):
        for idx, name in enumerate(names):
            if name == class_spec:
                return idx
    else:
        for idx, name in names.items():
            if name == class_spec:
                return idx
    return None


def _get_model_and_class(template_name: str):
    """Resolve (model, class_id, conf_override) for template_name from config.YOLO_TARGETS."""
    cfg = _target_config(template_name)
    if not cfg:
        msg = f"YOLO target not configured: {template_name} (missing in config.YOLO_TARGETS)"
        if _defer_validation_enabled():
            raise YoloRuntimeValidationError(msg)
        logger.warning(msg)
        return None, None, None
    model_path = cfg.get("model")
    class_spec = cfg.get("class")
    conf_override = cfg.get("conf")
    if not model_path or class_spec is None:
        msg = f"YOLO target {template_name} invalid config (needs model + class): {cfg}"
        if _defer_validation_enabled():
            raise YoloRuntimeValidationError(msg)
        logger.warning(msg)
        return None, None, conf_override
    model = _get_model(model_path)
    if model is None:
        return None, None, conf_override
    target_cls_id = _resolve_class_id(model, class_spec)
    if target_cls_id is None:
        msg = f"YOLO class {class_spec!r} not found in model {model_path} names={getattr(model, 'names', None)}"
        if _defer_validation_enabled():
            raise YoloRuntimeValidationError(msg)
        logger.warning(msg)
        return None, None, conf_override
    return model, target_cls_id, conf_override


def find_yolo(
    template_name: str,
    region: Optional[Region] = None,
    confidence: float = 0.5,
) -> Optional[Region]:
    """Find object using YOLO. Returns (x, y, w, h) in screen coordinates or None."""
    model, target_cls_id, conf_override = _get_model_and_class(template_name)
    if model is None:
        return None

    conf = conf_override if isinstance(conf_override, (int, float)) else confidence

    capture_region = region
    if region is None and getattr(config, "YOLO_CAPTURE_GAME_WINDOW", True):
        capture_region = game_io.get_game_window_region()

    used_dxcam = False
    if getattr(config, "USE_DXCAM_FOR_YOLO", False):
        frame = game_io.screenshot_dxcam(region=capture_region)
        if frame is not None:
            used_dxcam = True
        if frame is None:
            frame = game_io.screenshot(region=capture_region)
    else:
        frame = game_io.screenshot(region=capture_region)
    if frame is None or frame.size == 0:
        return None
    if not hasattr(find_yolo, "_logged_capture"):
        logger.info("YOLO capture: %s (frame %s)", "dxcam" if used_dxcam else "pyautogui", frame.shape)
        find_yolo._logged_capture = True

    gamma = getattr(config, "YOLO_HDR_GAMMA", None)
    if gamma is not None and isinstance(gamma, (int, float)):
        inv = 1.0 / float(gamma)
        frame = np.clip((frame.astype(np.float32) / 255.0) ** inv * 255.0, 0, 255).astype(np.uint8)

    results = model.predict(frame, verbose=False, conf=0.1)
    if not results or len(results) == 0:
        _save_debug_frame(frame, template_name)
        _log_yolo_fail(template_name, frame.shape, 0, target_cls_id, conf, None)
        logger.warning("YOLO %s: no detections. Check assets/templates/yolo/debug_frame_no_boxes.jpg", template_name)
        return None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        _save_debug_frame(frame, template_name)
        _log_yolo_fail(template_name, frame.shape, 0, target_cls_id, conf, None)
        logger.warning("YOLO %s: no boxes. Check assets/templates/yolo/debug_frame_no_boxes.jpg", template_name)
        return None

    boxes = r.boxes
    offset_x = capture_region[0] if capture_region else (region[0] if region else 0)
    offset_y = capture_region[1] if capture_region else (region[1] if region else 0)

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id != target_cls_id:
            continue
        conf_val = float(boxes.conf[i].item()) if boxes.conf is not None else 1.0
        if conf_val < conf:
            continue
        xyxy = boxes.xyxy[i]
        x1 = int(xyxy[0].item()) + offset_x
        y1 = int(xyxy[1].item()) + offset_y
        x2 = int(xyxy[2].item()) + offset_x
        y2 = int(xyxy[3].item()) + offset_y
        w, h = x2 - x1, y2 - y1
        return (x1, y1, w, h)

    box_list = [(int(boxes.cls[j].item()), float(boxes.conf[j].item()) if boxes.conf is not None else 0) for j in range(len(boxes))]
    target_confs = [c for cid, c in box_list if cid == target_cls_id]
    max_conf = max(target_confs) if target_confs else (max((c for _, c in box_list)) if box_list else 0.0)
    _save_debug_frame(frame, template_name)
    _log_yolo_fail(template_name, frame.shape, len(boxes), target_cls_id, conf, box_list)
    logger.warning(
        "YOLO %s: %d box(es), max conf %.2f (need >= %.2f).",
        template_name,
        len(boxes),
        max_conf,
        conf,
    )
    return None


def validate_targets(required_targets: List[str]) -> Dict[str, List[str]]:
    """
    Validate YOLO target configs for startup checks.
    Returns lists: missing_targets, missing_models, class_mismatches, invalid_entries.
    """
    report = {
        "missing_targets": [],
        "missing_models": [],
        "class_mismatches": [],
        "invalid_entries": [],
    }
    seen = set()
    for target in required_targets:
        if target in seen:
            continue
        seen.add(target)
        cfg = _target_config(target)
        if cfg is None:
            report["missing_targets"].append(target)
            continue
        model_path = cfg.get("model")
        class_spec = cfg.get("class")
        if not model_path or class_spec is None:
            report["invalid_entries"].append(target)
            continue
        if not os.path.isfile(model_path):
            report["missing_models"].append(f"{target} -> {model_path}")
            continue
        model = _get_model(model_path)
        if model is None:
            report["missing_models"].append(f"{target} -> {model_path}")
            continue
        cls_id = _resolve_class_id(model, class_spec)
        if cls_id is None:
            report["class_mismatches"].append(
                f"{target} -> class={class_spec!r}, model_names={getattr(model, 'names', None)}"
            )
    return report


_logged_fail = set()


def _save_debug_frame(frame, template_name: str):
    """Save the frame we fed to YOLO when we get 0 boxes, so user can verify capture."""
    if not getattr(config, "YOLO_DEBUG_SAVE_FRAME", False):
        return
    try:
        import cv2
        debug_dir = os.path.join(config.TEMPLATES_DIR, "yolo")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "debug_frame_no_boxes.jpg")
        cv2.imwrite(path, frame)
        logger.info("YOLO: saved capture to %s (check if game/mound is visible)", path)
    except Exception as e:
        logger.debug("Could not save debug frame: %s", e)


def _log_yolo_fail(template_name: str, shape, num_boxes: int, target_cls_id: int, conf_thresh: float, box_list):
    """Log once per session when YOLO returns None so we can diagnose."""
    if template_name in _logged_fail or not getattr(config, "YOLO_LOG_FAILURE", True):
        return
    _logged_fail.add(template_name)
    if num_boxes == 0:
        logger.info("YOLO %s: no boxes (frame %s, conf>=%.2f). Check debug_frame_no_boxes.jpg if YOLO_DEBUG_SAVE_FRAME=True. Multi-monitor? Put game on same display as when you trained.", template_name, shape, conf_thresh)
    else:
        logger.info("YOLO %s: %d boxes but none passed (target_cls=%s, conf>=%.2f). Boxes (cls,conf): %s", template_name, num_boxes, target_cls_id, conf_thresh, box_list[:10])
