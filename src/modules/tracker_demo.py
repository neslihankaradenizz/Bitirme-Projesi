"""
tracker_demo.py — Part 2: YOLOv8 + ByteTrack real-time tracking demo.

Runs YOLOv8n with ByteTrack on a live webcam feed.
Annotates each frame with:
  - Bounding boxes (colour-coded by track ID)
  - Class name + confidence score
  - Persistent tracking ID (#N)
  - EMA-smoothed FPS counter

Saves annotated frames to a video file via OpenCV VideoWriter.
Prints per-frame stats every 30 frames.

Run from the project root:
    python src/modules/tracker_demo.py

CLI options:
    --camera  INT    Camera index (default: 0)
    --conf    FLOAT  Confidence threshold (default: 0.4)
    --output  STR    Output video path (default: outputs/tracked_output.mp4)
    --detect         Use detection-only mode (no ByteTrack), for comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python  →  pip install opencv-python")
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        missing.append("ultralytics    →  pip install ultralytics")
    if missing:
        print("\n[tracker_demo] Missing dependencies:\n")
        for m in missing:
            print(f"  • {m}")
        print("\nInstall:  pip install -r requirements.txt\n")
        sys.exit(1)

_check_deps()

import cv2  # noqa: E402
from src.modules.object_tracker import ObjectTracker  # noqa: E402

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# 20-colour palette for track IDs — gives distinct colours even for crowded scenes
_ID_PALETTE = [
    (220,  20,  60), (255, 127,  14), ( 44, 160,  44), ( 31, 119, 180),
    (148, 103, 189), (140,  86,  75), (227, 119, 194), (127, 127, 127),
    (188, 189,  34), ( 23, 190, 207), (255, 187, 120), (152, 223, 138),
    (174, 199, 232), (197, 176, 213), (196, 156, 148), (247, 182, 210),
    (199, 199, 199), (219, 219, 141), (158, 218, 229), (255, 152, 150),
]


def _bgr_for_id(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour: per track_id when tracking, per class otherwise."""
    idx = track_id if track_id is not None else class_id
    r, g, b = _ID_PALETTE[idx % len(_ID_PALETTE)]
    return (b, g, r)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_tracked(frame, detections: list[dict], tracking: bool) -> None:
    """
    Annotate *frame* with detection / tracking overlays (in-place).

    Shows:  [#ID] class  conf%   (track ID shown only when tracking=True)
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid    = det["track_id"]
        color  = _bgr_for_id(tid, det["class_id"])

        # Build label
        id_tag = f"#{tid} " if (tracking and tid is not None) else ""
        label  = f"{id_tag}{det['class_name']} {det['confidence']:.0%}"

        # Bounding box — thicker for tracked objects
        thickness = 3 if (tracking and tid is not None) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label pill background
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Small dot at centroid (useful for trajectory visualisation)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, color, -1)


def draw_fps_and_mode(frame, fps: float, tracking: bool, n_det: int) -> None:
    """Top-right HUD: mode badge + FPS + object count."""
    h, w = frame.shape[:2]
    mode_text = "TRACKING" if tracking else "DETECT"
    mode_color = (0, 200, 255) if tracking else (200, 200, 200)

    lines = [
        (f"Mode : {mode_text}", mode_color),
        (f"FPS  : {fps:5.1f}",  (0, 220, 0) if fps >= 15 else (0, 140, 255)),
        (f"Objs : {n_det}",     (220, 220, 220)),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    for text, color in lines:
        (tw, _), _ = cv2.getTextSize(text, font, 0.65, 2)
        x = w - tw - 12
        cv2.putText(frame, text, (x + 1, y + 1), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),          font, 0.65, color,     2, cv2.LINE_AA)
        y += 28


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(
    camera_index: int = 0,
    conf_threshold: float = 0.4,
    output_path: str = "outputs/tracked_output.mp4",
    tracking: bool = True,
) -> None:
    """
    Capture webcam → (detect or track) → annotate → display + save.

    Args:
        camera_index:    cv2 camera index.
        conf_threshold:  YOLO confidence gate.
        output_path:     Destination .mp4 file.
        tracking:        True → ByteTrack (Part 2), False → detect-only (Part 1).
    """
    tracker = ObjectTracker(conf_threshold=conf_threshold)
    infer_fn = tracker.track if tracking else tracker.detect
    mode_str = "ByteTrack" if tracking else "detect-only"
    print(f"[tracker_demo] Mode: {mode_str}  conf={conf_threshold:.2f}")

    # --- Open camera ---
    print(f"[tracker_demo] Opening camera {camera_index} …")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[tracker_demo] ERROR: Cannot open camera {camera_index}.")
        sys.exit(1)

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[tracker_demo] Camera: {fw}×{fh} @ {cam_fps:.0f} fps")

    # --- VideoWriter ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, cam_fps, (fw, fh))
    print(f"[tracker_demo] Saving → {output_path}\n")
    print("[tracker_demo] Press 'q' to quit.\n")

    # --- EMA FPS ---
    fps_ema  = 0.0
    alpha    = 0.1
    n_frames = 0

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[tracker_demo] Frame grab failed — stopping.")
                break
            n_frames += 1

            # ---- Inference ----
            detections = infer_fn(frame)

            # ---- Annotate ----
            draw_tracked(frame, detections, tracking=tracking)
            draw_fps_and_mode(frame, fps_ema, tracking=tracking, n_det=len(detections))

            # ---- Display + save ----
            window_title = "Hazard Detection — Part 2: YOLO + ByteTrack"
            cv2.imshow(window_title, frame)
            writer.write(frame)

            # ---- FPS ----
            elapsed = time.perf_counter() - t0
            fps_ema = alpha * (1.0 / elapsed if elapsed > 0 else 0.0) + (1 - alpha) * fps_ema

            if n_frames % 30 == 0:
                ids = [d["track_id"] for d in detections if d["track_id"] is not None]
                print(
                    f"  frame {n_frames:>5}  |  FPS {fps_ema:5.1f}  "
                    f"|  objects: {len(detections)}  |  track_ids: {ids}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[tracker_demo] Interrupted.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n[tracker_demo] Done. {n_frames} frames processed.")
        print(f"[tracker_demo] Video saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 2 — YOLOv8 + ByteTrack demo")
    p.add_argument("--camera",  type=int,   default=0,                              help="Camera index (default: 0)")
    p.add_argument("--conf",    type=float, default=0.4,                            help="Confidence threshold (default: 0.4)")
    p.add_argument("--output",  type=str,   default="outputs/tracked_output.mp4",   help="Output video path")
    p.add_argument("--detect",  action="store_true",                                help="Disable ByteTrack (detection-only mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        camera_index=args.camera,
        conf_threshold=args.conf,
        output_path=args.output,
        tracking=not args.detect,
    )
