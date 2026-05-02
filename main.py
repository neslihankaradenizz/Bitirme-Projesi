"""
main.py — Hazard Detection System entry point.

Pipeline:
  depth estimation → optical flow → danger analysis → HUD overlay
  + YOLOv8 detection (ENABLE_YOLO = True)
  + ByteTrack persistent tracking (ENABLE_BYTETRACK = True)

Press 'q' to quit.
"""

import collections
import os
import sys
import time

import cv2

from src.utils import config
from src.utils.logger import CSVLogger
from src.utils.overlay import draw_hud
from src.modules.depth_estimator import DepthEstimator
from src.modules.optical_flow import OpticalFlowEstimator
from src.core.danger_analyzer import DangerAnalyzer


# ---------------------------------------------------------------------------
# Optional: YOLO detection drawing helper
# ---------------------------------------------------------------------------

_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 77, 255, 141), ( 51, 153, 255),
    (255,  51, 153), (153, 102, 255), ( 51, 255, 255), (255, 203,  51),
]


def _bgr(r, g, b):
    return (b, g, r)


def _draw_yolo(frame, detections: list[dict]) -> None:
    """Annotate *frame* with bounding boxes, class labels, confidence and tracking IDs."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid   = det.get("track_id")
        # Colour by track_id when available, otherwise by class
        idx   = tid if tid is not None else det["class_id"]
        color = _bgr(*_PALETTE[idx % len(_PALETTE)][:3])

        # Build label: prepend '#ID' when tracking
        id_tag = f"#{tid} " if tid is not None else ""
        label  = f"{id_tag}{det['class_name']} {det['confidence']:.0%}"

        thickness = 3 if tid is not None else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Ensure output directories exist
    os.makedirs(config.LOG_DIR,    exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR,  exist_ok=True)

    print("Initializing components …")
    logger          = CSVLogger(config.LOG_DIR)
    depth_estimator = DepthEstimator()
    flow_estimator  = OpticalFlowEstimator()
    analyzer        = DangerAnalyzer()

    # 2. Conditionally load YOLO detector
    detector  = None
    use_track = False
    if config.ENABLE_YOLO:
        try:
            from src.modules.object_tracker import ObjectTracker
            detector  = ObjectTracker()
            use_track = config.ENABLE_BYTETRACK
            mode_str  = "YOLO + ByteTrack" if use_track else "YOLO (detect-only)"
            print(f"[main] {mode_str} ready.")
        except ImportError as exc:
            print(f"[main] WARNING: {exc}\n       Continuing without YOLO.")

    # Occlusion test: rolling window of track-ID sets over the last 10 frames
    _occ_window: collections.deque[set[int]] = collections.deque(maxlen=10)

    # 3. Open camera
    print(f"Opening camera {config.CAMERA_INDEX} …")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {config.CAMERA_INDEX}.")
        sys.exit(1)

    # 4. VideoWriter — always on when YOLO is active
    writer = None
    if config.ENABLE_YOLO:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        # Save to tracked path when ByteTrack is on, else detection path
        out_path = (
            config.OUTPUT_TRACKED_VIDEO_PATH if use_track
            else config.OUTPUT_VIDEO_PATH
        )
        writer = cv2.VideoWriter(out_path, fourcc, fps_cam, (fw, fh))
        print(f"[main] Saving annotated output → {out_path}")

    # 5. Main loop
    frame_num    = 0
    fps_smoothed = 0.0
    alpha        = 0.1
    print("Application started. Press 'q' to quit.\n")

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — exiting.")
                break

            frame_num += 1

            # --- Core pipeline ---
            depth_map   = depth_estimator.estimate(frame)
            motion_map  = flow_estimator.estimate(frame)
            motion_score, depth_score, delta_d, approach_score, danger_score = \
                analyzer.analyze(motion_map, depth_map)

            # --- Logging ---
            logger.log(frame_num, motion_score, depth_score, delta_d, approach_score, danger_score)

            # --- Build display frame ---
            display_frame = frame.copy()

            # YOLO detections drawn first (below the HUD text)
            if detector is not None:
                # Call track() for ByteTrack IDs, detect() otherwise
                detections = (
                    detector.track(display_frame) if use_track
                    else detector.detect(display_frame)
                )
                # --- Danger integration hook ---
                # Enrich each detection with the current frame's danger scores
                # so future sprints can correlate per-object risk levels.
                for det in detections:
                    det["depth_score"]  = depth_score
                    det["motion_score"] = motion_score

                _draw_yolo(display_frame, detections)

            # --- ID-switch telemetry (every 30 frames, ByteTrack only) ---
            if frame_num % 30 == 0 and detector is not None:
                logger.log_id_switches(frame_num, detector.get_id_switch_count())

            # --- Occlusion diagnostic overlay ---
            if config.OCCLUSION_TEST_MODE and use_track and detector is not None:
                visible_ids: set[int] = {
                    det["track_id"] for det in detections
                    if det.get("track_id") is not None
                }
                _occ_window.append(visible_ids)
                expected_ids: set[int] = set().union(*_occ_window)
                hidden_count = len(expected_ids - visible_ids)
                logger.log_occlusion(frame_num, hidden_count)

                _yel = (0, 255, 255)  # yellow in BGR
                cv2.putText(display_frame, "OCCLUSION TEST ACTIVE",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _yel, 2, cv2.LINE_AA)
                if hidden_count > 0:
                    cv2.putText(display_frame, f"Hidden objects: {hidden_count}",
                                (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _yel, 2, cv2.LINE_AA)

            # HUD overlay on top
            display_frame = draw_hud(display_frame, danger_score, config.DANGER_THRESHOLD, frame_num)

            # FPS counter
            elapsed       = time.perf_counter() - t0
            instant_fps   = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_smoothed  = alpha * instant_fps + (1 - alpha) * fps_smoothed
            fps_color     = (0, 220, 0) if fps_smoothed >= 15 else (0, 140, 255)
            cv2.putText(display_frame, f"FPS {fps_smoothed:5.1f}",
                        (display_frame.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

            # --- Show & save ---
            cv2.imshow("Hazard Detection", display_frame)
            if writer is not None:
                writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nShutdown complete. {frame_num} frames processed.")
        if writer is not None:
            out_path = (
                config.OUTPUT_TRACKED_VIDEO_PATH if use_track
                else config.OUTPUT_VIDEO_PATH
            )
            print(f"Output saved → {out_path}")


if __name__ == "__main__":
    main()
