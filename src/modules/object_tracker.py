"""
object_tracker.py — YOLOv8 detection + ByteTrack persistent tracking.

Part 1: detect(frame)  → detections without track IDs  (track_id = None)
Part 2: track(frame)   → detections WITH persistent IDs from ByteTrack

Bu kod, ekrandaki engelleri  YOLOv8 ile kutu içine alır (tespit eder) ve 
ByteTrack algoritması ile bu nesneleri kareler boyunca takip eder.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Dependency guard-bagimlilik kontrolu
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore[import]
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np


def _require_ultralytics() -> None:
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "\n[ObjectTracker] 'ultralytics' is not installed.\n"
            "Fix:  pip install ultralytics\n"
            "      — or — pip install -r requirements.txt\n"
        )


# ---------------------------------------------------------------------------
# ObjectTracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    YOLOv8n-based detector / tracker.

    * ``detect(frame)``  — Part 1: detection only, no IDs.
    * ``track(frame)``   — Part 2: YOLOv8 + ByteTrack, persistent IDs.

    Both methods return a list of the same dict structure::

        {
            "track_id":    int | None,   # None for detect(), int for track()
            "class_id":    int,
            "class_name":  str,
            "bbox":        [x1, y1, x2, y2],   # pixel coords
            "confidence":  float,
            # Danger-integration placeholders (populated by caller):
            "depth_score": None,
            "motion_score": None,
        }
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float | None = None,
    ) -> None:
        """
        Args:
            model_path:      Ultralytics model or local .pt file.
            conf_threshold:  Overrides config.YOLO_CONF_THRESHOLD when set.

        Raises:
            ImportError: If ``ultralytics`` is not installed.
        """
        _require_ultralytics()

        try:
            from src.utils import config as _cfg
            _default_conf = _cfg.YOLO_CONF_THRESHOLD
        except ImportError:
            _default_conf = 0.4

        self.conf_threshold: float = (
            conf_threshold if conf_threshold is not None else _default_conf
        )
        self.model = YOLO(model_path)
        print(
            f"[ObjectTracker] Model '{model_path}' loaded  "
            f"conf={self.conf_threshold:.2f}"
        )

    # ------------------------------------------------------------------
    # Part 1: detection only
    # ------------------------------------------------------------------

    def detect(self, frame: "np.ndarray") -> list[dict]:
        """
        YOLOv8 inference — no tracking, track_id is always None.

        Args:
            frame: BGR image (e.g. from cv2.VideoCapture.read()).

        Returns:
            List of detection dicts (see class docstring).
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return self._parse(results, with_ids=False)

    # ------------------------------------------------------------------
    # Part 2: detection + ByteTrack
    # ------------------------------------------------------------------

    def track(self, frame: "np.ndarray") -> list[dict]:
        """
        YOLOv8 + ByteTrack — each object gets a persistent ``track_id``
        that is stable across frames (until the track is lost).

        Uses YOLOv8's built-in ByteTrack integration (``model.track()``).
        No extra package required beyond ``ultralytics``.

        Args:
            frame: BGR image.

        Returns:
            List of detection dicts with track_id filled.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",   # ships with ultralytics
            persist=True,               # keeps the tracker state across calls
            verbose=False,
        )
        return self._parse(results, with_ids=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, results, *, with_ids: bool) -> list[dict]:
        """Convert ultralytics Results objects into plain dicts."""
        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))

                track_id: int | None = None
                if with_ids and box.id is not None:
                    track_id = int(box.id[0])

                detections.append({
                    # --- Core detection fields ---
                    "track_id":    track_id,
                    "class_id":    cls_id,
                    "class_name":  cls_name,
                    "bbox":        [x1, y1, x2, y2],
                    "confidence":  float(f"{conf:.4f}"),
                    # --- Danger-integration placeholders ---
                    # danger_analyzer.py will fill these in a future sprint
                    "depth_score":  None,
                    "motion_score": None,
                })
        return detections
