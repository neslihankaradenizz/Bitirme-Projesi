import cv2

# Danger level thresholds (mirrors config.DANGER_THRESHOLD = 0.45)
_DIKKAT_THRESHOLD  = 0.30   # score >= 0.30 → DIKKAT  (warning)
_TEHLIKE_THRESHOLD = 0.45   # score >= 0.45 → TEHLIKE (danger)

def draw_hud(frame, danger_score, danger_threshold, frame_count=0):
    """
    Draws the heads-up display overlay on the frame with 3 granular alert levels.

    Levels:
        score < 0.30            → "GUVENLI"  (green)
        0.30 <= score < 0.45    → "DIKKAT"   (orange)
        score >= 0.45           → "TEHLIKE"  (red, flashing border)

    Args:
        frame         (np.ndarray): BGR frame to annotate (modified in-place).
        danger_score  (float):      Smoothed danger score in [0, 1].
        danger_threshold (float):   Configured danger threshold (used in metrics text).
        frame_count   (int):        Monotonically increasing frame counter used to
                                    drive the border flash effect on TEHLIKE.

    Returns:
        np.ndarray: Annotated frame.
    """
    # --- Determine alert level ---
    if danger_score >= _TEHLIKE_THRESHOLD:
        text  = "TEHLIKE"
        color = (0, 0, 255)          # Red   (BGR)
        level = "TEHLIKE"
    elif danger_score >= _DIKKAT_THRESHOLD:
        text  = "DIKKAT"
        color = (0, 165, 255)        # Orange (BGR)
        level = "DIKKAT"
    else:
        text  = "GUVENLI"
        color = (0, 255, 0)          # Green  (BGR)
        level = "GUVENLI"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness  = 3

    # Shadow for readability
    cv2.putText(frame, text, (52, 52), font, font_scale, (0, 0, 0), thickness + 2)
    # Primary text
    cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness)

    # Metrics line
    metrics_text = f"Danger Score: {danger_score:.2f} / {danger_threshold:.2f}"
    cv2.putText(frame, metrics_text, (50, 90), font, 0.7, (255, 255, 255), 2)

    # --- Border effects ---
    h, w = frame.shape[:2]

    if level == "TEHLIKE":
        # Flashing border: visible on even frames, hidden on odd frames
        if (frame_count // 15) % 2 == 0:          # toggles ~every 15 frames
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    elif level == "DIKKAT":
        # Solid orange border — always visible at warning level
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 6)

    return frame
