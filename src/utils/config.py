import os

# --- Hardware / Input Configuration ---
#Sistemin hangi kamerayı kullanacağını belirler. Genelde dizüstü bilgisayarların dahili kamerası 0'dır. Harici bir USB kamera takarsan bunu 1 veya 2 yapman gerekir.
CAMERA_INDEX = 0

# --- Directory Configuration ---
LOG_DIR    = "logs"
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"

# --- Danger Analyzer Configuration ---
DANGER_THRESHOLD  = 0.45   # min tehlike orani
MOTION_WEIGHT     = 0.3    # hareket agirligi
DEPTH_WEIGHT      = 0.3    # derinlik agirligi
APPROACH_WEIGHT   = 0.4    # yaklasim agirligi
NEAR_REGION_THRESHOLD = 0.6  # nesnenin yaklasik bolgesi

DEPTH_TAU = 0.02 # "Yaklaşma"  olarak sayılması için gereken minimum zamansal derinlik değişimi

# --- YOLO / Detection Configuration ---
# Minimum confidence for a detection to be kept.
# Lower  → more detections, more false positives.
# Higher → fewer detections, higher precision.
YOLO_CONF_THRESHOLD = 0.4

# Path for plain detection output (Part 1)
OUTPUT_VIDEO_PATH = "outputs/detection_output.mp4"

# Path for tracked output with ByteTrack IDs (Part 2)
OUTPUT_TRACKED_VIDEO_PATH = "outputs/tracked_output.mp4"

# --- Feature Flags ---
ENABLE_YOLO      = True   # YOLOv8 detection
ENABLE_BYTETRACK = True   # ByteTrack persistent tracking (Part 2)
OCCLUSION_TEST_MODE = False  # Set True to enable occlusion diagnostics overlay

# --- Performance Tuning ---
DEPTH_EVERY_N_FRAMES: int = 3   # run MiDaS depth estimation only every Nth frame to reduce CPU load
FLOW_EVERY_N_FRAMES:  int = 2   # run Optical Flow estimation only every Nth frame to reduce CPU load
FRAME_SCALE: float = 0.5        # downscale input frames before all processing (0.5 = 50% resolution)
