# Real-Time Hazard and Collision Detection MVP

Real-time hazard and collision detection system prototype designed to assist visually impaired users. This system combines Monocular Depth Estimation (MiDaS) and Optical Flow to calculate danger scores from a live webcam feed.

## Setup Instructions

1. **Clone the repository and enter the project folder:**
   ```bash
   cd bitirme-projesi
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Downloads:**
   - The MiDaS model will automatically be downloaded by PyTorch Hub during your first run. Ensure you have an active internet connection.
   - For YOLO and ByteTrack (coming soon), weights will be placed in the `models/` directory.

## How to Run the Project

1. Run the main processing pipeline:
   ```bash
   python main.py
   ```
   Press `q` on your keyboard to stop the webcam feed and exit the program.
   
2. If you want to visualize your session data, run the plotting script:
   ```bash
   python scripts/plot_logs.py
   ```

## Directory Structure
- `src/`: Source code
  - `core/`: Hazard detection logic (includes Temporal Depth differencing and Near-Region motion masks)
  - `modules/`: Computer vision wrappers (Optical Flow, Depth)
  - `utils/`: Overlays, config, logging
- `logs/`: CSV logs output by the project
- `models/`: Storage directory for downloaded weights
- `outputs/`: Output media directory

## Troubleshooting
- If no camera feed appears, check `CAMERA_INDEX` in `src/utils/config.py`.
- If the app is slow, ensure you have a CUDA-compatible GPU properly configured with PyTorch. It drops back to CPU seamlessly if no GPU is found, but performance will be lower.
