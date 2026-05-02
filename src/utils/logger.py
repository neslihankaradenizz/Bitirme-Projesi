import csv
import os
import time

class CSVLogger:
    def __init__(self, log_dir):
        """
        Initializes the CSV Logger.
        Creates a new CSV file with a timestamp in the specified directory.
        """
        self.log_dir = log_dir
        
        # Ensure log directory exists (redundant with main.py but safe)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"hazard_log_{timestamp}.csv")
        
        # Write headers
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_num", "motion_score", "depth_score", "delta_d", "approach_score", "danger_score", "timestamp"])

    def log(self, frame_num, motion_score, depth_score, delta_d, approach_score, danger_score):
        """
        Appends a row of data for the current frame.
        """
        with open(self.log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_num, f"{motion_score:.4f}", f"{depth_score:.4f}", f"{delta_d:.4f}", f"{approach_score:.4f}", f"{danger_score:.4f}", time.time()])

    def log_id_switches(self, frame_num: int, id_switch_count: int) -> None:
        """Appends to id_switches.csv → columns: frame, cumulative_switches"""
        msg = f"[IDSwitch] frame={frame_num}  cumulative_switches={id_switch_count}"
        print(msg)
        path = os.path.join(self.log_dir, "id_switches.csv")
        self._append_or_create(path, ["frame", "cumulative_switches"], [frame_num, id_switch_count])

    def log_occlusion(self, frame_num: int, hidden_count: int) -> None:
        """Appends to occlusion_log.csv → columns: frame, hidden_object_count"""
        path = os.path.join(self.log_dir, "occlusion_log.csv")
        self._append_or_create(path, ["frame", "hidden_object_count"], [frame_num, hidden_count])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_or_create(self, path: str, headers: list, row: list) -> None:
        """Write row to path, prepending a header row if the file does not yet exist."""
        file_exists = os.path.isfile(path)
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

