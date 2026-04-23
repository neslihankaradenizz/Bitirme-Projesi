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
