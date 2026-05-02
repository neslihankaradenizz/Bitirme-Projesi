"""
sistemin calisirken hesapladigi tehlike ve yaklasma verilerini grafiklere doker.

"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_object_count(log_path: str) -> None:
    """Plot per-frame object count (left axis) and cumulative ID switches (right axis) — Figure 7."""
    df = pd.read_csv(log_path)
    if df.empty:
        print(f"[plot_object_count] File is empty: {log_path}")
        return

    # Normalise column names to lower-case stripped strings
    df.columns = [c.strip().lower() for c in df.columns]

    if "frame" not in df.columns or "object_count" not in df.columns:
        print(
            f"[plot_object_count] Expected columns 'frame' and 'object_count' in {log_path}.\n"
            f"  Found: {list(df.columns)}"
        )
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Primary Y-axis — detected object count (blue solid)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Detected Object Count", color="steelblue")
    ax1.plot(df["frame"], df["object_count"], color="steelblue",
             linewidth=1.8, label="Object Count")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Secondary Y-axis — cumulative ID switch count (red dashed)
    if "id_switch_count" in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Cumulative ID Switch Count", color="crimson")
        ax2.plot(df["frame"], df["id_switch_count"], color="crimson",
                 linewidth=1.5, linestyle="--", label="ID Switches (cumulative)")
        ax2.tick_params(axis="y", labelcolor="crimson")
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")
        print("[plot_object_count] Column 'id_switch_count' not found — skipping secondary axis.")

    plt.title("Detected Object Count Over Time (Figure 7)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "figure7_object_count.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_object_count] Saved → {out_path}")
    plt.show()


def main():
    """
    logs/ dizinindeki en sondosyasini  bulur ve 
    tehlike puanini zaman icinde grafik olarak gösterir.
    """
    log_files = glob.glob(os.path.join("logs", "hazard_log_*.csv"))
    if not log_files:
        print("No log files found in 'logs' folder.")
        return
        
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Plotting latest log: {latest_log}")
    
    #veri okuma 
    df = pd.read_csv(latest_log)
    
    if len(df) == 0:
        print("Log file is empty.")
        return
 
    """ Burada x ekseninde her zaman 
    frame_num (kameradan/videodan alinan kare numarasi) var. 
    y ekseninde ise farklı metrikler çizdiriliyor:
    Kirmizi: Toplam Tehlike Skoru. 
    Mavi : Hareketlilik Skoru.
    Turuncu : Derinlik Degisimi objelerin kameraya ne kadar hizli yaklastigini ifade eden matematiksel ifade
    Mor : Yaklasma Skoru.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['frame_num'], df['danger_score'], label='Total Danger Score', color='red', linewidth=2)
    plt.plot(df['frame_num'], df['motion_score'], label='Motion Component (Near)', color='blue', alpha=0.5, linestyle='--')
    plt.plot(df['frame_num'], df['delta_d'], label='Temporal Depth Delta (ΔD)', color='orange', alpha=0.5, linestyle='--')
    plt.plot(df['frame_num'], df['approach_score'], label='Approach Component', color='purple', alpha=0.5, linestyle='--')
    
    # threshold line
    plt.axhline(y=0.45, color='black', linestyle=':', label='Alert Threshold')

    plt.title("Hazard Detection Metrics Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    
    # Save the plot in outputs
    output_path = os.path.join("outputs", "recent_session_plot.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    plt.show()

    # --- Figure 7: object count over time ---
    id_switch_csv = os.path.join("logs", "id_switches.csv")
    if os.path.isfile(id_switch_csv):
        plot_object_count(id_switch_csv)
    else:
        print(f"[main] id_switches.csv not found — skipping Figure 7 (run with ENABLE_BYTETRACK=True first).")

if __name__ == "__main__":
    main()
