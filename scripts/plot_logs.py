"""
sistemin calisirken hesapladigi tehlike ve yaklasma verilerini grafiklere doker.

"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    main()
