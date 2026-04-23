import numpy as np
from src.utils import config

class DangerAnalyzer:
    def __init__(self):
        
        """
        Danger Analyzer 
        Hareket olcumlerini, derinlik olcummlerini ve yaklasma hizini
        birlestirerek tek bir tehlike puani hesaplar.
        """
        self.danger_threshold = config.DANGER_THRESHOLD
        self.motion_weight = getattr(config, 'MOTION_WEIGHT', 0.3)
        self.depth_weight = getattr(config, 'DEPTH_WEIGHT', 0.3)
        self.approach_weight = getattr(config, 'APPROACH_WEIGHT', 0.4)
        self.near_region_threshold = getattr(config, 'NEAR_REGION_THRESHOLD', 0.6)
        # Fix (Problem 2): load explicit tau; fall back to 0.02 if config is old 
        # Sensörden veya derinlik modelinden  gelen ufak titremeleri (gürültüyü) "yaklaşma" olarak algılamamak için konulmuş bir tolerans değeri.
        self.depth_tau = getattr(config, 'DEPTH_TAU', 0.02)

        # # Zamansal farklilik icin durumu takip edt
        self.prev_depth_score = None
        self.danger_history = []
        self.smoothing_window = 5

    def analyze(self, motion_map, depth_map):
        """
        Yakin bolge maskelemesi ve zamansal derinlik farklarini kullanarak tehlike puanini belirlemek icin sahneyi analiz eder.
        Argümanlar:
        m motion_map (np.ndarray): Hareketin yogun buyukluk map
        depth_map (np.ndarray): Goreceli derinlik msp
        tuple: (motion_score, depth_score, delta_d, approach_score, final_danger_score)
        """
        # # 1. Derinlik haritasini mevcut karenin minimum/maksimum degerlerine göre 0,0 – 1,0 araligina normalleştirin.
        d_min = np.min(depth_map)
        d_max = np.max(depth_map)
        if d_max > d_min:
            norm_depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            norm_depth_map = np.zeros_like(depth_map)

        # 2. Extract Depth Proximity Score- Derinlik Yakınlık Puanını Çıkarma
        # Normalleştirilmiş derinlik haritasının 90. yüzdelik dilimi, en yakın önemli cismi temsil eder.
        depth_score = float(np.percentile(norm_depth_map, 90))

        # 3. Temporal Depth Difference (ΔD = Dt − Dt-1)
        delta_d = 0.0
        if self.prev_depth_score is not None:
            delta_d = depth_score - self.prev_depth_score

        self.prev_depth_score = depth_score

        # Fix (Problem 2): Apply tau — treat sub-threshold changes as noise
        # approach_score is non-zero ONLY when delta_d exceeds DEPTH_TAU
        if delta_d > self.depth_tau:
            approach_score = min((delta_d - self.depth_tau) * 5.0, 1.0)
        else:
            approach_score = 0.0

        # 4. Near-region mask for Motion-Hareket Maskeleme
        # Select motion only inside areas where depth > near_region_threshold
        near_mask = norm_depth_map > self.near_region_threshold

        if np.any(near_mask):
            near_motion_map = motion_map[near_mask]
            # 95th percentile of motion inside the near region to ignore noise
            avg_motion = float(np.percentile(near_motion_map, 95))
        else:
            avg_motion = 0.0

        max_expected_motion = 15.0  # Tunable heuristic
        motion_score = min(avg_motion / max_expected_motion, 1.0)

        # 5. Fix (Problem 1): Gate on BOTH near AND approaching conditions.
        # A stationary nearby object must NEVER raise the danger score.
        # The weighted formula is only applied when:
        #   - depth_score exceeds the near-region threshold (object is close), AND
        #   - delta_d > tau AND motion_score > 0.1 (object is actively approaching)

        """Koşul: Cismin ağırlıklı tehlike hesabına girmesi için hem YAKIN olması 
        hem de aktif olarak YAKLAŞIYOR olması gerekiyor. 
        Aksi takdirde sadece genel bir hareket skoru hesaplanıyor ve tehlike düşük tutuluyor."""

        object_is_near = depth_score > self.near_region_threshold
        object_is_approaching = (delta_d > self.depth_tau) and (motion_score > 0.1)

        if object_is_near and object_is_approaching:
            raw_danger_score = (
                motion_score   * self.motion_weight +
                depth_score    * self.depth_weight  +
                approach_score * self.approach_weight
            )
        else:
            # Nesne ya çok uzakta ya da hareketsizdir; yalnızca hareket katkısını taşır.
            raw_danger_score = motion_score * self.motion_weight

        # 6. Smooth results over a short history window
        self.danger_history.append(raw_danger_score)
        if len(self.danger_history) > self.smoothing_window:
            self.danger_history.pop(0)

        final_danger_score = sum(self.danger_history) / len(self.danger_history)

        return motion_score, depth_score, delta_d, approach_score, final_danger_score
