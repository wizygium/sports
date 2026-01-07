import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple, Dict
from ..common.team import TeamClassifier

class TeamClassifierV15(TeamClassifier):
    """
    Handball-specific Team Classifier (V15) focusing on the upper body (jersey).
    Inherits from the base TeamClassifier but implements a color-histogram fallback.
    """
    def __init__(self, device: str = 'cpu', n_clusters: int = 2):
        # We don't necessarily need the full SigLIP model for this V15 color-only class,
        # but we inherit to keep the interface consistent.
        self.device = device
        self.n_clusters = n_clusters
        self.kmeans = None
        self.label_map = {} # cluster_id -> team_id
        self.cluster_centers_ = None

    def compute_histogram(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute color histogram, focusing on the upper 50% of the crop."""
        h, w = image.shape[:2]
        if mask is not None:
             target_img = image
             target_mask = mask.copy()
             # Refine mask: Keep only UPPER 50% (Jersey focus)
             target_mask[h//2:, :] = 0
        else:
             # Fallback to UPPER-BIASED crop
             c_h_start = int(h*0.1)
             c_h_end = int(h*0.6) 
             c_w = int(w*0.2)
             target_img = image[c_h_start:c_h_end, c_w:w-c_w]
             if target_img.size == 0: target_img = image
             target_mask = None 

        hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
        # H (16 bins), S (8 bins), V (8 bins)
        hist_h = cv2.calcHist([hsv], [0], target_mask, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], target_mask, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], target_mask, [8], [0, 256])

        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return np.concatenate([hist_h, hist_s, hist_v]).flatten()

    def fit(self, crops: List[np.ndarray], masks: Optional[List[np.ndarray]] = None):
        if not crops: return
        
        features = []
        for i in range(len(crops)):
            m = masks[i] if masks else None
            features.append(self.compute_histogram(crops[i], m))
        
        features = np.array(features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features)
        
        # Heuristic: Assign Team 0 (White) to the brighter cluster
        labels = self.kmeans.labels_
        b_scores = []
        for k in range(self.n_clusters):
            idx = np.where(labels == k)[0]
            if len(idx) > 0:
                # Average V (Brightness) across samples
                brightness = np.mean([np.mean(cv2.cvtColor(crops[i], cv2.COLOR_BGR2HSV)[:,:,2]) for i in idx])
                b_scores.append(brightness)
            else:
                b_scores.append(0)
        
        if b_scores[0] > b_scores[1]:
            self.label_map = {0: 0, 1: 1}
        else:
            self.label_map = {0: 1, 1: 0}
            
    def predict(self, crops: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        if not self.kmeans: return np.zeros(len(crops), dtype=int)
        features = []
        for i in range(len(crops)):
            m = masks[i] if masks else None
            features.append(self.compute_histogram(crops[i], m))
        
        raw_preds = self.kmeans.predict(np.array(features))
        return np.array([self.label_map.get(p, p) for p in raw_preds])

    def save_debug_grid(self, crops: List[np.ndarray], masks: Optional[List[np.ndarray]], output_path: str):
        """Save a visual grid of classified crops for verification."""
        if not crops: return
        labels = self.predict(crops, masks)
        
        for k in range(self.n_clusters):
            k_idx = np.where(labels == k)[0]
            if len(k_idx) == 0: continue
            
            sampled = random.sample(list(k_idx), min(len(k_idx), 64))
            vis_list = []
            for i in sampled:
                c = crops[i].copy()
                if masks and masks[i] is not None:
                    m = masks[i]
                    if m.shape[:2] != c.shape[:2]: m = cv2.resize(m, (c.shape[1], c.shape[0]))
                    if m.dtype == bool: m = m.astype(np.uint8) * 255
                    # Apply upper mask focus for visualization too
                    m[m.shape[0]//2:, :] = 0
                    c = cv2.bitwise_and(c, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
                vis_list.append(cv2.resize(c, (64, 64)))
            
            # Form grid
            rows = [np.hstack(vis_list[i:i+8]) for i in range(0, len(vis_list), 8) if i+8 <= len(vis_list)]
            if rows:
                grid = np.vstack(rows)
                cv2.imwrite(output_path.replace(".jpg", f"_team_{k}.jpg"), grid)
