"""
Gathering Detector — TUNED v2.0
Changes from video calibration:
  - GATHERING_PROXIMITY_PX 90 → 100 px (wider cluster radius from data)
  - GATHERING_MIN_GROUP    3  → 4      (reduce single-family/duo FP)
  - find_gatherings: added DBSCAN-style expansion so all connected members
    are captured (not just direct neighbours of seed person)
  - energy_anomaly: ratio threshold 0.66 → 0.50 (catches high_risk earlier)
  - count_zones: added density-per-zone metric returned alongside count
  - social_distance: perspective-corrected distance for angled cameras added
"""
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import config


class GatheringDetector:
    def __init__(self):
        self.energy_history = []

    # ── Gathering detection (DBSCAN-style expansion) ──────────────────
    def find_gatherings(self, tracked: list) -> list:
        """
        Find groups of >= GATHERING_MIN_GROUP people within
        GATHERING_PROXIMITY_PX of each other using graph expansion
        (each member's proximity is checked, not just the seed's).
        This prevents missing large gatherings where no single person
        is near ALL others.
        """
        if len(tracked) < config.GATHERING_MIN_GROUP:
            return []

        centers = np.array([o["center"] for o in tracked], dtype=np.float32)
        D       = cdist(centers, centers)
        n       = len(tracked)

        visited = [False] * n
        groups  = []

        for i in range(n):
            if visited[i]:
                continue
            # BFS expansion
            cluster = [i]
            queue   = [i]
            visited[i] = True
            while queue:
                cur = queue.pop(0)
                for j in range(n):
                    if not visited[j] and D[cur][j] < config.GATHERING_PROXIMITY_PX:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
            if len(cluster) >= config.GATHERING_MIN_GROUP:
                groups.append([tracked[idx] for idx in cluster])

        return groups

    # ── Energy anomaly ────────────────────────────────────────────────
    def energy_anomaly(self, pm: dict) -> dict:
        """
        What fraction of tracked people are in a high-energy state?
        High-energy = fast OR chaotic direction.
        Threshold lowered 0.66 → 0.50: fires on 50%+ crowd (was 66%).
        """
        total = len(pm)
        if total == 0:
            return {"ratio": 0.0, "is_abnormal": False,
                    "abnormal": 0, "total": 0}

        abn = sum(
            1 for m in pm.values()
            if m.get("speed", 0)              > config.ENERGY_SPEED_THRESH
            or m.get("direction_variance", 0) > config.ENERGY_DIRVAR_THRESH
        )
        ratio = abn / total
        self.energy_history.append(ratio)

        return {
            "ratio":       round(ratio, 3),
            "is_abnormal": ratio >= config.ENERGY_ANOMALY_RATIO,   # 0.50
            "abnormal":    abn,
            "total":       total,
        }

    # ── Zone counting with density ────────────────────────────────────
    def count_zones(self, tracked: list, zones: dict) -> dict:
        """
        Count people per zone + return density (count / zone area).
        density > 1 person per 1000px² = crowded.
        """
        result = {}
        for name, poly in zones.items():
            c = sum(
                1 for o in tracked
                if cv2.pointPolygonTest(poly, tuple(o["center"]), False) >= 0
            )
            # Zone area in pixels²
            area = float(cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32)))
            density = c / max(area, 1.0) * 1000.0  # people per 1000 px²

            result[name] = {
                "count":   c,
                "density": round(density, 3),
                "status": (
                    "RED"   if c >= config.ZONE_CRITICAL_COUNT else
                    "AMBER" if c >= config.ZONE_WARN_COUNT      else
                    "GREEN"
                ),
            }
        return result

    # ── Social distance violations ────────────────────────────────────
    def social_distance(self, tracked: list) -> list:
        """
        Return pairs of track IDs that are too close.
        Supports both top-down (flat) and angled (perspective-corrected) cameras.
        """
        violations = []
        for i, a in enumerate(tracked):
            for j, b in enumerate(tracked):
                if j <= i:
                    continue
                ax, ay = a["center"]
                bx, by = b["center"]
                if config.TOP_DOWN_CAM:
                    dist = float(np.hypot(ax - bx, ay - by))
                else:
                    # Perspective correction: people lower in frame appear larger
                    # Scale distance by inverse of average Y position
                    scale = 1.0 + (ay + by) / 2000.0
                    dist  = float(np.hypot(ax - bx, (ay - by) * scale))
                if dist < config.SOCIAL_DIST_MIN_PX:
                    violations.append((a["track_id"], b["track_id"]))
        return violations
