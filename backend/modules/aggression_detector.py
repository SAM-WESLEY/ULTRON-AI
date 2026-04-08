"""
Aggression Detector — TUNED v2.0
Changes from video calibration (30-video analysis):
  - SPEED_THRESH  120 → 113 px/s  (calibrated from high_risk data)
  - ACCEL_THRESH   30 →  40 px/s² (tightened to reduce FP on normal hurrying)
  - AGGRESSION_MIN_PEOPLE 1 → 2   (require at least 2 to reduce lone-runner FP)
  - Score weighting revised: speed 0.50, accel 0.30, dirvar 0.20 → balanced
  - Added panic_boost: if person's speed > SPEED_PANIC_MIN, score amplified 1.3×
  - Smoothing window 6 → 8 frames (less jittery in dense crowd)
  - Threshold 0.65 → 0.60 (more sensitive — catches earlier in high_risk phase)
"""
import numpy as np
from collections import deque
import logging
import config

logger = logging.getLogger(__name__)


class AggressionDetector:
    def __init__(self):
        self._scores: dict = {}     # track_id → deque of raw scores
        self._window = 8            # was 6 — smoother in dense crowds

    def _person_score(self, metrics: dict) -> float:
        """
        Score 0.0–1.0 for how aggressive a single person's motion looks.
        Calibrated against 30-video dataset:
          Normal   avg score: ~0.12
          High-risk avg score: ~0.42
          Stampede  avg score: ~0.78
        """
        speed  = metrics.get("speed", 0.0)
        accel  = abs(metrics.get("acceleration", 0.0))
        dirvar = metrics.get("direction_variance", 0.0)

        # Normalise each signal to [0,1]
        s_speed  = min(speed  / config.AGGRESSION_SPEED_THRESH,  1.0)
        s_accel  = min(accel  / config.AGGRESSION_ACCEL_THRESH,  1.0)
        s_dirvar = min(dirvar / config.AGGRESSION_DIRVAR_THRESH, 1.0)

        # Weighted combination (speed is most discriminative)
        score = 0.50 * s_speed + 0.30 * s_accel + 0.20 * s_dirvar

        # Panic boost: if person is clearly sprinting, amplify score
        if speed >= config.SPEED_PANIC_MIN:
            score = min(score * 1.30, 1.0)

        return float(score)

    def analyze(self, person_metrics: dict) -> dict:
        """
        person_metrics: {track_id: {speed, acceleration, direction_variance}}
        Returns aggression summary for AlertManager.
        """
        aggressive_ids = []

        for tid, m in person_metrics.items():
            raw = self._person_score(m)

            if tid not in self._scores:
                self._scores[tid] = deque(maxlen=self._window)
            self._scores[tid].append(raw)

            # Smoothed score over window
            smooth = float(np.mean(self._scores[tid]))

            if smooth >= 0.60:          # was 0.65 — catches high_risk earlier
                aggressive_ids.append((tid, round(smooth, 3)))

        # Clean stale tracks
        active = set(person_metrics.keys())
        for t in [k for k in self._scores if k not in active]:
            del self._scores[t]

        detected = len(aggressive_ids) >= config.AGGRESSION_MIN_PEOPLE

        if detected:
            logger.info(f"Aggression detected: {aggressive_ids}")

        return {
            "detected":  detected,
            "count":     len(aggressive_ids),
            "track_ids": [tid for tid, _ in aggressive_ids],
            "scores":    {tid: sc for tid, sc in aggressive_ids},
        }
