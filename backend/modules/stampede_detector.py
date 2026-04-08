"""
Stampede Detector — NEW MODULE v1.0
Physics-based multi-signal stampede detection, NOT using IsolationForest.
Works from frame 1 with zero warmup needed.

Detection model (all 4 signals must exceed threshold simultaneously):
  1. avg_speed      > STAMPEDE_SPEED_THRESH       (88 px/s  — early running)
  2. incoherence    > STAMPEDE_INCOHERENCE_THRESH  (0.55 — chaotic flow)
  3. fast_ratio     > STAMPEDE_FAST_RATIO          (35% of crowd running)
  4. max_density    > STAMPEDE_DENSITY_THRESH      (dense cluster present)

Additional: Farneback surge detection (MOG2 surge_ratio) from main pipeline
is factored in via the 'surge' flag from the caller.

Risk escalation levels:
  WATCH    — 2 of 4 signals active (early warning)
  WARNING  — 3 of 4 signals active
  CRITICAL — all 4 signals active for STAMPEDE_CONFIRM_FRAMES frames
"""
import numpy as np
from collections import deque
import logging
import config

logger = logging.getLogger(__name__)


class StampedeDetector:
    def __init__(self):
        self._confirm_buf   = deque(maxlen=config.STAMPEDE_CONFIRM_FRAMES)
        self._consec_crit   = 0
        self._last_risk     = "NONE"
        # Rolling window for smoothed signal values
        self._speed_buf     = deque(maxlen=8)
        self._incoh_buf     = deque(maxlen=8)
        self._fast_buf      = deque(maxlen=8)
        self._density_buf   = deque(maxlen=8)

    def _smooth(self, buf, val):
        buf.append(val)
        return float(np.mean(buf))

    def analyze(self, feats: dict, person_metrics: dict,
                surge_ratio: float = 0.0) -> dict:
        """
        feats: output of MotionAnalyzer.analyze()
        person_metrics: {tid: {speed, direction_variance, ...}}
        surge_ratio: optional MOG2 foreground ratio from main pipeline

        Returns:
          detected    bool   — confirmed stampede
          risk_level  str    — NONE / WATCH / WARNING / CRITICAL
          signals     dict   — individual signal values and pass/fail
          score       float  — 0–4 (count of triggered signals)
        """
        # ── Raw signal values ─────────────────────────────────────────
        avg_spd     = feats.get("avg_speed",    0.0)
        incoherence = feats.get("incoherence",  1.0 - feats.get("flow_coherence", 1.0))
        max_dens    = feats.get("max_density",  0.0)
        panic_ratio = feats.get("panic_ratio",  0.0)
        running_ratio = feats.get("running_ratio", 0.0)

        # fast_ratio: fraction of crowd at or above run speed (either source)
        fast_ratio = max(running_ratio, panic_ratio)

        # Smooth each signal over 8 frames to reduce single-frame spikes
        s_speed   = self._smooth(self._speed_buf,   avg_spd)
        s_incoh   = self._smooth(self._incoh_buf,   incoherence)
        s_fast    = self._smooth(self._fast_buf,    fast_ratio)
        s_density = self._smooth(self._density_buf, max_dens)

        # ── Signal tests ──────────────────────────────────────────────
        sig_speed   = s_speed   >= config.STAMPEDE_SPEED_THRESH        # 88 px/s
        sig_incoh   = s_incoh   >= config.STAMPEDE_INCOHERENCE_THRESH  # 0.55
        sig_fast    = s_fast    >= config.STAMPEDE_FAST_RATIO          # 0.35
        sig_density = s_density >= config.STAMPEDE_DENSITY_THRESH      # 0.034

        # Bonus signal: MOG2 surge
        sig_surge   = surge_ratio >= 0.032                # from 30-video analysis

        score = sum([sig_speed, sig_incoh, sig_fast, sig_density])

        # ── Risk escalation ───────────────────────────────────────────
        if score >= 4:
            risk = "CRITICAL"
        elif score == 3 or (score >= 2 and sig_surge):
            risk = "WARNING"
        elif score == 2:
            risk = "WATCH"
        else:
            risk = "NONE"

        # ── Confirmation window for CRITICAL ─────────────────────────
        self._confirm_buf.append(risk == "CRITICAL")
        confirmed_critical = (
            len(self._confirm_buf) >= config.STAMPEDE_CONFIRM_FRAMES
            and all(self._confirm_buf)
        )

        # Gradual counter (also fires if CRITICAL for ≥ half the confirm window)
        half_confirm = config.STAMPEDE_CONFIRM_FRAMES // 2
        partial_confirm = (
            len(self._confirm_buf) >= half_confirm
            and sum(list(self._confirm_buf)[-half_confirm:]) >= half_confirm
        )
        detected = confirmed_critical or partial_confirm

        if detected and self._last_risk != "CRITICAL":
            logger.warning(
                f"STAMPEDE DETECTED — speed={s_speed:.1f}px/s "
                f"incoherence={s_incoh:.3f} fast_ratio={s_fast:.2f} "
                f"density={s_density:.3f}"
            )
        self._last_risk = risk

        return {
            "detected":    detected,
            "risk_level":  "CRITICAL" if detected else risk,
            "score":       score,
            "signals": {
                "avg_speed":       round(s_speed, 1),
                "speed_pass":      bool(sig_speed),
                "incoherence":     round(s_incoh, 3),
                "incoherence_pass":bool(sig_incoh),
                "fast_ratio":      round(s_fast, 3),
                "fast_pass":       bool(sig_fast),
                "max_density":     round(s_density, 3),
                "density_pass":    bool(sig_density),
                "surge_ratio":     round(surge_ratio, 3),
                "surge_pass":      bool(sig_surge),
            },
            "new_alert": detected,
        }
