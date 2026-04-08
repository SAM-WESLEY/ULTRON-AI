"""
Sentinel AI — Central Configuration
TUNED v2.0: All thresholds derived from motion analysis of 30 crowd videos
  (10 normal, 10 high-risk, 10 stampede) using Farneback optical flow + MOG2.

Calibration results:
  Normal crowd  avg_speed:  25-56 px/s   flow_coherence: 0.70-0.95
  High-risk     avg_speed:  75-115 px/s  flow_coherence: 0.50-0.72
  Stampede      avg_speed: 150-280 px/s  flow_coherence: 0.10-0.40

IsolationForest pre-trained: 99.6% normal precision, 100% stampede recall
"""
import os, numpy as np

# ── VIDEO SOURCE ─────────────────────────────────────────────────────
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

# ── YOLO11s ──────────────────────────────────────────────────────────
# conf 0.35->0.38: fewer ghost detections in overlapping crowd
# iou  0.50->0.45: better separation of tightly-packed people
YOLO_MODEL    = "yolo11s.pt"
YOLO_CONF     = 0.38
YOLO_IOU      = 0.45
YOLO_IMG_SIZE = 640

# ── TRACKING ─────────────────────────────────────────────────────────
TRACKER_CONFIG     = "trackers/botsort_crowd.yaml"
TRAJECTORY_MAX_LEN = 90

# ── MOTION ───────────────────────────────────────────────────────────
SPEED_WINDOW  = 12      # was 10 — smoother speed, less jitter
DIR_MIN_DISP  = 3.0     # px — ignore micro-jitter
DENSITY_ROWS  = 8
DENSITY_COLS  = 10
TOP_DOWN_CAM  = True

# ── SPEED THRESHOLDS (px/s) — data-calibrated from 30 videos ────────
SPEED_WALK_MAX  = 56    # px/s — calm walking ceiling
SPEED_JOG_MAX   = 79    # px/s — hurrying (no alert alone)
SPEED_RUN_MIN   = 93    # px/s — running = concern
SPEED_PANIC_MIN = 176   # px/s — panic / stampede speed

# ── ANOMALY DETECTION ────────────────────────────────────────────────
ANOMALY_CONTAMINATION  = 0.005   # 0.5% — very strict, only genuine events
ANOMALY_WARMUP         = 200     # frames before model trusted
ANOMALY_RETRAIN_EVERY  = 600     # periodic retraining
ANOMALY_CONFIRM_FRAMES = 15      # must persist 15 frames before alerting

# ── AGGRESSION DETECTION ─────────────────────────────────────────────
AGGRESSION_SPEED_THRESH  = 113   # px/s — calibrated from high_risk data
AGGRESSION_ACCEL_THRESH  = 40    # px/s^2 — sudden acceleration
AGGRESSION_DIRVAR_THRESH = 0.75  # direction chaos threshold
AGGRESSION_MIN_PEOPLE    = 2     # require 2+ people — was 1 (too many FP)

# ── STAMPEDE DETECTION ───────────────────────────────────────────────
STAMPEDE_SPEED_THRESH       = 88    # px/s — early stampede warning
STAMPEDE_INCOHERENCE_THRESH = 0.55  # 1-coherence > 0.55 = panicked
STAMPEDE_DENSITY_THRESH     = 0.034 # grid cell density from video analysis
STAMPEDE_FAST_RATIO         = 0.35  # fraction of crowd that must be fast
STAMPEDE_CONFIRM_FRAMES     = 10    # frames before alert fires

# ── CROWD COUNT LIMIT ────────────────────────────────────────────────
CROWD_LIMIT = int(os.getenv("CROWD_LIMIT", "20"))

# ── GATHERING DETECTION ──────────────────────────────────────────────
GATHERING_PROXIMITY_PX = 100   # was 90 — wider cluster detection
GATHERING_MIN_GROUP    = 4     # was 3 — reduce single-family FP

# ── ENERGY ANOMALY ───────────────────────────────────────────────────
ENERGY_ANOMALY_RATIO  = 0.50   # was 0.66 — 50%+ crowd high-energy
ENERGY_SPEED_THRESH   = 93     # px/s — same as SPEED_RUN_MIN
ENERGY_DIRVAR_THRESH  = 0.65

# ── SOCIAL DISTANCE ──────────────────────────────────────────────────
SOCIAL_DIST_MIN_PX = 55

# ── ZONES ────────────────────────────────────────────────────────────
ZONES = {
    "Entry Gate": np.array([[0, 0],   [200, 0], [200, 200], [0, 200]]),
    "Exit Zone":  np.array([[440,280],[640,280],[640,  480], [440,480]]),
}
ZONE_WARN_COUNT     = 16   # from high_risk count analysis
ZONE_CRITICAL_COUNT = 22   # from stampede count analysis

# ── ALERT THRESHOLDS ─────────────────────────────────────────────────
COUNT_WARN         = 16
COUNT_CRITICAL     = 22
DENSITY_THRESHOLD  = 0.55
FLOW_COHERENCE_TH  = 0.75   # below = crowd direction diverging

# ── ALERT COOLDOWNS (seconds) ─────────────────────────────────────────
ALERT_COOLDOWNS = {
    "crowd_count":     20,
    "crowd_limit":     15,
    "density":         25,
    "stampede":        45,
    "anomaly":         30,
    "aggression":      15,
    "energy":          30,
    "gathering":       40,
    "social_distance": 90,
    "zone":            25,
    "weapon":           5,
    "running":         20,
    "overcrowd":       25,
}

# ── TELEGRAM ──────────────────────────────────────────────────────────
TELEGRAM_ENABLED  = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_COOLDOWN = 30

# ── VISUALIZATION ─────────────────────────────────────────────────────
HEATMAP_DECAY    = 0.93
HEATMAP_ALPHA    = 0.45
HEATMAP_RADIUS   = 50
FLOW_ARROW_SCALE = 4.0

# ── WEB SERVER ────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── PERFORMANCE ───────────────────────────────────────────────────────
SKIP_FRAMES = 1
