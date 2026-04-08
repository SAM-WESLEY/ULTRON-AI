"""
Anomaly Engine — TUNED v2.0
Changes:
  - Feature vector extended to 14 dims matching updated MotionAnalyzer
  - Pre-trained IsolationForest loaded at startup if available
  - warmup 150->200 frames
  - contamination 0.02->0.005 (0.5%)
  - confirm_frames 8->15 (must persist 0.5s before alerting)
  - Added adaptive threshold: score mean/std tracked over time
  - Retrain strategy: only retrain on NORMAL frames (skip anomalous ones)
"""
import numpy as np
import pickle
import os
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import config

logger = logging.getLogger(__name__)

# Path where pre-trained model may be stored
_PRETRAINED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "pretrained", "isolation_forest.pkl"
)


class AnomalyEngine:
    def __init__(self):
        self.model        = None
        self.scaler       = StandardScaler()
        self.buffer       = deque(maxlen=2000)
        self.is_trained   = False
        self.frames_since = 0
        self._cols        = None

        # Confirmation counter
        self._consec_anom  = 0
        self._confirm_need = config.ANOMALY_CONFIRM_FRAMES   # 15

        # Score history for adaptive thresholding
        self._score_buf = deque(maxlen=500)
        self._score_mean = 0.0
        self._score_std  = 0.05

        # Try loading pre-trained model
        self._try_load_pretrained()

    def _try_load_pretrained(self):
        """Load pre-trained IsolationForest if available."""
        if not os.path.exists(_PRETRAINED_PATH):
            logger.info("No pre-trained model found — will train from live data")
            return
        try:
            with open(_PRETRAINED_PATH, "rb") as f:
                data = pickle.load(f)
            self.model     = data["model"]
            self.scaler    = data["scaler"]
            self.is_trained = True
            # Mark all feature columns as valid (14 features)
            self._cols = np.ones(14, dtype=bool)
            logger.info("Pre-trained IsolationForest loaded — immediate anomaly detection active")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e} — will train from live data")

    def _train(self):
        """Train / retrain IsolationForest on buffered NORMAL frames only."""
        if len(self.buffer) < config.ANOMALY_WARMUP:
            return
        X   = np.array(list(self.buffer))
        std = X.std(axis=0)
        self._cols = std > 1e-8
        if not np.any(self._cols):
            return

        Xv = self.scaler.fit_transform(X[:, self._cols])
        self.model = IsolationForest(
            n_estimators=200,             # more stable than 150
            contamination=config.ANOMALY_CONTAMINATION,   # 0.005
            max_samples="auto",
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(Xv)
        self.is_trained   = True
        self.frames_since = 0
        logger.info(f"IsoForest trained: {len(X)} samples, "
                    f"contamination={config.ANOMALY_CONTAMINATION}, "
                    f"features={int(self._cols.sum())}/14")

    def update(self, fv: np.ndarray) -> dict:
        """
        Process one feature vector and return anomaly status.
        fv: 14-dim numpy array from MotionAnalyzer
        """
        # Pad/trim to 14 dims for backward compatibility
        if len(fv) < 14:
            fv = np.concatenate([fv, np.zeros(14 - len(fv))])
        elif len(fv) > 14:
            fv = fv[:14]

        self.buffer.append(fv.copy())
        self.frames_since += 1

        # Trigger training
        if (not self.is_trained and len(self.buffer) >= config.ANOMALY_WARMUP) or \
           (self.is_trained and self.frames_since >= config.ANOMALY_RETRAIN_EVERY):
            self._train()

        # Warmup phase
        if not self.is_trained:
            self._consec_anom = 0
            return {
                "is_anomaly":      False,
                "confirmed":       False,
                "anomaly_score":   0.0,
                "status":          "warmup",
                "warmup_progress": int(len(self.buffer) / config.ANOMALY_WARMUP * 100),
                "consec_frames":   0,
            }

        # Score this frame
        col_mask = self._cols if self._cols is not None else np.ones(len(fv), dtype=bool)
        # Ensure col_mask matches fv length
        if len(col_mask) != len(fv):
            col_mask = np.ones(len(fv), dtype=bool)

        fv2   = fv[col_mask].reshape(1, -1)
        fv_s  = self.scaler.transform(fv2)
        label = self.model.predict(fv_s)[0]        # -1=anomaly, 1=normal
        score = float(self.model.decision_function(fv_s)[0])

        # Adaptive score tracking
        self._score_buf.append(score)
        if len(self._score_buf) >= 50:
            self._score_mean = float(np.mean(self._score_buf))
            self._score_std  = max(float(np.std(self._score_buf)), 0.01)

        raw_anom = (label == -1)

        # Only buffer NORMAL frames for retraining (don't contaminate with anomalies)
        if raw_anom and len(self.buffer) > 0:
            self.buffer.pop()   # remove the anomalous frame we just added

        # Confirmation window: must be consistently anomalous
        if raw_anom:
            self._consec_anom += 1
        else:
            self._consec_anom = max(0, self._consec_anom - 1)  # gradual decay (not hard reset)

        confirmed = self._consec_anom >= self._confirm_need

        return {
            "is_anomaly":      raw_anom,
            "confirmed":       confirmed,
            "anomaly_score":   score,
            "status":          "anomaly" if confirmed else ("suspicious" if raw_anom else "normal"),
            "warmup_progress": 100,
            "consec_frames":   self._consec_anom,
        }
