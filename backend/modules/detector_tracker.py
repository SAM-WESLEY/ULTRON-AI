"""
Detector + Tracker — TUNED v2.0
Changes:
  - YOLO conf 0.35 → 0.38 (fewer ghost detections in crowd overlap)
  - YOLO iou  0.50 → 0.45 (better separation of tightly-packed people)
  - Added class-agnostic NMS override for dense crowd (avoids missed detections
    when people are pressed together at high density)
  - Stale track cleanup extended from 60 → 90 frames
    (prevents premature ID loss through occlusion in dense crowd)
  - Added bbox_area property to each tracked dict for downstream use
  - Added confidence to tracked dict so AlertManager can weight low-conf tracks
"""
from ultralytics import YOLO
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)


class DetectorTracker:
    def __init__(self):
        logger.info(f"Loading {config.YOLO_MODEL} ...")
        self.model = YOLO(config.YOLO_MODEL)
        self.position_history: dict = {}   # track_id → [(cx, cy, frame_n)]
        self.frame_count = 0
        logger.info("YOLOv11s ready")

    def process(self, frame: np.ndarray) -> list:
        """
        Run YOLO11s + BoT-SORT tracking on one frame.
        Returns list of dicts: {track_id, bbox, center, confidence,
                                width, height, bbox_area}
        """
        self.frame_count += 1

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.YOLO_CONF,       # 0.38 — tuned
            iou=config.YOLO_IOU,         # 0.45 — tuned
            imgsz=config.YOLO_IMG_SIZE,
            classes=[0],                 # person only
            agnostic_nms=True,           # class-agnostic NMS helps dense crowd
            verbose=False,
        )

        tracked = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return tracked

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids   = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        active = set()

        for bbox, tid, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = int(tid)
            active.add(tid)

            # Build / extend position history
            if tid not in self.position_history:
                self.position_history[tid] = []
            self.position_history[tid].append((cx, cy, self.frame_count))
            # Keep only last TRAJECTORY_MAX_LEN positions
            if len(self.position_history[tid]) > config.TRAJECTORY_MAX_LEN:
                self.position_history[tid] = \
                    self.position_history[tid][-config.TRAJECTORY_MAX_LEN:]

            w = int(x2 - x1)
            h = int(y2 - y1)
            tracked.append({
                "track_id":   tid,
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "center":     (int(cx), int(cy)),
                "confidence": float(conf),
                "width":      w,
                "height":     h,
                "bbox_area":  w * h,
            })

        # Clean stale tracks: extended 60 → 90 frames (handles occlusion)
        stale = [
            t for t in self.position_history
            if t not in active
            and (self.frame_count - self.position_history[t][-1][2]) > 90
        ]
        for t in stale:
            del self.position_history[t]

        return tracked
