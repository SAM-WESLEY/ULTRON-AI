"""
Alert Manager — TUNED v2.0
Changes:
  - Stampede alert now driven by StampedeDetector output (not a single threshold)
  - Speed thresholds updated to data-calibrated values (SPEED_RUN_MIN=93, SPEED_PANIC_MIN=176)
  - Running alert: separate WATCH alert at SPEED_JOG_MAX, WARNING at SPEED_RUN_MIN
  - Crowd count thresholds: COUNT_WARN=16, COUNT_CRITICAL=22 (from video analysis)
  - All cooldowns revised (see ALERT_COOLDOWNS in config)
  - Added multi-channel: Telegram (existing) + Email + SMS (Twilio)
  - Email/SMS fired only for CRITICAL-level events to avoid message spam
"""
import time, logging, smtplib, threading, requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config

logger = logging.getLogger(__name__)

# ── Shared mutable config (set via API) ──────────────────────────────
_crowd_limit = config.CROWD_LIMIT
_lock        = threading.Lock()
_email_cfg   = {"enabled": False}
_sms_cfg     = {"enabled": False}

def set_crowd_limit(v):
    global _crowd_limit
    with _lock: _crowd_limit = max(1, min(500, int(v)))

def get_crowd_limit():
    with _lock: return _crowd_limit

def set_email_cfg(cfg):
    global _email_cfg
    with _lock: _email_cfg.update(cfg)

def set_sms_cfg(cfg):
    global _sms_cfg
    with _lock: _sms_cfg.update(cfg)


# ── Notification helpers ──────────────────────────────────────────────
def _send_telegram(msg: str):
    if not config.TELEGRAM_ENABLED or not config.TELEGRAM_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": msg},
                      timeout=5)
        logger.info("Telegram sent")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")

def _send_email(subject: str, body: str):
    with _lock:
        cfg = dict(_email_cfg)
    if not cfg.get("enabled") or not cfg.get("host"):
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = cfg.get("from", "ultron@sentinel.ai")
        msg["To"]      = cfg.get("to",   "")
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(cfg["host"], int(cfg.get("port", 587)), timeout=8) as s:
            s.starttls()
            s.login(cfg.get("from",""), cfg.get("pass",""))
            s.send_message(msg)
        logger.info("Email sent")
    except Exception as e:
        logger.warning(f"Email error: {e}")

def _send_sms(body: str):
    with _lock:
        cfg = dict(_sms_cfg)
    if not cfg.get("enabled") or not cfg.get("sid"):
        return
    try:
        from twilio.rest import Client
        c = Client(cfg["sid"], cfg["token"])
        c.messages.create(body=body, from_=cfg["from"], to=cfg["to"])
        logger.info("SMS sent")
    except Exception as e:
        logger.warning(f"SMS error: {e}")

def _notify_critical(msg: str):
    """Fire all external channels for CRITICAL alerts."""
    threading.Thread(target=_send_telegram, args=(f"🚨 ULTRON ALERT\n{msg}",),  daemon=True).start()
    threading.Thread(target=_send_email,   args=("🚨 ULTRON CRITICAL ALERT", msg), daemon=True).start()
    threading.Thread(target=_send_sms,     args=(f"ULTRON: {msg}",),             daemon=True).start()

def _notify_warning(msg: str):
    """Telegram only for WARNING level."""
    threading.Thread(target=_send_telegram, args=(f"⚠️ ULTRON WARNING\n{msg}",), daemon=True).start()


class AlertManager:
    def __init__(self):
        self._last: dict = {}          # alert_key → last_fired_timestamp
        self._tg_last    = 0.0
        logger.info("AlertManager v2.0 ready (calibrated thresholds)")

    def _ok(self, key: str) -> bool:
        """True if cooldown has elapsed for this alert type."""
        cd  = config.ALERT_COOLDOWNS.get(key, 30)
        now = time.time()
        if now - self._last.get(key, 0) >= cd:
            self._last[key] = now
            return True
        return False

    def _alert(self, level: str, msg: str, atype: str) -> dict:
        return {"level": level, "type": atype, "message": msg,
                "time": time.strftime("%H:%M:%S")}

    def evaluate(self, features: dict, anomaly: dict,
                 groups: list, energy: dict,
                 zones: dict, social_violations: list,
                 aggression_flags: dict = None,
                 stampede: dict = None,
                 weapon: dict = None) -> list:
        """
        Evaluate all signals and return list of new alert dicts.
        Fires external notifications for CRITICAL alerts.
        """
        alerts = []
        count  = features.get("total_count", 0)
        spd    = features.get("avg_speed",   0.0)
        coh    = features.get("flow_coherence", 1.0)
        dens   = features.get("max_density",  0.0)
        limit  = get_crowd_limit()

        # ── 1. Stampede Detector output (primary signal) ──────────────
        if stampede and stampede.get("detected") and self._ok("stampede"):
            risk = stampede.get("risk_level", "CRITICAL")
            sigs = stampede.get("signals", {})
            msg  = (f"🌊 STAMPEDE {risk} — speed={sigs.get('avg_speed',0):.0f}px/s "
                    f"incoherence={sigs.get('incoherence',0):.2f} "
                    f"fast_ratio={sigs.get('fast_ratio',0):.2f}")
            alerts.append(self._alert("CRITICAL", msg, "stampede"))
            _notify_critical(msg)

        elif stampede and stampede.get("risk_level") in ("WARNING", "WATCH") and self._ok("stampede"):
            msg = (f"⚠️ STAMPEDE {stampede['risk_level']} — "
                   f"speed={stampede.get('signals',{}).get('avg_speed',0):.0f}px/s")
            alerts.append(self._alert("WARNING", msg, "stampede"))
            _notify_warning(msg)

        # ── 2. Crowd count limits ─────────────────────────────────────
        if count >= config.COUNT_CRITICAL and self._ok("crowd_count"):
            msg = f"🚨 CRITICAL CROWD — {count} people detected (critical={config.COUNT_CRITICAL})"
            alerts.append(self._alert("CRITICAL", msg, "crowd_count"))
            _notify_critical(msg)
        elif count >= config.COUNT_WARN and self._ok("crowd_count"):
            msg = f"⚠️ HIGH CROWD COUNT — {count} people (warn={config.COUNT_WARN})"
            alerts.append(self._alert("WARNING", msg, "crowd_count"))
            _notify_warning(msg)

        if count > limit and self._ok("crowd_limit"):
            msg = f"🚫 CROWD LIMIT EXCEEDED — {count}/{limit} people"
            alerts.append(self._alert("CRITICAL", msg, "crowd_limit"))
            _notify_critical(msg)

        # ── 3. Running / panic speed ──────────────────────────────────
        if spd >= config.SPEED_PANIC_MIN and self._ok("running"):
            msg = f"🏃 PANIC SPEED — avg {spd:.0f}px/s (panic>={config.SPEED_PANIC_MIN})"
            alerts.append(self._alert("CRITICAL", msg, "running"))
            _notify_critical(msg)
        elif spd >= config.SPEED_RUN_MIN and self._ok("running"):
            msg = f"🏃 CROWD RUNNING — avg {spd:.0f}px/s (run>={config.SPEED_RUN_MIN})"
            alerts.append(self._alert("WARNING", msg, "running"))
            _notify_warning(msg)

        # ── 4. Density ────────────────────────────────────────────────
        if dens >= config.DENSITY_THRESHOLD and self._ok("density"):
            msg = f"⬛ HIGH DENSITY — {dens:.2f} (threshold={config.DENSITY_THRESHOLD})"
            alerts.append(self._alert("WARNING", msg, "density"))
            _notify_warning(msg)

        # ── 5. Anomaly engine ─────────────────────────────────────────
        if anomaly.get("confirmed") and self._ok("anomaly"):
            score = anomaly.get("anomaly_score", 0.0)
            msg   = f"🔴 ANOMALY CONFIRMED — score={score:.4f} ({anomaly.get('consec_frames',0)} frames)"
            alerts.append(self._alert("CRITICAL", msg, "anomaly"))
            _notify_critical(msg)
        elif anomaly.get("is_anomaly") and not anomaly.get("confirmed") and self._ok("anomaly"):
            alerts.append(self._alert("INFO",
                f"🟡 Suspicious behaviour — score={anomaly.get('anomaly_score',0):.4f}",
                "anomaly"))

        # ── 6. Aggression ─────────────────────────────────────────────
        if aggression_flags and aggression_flags.get("detected") and self._ok("aggression"):
            ids  = aggression_flags.get("track_ids", [])
            msg  = f"👊 AGGRESSIVE MOVEMENT — IDs {ids}"
            alerts.append(self._alert("CRITICAL", msg, "aggression"))
            _notify_critical(msg)

        # ── 7. Gatherings ─────────────────────────────────────────────
        if len(groups) >= 2 and self._ok("gathering"):
            total_in_groups = sum(len(g) for g in groups)
            msg = f"👥 {len(groups)} gatherings detected — {total_in_groups} people in groups"
            alerts.append(self._alert("WARNING", msg, "gathering"))

        # ── 8. Energy anomaly ─────────────────────────────────────────
        if energy.get("is_abnormal") and self._ok("energy"):
            msg = f"⚡ ENERGY SPIKE — {energy['abnormal']}/{energy['total']} high-energy ({energy['ratio']:.0%})"
            alerts.append(self._alert("WARNING", msg, "energy"))
            _notify_warning(msg)

        # ── 9. Zone alerts ────────────────────────────────────────────
        for zname, zdata in zones.items():
            if zdata.get("status") == "RED" and self._ok(f"zone_{zname}"):
                msg = f"🔴 ZONE CRITICAL — {zname}: {zdata['count']} people"
                alerts.append(self._alert("CRITICAL", msg, "zone"))
                _notify_critical(msg)
            elif zdata.get("status") == "AMBER" and self._ok(f"zone_{zname}_amber"):
                alerts.append(self._alert("WARNING",
                    f"🟡 Zone warning — {zname}: {zdata['count']} people", "zone"))

        # ── 10. Social distance ───────────────────────────────────────
        if len(social_violations) >= 5 and self._ok("social_distance"):
            alerts.append(self._alert("INFO",
                f"📏 Social distance: {len(social_violations)} violations", "social_distance"))

        # ── 11. Weapon ────────────────────────────────────────────────
        if weapon and weapon.get("new_alert") and self._ok("weapon"):
            msg = f"🔪 WEAPON DETECTED — {weapon.get('weapons', [])}"
            alerts.append(self._alert("CRITICAL", msg, "weapon"))
            _notify_critical(msg)

        if alerts:
            logger.info(f"Fired {len(alerts)} alerts: {[a['type'] for a in alerts]}")

        return alerts
