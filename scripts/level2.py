# Gustavo Ojeda -- Predictive Assistance Alerts: Level 2, does this shopper need help? (live part)
"""Level 2 for the live program: turns the last 20 s of each person into a "needs help"
probability, and decides when to send an alert.

The behavior of the last 20 s arrives by TWO PATHS:
  - Path A (actions, from Level 1): % of the time looking, touching, hands busy
    (picking + phone, which a skeleton cannot tell apart) and not engaged.
  - Path B (body movement, straight from the skeleton): how much the person walks along the
    aisle, how much the head goes up and down (scanning shelves), how often a hand is above the
    shoulders (reaching high shelves) and how much of the time the person stands still.
    v1.1 measures path B over the last 60 s (numbers called "L_..."): a minute shows the search
    pattern better than 20 s.

Pieces:
  - IdRelinker: if the tracker loses a person for a moment and gives them a new ID, keep the old one
    (same place, similar size, a few seconds later). Uses position and size only, no face.
  - BehaviorTracker: keeps, per person, every frame's head and wrist height (path B) and every
    Level 1 prediction with the person's box (path A + walking), and computes the numbers above.
  - Level2Model: a logistic regression saved as plain numbers in a JSON file (one weight per
    number), so it can be read and explained. The JSON also holds the alert settings.
  - AlertLogic: an alert only when the probability stays high for a few seconds, the person has
    been there long enough, and at most `max_alerts` alerts per person (v1: one).
"""

import json
import math
from collections import deque

import numpy as np

HANDS_BUSY = ("picking", "phone")   # in a skeleton, holding a product and holding a phone look the same
NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10
MIN_CONF = 0.3                      # keypoints below this confidence are treated as missing
STILL_STEP = 0.02                   # a step smaller than 2% of the body height counts as "still"


class IdRelinker:
    """Map the tracker's IDs to stable IDs, joining short interruptions of the same person."""

    def __init__(self, max_gap_s=3.0, max_dist=0.6):
        self.max_gap_s, self.max_dist = max_gap_s, max_dist
        self.stable = {}      # raw tracker id -> stable id
        self.last = {}        # stable id -> (time, foot x, foot y, box height)

    def update(self, raw_id, t, box):
        foot_x, foot_y, height = (box[0] + box[2]) / 2, box[3], box[3] - box[1]
        if raw_id not in self.stable:
            best = None
            for sid, (lt, fx, fy, h) in self.last.items():
                gap = t - lt
                # a lost person: not seen in this moment, gone for less than max_gap_s, close and of similar size
                if 0.3 < gap <= self.max_gap_s and 0.7 < height / h < 1.4 and \
                        math.hypot(foot_x - fx, foot_y - fy) / h <= self.max_dist:
                    if best is None or gap < best[0]:
                        best = (gap, sid)
            self.stable[raw_id] = best[1] if best else raw_id
        sid = self.stable[raw_id]
        self.last[sid] = (t, foot_x, foot_y, height)
        return sid


def pose_numbers(xy, conf, box):
    """Head height and highest-wrist height above the shoulders, in body heights (NaN if not visible)."""
    height = max(box[3] - box[1], 1.0)
    if conf[L_SHOULDER] < MIN_CONF or conf[R_SHOULDER] < MIN_CONF:
        return np.nan, np.nan
    shoulder_y = (xy[L_SHOULDER, 1] + xy[R_SHOULDER, 1]) / 2          # image y grows downwards
    head_up = (shoulder_y - xy[NOSE, 1]) / height if conf[NOSE] >= MIN_CONF else np.nan
    wrists = [xy[w, 1] for w in (L_WRIST, R_WRIST) if conf[w] >= MIN_CONF]
    wrist_up = (shoulder_y - min(wrists)) / height if wrists else np.nan
    return head_up, wrist_up


class BehaviorTracker:
    """Per person: frames (path B) and Level 1 predictions (path A + walking), kept for the last 60 s.

    The main numbers use the last 20 s. The "L_" numbers use the last 60 s (longer memory): a lost
    shopper tends to cover the whole aisle and come back to places already checked.
    """

    def __init__(self, seconds=20.0, every_seconds=0.67, long_seconds=60.0, cell=0.4):
        self.seconds, self.every, self.long_seconds, self.cell = seconds, every_seconds, long_seconds, cell
        self.frames = {}      # id -> deque of (time, head_up, wrist_up)           every frame
        self.history = {}     # id -> deque of (time, action, foot x, foot y, box height)  every Level 1 prediction
        self.first_seen = {}  # id -> time

    def add_frame(self, pid, t, xy, conf, box):
        """Call on EVERY frame of a person (head and wrist heights)."""
        f = self.frames.setdefault(pid, deque())
        self.first_seen.setdefault(pid, t)
        f.append((t, *pose_numbers(xy, conf, box)))
        while t - f[0][0] > self.long_seconds:
            f.popleft()

    def add(self, pid, t, action, box):
        """Call every time Level 1 gives a new action for this person."""
        h = self.history.setdefault(pid, deque())
        self.first_seen.setdefault(pid, t)
        h.append((t, action, (box[0] + box[2]) / 2, box[3], box[3] - box[1]))
        while t - h[0][0] > self.long_seconds:
            h.popleft()

    def _numbers(self, h, f, prefix=""):
        actions = [a for _, a, _, _, _ in h]
        n = len(actions)
        cx = np.array([x for _, _, x, _, _ in h]); foot = np.array([y for _, _, _, y, _ in h])
        height = np.median([b for _, _, _, _, b in h]); duration = max(h[-1][0] - h[0][0], 1e-6)
        dy, dx = np.diff(foot) / height, np.diff(cx) / height
        head = np.array([v for _, v, _ in f], float); wrist = np.array([v for _, _, v in f], float)
        out = {
            # path A: actions from Level 1
            "pct_looking": actions.count("looking") / n,
            "pct_touching": actions.count("touching") / n,
            "pct_hands_busy": sum(a in HANDS_BUSY for a in actions) / n,
            "pct_not engaged": actions.count("not engaged") / n,
            # path B: body movement from the skeleton
            "aisle_walk": float(np.abs(dy).sum() / duration * self.seconds),   # body heights walked along the aisle per 20 s
            "head_scan": float(np.nanstd(head)) if np.isfinite(head).sum() > 5 else 0.0,  # head going up and down
            "hand_high": float(np.mean(wrist > 0)),                            # share of frames with a hand above the shoulders
            "still": float((np.hypot(dx, dy) < STILL_STEP).mean()),           # share of moments standing still
        }
        if prefix:
            # longer memory: how much of the aisle was covered and how often the person came back to a place
            cells = list(zip((cx / (self.cell * height)).astype(int), (foot / (self.cell * height)).astype(int)))
            seen, revisits, prev = set(), 0, None
            for c in cells:
                if c != prev:
                    revisits += c in seen
                    seen.add(c)
                prev = c
            minutes = duration / 60
            out.update({"x_range": float((cx.max() - cx.min()) / height), "y_range": float((foot.max() - foot.min()) / height),
                        "places_per_min": len(seen) / minutes, "revisits_per_min": revisits / minutes})
        else:
            # v0 numbers, kept so level2_v0.json still runs (for comparisons)
            out.update({"distance": float(np.abs(dy).sum() / n * 30), "stillness": float((np.abs(dy) < STILL_STEP).mean())})
        return {prefix + k: v for k, v in out.items()}

    def features(self, pid):
        """The behavior numbers (last 20 s, plus "L_" numbers over the last 60 s), or None if there is not enough history."""
        h, f = self.history.get(pid), self.frames.get(pid)
        if not h or not f:
            return None
        h20 = [x for x in h if x[0] >= h[-1][0] - self.seconds]
        f20 = [x for x in f if x[0] >= f[-1][0] - self.seconds]
        if h[-1][0] - self.first_seen[pid] < 0.9 * self.seconds or len(h20) < 0.6 * self.seconds / self.every or not f20:
            return None
        return {**self._numbers(h20, f20), **self._numbers(list(h), list(f), prefix="L_")}

    def age(self, pid, t):
        return t - self.first_seen.get(pid, t)


class Level2Model:
    """Logistic regression stored in JSON: probability = 1 / (1 + e^-(sum of weight x standardized number + intercept))."""

    def __init__(self, path):
        m = json.loads(open(path).read())
        self.name, self.features, self.threshold = m["name"], m["features"], m["threshold"]
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.weights, self.intercept = np.array(m["weights"]), m["intercept"]
        self.alert = m.get("alert", {})          # alert settings (v1); v0 files use the defaults

    def probability(self, values):
        x = (np.array([values[f] for f in self.features]) - self.mean) / self.scale
        return float(1 / (1 + math.exp(-(x @ self.weights + self.intercept))))

    def alert_logic(self):
        a = self.alert
        return AlertLogic(self.threshold, hold_s=a.get("hold_s", 5.0), smooth=a.get("smooth", 5),
                          cooldown_s=a.get("cooldown_s", 60.0), min_age_s=a.get("min_age_s", 20.0),
                          max_alerts=a.get("max_alerts", None))


class AlertLogic:
    """Decide when a person's 'needs help' probability becomes an alert."""

    def __init__(self, threshold, hold_s=5.0, smooth=5, cooldown_s=60.0, min_age_s=20.0, max_alerts=None):
        self.threshold, self.hold_s, self.cooldown_s, self.min_age_s = threshold, hold_s, cooldown_s, min_age_s
        self.smooth, self.max_alerts = smooth, max_alerts
        self.recent = {}      # id -> last probabilities (smoothing)
        self.high_since = {}  # id -> time the smoothed probability went above the threshold
        self.last_alert = {}  # id -> time of the last alert
        self.count = {}       # id -> alerts sent

    def update(self, pid, t, probability, age_s):
        """Return (smoothed probability, True if an alert must be sent now)."""
        r = self.recent.setdefault(pid, deque(maxlen=self.smooth))
        r.append(probability)
        smoothed = sum(r) / len(r)
        if smoothed < self.threshold or age_s < self.min_age_s:
            self.high_since.pop(pid, None)
            return smoothed, False
        if self.max_alerts is not None and self.count.get(pid, 0) >= self.max_alerts:
            return smoothed, False                      # this person already got their alert(s)
        start = self.high_since.setdefault(pid, t)
        due = t - start >= self.hold_s and t - self.last_alert.get(pid, -1e9) >= self.cooldown_s
        if due:
            self.last_alert[pid] = t
            self.count[pid] = self.count.get(pid, 0) + 1
        return smoothed, due

    def alerted(self, pid, t):
        """True during the 60 s after this person's last alert (to color them on screen)."""
        return t - self.last_alert.get(pid, -1e9) < self.cooldown_s
