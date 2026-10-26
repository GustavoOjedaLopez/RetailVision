# Gustavo Ojeda -- Predictive Assistance Alerts: run the Level 1 model live on a camera (pilot test)
"""Show, over each person, which action group the Level 1 model thinks they are doing.

For every tracked person the last ~2.7 s of keypoints are kept (the same duration as a
60-frame CARR window). About every 0.7 s the same features used in training
(build_carr_windows.window_features) are computed, with speeds converted to CARR's fps,
and the saved model predicts: looking / touching / picking / not engaged.

Pilot recording (optional, --record): every frame of every person is saved to a CSV
(keypoints only, no images) together with the model's current prediction and two
labels you set with the keyboard while recording:
    action (Level 1, what you are doing now):
        1 = looking   2 = touching   3 = picking   4 = not engaged   5 = phone
    task (Level 2, which kind of task you are doing):
        n = normal task      h = needs-help task
    0 = clear both labels (between tasks)      q = quit
--video also saves the annotated video, only for self-tests (never with volunteers
without their consent).
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from scripts.build_carr_windows import CARR_FPS, STRIDE, WINDOW, window_features

TASK_KEYS = {ord("n"): "normal", ord("h"): "needs_help"}
# Gustavo Ojeda -- Predictive Assistance Alerts: action labels for our own lab data (Level 1 adaptation)
ACTION_KEYS = {ord("1"): "looking", ord("2"): "touching", ord("3"): "picking", ord("4"): "not engaged",
               ord("5"): "phone"}  # phone is not in CARR: only our lab data can teach it


class ActionRecognizer:
    """Keeps a short movement history per person and asks the Level 1 model what they are doing.

    Gustavo Ojeda -- Predictive Assistance Alerts: windows are measured in SECONDS, not frames,
    so the model sees the same amount of movement as in CARR (60 frames at 22.5 fps = 2.7 s)
    whatever the live camera's fps is. Speeds are converted to CARR frames with speed_scale.
    """

    def __init__(self, model_path, seconds=WINDOW / CARR_FPS, every_seconds=STRIDE / CARR_FPS):
        saved = joblib.load(model_path)
        self.model, self.features = saved["model"], saved["features"]
        self.seconds, self.every_seconds = seconds, every_seconds
        self.history = {}      # track id -> deque of (time, xy, conf, box)
        self.last_seen = {}    # track id -> frame index
        self.last_predicted = {}  # track id -> time of the last prediction
        self.prediction = {}   # track id -> (label, probability)

    def update(self, track_id, frame_index, timestamp, xy, conf, box):
        """Add one frame of one person; return their latest (label, probability) or None."""
        if self.last_seen.get(track_id) != frame_index - 1:
            # The person was lost for a moment: start a fresh history (windows never cross gaps)
            self.history[track_id] = deque()
            self.last_predicted[track_id] = None
        self.last_seen[track_id] = frame_index
        frames = self.history[track_id]
        frames.append((timestamp, xy, conf, box))
        while timestamp - frames[0][0] > self.seconds:   # keep only the last `seconds` of movement
            frames.popleft()
        covered = timestamp - frames[0][0]
        due = self.last_predicted[track_id] is None or timestamp - self.last_predicted[track_id] >= self.every_seconds
        if covered >= 0.9 * self.seconds and len(frames) >= 5 and due:
            live_fps = (len(frames) - 1) / covered
            xy_w = np.stack([f[1] for f in frames])
            conf_w = np.stack([f[2] for f in frames])
            box_w = np.stack([f[3] for f in frames])
            features = window_features(xy_w, conf_w, box_w, speed_scale=live_fps / CARR_FPS)
            probabilities = self.model.predict_proba(pd.DataFrame([features])[self.features])[0]
            best = int(np.argmax(probabilities))
            self.prediction[track_id] = (self.model.classes_[best], float(probabilities[best]))
            self.last_predicted[track_id] = timestamp
        return self.prediction.get(track_id)


def open_video(path, frames, fps):
    """Create the video file at the measured fps and write the frames kept so far."""
    height, width = frames[0].shape[:2]
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, round(fps, 1)), (width, height))
    for frame in frames:
        video.write(frame)
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=int, default=0, help="camera index")
    parser.add_argument("--model", type=Path, default=Path("data/carr/level1_model.joblib"))
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--record", type=Path, help="save keypoints + predictions + action/task labels to this CSV")
    parser.add_argument("--video", type=Path, help="also save the annotated video (self-tests only)")
    args = parser.parse_args()

    recognizer = ActionRecognizer(args.model)
    pose = YOLO(args.pose_model)
    camera = cv2.VideoCapture(args.source)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.source}")

    writer, csv_file, video, pending = None, None, None, []
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.record, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["time", "frame", "track_id", "action", "task", "predicted", "probability",
                         "box_x1", "box_y1", "box_x2", "box_y2"]
                        + [f"kp{i}_{a}" for i in range(17) for a in ("x", "y", "conf")])

    action, task, frame_index, start = "none", "none", 0, time.time()
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame_index += 1
        result = pose.track(frame, persist=True, conf=args.conf, verbose=False)[0]
        annotated = result.plot(boxes=False)

        if result.boxes.id is not None and result.keypoints is not None:
            ids = result.boxes.id.int().tolist()
            boxes = result.boxes.xyxy.numpy()
            xys = result.keypoints.xy.numpy()
            confs = result.keypoints.conf.numpy()
            for track_id, box, xy, conf in zip(ids, boxes, xys, confs):
                prediction = recognizer.update(track_id, frame_index, time.time() - start, xy, conf, box)
                label = f"ID {track_id}: " + (f"{prediction[0]} {prediction[1]:.0%}" if prediction else "...")
                x1, y1 = int(box[0]), int(box[1])
                cv2.rectangle(annotated, (x1, y1), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                # Label above the box, or just inside it when the box touches the top of the image
                label_y = y1 - 10 if y1 > 30 else y1 + 25
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, label_y - label_h - 6), (x1 + label_w + 6, label_y + 6), (0, 0, 0), -1)
                cv2.putText(annotated, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if writer:
                    writer.writerow([round(time.time() - start, 3), frame_index, track_id, action, task,
                                     prediction[0] if prediction else "", round(prediction[1], 3) if prediction else "",
                                     *[round(float(v), 1) for v in box],
                                     *[round(float(v), 3) for pair in zip(xy[:, 0], xy[:, 1], conf) for v in pair]])

        # Gustavo Ojeda -- Predictive Assistance Alerts: small status text in the bottom-right corner,
        # so it never covers the prediction written over the person
        fps = frame_index / (time.time() - start)
        status = f"FPS {fps:.1f} | action: {action} | task: {task}" + (" | REC" if writer else "")
        (text_w, text_h), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        height, width = annotated.shape[:2]
        corner = (width - text_w - 10, height - 10)
        cv2.rectangle(annotated, (corner[0] - 5, corner[1] - text_h - 5), (width - 5, height - 5), (0, 0, 0), -1)
        cv2.putText(annotated, status, corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if args.video:
            # Gustavo Ojeda -- Predictive Assistance Alerts: the saved video must play at the REAL speed.
            # Keep the first 5 s in memory, measure the fps, then open the video file with that fps.
            if video is None:
                pending.append(annotated)
                if time.time() - start >= 5:
                    video = open_video(args.video, pending, frame_index / (time.time() - start))
                    pending = []
            else:
                video.write(annotated)
        cv2.imshow("Level 1 live", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in TASK_KEYS:
            task = TASK_KEYS[key]
            print(f"[{time.time() - start:7.1f} s] task -> {task}")
        if key in ACTION_KEYS:
            action = ACTION_KEYS[key]
            print(f"[{time.time() - start:7.1f} s] action -> {action}")
        if key == ord("0"):
            action, task = "none", "none"
            print(f"[{time.time() - start:7.1f} s] labels cleared")

    camera.release()
    if csv_file:
        csv_file.close()
        print(f"Saved recording to {args.record}")
    if video is None and pending:  # very short run: save what we have
        video = open_video(args.video, pending, frame_index / (time.time() - start))
    if video is not None:
        video.release()
        print(f"Saved video to {args.video}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
