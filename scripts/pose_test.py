"""
pose_test.py

Gustavo Ojeda -- Predictive Assistance Alerts: first look at body pose estimation.
Opens a camera, runs a YOLO pose model on every frame, draws each person's
skeleton, and reports the average FPS, to check whether pose estimation runs
fast enough on this laptop's CPU.
"""

import argparse
import time

import cv2
from ultralytics import YOLO


# Gustavo Ojeda -- Predictive Assistance Alerts: COCO keypoint indices used by YOLO
# pose models (0 nose, 1-4 eyes/ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
# 11-12 hips, 13-14 knees, 15-16 ankles).
LEFT_SHOULDER, RIGHT_SHOULDER = 5,6
LEFT_WRIST, RIGHT_WRIST = 9,10


def hand_raised(xy, conf, min_conf: float = 0.5) -> bool:
    """True if either wrist is above its shoulder, using only keypoints the model is confident about"""
    for wrist, shoulder in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
        if conf[wrist] < min_conf or conf[shoulder] < min_conf:
            continue
        if xy[wrist][1] < xy[shoulder][1]:
            return True
    return False


def main() -> None:
    """Open the camera, draw every person's skeleton live, and report the average FPS."""
    parser = argparse.ArgumentParser(description="Live body pose test")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Pose model weights; download automatically the first time (default: yolov8n-pose.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Pose confidence threshold(default = 0.5)" )
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.source}")

    frames = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=args.conf, verbose=False)
        annotated = results[0].plot()
        # Gustavo Ojeda -- Predictive Assistance Alerts: read each person's keypoints as numbers
        # and flag anyone with a hand raised -- the first hand-crafted body feature.
        result = results[0]
        boxes = result.boxes.xyxy.int().tolist()
        all_xy = result.keypoints.xy.tolist()
        all_conf = result.keypoints.conf.tolist()
        for box, xy, kp_conf in zip(boxes, all_xy, all_conf):
            if hand_raised(xy, kp_conf):
                cv2.putText(annotated, "HAND UP", (box[0], box[3] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),2)

        frames += 1
        fps = frames / (time.perf_counter() - start)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pose test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frames} frames in {elapsed:.1f}s ({frames/elapsed:.2f} FPS average)")


if __name__ == "__main__":
    main()
