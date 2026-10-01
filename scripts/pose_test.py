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

        results = model(frame, verbose=False, conf=args.conf)
        annotated = results[0].plot()

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
