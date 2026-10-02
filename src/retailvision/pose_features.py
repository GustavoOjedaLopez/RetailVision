"""
pose_features.py

Gustavo Ojeda -- Predictive Assistance Alerts: body features computed from YOLO pose
keypoints (the 17 COCO points per person), used to describe what a person's body is
doing -- the inputs the assistance model will learn from.
"""


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