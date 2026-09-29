"""
pipeline_demo.py

End-to-end demo of the RetailVision pipeline: opens a camera or a
pre-recorded video file, runs each frame through InferencePipeline (face
detection plus age/gender/emotion classification), tracks each detected
face across frames and counts virtual-line crossings (see tracking.py,
see counter.py), shows a live preview with each detected person's bounding
box, predictions, the counting line, and the running occupancy drawn, and
appends an anonymized log record for each detection (see output_log.py).
Optionally also ships the same records to a central server in real time
(see remote_log.py), for deployments running more than one camera node --
this is a second, independent sink alongside the local log file, not a
replacement for it.
Optionally also streams the annotated frame to that server for live
viewing in the dashboard (see frame_stream.py). This is the one path
that moves pixels off this machine, so it is opt-in behind
--stream-frames, off by default, and independent of the anonymized
record shipping above.
Supports a headless --benchmark mode (no display) to measure sustained
FPS, since imshow overhead would otherwise skew the number -- benchmark
mode has no window to capture a 'q' keypress, so it stops automatically
after --duration seconds (or when the source runs out, for a video file)
rather than waiting on unreachable input. Per the project's testing
convention, point --source at a pre-recorded video file first -- a
deterministic input makes debugging far easier than a live camera feed --
before testing against the live camera.
"""

import argparse
import time
from pathlib import Path

import cv2

from .counter import LineCounter
from .calibration import CameraCalibration
from .frame_stream import FrameStreamer
from .marker_map import DEFAULT_HEAD_HEIGHT_METERS, FLOOR_MOUNTING, WALL_MOUNTING, MarkerMap
from .marker_pose import MarkerPoseEstimator
from .inference import InferencePipeline
from .output_log import log_detection
from .person_track import TrackRegistry, reported_detection
from .remote_log import RemoteLogShipper
from .zones import ZoneMap, ZoneResolver, load_zones
from .tracking import CentroidTracker, bbox_centroid


def parse_args() -> argparse.Namespace:
    """Parse --source, --benchmark, --duration, the --line-* counting line options, the optional --server-url/--camera-node-id/--api-key trio, and --stream-frames."""
    parser = argparse.ArgumentParser(description="Run the RetailVision inference pipeline")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run headless (no preview window) and report average FPS on exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Benchmark mode only: seconds to run before stopping automatically (default: 15)",
    )
    parser.add_argument(
        "--line-axis",
        choices=["x", "y"],
        default="x",
        help="Axis the virtual counting line spans perpendicular to: 'x' for a vertical line, 'y' for a horizontal one (default: x)",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=None,
        help="Pixel coordinate of the counting line along --line-axis (default: middle of the frame)",
    )
    parser.add_argument(
        "--line-direction",
        choices=["increasing", "decreasing"],
        default="increasing",
        help="Crossing direction counted as an entry: 'increasing' (e.g. left-to-right) or 'decreasing' (default: increasing)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Central server base URL to also ship anonymized records to (e.g. http://localhost:8000). "
        "Omit to log locally only.",
    )
    parser.add_argument(
        "--camera-node-id",
        default=None,
        help="This machine's identifier, sent alongside shipped records. Required if --server-url is set.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the central server. Required if --server-url is set.",
    )
    # Marker-based zones. All optional: without --zones the pipeline behaves
    # exactly as before and zone_id stays null, as it has since the schema was
    # frozen. See the README's marker-based zones section.
    parser.add_argument("--zones", type=Path, default=None, help="Zone definition JSON; enables per-zone occupancy and populates zone_id")
    parser.add_argument(
        "--marker-map",
        type=Path,
        default=None,
        help="Surveyed marker map from 'scripts/setup/aruco_pose_test.py --save-map'. Required with --zones on a multi-camera setup, "
             "since a node that builds its own map would not share a world frame with the others",
    )
    parser.add_argument("--calibration", default=None, help="This camera's calibration JSON, required with --zones")
    parser.add_argument("--marker-size", type=float, default=0.14, help="Printed marker side length in meters (default: 0.14)")
    parser.add_argument("--anchor", type=int, default=None, help="Marker ID fixed as the world origin (default: first seen)")
    parser.add_argument(
        "--marker-mounting",
        choices=[FLOOR_MOUNTING, WALL_MOUNTING],
        default=FLOOR_MOUNTING,
        help="How markers are mounted: flat on the floor, or upright on walls (default: floor)",
    )
    parser.add_argument("--marker-height", type=float, default=0.0, help="Height in meters of the anchor marker above the floor")
    parser.add_argument(
        "--head-height",
        type=float,
        default=DEFAULT_HEAD_HEIGHT_METERS,
        help=f"Plane height in meters that detections back-project onto (default: {DEFAULT_HEAD_HEIGHT_METERS})",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "centroid"),
        default="bytetrack",
        help="How faces are followed between frames: 'bytetrack' uses the detector's own "
        "confidence-aware tracker, 'centroid' the simpler nearest-centroid matcher (default: bytetrack)",
    )
    parser.add_argument(
        "--stream-frames",
        action="store_true",
        help="Also stream the annotated frame to the server for live viewing in the dashboard. "
        "Requires --server-url. Off by default -- this is the only path that sends pixels off this machine.",
    )
    parser.add_argument(
        "--stream-fps",
        type=float,
        default=15.0,
        help="Cap on frames per second sent with --stream-frames (default: 15)",
    )
    args = parser.parse_args()
    if args.stream_frames and not args.server_url:
        parser.error("--stream-frames requires --server-url")
    if args.server_url and not (args.camera_node_id and args.api_key):
        parser.error("--server-url requires both --camera-node-id and --api-key")
    if args.zones and not args.calibration:
        parser.error("--zones requires --calibration, since zone positions are measured in real units")
    if args.marker_map and not args.zones:
        parser.error("--marker-map is only meaningful with --zones")
    return args


# A camera comes up at its own default resolution, which is rarely the one its
# calibration was captured at. Focal length is measured in pixels, so it only
# describes the camera at that resolution -- running at another one rescales
# every distance instead of failing, and the marker map was surveyed at the
# calibrated resolution, so the two must agree or no position lands in a zone.
CAPTURE_WARMUP_TIMEOUT = 5.0


def capture_resolution(capture: cv2.VideoCapture) -> tuple[int, int]:
    """The resolution the source is currently delivering."""
    return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def wait_for_first_frame(capture: cv2.VideoCapture, timeout: float = CAPTURE_WARMUP_TIMEOUT) -> bool:
    """Block until the source delivers a frame after a resolution change, or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, _ = capture.read()
        if ok:
            return True
        time.sleep(0.1)
    return False


def open_source(source: str, calibration: CameraCalibration | None = None) -> cv2.VideoCapture:
    """Open a camera by index or a video file by path, at the resolution its calibration was captured at."""
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    # Only a live camera has a resolution to ask for; a file delivers whatever
    # it was recorded at.
    if calibration is None or not source.isdigit() or not capture.isOpened():
        return capture
    if capture_resolution(capture) != tuple(calibration.image_size):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, calibration.image_size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration.image_size[1])
        wait_for_first_frame(capture)
    return capture


# A person's position comes from intersecting the ray through their face with a
# horizontal plane at head height. The closer the camera sits to that plane, the
# shallower that intersection, and the more a pixel of detection noise moves the
# result -- at zero separation the ray never meets the plane at all.
MIN_CAMERA_HEIGHT_ABOVE_PLANE = 0.75


def draw_zone_counts(frame, resolver: ZoneResolver, zone_counts: dict[str, int]) -> None:
    """Draw the live per-zone headcount, or why it is unavailable."""
    if resolver.camera_pose is None:
        cv2.putText(frame, "zones: camera not localized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    height_above_plane = float(resolver.camera_pose[2, 3]) - resolver.head_height
    if height_above_plane < MIN_CAMERA_HEIGHT_ABOVE_PLANE:
        cv2.putText(
            frame,
            f"camera only {height_above_plane:+.2f}m above the {resolver.head_height:.2f}m plane -- positions unreliable",
            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )
    if not zone_counts:
        waiting = {z: resolver.zone_map.missing_markers(z) for z in resolver.zone_map.zone_ids}
        cv2.putText(frame, f"zones waiting on markers {waiting}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return
    for index, (zone_id, count) in enumerate(sorted(zone_counts.items())):
        cv2.putText(frame, f"{zone_id}: {count}", (10, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Gustavo Ojeda -- Predictive Assistance Alerts: new parameter to draw each detected
# person's zone dwell time on screen.
def draw_detections(
    frame,
    detections: list[dict],
    zone_ids: list[str | None] | None = None,
    world_positions: list[tuple[float, float] | None] | None = None,
    dwell_seconds: list[float] | None = None,
) -> None:
    """Draw each detected person's bounding box, predictions, resolved zone, and world position onto the frame."""
    zone_ids = zone_ids if zone_ids is not None else [None] * len(detections)
    world_positions = world_positions if world_positions is not None else [None] * len(detections)
    dwell_seconds = dwell_seconds if dwell_seconds is not None else [None] * len(detections)
    #Adding dwell_seconds parameter- Gustavo Ojeda
    for det, zone_id, position, dwell in zip(detections, zone_ids, world_positions, dwell_seconds):
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        line1 = f"{det['age_group']} ({conf['age']:.2f}) / {det['gender']} ({conf['gender']:.2f})"
        line2 = f"{det['emotion']} ({conf['emotion']:.2f})"
        cv2.putText(frame, line1, (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, line2, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        # Showing the world position alongside the zone separates "the geometry put
        # this person somewhere wrong" from "the polygon is in the wrong place",
        # which look identical from the zone label alone.
        if position is not None:
            label = f"({position[0]:+.1f}, {position[1]:+.1f})m {zone_id or 'no zone'}"
            colour = (255, 0, 0) if zone_id else (0, 165, 255)
            cv2.putText(frame, label, (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if dwell is not None:
            # Gustavo Ojeda -- Predictive Assistance Alerts: show this person's zone dwell
            # time on screen, so it's visible while testing with the live camera.
            cv2.putText(frame, f"dwell: {dwell:.1f}s", (x, y + h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def draw_counter(frame, counter: LineCounter, width: int, height: int) -> None:
    """Draw the virtual counting line and the current occupancy count onto the frame."""
    position = int(counter.position)
    if counter.axis == "x":
        cv2.line(frame, (position, 0), (position, height), (255, 0, 0), 2)
    else:
        cv2.line(frame, (0, position), (width, position), (255, 0, 0), 2)
    cv2.putText(frame, f"Occupancy: {counter.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def main() -> None:
    """Run the inference pipeline over a camera or video source until 'q', EOF, or --duration elapses."""
    args = parse_args()
    # Loaded before the source is opened, so the camera can be asked for the
    # resolution the calibration describes. The same object then feeds the pose
    # estimator, so the frames and the intrinsics cannot disagree.
    calibration = CameraCalibration.load(args.calibration) if args.calibration else None
    cap = open_source(args.source, calibration)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width, height = capture_resolution(cap)
    if calibration is not None and (width, height) != tuple(calibration.image_size):
        print(
            f"WARNING: source is running at {width}x{height} but was calibrated at "
            f"{calibration.image_size[0]}x{calibration.image_size[1]}. Focal length is in pixels, "
            "so every distance will be wrong and no detection will land inside a zone. "
            "Some cameras only run at their native resolution -- recalibrate this one there."
        )

    line_position = args.line_position
    if line_position is None:
        line_position = width / 2 if args.line_axis == "x" else height / 2

    use_bytetrack = args.tracker == "bytetrack"
    pipeline = InferencePipeline(track=use_bytetrack)
    # Only built for --tracker centroid; ByteTrack assigns ids inside the
    # detector, where it can use the confidence scores this one never sees.
    #
    # The centroid tracker's match radius is how far a person may move between frames
    # and still be the same track. It is measured in pixels, so a fixed number
    # only means something at one resolution -- the same physical motion covers
    # three times as many pixels at 1920 wide as at 640. Expressed as a
    # fraction of frame width it survives resolution changes: 0.15 is roughly
    # half a metre of lateral motion at a few metres' range, generous enough
    # for slow movement at low frame rates while staying under typical
    # person-to-person spacing so two neighbours don't swap IDs.
    tracker = None if use_bytetrack else CentroidTracker(max_distance=width * 0.15)
    registry = TrackRegistry()
    counter = LineCounter(axis=args.line_axis, position=line_position, entry_direction=args.line_direction)
    shipper = None
    if args.server_url:
        shipper = RemoteLogShipper(args.server_url, args.camera_node_id, args.api_key)
        print(f"Shipping records to {args.server_url} as camera node '{args.camera_node_id}'.")

    resolver = None
    if args.zones:
        zones = load_zones(args.zones)
        if args.marker_map is not None:
            marker_map = MarkerMap.load(args.marker_map)
            print(f"Loaded marker map from {args.marker_map}: markers {marker_map.marker_ids}, anchor {marker_map.anchor_id}.")
        else:
            marker_map = MarkerMap(
                anchor_id=args.anchor, mounting=args.marker_mounting, anchor_height=args.marker_height
            )
            print("No --marker-map given; this node will survey its own map, which will not match other nodes.")
        estimator = MarkerPoseEstimator(
            calibration,
            args.marker_size,
            allowed_ids={m for zone in zones for m in zone.marker_ids},
        )
        resolver = ZoneResolver(estimator, marker_map, ZoneMap(zones, marker_map), head_height=args.head_height)
        print(f"Zone occupancy enabled: {[z.zone_id for z in zones]} from {args.zones}.")
        # The server never sees the marker map, so it learns each zone's floor
        # shape from the nodes -- shipped once here, at startup, for the
        # dashboard's floor map.
        if shipper is not None:
            shipper.ship_zone_geometry(
                [
                    {"zone_id": zone_id, "polygon": resolver.zone_map.polygon(zone_id).tolist()}
                    for zone_id in resolver.zone_map.ready_zone_ids()
                ]
            )

    streamer = None
    if args.stream_frames:
        streamer = FrameStreamer(args.server_url, args.camera_node_id, args.api_key, max_fps=args.stream_fps)
        print(f"Streaming frames to {args.server_url} as '{args.camera_node_id}' at up to {args.stream_fps:g} fps.")

    print(f"Source opened at {width}x{height} on device: {pipeline.device}.")
    print(f"Counting line: {args.line_axis}={line_position:.0f}, entry direction: {args.line_direction}.")
    if args.benchmark:
        print(f"Benchmark mode: running for up to {args.duration:.0f}s (Ctrl+C also stops cleanly).")
    else:
        print("Press 'q' to quit.")

    frame_count = 0
    total_faces = 0
    start = time.perf_counter()
    last_report = start
    try:
        while True:
            if args.benchmark and (time.perf_counter() - start) >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            detections = pipeline.process_frame(frame)
            total_faces += len(detections)

            if tracker is None:
                # ByteTrack reports a null id for a face it has seen but not yet
                # confirmed as a track. Those are dropped rather than counted:
                # everything downstream -- occupancy, dwell, the identity vote --
                # is about people, and a detection the tracker will not vouch for
                # is not yet a person.
                confirmed = [det for det in detections if det["track_id"] is not None]
                bboxes = [det["bbox"] for det in confirmed]
                track_ids = [det["track_id"] for det in confirmed]
                detections = confirmed
            else:
                bboxes = [det["bbox"] for det in detections]
                track_ids = tracker.update(bboxes)
            timestamp = time.time()
            tracks = {track_id: bbox_centroid(bbox) for track_id, bbox in zip(track_ids, bboxes)}
            for track_id, event in counter.update(tracks, timestamp):
                print(f"  Track {track_id} {event} -- occupancy now {counter.count}")

            # Fold this frame into each person's running identity vote before
            # anything is counted or reported, so both are answered about
            # people rather than about detections.
            registry.retire(set(track_ids))
            people = [
                registry.observe(track_id, det, timestamp)
                for track_id, det in zip(track_ids, detections)
            ]

            # With zones configured, each detection carries the zone it is standing
            # in and that zone's live headcount, rather than the line counter's
            # running total -- a headcount cannot drift the way a net count can.
            zone_ids: list[str | None] = [None] * len(detections)
            zone_counts: dict[str, int] = {}
            world_positions: list[tuple[float, float] | None] = [None] * len(detections)
            if resolver is not None:
                resolver.update(frame)
                world_positions = [resolver.world_position(bbox) for bbox in bboxes]
                # Membership tests the ray across the whole plausible band of
                # face heights, so seated and standing people both land in the
                # zone; the coordinate above stays committed to one plane.
                zone_ids = [resolver.zone_for_bbox(bbox) for bbox in bboxes]
                # Occupancy counts confirmed people, so a one-frame false
                # positive never appears in a headcount it would inflate.
                zone_counts = resolver.occupancy(
                    [zone for zone, person in zip(zone_ids, people) if person.confirmed]
                )

            # One record per person per change, not one per person per frame.
            # A track is reported only once its age/gender vote has settled,
            # and then only when its emotion or zone moves, or the heartbeat
            # falls due.
            for det, track_id, zone_id, position, person in zip(
                detections, track_ids, zone_ids, world_positions, people
            ):
                if not person.should_emit(zone_id, timestamp):
                    continue
                # Gustavo Ojeda -- Predictive Assistance Alerts: use the person's own zone dwell timer
                # instead of the line counter, so dwell reflects real time in the ArUco zone.
                dwell = person.dwell_in_zone(timestamp)
                present = registry.confirmed_count()
                count = zone_counts.get(zone_id, present) if resolver is not None else present
                reported = reported_detection(det, person)
                log_detection(
                    reported, count=count, dwell_seconds=dwell, zone_id=zone_id,
                    world_position=position, track_id=person.track_id,
                )
                if shipper is not None:
                    shipper.ship(
                        reported, count=count, dwell_seconds=dwell, zone_id=zone_id,
                        world_position=position, track_id=person.track_id,
                    )
                person.mark_emitted(zone_id, timestamp)

            # Gustavo Ojeda -- Predictive Assistance Alerts: drawing happens once per frame, outside
            # the per-person emission loop above -- inside it, the overlay was only drawn on the
            # frames where someone emitted a record, which made the preview flicker.
            if not args.benchmark or streamer is not None:
                # Gustavo Ojeda -- Predictive Assistance Alerts: compute each visible
                # person's current zone dwell time, to draw it on screen for live testing.
                dwell_times = [person.dwell_in_zone(timestamp) for person in people]
                draw_detections(frame, detections, zone_ids, world_positions, dwell_times)
                if resolver is None:
                    draw_counter(frame, counter, width, height)
                else:
                    draw_zone_counts(frame, resolver, zone_counts)

            if streamer is not None:
                streamer.send(frame)

            if args.benchmark:
                now = time.perf_counter()
                if now - last_report >= 2.0:
                    running_fps = frame_count / (now - start)
                    print(f"  {frame_count} frames, {running_fps:.2f} FPS running avg, {len(detections)} faces this frame")
                    last_report = now
            else:
                cv2.imshow("RetailVision - inference pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
        cv2.destroyAllWindows()
        if shipper is not None:
            shipper.close()
        if streamer is not None:
            streamer.close()
        if frame_count:
            avg_faces = total_faces / frame_count
            print(
                f"Processed {frame_count} frames in {elapsed:.1f}s "
                f"({frame_count / elapsed:.2f} FPS, avg {avg_faces:.2f} faces/frame)"
            )


if __name__ == "__main__":
    main()
