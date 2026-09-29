# RetailVision

[![CI](https://github.com/GustavoOjedaLopez/RetailVision/actions/workflows/ci.yml/badge.svg)](https://github.com/GustavoOjedaLopez/RetailVision/actions/workflows/ci.yml)

Real-time computer vision system that turns retail camera feeds into
anonymized occupancy and demographic analytics — measuring **how many
people are in a space, where they are standing, how long they stay, and
what mood they are in**, without any footage or biometric identifier
leaving the camera.

Three components, each independently runnable:

| Component | Stack | What it does |
|---|---|---|
| **Camera node** (`src/`) | Python, OpenCV, PyTorch/YOLOv8 | Runs inference at the edge; ships anonymized records only |
| **Server** (`server/`) | FastAPI, SQLAlchemy, TimescaleDB | Ingests from many nodes, persists, serves aggregates |
| **Dashboard** (`retailVision/`) | React 19, TypeScript, Vite, Tailwind | Live occupancy, floor heatmap, visit analytics |

Model accuracy, evaluation methodology, and known limitations:
**[RESULTS.md](RESULTS.md)**.

## Credits and project history

RetailVision was originally designed and built by **Pedro**
([@Pedrosan26](https://github.com/Pedrosan26/RetailVision)) at Mirai Innovation
Lab: the camera pipeline, the face, age, gender and emotion models, tracking,
ArUco marker-based zones, the server and the dashboard.

This repository continues that work. **Gustavo Ojeda**
([@GustavoOjedaLopez](https://github.com/GustavoOjedaLopez)) is extending it with
**Predictive Assistance Alerts**: a layer that detects when a customer probably
needs help (for example, staying in one zone for a long time, or moving back and
forth between zones) and notifies staff in real time, without identifying anyone.
Changes made as part of that work are marked in the code with
`# Gustavo Ojeda -- Predictive Assistance Alerts:` comments.

---

## Architecture

```
 camera 0 ──┐
 camera 1 ──┤  each node: detect → classify → track → localize
 camera 2 ──┘        │
                     │  anonymized records only (no pixels)
                     ▼
              FastAPI server ──► TimescaleDB (hypertable)
                     │
                     ▼
               React dashboard
```

**Inference runs at the edge.** Each camera node runs the full pipeline
locally and transmits only labels, counts and floor coordinates. Frames
and face crops never leave the machine, so the privacy boundary is
architectural rather than a policy applied after the fact.

**Multiple cameras share one coordinate system.** Printed ArUco markers
are pose-estimated individually (`cv2.solvePnP` plus each marker's known
printed size — the only physical measurement the system needs), then
merged into a common world frame. Two cameras need **one marker in
common**, not four, so a zone larger than any single camera's view can be
assembled from partial views. This is what lets a person seen by three
cameras be counted once.

**A person is a track, not a frame.** Age and gender are decided once per
person by majority vote over their first few frames and then frozen — the
classifiers disagree with themselves frame to frame, so voting is both
steadier and more accurate than keeping whichever answer landed last.
Records are emitted on change plus a slow heartbeat, which took the data
rate from roughly 300 records per person per minute to about 6.

---

## Quick start

The server stack -- database, API and dashboard -- runs in Docker:

```bash
cp .env.example .env        # set database credentials and camera-node API keys
docker compose up
```

| | |
|---|---|
| Dashboard | http://localhost:8080 |
| API docs | http://localhost:8000/docs |
| Database | `localhost:5433` |

The server container applies `alembic upgrade head` before it starts, so
a fresh checkout builds its own schema and an existing one is left
alone -- there is no separate migration step to forget.

Camera nodes are **not** compose services. A node needs the camera
attached to the machine it runs on, so it belongs on that hardware
rather than in a container here. Run one against the stack:

```bash
python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt

PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo \
  --source 0 \
  --server-url http://localhost:8000 --camera-node-id cam-1 --api-key <key>
```

`--source` takes a camera index or a video file path. The `--api-key`
must match an entry in `CAMERA_NODE_API_KEYS`. Press `q` to quit.

<details>
<summary><b>Running the server or dashboard on the host instead</b></summary>

Faster loop when developing either one -- the database still comes from
compose, and naming a single service starts only that service:

```bash
docker compose up -d timescaledb

cd server && python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env && ./venv/bin/alembic upgrade head
./venv/bin/uvicorn app.main:app --reload --port 8000

cd retailVision && nvm use && npm install
cp .env.example .env && npm run dev        # http://localhost:5173
```

`server/.env` points at `localhost:5433`; the compose file overrides
`DATABASE_URL` to `timescaledb:5432` for the containerized server, since
the published port only exists for host processes. The API allows both
origins, so a host dev server can talk to the containerized API.
</details>

<details>
<summary><b>Environment notes</b></summary>

- **Python 3.12** specifically — not the system Python 3.9, which is too
  old for the type syntax used throughout.
- **Node 22** (pinned in `.nvmrc`). The Vite 8 / oxlint toolchain needs
  `^20.19 || >=22.12`; an older Node does not fail loudly, it silently
  skips installing platform-specific native bindings, so `npm install`
  "succeeds" and `npm run dev` then crashes on a missing binding.
- **TimescaleDB is on host port 5433**, not 5432, to avoid clashing with
  a locally installed Postgres.
- **The dashboard is on 8080 in Docker, 5173 under `npm run dev`** — two
  ports so a container and a dev server can run side by side. The API
  allows both origins.
- `opencv-python` is pinned. A later release shipped a macOS build
  missing `cv2.CascadeClassifier` and its bundled cascade files.
  Production code no longer uses either, but the pin has not been
  re-verified against a newer release for the capture/display/video-I/O
  functionality still used throughout — test broadly before bumping it.
</details>

---

## Marker-based zones

A zone is a floor area defined by printed ArUco markers placed around
its edges. Setup is a one-time survey, and the resulting map is shared
by every node.

```bash
# 1. Calibrate each camera's intrinsics (once per physical camera)
PYTHONPATH=. ./venv/bin/python3 scripts/setup/calibrate_camera.py --source 0 --output calibration/camera_0.json

# 2. Survey the markers into a shared world frame
PYTHONPATH=. ./venv/bin/python3 scripts/setup/aruco_pose_test.py \
  --source 0 1 2 \
  --calibration calibration/camera_0.json calibration/camera_1.json calibration/camera_2.json \
  --marker-size 0.14 --anchor 3 --marker-mounting wall --marker-height 2.4 \
  --zones config/zones.json --save-map config/marker_map.json

# 3. Run nodes against the shared map
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo \
  --source 0 --zones config/zones.json --marker-map config/marker_map.json \
  --calibration calibration/camera_0.json --marker-size 0.14 --head-height 1.2 \
  --server-url http://localhost:8000 --camera-node-id cam-1 --api-key <key>
```

Three details that decide whether this works at all:

- **`--marker-height` is the anchor marker's centre height above the
  floor.** It is the datum for the entire world frame. Get it wrong and
  every zone floats at the wrong height, so nobody is ever inside one.
  The overlay warns in red if a camera computes to below the floor.
- **Camera height versus head height.** Positions come from intersecting
  a viewing ray with a horizontal plane at head height. The closer a
  camera sits to that plane, the more a small posture difference moves
  the reported position: the error scales with
  `(h_camera − h_assumed) / (h_camera − h_face)`. Mount cameras high.
- **Calibration is per-resolution.** Focal length is measured in pixels,
  so a calibration only describes a camera at the resolution it was
  captured at. Running at another resolution rescales every distance
  silently instead of failing.

Zone membership samples the viewing ray across a band of plausible face
heights rather than a single plane, so seated and standing people are
both counted. Zone polygons are the convex hull of their markers, so
markers can be listed in any order and a marker on an interior partition
cannot fold a dent into the perimeter.

---

## What the dashboard shows

- **Overview** — live per-zone occupancy (deduplicated across cameras),
  KPI strip with week-over-week deltas, per-node reporting status,
  configurable crowding alerts
- **Zones** — per-zone headcount with per-camera contributions, a
  top-down **floor heatmap** (surveyed polygon, position density, live
  person dots, zoom/pan/hover), and historical charts
- **Visits** — one row per person's stay rather than per detection
  event: arrival, duration, zone, dominant mood, plus stay-duration
  distribution and hour-of-day rhythm
- **Cameras** — one node's annotated frame at a time, full width, over a
  WebSocket. Only shows nodes started with `--stream-frames`; see
  [Privacy design](#privacy-design)

Page sections are reorderable and the arrangement persists per browser.

---

## Privacy design

The constraint is that **the analytics pipeline moves no pixels**. What
leaves a camera node as data is labels, counts and coordinates — nothing
from which a face could be reconstructed or matched:

- Inference is local; the record stream carries only labels, counts and
  floor coordinates
- Log records carry no bounding boxes, crops, or face embeddings
- `track_id` groups one person's records **within one camera and one
  process run**. It is random, survives no restart, and is not comparable
  between cameras — enough to count someone once, useless for following
  them
- Recognising the same person across cameras is solved **spatially**,
  from world position, never by appearance matching

Re-identification across visits is deliberately not implemented. It
would make several metrics better and is the one capability this
architecture rules out on purpose.

**Live view is the exception, and it is deliberately a separate channel.**
`--stream-frames` opens a WebSocket that sends the annotated preview to
the server so an operator can see what a camera is actually pointed at.
It is off unless asked for, the frames are held in the server's memory
and never written to disk or the database, and they are not part of the
record stream above — the analytics data is identical whether streaming
is on or off. A deployment that does not need to check camera aim should
leave it off, and nothing else stops working.

---

## Layout

```
src/retailvision/      camera node pipeline
  detection.py           YOLOv8 face detector, with ByteTrack association (default)
  inference.py           detector + age/gender/emotion classifiers, one call per frame
  tracking.py            centroid tracker, Hungarian matching (--tracker centroid)
  person_track.py        per-person identity voting and change-based emission
  calibration.py         intrinsics, with a self-consistency check
  marker_pose.py         per-marker 3D pose from solvePnP
  marker_map.py          shared world frame across cameras
  zones.py               floor polygons and zone membership
  counter.py             virtual-line crossing counter
  output_log.py          anonymized local record sink
  remote_log.py          batched shipping to the server
  frame_stream.py        opt-in live preview over a WebSocket
  pipeline_demo.py       wires it together

server/app/            FastAPI service
  routers/               ingest, detections, occupancy, aggregates, summary, visits,
                         zone geometry, frames (live preview WebSockets)
  models/, schemas/      SQLAlchemy ORM and Pydantic models
  dedup.py               cross-camera spatial deduplication
migrations/            Alembic

retailVision/src/      React dashboard
  api/, hooks/           typed client and TanStack Query polling
  components/, pages/    charts, occupancy, cameras, layout
  store/                 Zustand, UI state only

scripts/               dataset prep, training and evaluation that produced the models
  setup/                 calibration, marker generation, camera and marker survey --
                         the tools you run to deploy a node

docker-compose.yml     the server stack
  server/Dockerfile      migrations then uvicorn; no torch, inference is on the nodes
  retailVision/Dockerfile  Vite build served by nginx, with SPA-route fallback
```

---

## Tests

```bash
PYTHONPATH=. ./venv/bin/python3 -m unittest discover -s tests -v   # camera node
cd server && ./venv/bin/python -m pytest tests/ -v                 # server
cd retailVision && npm run build && npm run lint                   # dashboard
```

Server tests run against in-memory SQLite, which exercises the
application and ORM logic.

All three run on every push and pull request, alongside a fourth job that
builds the Docker images, brings the stack up on an empty volume and
checks the hypertable exists afterwards — which is the only place the
TimescaleDB-specific DDL actually executes, and the only way the claim
that `docker compose up` runs the system stays true.

---

## Engineering notes

**Two trackers, and why both are still here.** Faces are followed between
frames by ByteTrack (`--tracker bytetrack`, the default), which runs
inside the detector where the per-detection confidence scores still
exist. Its two-stage association matches confident detections first, then
tries the low-confidence leftovers against tracks still unmatched instead
of discarding them — a face that turns or blurs loses confidence well
before it disappears, and that second pass is what keeps it on one track.

The original centroid tracker (`--tracker centroid`) is kept as the
baseline, which is what makes the claim checkable. Measured over the
recorded evaluation clips, same frames through both:

| clip | centroid | ByteTrack |
|---|---|---|
| `side_view` | 3 tracks | **2 tracks** |
| `looking_down` | 2 tracks | 2 tracks |
| `close_1m` | 1 track | 1 track |

Fewer tracks over identical detections means fewer identity breaks. The
gain shows up only on the turned-face clip, which is the case the second
stage exists for, and it costs nothing measurable in frame rate. That is
a modest result honestly obtained, not a transformation — and worth
having, because a track that breaks in two reports one visitor as two,
restarts their dwell timer, and splits their age/gender vote.


A few problems whose solutions shaped the system:

**A good reprojection error does not mean a good calibration.** Three of
four camera calibrations reported 0.31–0.86 px error while being
physically impossible — distortion coefficients had absorbed capture
noise into enormous values. Reprojection error cannot catch this,
because it is the metric the overfit was optimised against. The fix was
a second, independent check: undistort a dense grid of points, redistort
them, and measure whether the round trip agrees. The broken models were
110–14,412 px out; the corrected ones, under 0.5 px.

**A flat square has two valid poses.** Both project to nearly identical
pixels, and choosing by reprojection error alone picked an orientation
128° wrong — separated from the correct one by 0.006 px. Resolving it
needs information the marker itself does not carry: agreement with other
markers in view, the plane they are mounted on, and continuity with the
previous frame.

**Pixel thresholds are resolution-dependent.** After moving capture to
1080p to match the calibrations, the tracker's fixed 75-pixel match
radius silently became far too tight — a seated person shifting 20 cm
moved 106 px and was issued a new identity almost every frame, so one
person read as sixty. Thresholds in pixels now scale with frame width.

**Emit on change, not per frame.** See the emotion-label consolidation
in [RESULTS.md](RESULTS.md) for the same principle applied to model
output: the honest unit of measurement is a person, not a frame.
