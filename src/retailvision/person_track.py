"""
person_track.py

Turns the per-frame stream of detections into a per-person stream of events.

The pipeline detects faces frame by frame, so a single person sitting still
produces one detection per frame for as long as they are there. Logging each
one asks the same question hundreds of times and records hundreds of answers
to it, which makes a person who stayed a long time indistinguishable from a
crowd who passed through quickly, and makes any count of "people" a count of
frames instead.

Two things fix that, and both need a notion of a person rather than a
detection:

Identity is voted, not sampled. The age and gender classifiers disagree with
themselves between frames -- the same face can read 18-40 and then 41-64 --
so a track's labels are decided by majority over its first few frames and
then fixed. That is more accurate than re-asking every frame and keeping
whichever answer happened to land last, as well as far cheaper. A track is
not reported at all until it has been seen enough times to vote, which also
discards the single-frame false positives that never survive to a second
frame.

Emission is by change, not by frame. Emotion is the field that genuinely
moves, so a record is emitted when a track's smoothed emotion or its zone
changes, and otherwise on a slow heartbeat so a stationary person still reads
as present. Emotion is smoothed over a short window first: raw per-frame
predictions flicker, and emitting on every flicker would reproduce the
per-frame stream this exists to avoid.

Track IDs are random per track and per process run. They are stable enough to
group one person's records together while they are in view and nothing more --
they carry no identity, survive no restart, and are not comparable between
cameras. Deciding that one person is in two cameras at once remains the
server's spatial job (server/app/dedup.py), which is what the shared world
frame is for.
"""

from __future__ import annotations

import uuid
from collections import Counter, deque

# Frames a track must be seen for before it is reported at all. Also the
# number of votes its age/gender are decided from, since they are gathered
# over exactly this period. Three is enough to outvote a single bad frame
# without holding a real person back for long even at low frame rates.
CONFIRM_FRAMES = 3

# How many recent frames the reported emotion is the majority of. Wider is
# steadier but slower to react to a real change.
EMOTION_WINDOW = 5

# A present, unchanging person still emits this often, so the server can tell
# "still here, still neutral" from "gone".
HEARTBEAT_SECONDS = 10.0

# Gustavo Ojeda -- Predictive Assistance Alerts: how many recent zones to remember per
# track, so back-and-forth movement between zones can be detected.
#This will remember the 5 last zones where each person has passed
ZONE_HISTORY_LENGTH = 5

# Gustavo Ojeda -- Predictive Assistance Alerts: how long someone must be seen outside their
# zone before it counts as leaving it. Shorter absences (leaning, turning the head, a
# noisy position estimate) are ignored, so they no longer restart the dwell timer.
ZONE_GRACE_SECONDS = 3.0


class PersonTrack:
    """One tracked person: votes on their identity, smooths their emotion, and remembers what was last reported."""

    def __init__(self, timestamp: float) -> None:
        """Start an unconfirmed track that has not yet been reported."""
        self.track_id = uuid.uuid4().hex[:12]
        self.frames = 0
        self.first_seen = timestamp
        self._age_votes: Counter[str] = Counter()
        self._gender_votes: Counter[str] = Counter()
        self._emotions: deque[str] = deque(maxlen=EMOTION_WINDOW)
        self.age_group: str | None = None
        self.gender: str | None = None
        self.last_emitted_at: float | None = None
        self.last_emitted_emotion: str | None = None
        self.last_emitted_zone: str | None = None
        # Gustavo Ojeda -- Predictive Assistance Alerts: zone dwell timer (when the track entered
        # its current zone) and a short history of recently visited zones, for measuring
        # how long someone stays in a zone.
        self.zone_entered_at: float = timestamp
        self.zone_history: deque[str] = deque(maxlen=ZONE_HISTORY_LENGTH)
        # Gustavo Ojeda -- Predictive Assistance Alerts: the zone the dwell timer belongs to, and
        # when the track was first seen somewhere else (None while it is still in that zone).
        self.dwell_zone: str | None = None
        self._away_since: float | None = None

    @property
    def confirmed(self) -> bool:
        """True once the track has been seen enough times for its identity to be settled."""
        return self.age_group is not None

    def observe(self, detection: dict, timestamp: float) -> None:
        """Fold one frame's prediction into the track's votes and emotion window."""
        self.frames += 1
        self._emotions.append(detection["emotion"])
        # Votes stop being collected once the identity is fixed, so a long
        # track cannot drift onto a different answer than it was reported under.
        if self.confirmed:
            return
        self._age_votes[detection["age_group"]] += 1
        self._gender_votes[detection["gender"]] += 1
        if self.frames >= CONFIRM_FRAMES:
            self.age_group = self._age_votes.most_common(1)[0][0]
            self.gender = self._gender_votes.most_common(1)[0][0]

    @property
    def emotion(self) -> str:
        """The majority emotion over the recent window, which is steadier than any single frame."""
        return Counter(self._emotions).most_common(1)[0][0]

    def should_emit(self, zone_id: str | None, timestamp: float) -> bool:
        """Whether this track has changed, or gone quiet for long enough, to be worth a record."""
        if not self.confirmed:
            return False
        if self.last_emitted_at is None:
            return True
        if self.emotion != self.last_emitted_emotion or zone_id != self.last_emitted_zone:
            return True
        return timestamp - self.last_emitted_at >= HEARTBEAT_SECONDS

    def mark_emitted(self, zone_id: str | None, timestamp: float) -> None:
        """Record what was just reported, so the next frame is compared against it."""
        # Gustavo Ojeda -- Predictive Assistance Alerts: the dwell timer and zone history used to be
        # updated here; they moved to observe_zone(), which sees every frame and can apply
        # the grace period.
        self.last_emitted_at = timestamp
        self.last_emitted_emotion = self.emotion
        self.last_emitted_zone = zone_id

    # Gustavo Ojeda -- Predictive Assistance Alerts: called once per frame with the zone the
    # person is in right now. Only a change that lasts ZONE_GRACE_SECONDS restarts the
    # dwell timer; a shorter absence is forgotten as soon as they are back.
    def observe_zone(self, zone_id: str | None, timestamp: float) -> None:
        """Fold this frame's zone into the dwell timer, ignoring zone changes shorter than the grace period."""
        if zone_id == self.dwell_zone:
            self._away_since = None
            return
        if self._away_since is None:
            self._away_since = timestamp
            return
        if timestamp - self._away_since >= ZONE_GRACE_SECONDS:
            self.dwell_zone = zone_id
            self.zone_entered_at = self._away_since
            self._away_since = None
            if zone_id is not None:
                self.zone_history.append(zone_id)

    # Gustavo Ojeda -- Predictive Assistance Alerts: exposes the zone dwell timer so callers
    # can read how long this track has continuously been in its current zone.
    def dwell_in_zone(self, timestamp: float)-> float:
        """How many senconds this track has continuously been in its current zone"""
        return timestamp - self.zone_entered_at


class TrackRegistry:
    """Holds a PersonTrack per live tracker ID and decides which of them are worth reporting this frame."""

    def __init__(self) -> None:
        """Start with no tracks."""
        self._tracks: dict[int, PersonTrack] = {}

    def observe(self, track_id: int, detection: dict, timestamp: float) -> PersonTrack:
        """Fold a detection into its track, creating the track if this is the first sighting."""
        track = self._tracks.get(track_id)
        if track is None:
            track = PersonTrack(timestamp)
            self._tracks[track_id] = track
        track.observe(detection, timestamp)
        return track

    def retire(self, live_ids: set[int]) -> None:
        """Forget tracks the tracker no longer reports, so their IDs are not reused by a later person."""
        for track_id in [t for t in self._tracks if t not in live_ids]:
            del self._tracks[track_id]

    def confirmed_count(self) -> int:
        """How many tracked people have been seen long enough to count as present."""
        return sum(1 for track in self._tracks.values() if track.confirmed)


def reported_detection(detection: dict, track: PersonTrack) -> dict:
    """Return the detection with its per-frame labels replaced by the track's settled ones."""
    return {
        **detection,
        "age_group": track.age_group,
        "gender": track.gender,
        "emotion": track.emotion,
    }
