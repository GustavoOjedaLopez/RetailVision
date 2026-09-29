"""
test_person_track.py

Tests for the per-person layer: that identity is voted rather than
sampled, that an unconfirmed track is never reported, and that emission
happens on change or heartbeat rather than on every frame.
"""

import unittest

from src.retailvision.person_track import (
    CONFIRM_FRAMES,
    HEARTBEAT_SECONDS,
    PersonTrack,
    TrackRegistry,
    reported_detection,
)


def detection(age_group="18-40", gender="Female", emotion="neutral") -> dict:
    """Build a minimal process_frame()-shaped detection."""
    return {"age_group": age_group, "gender": gender, "emotion": emotion, "bbox": (0, 0, 10, 10)}


class PersonTrackTests(unittest.TestCase):
    """Voting, confirmation and emission policy for a single track."""

    def test_track_is_not_reported_before_it_is_confirmed(self):
        """A track seen fewer than CONFIRM_FRAMES times emits nothing, so one-frame false positives never appear."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES - 1):
            track.observe(detection(), float(frame))
            self.assertFalse(track.confirmed)
            self.assertFalse(track.should_emit(None, float(frame)))

    def test_identity_is_the_majority_not_the_last_frame(self):
        """A single disagreeing frame does not decide age/gender."""
        track = PersonTrack(0.0)
        track.observe(detection(age_group="18-40"), 0.0)
        track.observe(detection(age_group="18-40"), 1.0)
        track.observe(detection(age_group="41-64"), 2.0)
        self.assertTrue(track.confirmed)
        self.assertEqual(track.age_group, "18-40")

    def test_identity_is_fixed_once_confirmed(self):
        """Later frames cannot move a track onto a different identity than it was reported under."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(age_group="18-40", gender="Female"), float(frame))
        for frame in range(20):
            track.observe(detection(age_group="41-64", gender="Male"), float(frame + CONFIRM_FRAMES))
        self.assertEqual(track.age_group, "18-40")
        self.assertEqual(track.gender, "Female")

    def test_emotion_follows_the_window_majority(self):
        """A single flickered frame does not change the reported emotion."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(emotion="neutral"), float(frame))
        track.observe(detection(emotion="happy"), 10.0)
        self.assertEqual(track.emotion, "neutral")

    def test_sustained_emotion_change_is_picked_up(self):
        """Enough consistent frames do move the reported emotion."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(emotion="neutral"), float(frame))
        for frame in range(10):
            track.observe(detection(emotion="happy"), float(frame + CONFIRM_FRAMES))
        self.assertEqual(track.emotion, "happy")

    def test_unchanged_track_is_quiet_until_the_heartbeat(self):
        """A settled, stationary person is not re-reported every frame."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(), float(frame))
        self.assertTrue(track.should_emit("zone-a", 5.0))
        track.mark_emitted("zone-a", 5.0)

        track.observe(detection(), 6.0)
        self.assertFalse(track.should_emit("zone-a", 6.0))
        self.assertTrue(track.should_emit("zone-a", 5.0 + HEARTBEAT_SECONDS))

    def test_zone_change_emits_immediately(self):
        """Moving between zones is reported without waiting for the heartbeat."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(), float(frame))
        track.mark_emitted("zone-a", 5.0)
        self.assertTrue(track.should_emit("zone-b", 5.1))

    def test_reported_detection_carries_the_settled_labels(self):
        """The record is built from the track's voted labels, not the current frame's."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(age_group="18-40", gender="Female", emotion="neutral"), float(frame))
        record = reported_detection(detection(age_group="65+", gender="Male", emotion="happy"), track)
        self.assertEqual(record["age_group"], "18-40")
        self.assertEqual(record["gender"], "Female")
        self.assertEqual(record["emotion"], "neutral")

    def test_dwell_in_zone_resets_on_zone_change(self):
        """dwell_in_zone measures time since the track's last zone change, not since it was first seen."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(), float(frame))
        track.mark_emitted("zone-a", 5.0)
        self.assertEqual(track.dwell_in_zone(15.0), 10.0)
        track.mark_emitted("zone-b", 20.0)
        self.assertEqual(track.dwell_in_zone(21.0), 1.0)

    def test_zone_history_records_zone_changes_only(self):
        """zone_history remembers the sequence of zones visited, not every heartbeat re-emit."""
        track = PersonTrack(0.0)
        for frame in range(CONFIRM_FRAMES):
            track.observe(detection(), float(frame))
        track.mark_emitted("zone-a", 5.0)
        track.mark_emitted("zone-a", 6.0)
        track.mark_emitted("zone-b", 7.0)
        track.mark_emitted("zone-a", 8.0)
        self.assertEqual(list(track.zone_history), ["zone-a", "zone-b", "zone-a"])


class TrackRegistryTests(unittest.TestCase):
    """Registry bookkeeping across frames."""

    def test_same_tracker_id_keeps_one_person(self):
        """Repeated sightings of a tracker ID accumulate into a single track with a stable public ID."""
        registry = TrackRegistry()
        first = registry.observe(1, detection(), 0.0)
        second = registry.observe(1, detection(), 1.0)
        self.assertIs(first, second)
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.frames, 2)

    def test_distinct_tracker_ids_are_distinct_people(self):
        """Two tracked faces get different public IDs."""
        registry = TrackRegistry()
        a = registry.observe(1, detection(), 0.0)
        b = registry.observe(2, detection(), 0.0)
        self.assertNotEqual(a.track_id, b.track_id)

    def test_retired_id_does_not_carry_over_to_a_later_person(self):
        """A reused tracker ID starts a fresh person rather than inheriting the previous one's votes."""
        registry = TrackRegistry()
        first = registry.observe(1, detection(), 0.0)
        registry.retire(set())
        second = registry.observe(1, detection(), 10.0)
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertEqual(second.frames, 1)

    def test_confirmed_count_excludes_unconfirmed_tracks(self):
        """Occupancy counts only people seen long enough to be real."""
        registry = TrackRegistry()
        for frame in range(CONFIRM_FRAMES):
            registry.observe(1, detection(), float(frame))
        registry.observe(2, detection(), 0.0)
        self.assertEqual(registry.confirmed_count(), 1)


if __name__ == "__main__":
    unittest.main()
