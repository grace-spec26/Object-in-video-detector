from scene_frame_cutter import (
    _scene_list_to_switches,
    detect_camera_switch_times,
    format_seconds_mmssmmm,
)


class FakeTimecode:
    def __init__(self, seconds, frames):
        self._seconds = seconds
        self._frames = frames

    def get_seconds(self):
        return self._seconds

    def get_frames(self):
        return self._frames


def test_format_seconds_mmssmmm_rounds_to_milliseconds():
    assert format_seconds_mmssmmm(0) == "00:00:000"
    assert format_seconds_mmssmmm(1.2344) == "00:01:234"
    assert format_seconds_mmssmmm(61.9996) == "01:02:000"
    assert format_seconds_mmssmmm(3723.456) == "62:03:456"


def test_scene_list_to_switches_skips_first_scene_start():
    scenes = [
        (FakeTimecode(0.0, 0), FakeTimecode(2.0, 48)),
        (FakeTimecode(2.0, 48), FakeTimecode(5.5, 132)),
        (FakeTimecode(5.5, 132), FakeTimecode(8.0, 192)),
    ]

    switches = _scene_list_to_switches(scenes)

    assert [switch.timestamp for switch in switches] == ["00:02:000", "00:05:500"]
    assert [switch.seconds for switch in switches] == [2.0, 5.5]
    assert [switch.frame for switch in switches] == [48, 132]
    assert [switch.scene_index for switch in switches] == [2, 3]


def test_detect_camera_switch_times_returns_formatted_timestamps(monkeypatch):
    def fake_detect_camera_switches(_video_path, **_kwargs):
        return _scene_list_to_switches(
            [
                (FakeTimecode(0, 0), FakeTimecode(12.345, 296)),
                (FakeTimecode(12.345, 296), FakeTimecode(20, 480)),
            ]
        )

    monkeypatch.setattr(
        "scene_frame_cutter.detect_camera_switches",
        fake_detect_camera_switches,
    )

    assert detect_camera_switch_times("example.mp4") == ["00:12:345"]
