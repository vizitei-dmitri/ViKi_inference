"""Sources never retimestamp held sensor frames as new observations."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from viki.cameras import CameraFrameSource, RecordedFrameSource
from viki.contracts import Frame, SyncedFrameGroup


class Capture:
    def __init__(self, path):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def grab(self):
        self.index += 1
        return self.index <= 4

    def read(self):
        image = np.full((2, 3, 3), self.index, np.uint8)
        self.index += 1
        return True, image

    def release(self):
        self.released = True


def recorded(tmp_path, monkeypatch):
    import viki.cameras.live as live
    monkeypatch.setattr(live.cv2, "VideoCapture", Capture)
    raw = tmp_path / "raw"
    raw.mkdir()
    intr = {"fx": 1., "fy": 1., "cx": 1., "cy": 1., "width": 3, "height": 2}
    (raw / "intrinsics.json").write_text(json.dumps({d: {"color": intr, "depth": intr}
                                                    for d in ("a", "b")}))
    rows = [{"sync_us": 100_000 + i * 33_000, "offsets_us": {"a": -1000, "b": -2000},
             "device_timestamps_us": {"a": 30_000 + i * 33_000, "b": 30_160 + i * 33_000}}
            for i in range(4)]
    (raw / "timestamps.json").write_text(json.dumps(rows))
    for device in ("a", "b"):
        (raw / f"{device}.mp4").touch()
        folder = raw / f"{device}_depth"
        folder.mkdir()
        for i in range(4):
            np.save(folder / f"{i:06d}.npy", np.full((2, 3), i, np.uint16))
    clock = [1_000_000]
    return RecordedFrameSource(tmp_path, clock_us=lambda: clock[0]), clock


def test_recorded_source_paces_groups_preserves_native_and_is_readonly(tmp_path, monkeypatch):
    source, clock = recorded(tmp_path, monkeypatch)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob("*") if p.is_file()}
    first = source.poll()
    assert source.source_summary["clock"] == {
        "host_start_us": 1_000_000, "recorded_start_us": 100_000, "playback_speed": 1.}
    assert first.sync_timestamp_us == 1_000_000
    assert first.frames["a"].host_timestamp_us == 999_000
    assert first.frames["a"].timestamp_us == 30_000
    assert source.poll() is None
    clock[0] += 33_000
    second = source.poll()
    assert second.frames["a"].timestamp_us == 63_000
    assert second.frames["a"].color[0, 0, 0] == 1
    assert second.frames["b"].depth[0, 0] == 1
    source.close()
    assert source.exhausted
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in tmp_path.rglob("*") if p.is_file()}


def test_recorded_source_discards_only_due_frames_never_peeks_future(tmp_path, monkeypatch):
    source, clock = recorded(tmp_path, monkeypatch)
    clock[0] += 75_000
    result = source.poll()
    assert result.frames["a"].color[0, 0, 0] == 2
    assert source.dropped_groups == 2
    assert result.frames["a"].host_timestamp_us <= clock[0]
    assert source.poll() is None
    clock[0] += 24_000
    assert source.poll().frames["a"].color[0, 0, 0] == 3
    assert source.exhausted


def synced(host, native):
    return SyncedFrameGroup(
        frames={d: Frame(np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.uint16),
                         native, d, host_timestamp_us=host) for d in ("a", "b")},
        sync_timestamp_us=host, offsets_us={"a": 0, "b": 0},
    )


def test_camera_source_deduplicates_native_time_does_not_stop_shared_manager(monkeypatch):
    import viki.cameras.live as live
    current = [synced(1_000_000, 10)]
    manager = SimpleNamespace(active_device_ids=lambda: ["a", "b"])
    monkeypatch.setattr(live, "MultiCameraSync", lambda *a, **k:
                        SimpleNamespace(get_synced_frame=lambda: current[0]))
    clock = [1_000_000]
    source = CameraFrameSource(manager, device_ids=["a", "b"], clock_us=lambda: clock[0])
    assert source.poll() is current[0]
    clock[0] += 33_000
    current[0] = synced(clock[0], 10)
    assert source.poll() is None
    current[0] = synced(clock[0], 11)
    assert source.poll() is current[0]
    current[0] = synced(clock[0], 9)
    with pytest.raises(RuntimeError, match="clock reset"):
        source.poll()
    source.close()
    assert source.poll() is None


def test_camera_source_stale_and_future_arrivals_are_not_fresh(monkeypatch):
    import viki.cameras.live as live
    current = [synced(700_000, 10)]
    manager = SimpleNamespace(active_device_ids=lambda: ["a", "b"])
    monkeypatch.setattr(live, "MultiCameraSync", lambda *a, **k:
                        SimpleNamespace(get_synced_frame=lambda: current[0]))
    source = CameraFrameSource(manager, device_ids=["a", "b"], clock_us=lambda: 1_000_000)
    assert source.poll() is None
    current[0] = synced(1_000_001, 11)
    assert source.poll() is None
    current[0] = synced(999_000, 12)
    assert source.poll() is current[0]
    manager.active_device_ids = lambda: ["a"]
    with pytest.raises(RuntimeError, match="required camera stopped"):
        source.poll()
