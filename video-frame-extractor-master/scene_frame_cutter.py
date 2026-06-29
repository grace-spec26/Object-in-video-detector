#!/usr/bin/env python3
"""Detect camera switches with PySceneDetect adaptive detection.

The public helper `detect_camera_switch_times()` returns switch timestamps in
MM:SS:MMM format, matching the format used by the browser extractor.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CameraSwitch:
    """A detected camera/shot switch."""

    scene_index: int
    seconds: float
    timestamp: str
    frame: Optional[int] = None
    frame_path: Optional[str] = None


def format_seconds_mmssmmm(seconds: float) -> str:
    """Format seconds as MM:SS:MMM.

    Hours are folded into the minute field so the output always follows the
    requested two-colon shape, e.g. 3723.456 -> 62:03:456.
    """

    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    total_millis = int(round(seconds * 1000.0))
    total_seconds, millis = divmod(total_millis, 1000)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}:{millis:03d}"


def _timecode_seconds(timecode: object) -> float:
    if hasattr(timecode, "get_seconds"):
        return float(timecode.get_seconds())
    if hasattr(timecode, "seconds"):
        return float(getattr(timecode, "seconds"))
    return float(timecode)


def _timecode_frame(timecode: object) -> Optional[int]:
    if hasattr(timecode, "get_frames"):
        return int(timecode.get_frames())
    for attr in ("frame_num", "frames"):
        if hasattr(timecode, attr):
            return int(getattr(timecode, attr))
    return None


def _scene_list_to_switches(scene_list: Sequence[Tuple[object, object]]) -> List[CameraSwitch]:
    """Convert PySceneDetect scenes into switch timestamps.

    PySceneDetect returns scenes as (start, end). The start of every scene after
    the first is a camera/shot switch.
    """

    switches: List[CameraSwitch] = []
    for scene_index, (scene_start, _scene_end) in enumerate(scene_list[1:], start=2):
        seconds = _timecode_seconds(scene_start)
        switches.append(
            CameraSwitch(
                scene_index=scene_index,
                seconds=seconds,
                timestamp=format_seconds_mmssmmm(seconds),
                frame=_timecode_frame(scene_start),
            )
        )
    return switches


def detect_camera_switches(
    video_path: str | Path,
    *,
    adaptive_threshold: float = 3.0,
    min_scene_len: int | float | str = "0.5s",
    window_width: int = 2,
    min_content_val: float = 15.0,
    show_progress: bool = False,
) -> List[CameraSwitch]:
    """Detect camera switches in a video using PySceneDetect AdaptiveDetector.

    Returns one item per detected switch. Each item includes the exact timestamp
    in MM:SS:MMM plus the raw seconds value for downstream processing.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is required. Install it with "
            "`python -m pip install -r video-frame-extractor-master/requirements.txt`."
        ) from exc

    scenes = detect(
        str(path),
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len,
            window_width=window_width,
            min_content_val=min_content_val,
        ),
        show_progress=show_progress,
        start_in_scene=True,
    )
    return _scene_list_to_switches(scenes)


def detect_camera_switch_times(video_path: str | Path, **kwargs: object) -> List[str]:
    """Return detected camera switch timestamps in MM:SS:MMM format."""

    return [switch.timestamp for switch in detect_camera_switches(video_path, **kwargs)]


def cut_frames_at_switches(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    image_extension: str = "jpg",
    jpeg_quality: int = 95,
    prefix: str = "switch",
    **detect_kwargs: object,
) -> List[CameraSwitch]:
    """Save one frame at each detected camera switch and return metadata."""

    switches = detect_camera_switches(video_path, **detect_kwargs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to cut frames. Install opencv-python.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    try:
        saved: List[CameraSwitch] = []
        extension = image_extension.lower().lstrip(".")
        for index, switch in enumerate(switches, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, switch.seconds * 1000.0))
            ok, frame = capture.read()
            if not ok:
                continue

            frame_name = f"{prefix}_{index:04d}_{switch.timestamp.replace(':', '-')}.{extension}"
            frame_path = output_path / frame_name
            write_params: List[int] = []
            if extension in {"jpg", "jpeg"}:
                write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            if not cv2.imwrite(str(frame_path), frame, write_params):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            saved.append(CameraSwitch(**{**asdict(switch), "frame_path": str(frame_path)}))
        return saved
    finally:
        capture.release()


def write_switches_csv(switches: Iterable[CameraSwitch], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["scene_index", "seconds", "timestamp", "frame", "frame_path"],
        )
        writer.writeheader()
        for switch in switches:
            writer.writerow(asdict(switch))


def write_switches_json(switches: Iterable[CameraSwitch], output_path: str | Path) -> None:
    payload = [asdict(switch) for switch in switches]
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect camera switches with PySceneDetect adaptive detection."
    )
    parser.add_argument("video", help="Input video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Optional directory to save one frame at each detected switch.",
    )
    parser.add_argument("--json", help="Optional JSON metadata output path.")
    parser.add_argument("--csv", help="Optional CSV metadata output path.")
    parser.add_argument("--adaptive-threshold", type=float, default=3.0)
    parser.add_argument("--min-scene-len", default="0.5s")
    parser.add_argument("--window-width", type=int, default=2)
    parser.add_argument("--min-content-val", type=float, default=15.0)
    parser.add_argument("--image-extension", default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    detect_kwargs = {
        "adaptive_threshold": args.adaptive_threshold,
        "min_scene_len": args.min_scene_len,
        "window_width": args.window_width,
        "min_content_val": args.min_content_val,
        "show_progress": args.show_progress,
    }

    if args.output_dir:
        switches = cut_frames_at_switches(
            args.video,
            args.output_dir,
            image_extension=args.image_extension,
            jpeg_quality=args.jpeg_quality,
            **detect_kwargs,
        )
    else:
        switches = detect_camera_switches(args.video, **detect_kwargs)

    for switch in switches:
        suffix = f" frame={switch.frame}" if switch.frame is not None else ""
        print(f"{switch.timestamp} ({switch.seconds:.3f}s){suffix}")

    if args.json:
        write_switches_json(switches, args.json)
    if args.csv:
        write_switches_csv(switches, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
