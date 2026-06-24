import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def parse_points(points_text: Optional[str]) -> List[Point]:
    """Parse CLI points in x,y;x,y format into pixel-coordinate (x, y) tuples."""
    if not points_text:
        return []

    points: List[Point] = []
    for chunk in points_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = [piece.strip() for piece in chunk.split(",")]
        if len(pieces) != 2:
            raise ValueError(
                f"Invalid point '{chunk}'. Expected points in 'x,y;x,y' format."
            )
        try:
            x, y = float(pieces[0]), float(pieces[1])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric point '{chunk}'.") from exc
        points.append((x, y))
    return points


def generate_negative_points_from_positive_points(
    positive_points: Sequence[Point],
    image_width: int,
    image_height: int,
    radius: float = 20,
    num_negative_per_positive: int = 1,
) -> List[Point]:
    """Generate nearby negative prompts in pixel-coordinate (x, y) format.

    Labels are not returned here; callers should assign label 0 to all generated
    points. Candidates are sampled around each positive point, kept inside image
    bounds, and rejected if they are too close to any positive prompt.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if num_negative_per_positive < 0:
        raise ValueError("num_negative_per_positive must be non-negative.")

    positives = [(float(x), float(y)) for x, y in positive_points]
    if not positives or num_negative_per_positive == 0:
        return []

    min_distance = max(4.0, radius * 0.75)
    angles = [0, math.pi, math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]
    radii = [radius, radius * 1.5, radius * 2.0, radius * 2.5]
    negatives: List[Point] = []

    def inside(x: float, y: float) -> bool:
        return 0 <= x <= image_width - 1 and 0 <= y <= image_height - 1

    def far_from_positives(x: float, y: float) -> bool:
        return all(math.hypot(x - px, y - py) >= min_distance for px, py in positives)

    for px, py in positives:
        point_negatives: List[Point] = []
        for candidate_radius in radii:
            for angle in angles:
                if len(point_negatives) >= num_negative_per_positive:
                    break
                x = px + candidate_radius * math.cos(angle)
                y = py + candidate_radius * math.sin(angle)
                if inside(x, y) and far_from_positives(x, y):
                    point_negatives.append((float(x), float(y)))
            if len(point_negatives) >= num_negative_per_positive:
                break

        # Last-resort bounded candidates for prompts near image edges.
        if len(point_negatives) < num_negative_per_positive:
            fallback_candidates = [
                (min(image_width - 1.0, px + radius), py),
                (max(0.0, px - radius), py),
                (px, min(image_height - 1.0, py + radius)),
                (px, max(0.0, py - radius)),
            ]
            for x, y in fallback_candidates:
                if len(point_negatives) >= num_negative_per_positive:
                    break
                if inside(x, y) and far_from_positives(x, y):
                    point_negatives.append((float(x), float(y)))

        negatives.extend(point_negatives[:num_negative_per_positive])

    return negatives


def read_video_rgb(video_path: Path) -> np.ndarray:
    """Read a video into an RGB uint8 array shaped (T, H, W, 3)."""
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")
    return np.stack(frames, axis=0)


def load_cotracker_model(checkpoint: Optional[Path], device: str):
    """Load CoTracker3 offline predictor with either a local checkpoint or hub weights."""
    import torch

    if checkpoint:
        from cotracker.predictor import CoTrackerPredictor

        if not checkpoint.exists():
            raise FileNotFoundError(f"CoTracker checkpoint not found: {checkpoint}")
        model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60)
    else:
        from hubconf import cotracker3_offline

        model = cotracker3_offline(pretrained=True)

    model = model.to(device)
    model.eval()
    return model


def track_points(
    frames_rgb: np.ndarray,
    points: Sequence[Point],
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track initial frame points through a video.

    Inputs and outputs use pixel-coordinate (x, y) format. Returned tracks have
    shape (T, N, 2), and returned visibilities have shape (T, N).
    """
    import torch

    if not points:
        raise ValueError("At least one prompt point is required for tracking.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    video_tensor = (
        torch.from_numpy(frames_rgb)
        .permute(0, 3, 1, 2)[None]
        .to(resolved_device, dtype)
    )
    queries = torch.tensor(
        [[[0.0, float(x), float(y)] for x, y in points]],
        dtype=dtype,
        device=resolved_device,
    )

    model = load_cotracker_model(checkpoint, resolved_device)
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video_tensor, queries=queries, grid_size=0)

    return pred_tracks[0].detach().cpu().numpy(), pred_visibility[0].detach().cpu().numpy()


def make_frame_record(
    frame_index: int,
    frame_path: Path,
    coords: np.ndarray,
    labels: Sequence[int],
    visible: Sequence[bool],
    image_width: int,
    image_height: int,
    object_id: str = "wound_1",
    class_id: int = 1,
) -> dict:
    clipped_coords = []
    point_records = []
    for (x, y), label, is_visible in zip(coords, labels, visible):
        x = float(np.clip(x, 0, image_width - 1))
        y = float(np.clip(y, 0, image_height - 1))
        label = int(label)
        is_visible = bool(is_visible)
        clipped_coords.append([x, y])
        point_records.append(
            {
                "x": x,
                "y": y,
                "label": label,
                "type": "positive" if label == 1 else "negative",
                "visible": is_visible,
                "confidence": 1.0 if is_visible else 0.0,
            }
        )

    return {
        "frame_index": int(frame_index),
        "frame_path": frame_path.as_posix(),
        "objects": [
            {
                "object_id": object_id,
                "class_id": int(class_id),
                # MobileSAM expects point coordinates in pixel-coordinate (x, y)
                # format and labels where 1=foreground wound, 0=background.
                "point_coords": clipped_coords,
                "point_labels": [int(label) for label in labels],
                "points": point_records,
            }
        ],
    }


def export_cotracker_data(
    video_path: Path,
    positive_points: Sequence[Point],
    output_dir: Path,
    negative_points: Optional[Sequence[Point]] = None,
    stride: int = 1,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    negative_radius: float = 20,
    num_negative_per_positive: int = 1,
) -> None:
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if not positive_points:
        raise ValueError("At least one positive point is required.")

    frames_rgb = read_video_rgb(video_path)
    total_frames, image_height, image_width = frames_rgb.shape[:3]
    negatives = list(negative_points or [])
    if not negatives:
        negatives = generate_negative_points_from_positive_points(
            positive_points,
            image_width=image_width,
            image_height=image_height,
            radius=negative_radius,
            num_negative_per_positive=num_negative_per_positive,
        )

    all_points: List[Point] = list(positive_points) + negatives
    labels = [1] * len(positive_points) + [0] * len(negatives)
    tracks, visibilities = track_points(frames_rgb, all_points, checkpoint, device)

    frame_dir = output_dir / "frame"
    coordinates_dir = output_dir / "coordinates"
    frame_dir.mkdir(parents=True, exist_ok=True)
    coordinates_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    for frame_index in range(0, total_frames, stride):
        frame_name = f"frame_{frame_index:06d}.jpg"
        frame_path = frame_dir / frame_name
        frame_bgr = cv2.cvtColor(frames_rgb[frame_index], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        record = make_frame_record(
            frame_index=frame_index,
            frame_path=frame_path,
            coords=tracks[frame_index],
            labels=labels,
            visible=visibilities[frame_index],
            image_width=image_width,
            image_height=image_height,
        )
        json_path = coordinates_dir / f"frame_{frame_index:06d}.json"
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def dry_run(video_path: Path, positive_points: Sequence[Point], negative_points: Sequence[Point]) -> None:
    frames_rgb = read_video_rgb(video_path)
    _, image_height, image_width = frames_rgb.shape[:3]
    negatives = list(negative_points) or generate_negative_points_from_positive_points(
        positive_points, image_width, image_height
    )
    print(f"video={video_path}")
    print(f"frames={frames_rgb.shape[0]} size={image_width}x{image_height}")
    print(f"positive_points={list(positive_points)}")
    print(f"negative_points={negatives}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track wound prompt points with CoTracker and export frames plus JSON coordinates."
    )
    parser.add_argument("--video", required=True, type=Path, help="Raw input video path.")
    parser.add_argument(
        "--positive-points",
        required=True,
        help="Positive wound foreground points in 'x,y;x,y' pixel-coordinate format.",
    )
    parser.add_argument(
        "--negative-points",
        default=None,
        help="Optional negative background points in 'x,y;x,y' pixel-coordinate format.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output data directory.")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth frame.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local CoTracker checkpoint.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    parser.add_argument("--negative-radius", type=float, default=20, help="Radius for generated negative points.")
    parser.add_argument(
        "--num-negative-per-positive",
        type=int,
        default=1,
        help="Number of generated negative points for each positive point.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading CoTracker.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    positive_points = parse_points(args.positive_points)
    negative_points = parse_points(args.negative_points)

    if args.dry_run:
        dry_run(args.video, positive_points, negative_points)
        return

    export_cotracker_data(
        video_path=args.video,
        positive_points=positive_points,
        negative_points=negative_points,
        output_dir=args.output_dir,
        stride=args.stride,
        checkpoint=args.checkpoint,
        device=args.device,
        negative_radius=args.negative_radius,
        num_negative_per_positive=args.num_negative_per_positive,
    )


if __name__ == "__main__":
    main()
