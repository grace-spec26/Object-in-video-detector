import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_MASK_DATA_DIR = PROJECT_ROOT / "raw-mask-data"
SUPPORTED_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_checkpoint(checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        resolved = checkpoint
    else:
        resolved = Path(__file__).resolve().parent / "weights" / "mobile_sam.pt"
    if not resolved.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint not found: {resolved}")
    return resolved


def load_predictor(checkpoint: Optional[Path] = None, device: Optional[str] = None):
    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = sam_model_registry["vit_t"](checkpoint=str(resolve_checkpoint(checkpoint)))
    model = model.to(device=resolved_device)
    model.eval()
    return SamPredictor(model)


def _clear_matching_files(output_dir: Path, patterns: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _json_ready_points(points: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points.tolist()]


def _as_point_array(points: Any, source: Path) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected an array of [x, y] points in {source}")
    if len(coords) == 0:
        raise ValueError(f"No positive points found in {source}")
    return coords


def _clamp_points(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    clamped = np.asarray(points, dtype=np.float32).copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, image_width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, image_height - 1)
    return clamped


def _extract_positive_point_sets(
    data: Any,
    source: Path,
    visible_only: bool,
) -> List[Tuple[np.ndarray, int]]:
    if isinstance(data, list):
        return [(_as_point_array(data, source), 1)]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported coordinate JSON format in {source}")

    objects = data.get("objects")
    if objects:
        point_sets: List[Tuple[np.ndarray, int]] = []
        for obj in objects:
            raw_coords = obj.get("positive_points", obj.get("point_coords", []))
            coords = _as_point_array(raw_coords, source)
            labels = np.asarray(obj.get("point_labels", []), dtype=np.int32)
            if "positive_points" not in obj and labels.ndim == 1 and len(labels) == len(coords):
                coords = coords[labels == 1]

            if visible_only and obj.get("points"):
                visible_mask = np.asarray(
                    [bool(point.get("visible", True)) for point in obj["points"]],
                    dtype=bool,
                )
                if len(visible_mask) == len(coords):
                    coords = coords[visible_mask]

            if len(coords) > 0:
                point_sets.append((coords, int(obj.get("class_id", 1))))

        if not point_sets:
            raise ValueError(f"No positive points found in {source}")
        return point_sets

    for key in ("point_coords", "points", "coordinates"):
        if key in data:
            return [(_as_point_array(data[key], source), int(data.get("class_id", 1)))]

    raise ValueError(f"Unsupported coordinate JSON format in {source}")


def generate_expanded_box_negative_points(
    positive_points: np.ndarray,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if padding_ratio < 0:
        raise ValueError("Padding ratio must be non-negative.")

    positives = _clamp_points(positive_points, image_width, image_height)
    min_x = float(np.min(positives[:, 0]))
    max_x = float(np.max(positives[:, 0]))
    min_y = float(np.min(positives[:, 1]))
    max_y = float(np.max(positives[:, 1]))

    width = max_x - min_x
    height = max_y - min_y
    pad_x = max(width * float(padding_ratio), 1.0 if width == 0 else 0.0)
    pad_y = max(height * float(padding_ratio), 1.0 if height == 0 else 0.0)

    expanded_min_x = max(0.0, min_x - pad_x)
    expanded_max_x = min(float(image_width - 1), max_x + pad_x)
    expanded_min_y = max(0.0, min_y - pad_y)
    expanded_max_y = min(float(image_height - 1), max_y + pad_y)

    return np.asarray(
        [
            [expanded_min_x, expanded_min_y],
            [expanded_max_x, expanded_min_y],
            [expanded_max_x, expanded_max_y],
            [expanded_min_x, expanded_max_y],
        ],
        dtype=np.float32,
    )


def build_augmented_prompt_object(
    positive_points: Any,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
    class_id: int = 1,
) -> Dict[str, Any]:
    positives = _clamp_points(
        _as_point_array(positive_points, Path("<coordinates>")),
        image_width=image_width,
        image_height=image_height,
    )
    negatives = generate_expanded_box_negative_points(
        positives,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
    )
    point_coords = np.concatenate([positives, negatives], axis=0)
    point_labels = [1] * len(positives) + [0] * len(negatives)

    return {
        "class_id": int(class_id),
        "positive_points": _json_ready_points(positives),
        "negative_points": _json_ready_points(negatives),
        "point_coords": _json_ready_points(point_coords),
        "point_labels": point_labels,
    }


def build_augmented_prompt_json(
    positive_points: Any,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
    class_id: int = 1,
    source_coordinate_json: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "source_coordinate_json": str(source_coordinate_json) if source_coordinate_json else None,
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "padding_ratio": float(padding_ratio),
        "objects": [
            build_augmented_prompt_object(
                positive_points=positive_points,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                class_id=class_id,
            )
        ],
    }


def _prompt_objects_from_augmented_json(
    augmented_prompt_json: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    prompt_objects: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for obj in augmented_prompt_json.get("objects", []):
        coords = _clamp_points(
            _as_point_array(obj.get("point_coords", []), Path("<augmented-coordinates>")),
            image_width=image_width,
            image_height=image_height,
        )
        labels = np.asarray(obj.get("point_labels", []), dtype=np.int32)
        if labels.ndim != 1 or len(labels) != len(coords):
            raise ValueError("point_labels length does not match point_coords")
        prompt_objects.append((coords, labels, int(obj.get("class_id", 1))))
    return prompt_objects


def prepare_coordinate_prompt_json(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    output_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
    visible_only: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    if not coordinate_json_path.exists():
        raise FileNotFoundError(f"Coordinate JSON not found: {coordinate_json_path}")

    data = json.loads(coordinate_json_path.read_text(encoding="utf-8"))
    objects = []
    for positive_points, class_id in _extract_positive_point_sets(
        data,
        source=coordinate_json_path,
        visible_only=visible_only,
    ):
        objects.append(
            build_augmented_prompt_object(
                positive_points=positive_points,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                class_id=class_id,
            )
        )

    augmented_prompt_json = {
        "source_coordinate_json": str(coordinate_json_path),
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "padding_ratio": float(padding_ratio),
        "objects": objects,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(augmented_prompt_json, indent=2),
            encoding="utf-8",
        )

    return _prompt_objects_from_augmented_json(
        augmented_prompt_json,
        image_width=image_width,
        image_height=image_height,
    )


def load_prompt_objects(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    visible_only: bool = True,
    augmented_output_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    return prepare_coordinate_prompt_json(
        coordinate_json_path=coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        output_path=augmented_output_path,
        padding_ratio=padding_ratio,
        visible_only=visible_only,
    )


def select_frames_for_target_fps(
    frame_paths: Sequence[Path],
    target_fps: float,
    source_fps: float = 30.0,
) -> List[Path]:
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than 0.")
    if source_fps <= 0:
        raise ValueError("source_fps must be greater than 0.")

    sorted_paths = sorted(frame_paths)
    stride = max(1, int(round(float(source_fps) / float(target_fps))))
    return list(sorted_paths[::stride])


def list_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in SUPPORTED_FRAME_EXTENSIONS
    )


def save_preview(frame_rgb: np.ndarray, mask: np.ndarray, preview_path: Path) -> None:
    overlay = frame_rgb.astype(np.float32).copy()
    wound = mask == 1
    ignore = mask == 255
    overlay[wound] = overlay[wound] * 0.45 + np.array([255, 40, 40], dtype=np.float32) * 0.55
    overlay[ignore] = overlay[ignore] * 0.45 + np.array([255, 220, 40], dtype=np.float32) * 0.55
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(preview_path)


def run_mobilesam_for_frame(
    predictor,
    frame_path: Path,
    coordinate_json_path: Path,
    mask_path: Path,
    preview_path: Optional[Path] = None,
    augmented_coordinate_json_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
) -> None:
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    image_height, image_width = frame_rgb.shape[:2]
    prompt_objects = load_prompt_objects(
        coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        visible_only=visible_only,
        augmented_output_path=augmented_coordinate_json_path,
        padding_ratio=padding_ratio,
    )

    output_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if not prompt_objects:
        output_mask[:, :] = 255
    else:
        predictor.set_image(frame_rgb)
        for point_coords, point_labels, class_id in prompt_objects:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords.astype(np.float32),
                point_labels=point_labels.astype(np.int32),
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            best_mask = masks[best_index].astype(bool)
            best_score = float(scores[best_index])
            if best_score < score_threshold:
                output_mask[best_mask] = 255
            else:
                output_mask[best_mask] = 1 if class_id == 1 else int(class_id)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_mask, mode="L").save(mask_path)
    if preview_path is not None:
        save_preview(frame_rgb, output_mask, preview_path)


def _make_zip_archive(source_dir: Path, archive_path: Path) -> Path:
    if archive_path.exists():
        archive_path.unlink()
    archive_base = archive_path.with_suffix("")
    created_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=source_dir.parent,
        base_dir=source_dir.name,
    )
    return Path(created_path)


def _progress_iter(items: Sequence[Path], progress: Optional[Any], desc: str) -> Iterable[Path]:
    if progress is not None and hasattr(progress, "tqdm"):
        return progress.tqdm(items, desc=desc)
    return items


def run_coordinate_prompt_folders(
    frames_dir: Path,
    coordinates_dir: Path,
    output_root: Path = DEFAULT_RAW_MASK_DATA_DIR,
    predictor=None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    target_fps: float = 5.0,
    source_fps: float = 30.0,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    frames_dir = Path(frames_dir)
    coordinates_dir = Path(coordinates_dir)
    output_root = Path(output_root)
    frames_output_dir = output_root / "frames"
    coordinates_output_dir = output_root / "coordinates"
    masks_output_dir = output_root / "mask"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    selected_frame_paths = select_frames_for_target_fps(
        frame_paths,
        target_fps=float(target_fps),
        source_fps=float(source_fps),
    )
    missing_json = [
        coordinates_dir / f"{frame_path.stem}.json"
        for frame_path in selected_frame_paths
        if not (coordinates_dir / f"{frame_path.stem}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Missing coordinate JSON for frame: {missing_json[0]}")

    _clear_matching_files(frames_output_dir, ("*.png", "*.jpg", "*.jpeg"))
    _clear_matching_files(coordinates_output_dir, ("*.json",))
    _clear_matching_files(masks_output_dir, ("*.png", "*.jpg", "*.jpeg"))

    if predictor is None:
        predictor = load_predictor(checkpoint=checkpoint, device=device)

    total = len(selected_frame_paths)
    for frame_index, frame_path in enumerate(
        _progress_iter(selected_frame_paths, progress, desc="MobileSAM frames"),
        start=1,
    ):
        if progress is not None and not hasattr(progress, "tqdm") and callable(progress):
            progress((frame_index - 1) / max(total, 1), desc="MobileSAM frames")

        copied_frame_path = frames_output_dir / frame_path.name
        shutil.copy2(frame_path, copied_frame_path)

        run_mobilesam_for_frame(
            predictor=predictor,
            frame_path=frame_path,
            coordinate_json_path=coordinates_dir / f"{frame_path.stem}.json",
            mask_path=masks_output_dir / f"{frame_path.stem}.png",
            augmented_coordinate_json_path=coordinates_output_dir / f"{frame_path.stem}.json",
            padding_ratio=float(padding_ratio),
            score_threshold=score_threshold,
            visible_only=visible_only,
        )

    if progress is not None and not hasattr(progress, "tqdm") and callable(progress):
        progress(1.0, desc="Packaging downloads")

    frames_zip = _make_zip_archive(frames_output_dir, output_root / "frames.zip")
    masks_zip = _make_zip_archive(masks_output_dir, output_root / "mask.zip")

    return {
        "processed_frames": len(selected_frame_paths),
        "frame_paths": selected_frame_paths,
        "frames_dir": frames_output_dir,
        "coordinates_dir": coordinates_output_dir,
        "masks_dir": masks_output_dir,
        "frames_zip": frames_zip,
        "masks_zip": masks_zip,
    }


def run_directory(
    frames_dir: Path,
    coordinates_dir: Path,
    masks_dir: Path,
    preview_dir: Optional[Path] = None,
    augmented_coordinates_dir: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
    dry_run: bool = False,
) -> None:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    missing_json = [
        coordinates_dir / f"{frame_path.stem}.json"
        for frame_path in frame_paths
        if not (coordinates_dir / f"{frame_path.stem}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Missing coordinate JSON for frame: {missing_json[0]}")

    if dry_run:
        print(f"frames={len(frame_paths)}")
        print(f"frames_dir={frames_dir}")
        print(f"coordinates_dir={coordinates_dir}")
        print(f"masks_dir={masks_dir}")
        print(f"preview_dir={preview_dir}")
        print(f"augmented_coordinates_dir={augmented_coordinates_dir}")
        return

    if augmented_coordinates_dir is None:
        augmented_coordinates_dir = DEFAULT_RAW_MASK_DATA_DIR / "coordinates"

    predictor = load_predictor(checkpoint=checkpoint, device=device)
    for frame_path in frame_paths:
        json_path = coordinates_dir / f"{frame_path.stem}.json"
        mask_path = masks_dir / f"{frame_path.stem}.png"
        preview_path = preview_dir / f"{frame_path.stem}.jpg" if preview_dir else None
        augmented_path = augmented_coordinates_dir / f"{frame_path.stem}.json"
        run_mobilesam_for_frame(
            predictor=predictor,
            frame_path=frame_path,
            coordinate_json_path=json_path,
            mask_path=mask_path,
            preview_path=preview_path,
            augmented_coordinate_json_path=augmented_path,
            padding_ratio=padding_ratio,
            score_threshold=score_threshold,
            visible_only=visible_only,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MobileSAM wound masks from raw frames and CoTracker coordinate JSON."
    )
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing raw exported frames.")
    parser.add_argument("--coordinates-dir", required=True, type=Path, help="Directory containing matching JSON prompts.")
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=DEFAULT_RAW_MASK_DATA_DIR / "mask",
        help="Output directory for single-channel PNG masks.",
    )
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional output directory for masked previews.")
    parser.add_argument(
        "--augmented-coordinates-dir",
        type=Path,
        default=DEFAULT_RAW_MASK_DATA_DIR / "coordinates",
        help="Output directory for augmented prompt JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="If set, copy sampled frames, augmented coordinates, masks, and zip downloads under this folder.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional sampled processing FPS. Uses --source-fps to compute the frame stride.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Source frame rate for target FPS sampling. Defaults to 30.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Expanded-box padding ratio used to create four negative corner prompts.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional MobileSAM checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Masks below this score are encoded as 255 ignore instead of wound.",
    )
    parser.add_argument(
        "--include-invisible-points",
        action="store_true",
        help="Use points marked visible=false in the coordinate JSON.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate frame/JSON matching without loading MobileSAM.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.output_root is not None or args.target_fps is not None:
        if args.dry_run:
            frame_paths = list_frame_paths(args.frames_dir)
            selected_frame_paths = select_frames_for_target_fps(
                frame_paths,
                target_fps=args.target_fps or args.source_fps,
                source_fps=args.source_fps,
            )
            print(f"frames={len(frame_paths)}")
            print(f"selected_frames={len(selected_frame_paths)}")
            print(f"frames_dir={args.frames_dir}")
            print(f"coordinates_dir={args.coordinates_dir}")
            print(f"output_root={args.output_root or DEFAULT_RAW_MASK_DATA_DIR}")
            print(f"target_fps={args.target_fps or args.source_fps}")
            return

        result = run_coordinate_prompt_folders(
            frames_dir=args.frames_dir,
            coordinates_dir=args.coordinates_dir,
            output_root=args.output_root or DEFAULT_RAW_MASK_DATA_DIR,
            checkpoint=args.checkpoint,
            device=args.device,
            target_fps=args.target_fps or args.source_fps,
            source_fps=args.source_fps,
            padding_ratio=args.padding_ratio,
            score_threshold=args.score_threshold,
            visible_only=not args.include_invisible_points,
        )
        print(f"processed_frames={result['processed_frames']}")
        print(f"frames_dir={result['frames_dir']}")
        print(f"coordinates_dir={result['coordinates_dir']}")
        print(f"masks_dir={result['masks_dir']}")
        print(f"frames_zip={result['frames_zip']}")
        print(f"masks_zip={result['masks_zip']}")
        return

    run_directory(
        frames_dir=args.frames_dir,
        coordinates_dir=args.coordinates_dir,
        masks_dir=args.masks_dir,
        preview_dir=args.preview_dir,
        augmented_coordinates_dir=args.augmented_coordinates_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        padding_ratio=args.padding_ratio,
        score_threshold=args.score_threshold,
        visible_only=not args.include_invisible_points,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
