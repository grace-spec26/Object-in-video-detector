import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


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


def load_prompt_objects(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    visible_only: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    if not coordinate_json_path.exists():
        raise FileNotFoundError(f"Coordinate JSON not found: {coordinate_json_path}")

    data = json.loads(coordinate_json_path.read_text(encoding="utf-8"))
    objects = data.get("objects", [])
    if not objects:
        raise ValueError(f"No objects found in coordinate JSON: {coordinate_json_path}")

    prompt_objects: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for obj in objects:
        coords = np.asarray(obj.get("point_coords", []), dtype=np.float32)
        labels = np.asarray(obj.get("point_labels", []), dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Invalid point_coords in {coordinate_json_path}")
        if labels.ndim != 1 or len(labels) != len(coords):
            raise ValueError(f"point_labels length does not match point_coords in {coordinate_json_path}")

        # Coordinates are pixel-coordinate (x, y). Labels use 1=wound foreground,
        # 0=nearby non-wound background.
        if visible_only and obj.get("points"):
            visible_mask = np.asarray(
                [bool(point.get("visible", True)) for point in obj["points"]],
                dtype=bool,
            )
            if len(visible_mask) == len(coords):
                coords = coords[visible_mask]
                labels = labels[visible_mask]

        if len(coords) == 0:
            continue

        coords[:, 0] = np.clip(coords[:, 0], 0, image_width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, image_height - 1)
        prompt_objects.append((coords, labels, int(obj.get("class_id", 1))))

    return prompt_objects


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


def run_directory(
    frames_dir: Path,
    coordinates_dir: Path,
    masks_dir: Path,
    preview_dir: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    score_threshold: float = 0.0,
    visible_only: bool = True,
    dry_run: bool = False,
) -> None:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
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
        return

    predictor = load_predictor(checkpoint=checkpoint, device=device)
    for frame_path in frame_paths:
        json_path = coordinates_dir / f"{frame_path.stem}.json"
        mask_path = masks_dir / f"{frame_path.stem}.png"
        preview_path = preview_dir / f"{frame_path.stem}.jpg" if preview_dir else None
        run_mobilesam_for_frame(
            predictor=predictor,
            frame_path=frame_path,
            coordinate_json_path=json_path,
            mask_path=mask_path,
            preview_path=preview_path,
            score_threshold=score_threshold,
            visible_only=visible_only,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MobileSAM wound masks from raw frames and CoTracker coordinate JSON."
    )
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing raw exported frames.")
    parser.add_argument("--coordinates-dir", required=True, type=Path, help="Directory containing matching JSON prompts.")
    parser.add_argument("--masks-dir", required=True, type=Path, help="Output directory for single-channel PNG masks.")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional output directory for masked previews.")
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
    run_directory(
        frames_dir=args.frames_dir,
        coordinates_dir=args.coordinates_dir,
        masks_dir=args.masks_dir,
        preview_dir=args.preview_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        visible_only=not args.include_invisible_points,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
