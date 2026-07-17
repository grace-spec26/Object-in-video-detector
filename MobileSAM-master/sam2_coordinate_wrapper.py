import argparse
import json
import shutil
import ssl
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from mobilesam_coordinate_wrapper import (
    DEFAULT_MAX_MASK_AREA_RATIO,
    DEFAULT_MIN_NEGATIVE_DISTANCE,
    DEFAULT_MIN_PADDING_PX,
    DEFAULT_NEGATIVE_MODE,
    DEFAULT_PADDING_RATIO,
    DEFAULT_RAW_MASK_DATA_DIR,
    FRAME_BOUNDARY_SKIP,
    NEGATIVE_MODES,
    SUPPORTED_FRAME_EXTENSIONS,
    build_augmented_prompt_object,
    format_coordinate_progress_html,
    prepare_coordinate_prompt_json,
    save_preview,
    select_best_mask,
    select_frames_for_frame_step,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_SOURCE_COORDINATES_DIR = PROJECT_ROOT / "data" / "coordinates"
SAM2_REPO_ROOT = PROJECT_ROOT / "sam2"
SAM2_CHECKPOINTS_DIR = SAM2_REPO_ROOT / "checkpoints"
SAM2_MODEL_DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
DEFAULT_SAM2_MODEL = "sam2.1_hiera_small.pt"
SAM2_MODEL_OPTIONS = {
    "sam2.1_hiera_tiny.pt": {
        "label": "SAM2.1 Hiera Tiny",
        "checkpoint_name": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": f"{SAM2_MODEL_DOWNLOAD_BASE_URL}/sam2.1_hiera_tiny.pt",
        "expected_size": 156008466,
    },
    "sam2.1_hiera_small.pt": {
        "label": "SAM2.1 Hiera Small",
        "checkpoint_name": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": f"{SAM2_MODEL_DOWNLOAD_BASE_URL}/sam2.1_hiera_small.pt",
        "expected_size": 184416285,
    },
    "sam2.1_hiera_base_plus.pt": {
        "label": "SAM2.1 Hiera Base Plus",
        "checkpoint_name": "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "url": f"{SAM2_MODEL_DOWNLOAD_BASE_URL}/sam2.1_hiera_base_plus.pt",
        "expected_size": 323606802,
    },
    "sam2.1_hiera_large.pt": {
        "label": "SAM2.1 Hiera Large",
        "checkpoint_name": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "url": f"{SAM2_MODEL_DOWNLOAD_BASE_URL}/sam2.1_hiera_large.pt",
        "expected_size": 898083611,
    },
}
SAM2_MODEL_CHOICES = tuple(SAM2_MODEL_OPTIONS.keys())
DEFAULT_SAM2_CHECKPOINT = SAM2_CHECKPOINTS_DIR / DEFAULT_SAM2_MODEL
DEFAULT_SAM2_CONFIG = SAM2_MODEL_OPTIONS[DEFAULT_SAM2_MODEL]["config"]


def resolve_sam2_model_option(model_name: Optional[str] = None) -> Dict[str, Any]:
    selected_model = str(model_name or DEFAULT_SAM2_MODEL)
    if selected_model not in SAM2_MODEL_OPTIONS:
        allowed = ", ".join(SAM2_MODEL_CHOICES)
        raise ValueError(f"SAM2 model must be one of: {allowed}. Got {selected_model!r}.")

    option = dict(SAM2_MODEL_OPTIONS[selected_model])
    option["name"] = selected_model
    option["checkpoint"] = SAM2_CHECKPOINTS_DIR / option["checkpoint_name"]
    return option


def checkpoint_file_looks_unavailable(checkpoint_path: Path) -> bool:
    """Detect cloud/sparse placeholder checkpoints before torch.load can hang."""
    try:
        stat_result = Path(checkpoint_path).stat()
    except FileNotFoundError:
        return False

    allocated_blocks = getattr(stat_result, "st_blocks", None)
    return stat_result.st_size > 0 and allocated_blocks == 0


def checkpoint_file_looks_incomplete(checkpoint_path: Path, model_option: Dict[str, Any]) -> bool:
    expected_size = model_option.get("expected_size")
    if not expected_size:
        return False
    try:
        stat_result = Path(checkpoint_path).stat()
    except FileNotFoundError:
        return False
    return 0 < stat_result.st_size < int(expected_size)


def _ensure_sam2_on_path() -> None:
    if not SAM2_REPO_ROOT.exists():
        raise FileNotFoundError(f"SAM2 repo not found: {SAM2_REPO_ROOT}")
    repo_path = str(SAM2_REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def import_torch_without_entry_point_scan():
    import importlib.metadata as importlib_metadata

    original_entry_points = importlib_metadata.entry_points
    importlib_metadata.entry_points = lambda *args, **kwargs: {}
    try:
        import torch
    finally:
        importlib_metadata.entry_points = original_entry_points
    return torch


def install_torchvision_transform_stub_for_sam2() -> None:
    """Avoid broken torchvision NMS registration; SAM2 only needs basic transforms."""
    torch = import_torch_without_entry_point_scan()
    import torch.nn as nn
    import torch.nn.functional as F

    class Resize(nn.Module):
        def __init__(self, size):
            super().__init__()
            if isinstance(size, int):
                self.size = (int(size), int(size))
            else:
                self.size = (int(size[0]), int(size[1]))

        def forward(self, image):
            return F.interpolate(
                image.unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(-1, 1, 1))

        def forward(self, image):
            return (image - self.mean) / self.std

    class ToTensor:
        def __call__(self, image):
            array = np.asarray(image)
            if array.ndim == 2:
                array = array[:, :, None]
            if array.dtype != np.float32:
                array = array.astype(np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            return tensor

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.Normalize = Normalize
    transforms_module.Resize = Resize
    transforms_module.ToTensor = ToTensor
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


def python_downloader_should_use_curl_first() -> bool:
    return "LibreSSL" in getattr(ssl, "OPENSSL_VERSION", "")


def download_url_with_curl(url: str, destination: Path, curl_path: str) -> None:
    command = [
        curl_path,
        "--fail",
        "--location",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
    ]
    if destination.exists() and destination.stat().st_size > 0:
        command.extend(["--continue-at", "-"])
    command.extend(["-o", str(destination), url])
    subprocess.run(command, check=True)


def download_sam2_checkpoint(model_name: Optional[str] = None) -> Path:
    model_option = resolve_sam2_model_option(model_name)
    checkpoint_path = Path(model_option["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".download")
    curl_path = shutil.which("curl")

    if checkpoint_file_looks_incomplete(checkpoint_path, model_option):
        checkpoint_size = checkpoint_path.stat().st_size
        temporary_size = temporary_path.stat().st_size if temporary_path.exists() else -1
        if checkpoint_size > temporary_size:
            if temporary_path.exists():
                temporary_path.unlink()
            checkpoint_path.replace(temporary_path)

    if curl_path and python_downloader_should_use_curl_first():
        download_url_with_curl(model_option["url"], temporary_path, curl_path)
    else:
        if temporary_path.exists():
            temporary_path.unlink()
        try:
            urllib.request.urlretrieve(model_option["url"], temporary_path)
        except Exception as urllib_error:
            if not curl_path:
                if temporary_path.exists() and temporary_path.stat().st_size == 0:
                    temporary_path.unlink()
                raise RuntimeError(
                    f"Failed to download {model_option['checkpoint_name']} with Python urllib, "
                    "and curl is not available for fallback."
                ) from urllib_error

            try:
                download_url_with_curl(model_option["url"], temporary_path, curl_path)
            except Exception as curl_error:
                if temporary_path.exists() and temporary_path.stat().st_size == 0:
                    temporary_path.unlink()
                raise RuntimeError(
                    f"Failed to download {model_option['checkpoint_name']} with Python urllib "
                    "or curl fallback."
                ) from curl_error

    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded checkpoint is empty: {temporary_path}")
    if checkpoint_file_looks_incomplete(temporary_path, model_option):
        raise RuntimeError(
            f"Downloaded checkpoint is incomplete: {temporary_path} "
            f"({temporary_path.stat().st_size} of {model_option['expected_size']} bytes)."
        )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def resolve_sam2_checkpoint(
    checkpoint: Optional[Path] = None,
    model_name: Optional[str] = None,
    download_checkpoint: bool = False,
) -> Path:
    if checkpoint is not None:
        resolved = Path(checkpoint)
        checkpoint_hint = str(resolved)
    else:
        model_option = resolve_sam2_model_option(model_name)
        resolved = Path(model_option["checkpoint"])
        checkpoint_hint = model_option["checkpoint_name"]

    if not resolved.exists():
        if checkpoint is None and download_checkpoint:
            return download_sam2_checkpoint(model_name)
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {resolved}. "
            f"Download {checkpoint_hint} into sam2/checkpoints/."
        )

    if checkpoint_file_looks_unavailable(resolved):
        if checkpoint is None and download_checkpoint:
            return download_sam2_checkpoint(model_name)
        raise FileNotFoundError(
            f"SAM2 checkpoint appears to be an unavailable sparse/cloud placeholder: {resolved}. "
            f"Re-download {checkpoint_hint} into sam2/checkpoints/."
        )
    if checkpoint is None and checkpoint_file_looks_incomplete(resolved, model_option):
        if download_checkpoint:
            return download_sam2_checkpoint(model_name)
        raise FileNotFoundError(
            f"SAM2 checkpoint appears incomplete: {resolved}. "
            f"Re-download {checkpoint_hint} into sam2/checkpoints/."
        )
    return resolved


def resolve_sam2_device(device: Optional[str] = None) -> str:
    if device:
        return str(device)

    torch = import_torch_without_entry_point_scan()

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam2_predictor(
    checkpoint: Optional[Path] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    download_checkpoint: bool = False,
):
    _ensure_sam2_on_path()
    install_torchvision_transform_stub_for_sam2()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    resolved_device = resolve_sam2_device(device)
    model_option = resolve_sam2_model_option(model_name)
    resolved_config = config or str(model_option["config"])
    model = build_sam2(
        resolved_config,
        ckpt_path=str(
            resolve_sam2_checkpoint(
                checkpoint=checkpoint,
                model_name=model_option["name"],
                download_checkpoint=download_checkpoint,
            )
        ),
        device=resolved_device,
        apply_postprocessing=False,
    )
    return SAM2ImagePredictor(model), resolved_device


def _json_ready_points(points: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in np.asarray(points, dtype=np.float32).tolist()]


def _as_point_array(points: Any, source: Path) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected [x, y] point coordinates in {source}")
    if len(coords) == 0:
        raise ValueError(f"No point coordinates found in {source}")
    return coords


def _clamp_points(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    clamped = np.asarray(points, dtype=np.float32).copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, image_width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, image_height - 1)
    return clamped


def _clamp_box(box: Any, image_width: int, image_height: int) -> np.ndarray:
    box_array = np.asarray(box, dtype=np.float32).reshape(-1)
    if box_array.shape != (4,):
        raise ValueError("Expected box prompt as [x1, y1, x2, y2].")
    x1, y1, x2, y2 = [float(value) for value in box_array.tolist()]
    min_x, max_x = sorted((x1, x2))
    min_y, max_y = sorted((y1, y2))
    return np.asarray(
        [
            np.clip(min_x, 0, image_width - 1),
            np.clip(min_y, 0, image_height - 1),
            np.clip(max_x, 0, image_width - 1),
            np.clip(max_y, 0, image_height - 1),
        ],
        dtype=np.float32,
    )


def _points_records_to_prompt(records: Sequence[Dict[str, Any]], source: Path) -> Tuple[np.ndarray, np.ndarray]:
    coords = []
    labels = []
    for point in records:
        if "x" not in point or "y" not in point:
            raise ValueError(f"Point record missing x/y in {source}")
        if not bool(point.get("visible", True)):
            continue
        coords.append([float(point["x"]), float(point["y"])])
        if "label" in point:
            labels.append(int(point["label"]))
        else:
            labels.append(0 if str(point.get("type", "")).lower() == "negative" else 1)
    if not coords:
        raise ValueError(f"No visible point records found in {source}")
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _prompt_from_positive_only(
    positive_points: Any,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding_px: float,
    min_negative_distance: float,
    negative_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    augmented = build_augmented_prompt_object(
        positive_points=positive_points,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        min_negative_distance=min_negative_distance,
        negative_mode=negative_mode,
        class_id=1,
    )
    coords = np.asarray(augmented["point_coords"], dtype=np.float32)
    labels = np.asarray(augmented["point_labels"], dtype=np.int32)
    return coords, labels, augmented


def _object_to_prompt(
    obj: Dict[str, Any],
    source: Path,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding_px: float,
    min_negative_distance: float,
    negative_mode: str,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray], Dict[str, Any]]:
    class_id = int(obj.get("class_id", 1))

    if "point_coords" in obj and "point_labels" in obj:
        coords = _clamp_points(_as_point_array(obj["point_coords"], source), image_width, image_height)
        labels = np.asarray(obj["point_labels"], dtype=np.int32)
        if labels.ndim != 1 or len(labels) != len(coords):
            raise ValueError(f"point_labels length does not match point_coords in {source}")
        prompt_json = {
            "class_id": class_id,
            "positive_points": _json_ready_points(coords[labels == 1]),
            "negative_points": _json_ready_points(coords[labels == 0]),
            "point_coords": _json_ready_points(coords),
            "point_labels": [int(label) for label in labels.tolist()],
        }
        prompt_box = None
        if obj.get("box") is not None:
            prompt_box = _clamp_box(obj["box"], image_width=image_width, image_height=image_height)
            prompt_json["box"] = [float(value) for value in prompt_box.tolist()]
        return coords, labels, class_id, prompt_box, prompt_json

    if "positive_points" in obj or "negative_points" in obj:
        positives = np.asarray(obj.get("positive_points", []), dtype=np.float32).reshape(-1, 2)
        negatives = np.asarray(obj.get("negative_points", []), dtype=np.float32).reshape(-1, 2)
        if len(negatives) == 0:
            coords, labels, prompt_json = _prompt_from_positive_only(
                positives,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                min_padding_px=min_padding_px,
                min_negative_distance=min_negative_distance,
                negative_mode=negative_mode,
            )
            prompt_json["class_id"] = class_id
            prompt_box = _clamp_box(prompt_json["box"], image_width=image_width, image_height=image_height)
            return coords, labels, class_id, prompt_box, prompt_json

        coords = _clamp_points(np.concatenate([positives, negatives], axis=0), image_width, image_height)
        labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
        prompt_json = {
            "class_id": class_id,
            "positive_points": _json_ready_points(coords[labels == 1]),
            "negative_points": _json_ready_points(coords[labels == 0]),
            "point_coords": _json_ready_points(coords),
            "point_labels": [int(label) for label in labels.tolist()],
        }
        prompt_box = None
        if obj.get("box") is not None:
            prompt_box = _clamp_box(obj["box"], image_width=image_width, image_height=image_height)
            prompt_json["box"] = [float(value) for value in prompt_box.tolist()]
        return coords, labels, class_id, prompt_box, prompt_json

    if obj.get("points"):
        coords, labels = _points_records_to_prompt(obj["points"], source)
        coords = _clamp_points(coords, image_width, image_height)
        prompt_json = {
            "class_id": class_id,
            "positive_points": _json_ready_points(coords[labels == 1]),
            "negative_points": _json_ready_points(coords[labels == 0]),
            "point_coords": _json_ready_points(coords),
            "point_labels": [int(label) for label in labels.tolist()],
        }
        prompt_box = None
        if obj.get("box") is not None:
            prompt_box = _clamp_box(obj["box"], image_width=image_width, image_height=image_height)
            prompt_json["box"] = [float(value) for value in prompt_box.tolist()]
        return coords, labels, class_id, prompt_box, prompt_json

    raise ValueError(f"Unsupported coordinate object format in {source}")


def load_sam2_prompt_objects(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_padding_px: float = DEFAULT_MIN_PADDING_PX,
    min_negative_distance: float = DEFAULT_MIN_NEGATIVE_DISTANCE,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]], Dict[str, Any]]:
    if not coordinate_json_path.exists():
        raise FileNotFoundError(f"Coordinate JSON not found: {coordinate_json_path}")

    data = json.loads(coordinate_json_path.read_text(encoding="utf-8"))
    prompt_objects: List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]] = []
    prompt_json_objects = []

    if isinstance(data, list):
        coords, labels, prompt_json = _prompt_from_positive_only(
            data,
            image_width=image_width,
            image_height=image_height,
            padding_ratio=padding_ratio,
            min_padding_px=min_padding_px,
            min_negative_distance=min_negative_distance,
            negative_mode=negative_mode,
        )
        prompt_objects.append((coords, labels, 1, _clamp_box(prompt_json["box"], image_width, image_height)))
        prompt_json_objects.append(prompt_json)
    elif isinstance(data, dict) and data.get("objects"):
        for obj in data["objects"]:
            coords, labels, class_id, prompt_box, prompt_json = _object_to_prompt(
                obj,
                source=coordinate_json_path,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                min_padding_px=min_padding_px,
                min_negative_distance=min_negative_distance,
                negative_mode=negative_mode,
            )
            prompt_objects.append((coords, labels, class_id, prompt_box))
            prompt_json_objects.append(prompt_json)
    elif isinstance(data, dict):
        if "point_coords" in data and "point_labels" in data:
            obj = {
                "class_id": int(data.get("class_id", 1)),
                "point_coords": data["point_coords"],
                "point_labels": data["point_labels"],
            }
            coords, labels, class_id, prompt_box, prompt_json = _object_to_prompt(
                obj,
                source=coordinate_json_path,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                min_padding_px=min_padding_px,
                min_negative_distance=min_negative_distance,
                negative_mode=negative_mode,
            )
            prompt_objects.append((coords, labels, class_id, prompt_box))
            prompt_json_objects.append(prompt_json)
        elif data.get("points") and isinstance(data["points"][0], dict):
            obj = {"class_id": int(data.get("class_id", 1)), "points": data["points"]}
            coords, labels, class_id, prompt_box, prompt_json = _object_to_prompt(
                obj,
                source=coordinate_json_path,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                min_padding_px=min_padding_px,
                min_negative_distance=min_negative_distance,
                negative_mode=negative_mode,
            )
            prompt_objects.append((coords, labels, class_id, prompt_box))
            prompt_json_objects.append(prompt_json)
        else:
            for key in ("positive_points", "coordinates", "points"):
                if key in data:
                    coords, labels, prompt_json = _prompt_from_positive_only(
                        data[key],
                        image_width=image_width,
                        image_height=image_height,
                        padding_ratio=padding_ratio,
                        min_padding_px=min_padding_px,
                        min_negative_distance=min_negative_distance,
                        negative_mode=negative_mode,
                    )
                    prompt_objects.append(
                        (
                            coords,
                            labels,
                            int(data.get("class_id", 1)),
                            _clamp_box(prompt_json["box"], image_width, image_height),
                        )
                    )
                    prompt_json_objects.append(prompt_json)
                    break
    else:
        raise ValueError(f"Unsupported coordinate JSON format in {coordinate_json_path}")

    if not prompt_objects:
        raise ValueError(f"No SAM2 prompt objects found in {coordinate_json_path}")

    prompt_json = {
        "source_coordinate_json": str(coordinate_json_path),
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "padding_ratio": float(padding_ratio),
        "min_padding_px": float(min_padding_px),
        "min_negative_distance": float(min_negative_distance),
        "negative_mode": negative_mode,
        "engine": "sam2",
        "objects": prompt_json_objects,
    }
    return prompt_objects, prompt_json


def list_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        path for path in Path(frames_dir).iterdir() if path.suffix.lower() in SUPPORTED_FRAME_EXTENSIONS
    )


def select_frames_for_target_fps(
    frame_paths: Sequence[Path],
    target_fps: float,
    source_fps: float = 30.0,
) -> List[Path]:
    return select_frames_for_frame_step(frame_paths, frame_step=target_fps)


def _clear_matching_files(output_dir: Path, patterns: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def _copy_frame_to_output(frame_path: Path, output_frames_dir: Path) -> Path:
    output_frame_path = output_frames_dir / frame_path.name
    output_frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not _same_path(frame_path, output_frame_path):
        shutil.copy2(frame_path, output_frame_path)
    return output_frame_path


def write_processed_sam2_coordinate_json(
    source_coordinate_json_path: Path,
    frame_path: Path,
    processed_coordinate_json_path: Path,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_padding_px: float = DEFAULT_MIN_PADDING_PX,
    min_negative_distance: float = DEFAULT_MIN_NEGATIVE_DISTANCE,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
) -> Path:
    """Write processed SAM2 prompts without overwriting source coordinate JSON."""
    frame = Image.open(frame_path).convert("RGB")
    image_width, image_height = frame.size
    prepare_coordinate_prompt_json(
        coordinate_json_path=source_coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        output_path=processed_coordinate_json_path,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        min_negative_distance=min_negative_distance,
        negative_mode=negative_mode,
        visible_only=True,
    )
    processed_data = json.loads(processed_coordinate_json_path.read_text(encoding="utf-8"))
    processed_data["engine"] = "sam2"
    processed_coordinate_json_path.write_text(
        json.dumps(processed_data, indent=2),
        encoding="utf-8",
    )
    return processed_coordinate_json_path


def run_sam2_for_frame(
    predictor,
    frame_path: Path,
    coordinate_json_path: Path,
    mask_path: Path,
    preview_path: Optional[Path] = None,
    normalized_coordinate_json_path: Optional[Path] = None,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_padding_px: float = DEFAULT_MIN_PADDING_PX,
    min_negative_distance: float = DEFAULT_MIN_NEGATIVE_DISTANCE,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
    score_threshold: float = 0.0,
    max_mask_area_ratio: float = DEFAULT_MAX_MASK_AREA_RATIO,
) -> None:
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    image_height, image_width = frame_rgb.shape[:2]
    prompt_objects, prompt_json = load_sam2_prompt_objects(
        coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        min_negative_distance=min_negative_distance,
        negative_mode=negative_mode,
    )

    output_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    predictor.set_image(frame_rgb)
    for point_coords, point_labels, class_id, prompt_box in prompt_objects:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            box=None if prompt_box is None else prompt_box.astype(np.float32),
            multimask_output=True,
            normalize_coords=True,
        )
        best_index = select_best_mask(
            masks=masks,
            scores=scores,
            point_coords=point_coords,
            point_labels=point_labels,
            prompt_box=prompt_box,
            max_mask_area_ratio=max_mask_area_ratio,
        )
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
    if normalized_coordinate_json_path is not None:
        normalized_coordinate_json_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_coordinate_json_path.write_text(json.dumps(prompt_json, indent=2), encoding="utf-8")


def iter_sam2_coordinate_prompt_folder_steps(
    frames_dir: Path,
    coordinates_dir: Path,
    output_root: Path = DEFAULT_RAW_MASK_DATA_DIR,
    predictor=None,
    checkpoint: Optional[Path] = None,
    config: Optional[str] = None,
    sam2_model: str = DEFAULT_SAM2_MODEL,
    device: Optional[str] = None,
    download_checkpoint: bool = False,
    frame_step: Optional[float] = None,
    target_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_padding_px: float = DEFAULT_MIN_PADDING_PX,
    min_negative_distance: float = DEFAULT_MIN_NEGATIVE_DISTANCE,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
    score_threshold: float = 0.0,
    max_mask_area_ratio: float = DEFAULT_MAX_MASK_AREA_RATIO,
) -> Iterable[Dict[str, Any]]:
    frames_dir = Path(frames_dir)
    coordinates_dir = Path(coordinates_dir)
    output_root = Path(output_root)
    frames_output_dir = output_root / "frames"
    masks_output_dir = output_root / "mask"
    previews_output_dir = output_root / "masked_frames"
    processed_coordinates_dir = output_root / "coordinates"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    effective_frame_step = frame_step if frame_step is not None else target_fps
    if effective_frame_step is None:
        effective_frame_step = 1
    selected_frame_paths = select_frames_for_frame_step(
        frame_paths,
        frame_step=effective_frame_step,
    )
    total = len(selected_frame_paths)
    yield {
        "stage": "scanned",
        "completed": 0,
        "total": total,
        "message": (
            f"Selected {total} frame(s) from {len(frame_paths)} input frame(s) "
            f"after skipping the first and last {FRAME_BOUNDARY_SKIP} frame(s) "
            f"with frame_step={int(float(effective_frame_step)):g}"
        ),
        "result": None,
    }
    missing_json = [
        coordinates_dir / f"{frame_path.stem}.json"
        for frame_path in selected_frame_paths
        if not (coordinates_dir / f"{frame_path.stem}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Missing coordinate JSON for frame: {missing_json[0]}")

    _clear_matching_files(masks_output_dir, ("*.png", "*.jpg", "*.jpeg"))
    _clear_matching_files(previews_output_dir, ("*.png", "*.jpg", "*.jpeg"))
    if not _same_path(frames_dir, frames_output_dir):
        _clear_matching_files(frames_output_dir, ("*.jpg", "*.jpeg", "*.png"))
    else:
        frames_output_dir.mkdir(parents=True, exist_ok=True)
    if not _same_path(coordinates_dir, processed_coordinates_dir):
        _clear_matching_files(processed_coordinates_dir, ("*.json",))
    else:
        processed_coordinates_dir.mkdir(parents=True, exist_ok=True)
    _clear_matching_files(output_root, ("frames.zip", "mask.zip", "masked_frame.zip", "masked_frames.zip"))

    yield {
        "stage": "prepared-output",
        "completed": 0,
        "total": total,
        "message": (
            f"Prepared outputs in {output_root}: frames, coordinates, mask, masked_frames"
        ),
        "result": None,
    }

    resolved_device = None
    if predictor is None:
        yield {
            "stage": "loading-model",
            "completed": 0,
            "total": total,
            "message": f"Loading SAM2 model {sam2_model}",
            "result": None,
        }
        predictor, resolved_device = load_sam2_predictor(
            checkpoint=checkpoint,
            config=config,
            model_name=sam2_model,
            device=device,
            download_checkpoint=download_checkpoint,
        )
    else:
        resolved_device = device or "provided predictor"

    yield {
        "stage": "starting",
        "completed": 0,
        "total": total,
        "message": f"Starting SAM2 on {resolved_device} for {total} frame(s)",
        "result": None,
    }

    for frame_index, frame_path in enumerate(selected_frame_paths, start=1):
        output_frame_path = _copy_frame_to_output(frame_path, frames_output_dir)
        processed_coordinate_json_path = write_processed_sam2_coordinate_json(
            source_coordinate_json_path=coordinates_dir / f"{frame_path.stem}.json",
            frame_path=output_frame_path,
            processed_coordinate_json_path=processed_coordinates_dir / f"{frame_path.stem}.json",
            padding_ratio=padding_ratio,
            min_padding_px=min_padding_px,
            min_negative_distance=min_negative_distance,
            negative_mode=negative_mode,
        )
        run_sam2_for_frame(
            predictor=predictor,
            frame_path=output_frame_path,
            coordinate_json_path=processed_coordinate_json_path,
            mask_path=masks_output_dir / f"{frame_path.stem}.png",
            preview_path=previews_output_dir / f"{frame_path.stem}.jpg",
            normalized_coordinate_json_path=None,
            padding_ratio=padding_ratio,
            min_padding_px=min_padding_px,
            min_negative_distance=min_negative_distance,
            negative_mode=negative_mode,
            score_threshold=score_threshold,
            max_mask_area_ratio=max_mask_area_ratio,
        )
        yield {
            "stage": "processing",
            "completed": frame_index,
            "total": total,
            "message": f"Processed {frame_path.name}",
            "result": None,
        }

    result = {
        "engine": "sam2",
        "sam2_model": sam2_model,
        "processed_frames": len(selected_frame_paths),
        "frame_paths": selected_frame_paths,
        "source_frames_dir": frames_dir,
        "frames_dir": frames_output_dir,
        "source_coordinates_dir": coordinates_dir,
        "coordinates_dir": processed_coordinates_dir,
        "masks_dir": masks_output_dir,
        "previews_dir": previews_output_dir,
        "frames_zip": None,
        "masks_zip": None,
        "previews_zip": None,
    }
    yield {
        "stage": "done",
        "completed": total,
        "total": total,
        "message": f"Completed {total} SAM2 frame(s)",
        "result": result,
    }


def run_sam2_coordinate_prompt_folders(
    frames_dir: Path,
    coordinates_dir: Path,
    output_root: Path = DEFAULT_RAW_MASK_DATA_DIR,
    predictor=None,
    checkpoint: Optional[Path] = None,
    config: Optional[str] = None,
    sam2_model: str = DEFAULT_SAM2_MODEL,
    device: Optional[str] = None,
    download_checkpoint: bool = False,
    frame_step: Optional[float] = None,
    target_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_padding_px: float = DEFAULT_MIN_PADDING_PX,
    min_negative_distance: float = DEFAULT_MIN_NEGATIVE_DISTANCE,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
    score_threshold: float = 0.0,
    max_mask_area_ratio: float = DEFAULT_MAX_MASK_AREA_RATIO,
) -> Dict[str, Any]:
    last_update = None
    for update in iter_sam2_coordinate_prompt_folder_steps(
        frames_dir=frames_dir,
        coordinates_dir=coordinates_dir,
        output_root=output_root,
        predictor=predictor,
        checkpoint=checkpoint,
        config=config,
        sam2_model=sam2_model,
        device=device,
        download_checkpoint=download_checkpoint,
        frame_step=frame_step,
        target_fps=target_fps,
        source_fps=source_fps,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        min_negative_distance=min_negative_distance,
        negative_mode=negative_mode,
        score_threshold=score_threshold,
        max_mask_area_ratio=max_mask_area_ratio,
    ):
        last_update = update

    if last_update and last_update.get("result"):
        return last_update["result"]
    raise RuntimeError("SAM2 coordinate folder processing did not complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SAM2 wound masks from raw frames and coordinate JSON prompts."
    )
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_SOURCE_FRAMES_DIR)
    parser.add_argument("--coordinates-dir", type=Path, default=DEFAULT_SOURCE_COORDINATES_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAW_MASK_DATA_DIR)
    parser.add_argument(
        "--frame-step",
        type=float,
        default=1,
        help="Process every Nth frame. For example, 3 processes frames 0, 3, 6...",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Deprecated alias for --frame-step.")
    parser.add_argument("--source-fps", type=float, default=None, help="Deprecated; frame sampling uses --frame-step.")
    parser.add_argument("--padding-ratio", type=float, default=DEFAULT_PADDING_RATIO)
    parser.add_argument("--min-padding-px", type=float, default=DEFAULT_MIN_PADDING_PX)
    parser.add_argument("--min-negative-distance", type=float, default=DEFAULT_MIN_NEGATIVE_DISTANCE)
    parser.add_argument("--negative-mode", choices=NEGATIVE_MODES, default=DEFAULT_NEGATIVE_MODE)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=DEFAULT_MAX_MASK_AREA_RATIO)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sam2-model", choices=SAM2_MODEL_CHOICES, default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    frame_step = args.frame_step if args.target_fps is None else args.target_fps
    result = run_sam2_coordinate_prompt_folders(
        frames_dir=args.frames_dir,
        coordinates_dir=args.coordinates_dir,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        config=args.config,
        sam2_model=args.sam2_model,
        device=args.device,
        download_checkpoint=args.download_checkpoint,
        frame_step=frame_step,
        source_fps=args.source_fps,
        padding_ratio=args.padding_ratio,
        min_padding_px=args.min_padding_px,
        min_negative_distance=args.min_negative_distance,
        negative_mode=args.negative_mode,
        score_threshold=args.score_threshold,
        max_mask_area_ratio=args.max_mask_area_ratio,
    )
    print(f"processed_frames={result['processed_frames']}")
    print(f"sam2_model={result['sam2_model']}")
    print(f"frames_dir={result['frames_dir']}")
    print(f"source_coordinates_dir={result['source_coordinates_dir']}")
    print(f"coordinates_dir={result['coordinates_dir']}")
    print(f"masks_dir={result['masks_dir']}")
    print(f"previews_dir={result['previews_dir']}")


if __name__ == "__main__":
    main()
