import shutil
import ssl
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def resolve_sam2_model_option(model_name=None):
    selected_model = str(model_name or DEFAULT_SAM2_MODEL)
    if selected_model not in SAM2_MODEL_OPTIONS:
        allowed = ", ".join(SAM2_MODEL_CHOICES)
        raise ValueError(f"SAM2 image model must be one of: {allowed}. Got {selected_model!r}.")

    option = dict(SAM2_MODEL_OPTIONS[selected_model])
    option["name"] = selected_model
    option["checkpoint"] = SAM2_CHECKPOINTS_DIR / option["checkpoint_name"]
    return option


def checkpoint_file_looks_unavailable(checkpoint_path):
    try:
        stat_result = Path(checkpoint_path).stat()
    except FileNotFoundError:
        return False

    allocated_blocks = getattr(stat_result, "st_blocks", None)
    if stat_result.st_size <= 0 or allocated_blocks is None:
        return False

    allocated_bytes = int(allocated_blocks) * 512
    return allocated_bytes < stat_result.st_size * 0.5


def checkpoint_file_looks_incomplete(checkpoint_path, model_option):
    expected_size = model_option.get("expected_size")
    if not expected_size:
        return False
    try:
        stat_result = Path(checkpoint_path).stat()
    except FileNotFoundError:
        return False
    return 0 < stat_result.st_size < int(expected_size)


def ensure_sam2_on_path():
    if not SAM2_REPO_ROOT.exists():
        raise FileNotFoundError(f"SAM2 repo not found: {SAM2_REPO_ROOT}")
    repo_path = str(SAM2_REPO_ROOT)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
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


def install_torchvision_transform_stub_for_sam2():
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
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.Normalize = Normalize
    transforms_module.Resize = Resize
    transforms_module.ToTensor = ToTensor
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


def python_downloader_should_use_curl_first():
    return "LibreSSL" in getattr(ssl, "OPENSSL_VERSION", "")


def checkpoint_range_tail_path(destination):
    return Path(destination).with_suffix(Path(destination).suffix + ".range")


def download_url_with_curl(url, destination, curl_path):
    destination = Path(destination)
    existing_size = destination.stat().st_size if destination.exists() else 0
    output_path = destination
    range_tail_path = None

    command = [
        curl_path,
        "--fail",
        "--location",
        "--http1.1",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
    ]

    if existing_size > 0:
        range_tail_path = checkpoint_range_tail_path(destination)
        if range_tail_path.exists():
            range_tail_path.unlink()
        output_path = range_tail_path
        command.extend(["--range", f"{existing_size}-"])

    command.extend(["-o", str(output_path), url])
    try:
        subprocess.run(command, check=True)
    except Exception:
        if range_tail_path is not None and range_tail_path.exists():
            range_tail_path.unlink()
        raise

    if range_tail_path is not None:
        with destination.open("ab") as destination_file:
            with range_tail_path.open("rb") as range_tail_file:
                shutil.copyfileobj(range_tail_file, destination_file)
        range_tail_path.unlink()


def format_checkpoint_download_progress(checkpoint_path, model_option):
    expected_size = int(model_option.get("expected_size") or 0)
    try:
        downloaded_size = Path(checkpoint_path).stat().st_size
    except FileNotFoundError:
        downloaded_size = 0

    if expected_size > 0:
        percent = int(round((downloaded_size / expected_size) * 100))
        return f"{downloaded_size}/{expected_size} bytes ({percent}%)"
    return f"{downloaded_size} bytes"


def raise_checkpoint_download_error(model_option, temporary_path, cause):
    progress = format_checkpoint_download_progress(temporary_path, model_option)
    if isinstance(cause, subprocess.CalledProcessError):
        reason = f"curl exited with status {cause.returncode}"
    else:
        reason = str(cause)
    raise RuntimeError(
        f"Failed to download {model_option['checkpoint_name']}: {progress}; {reason}."
    ) from cause


def download_sam2_checkpoint(model_name=None):
    model_option = resolve_sam2_model_option(model_name)
    checkpoint_path = Path(model_option["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".download")
    curl_path = shutil.which("curl")

    if temporary_path.exists() and checkpoint_file_looks_unavailable(temporary_path):
        temporary_path.unlink()
        range_tail_path = checkpoint_range_tail_path(temporary_path)
        if range_tail_path.exists():
            range_tail_path.unlink()

    if checkpoint_file_looks_incomplete(checkpoint_path, model_option):
        checkpoint_size = checkpoint_path.stat().st_size
        temporary_size = temporary_path.stat().st_size if temporary_path.exists() else -1
        if checkpoint_size > temporary_size:
            if temporary_path.exists():
                temporary_path.unlink()
            checkpoint_path.replace(temporary_path)

    if curl_path and python_downloader_should_use_curl_first():
        try:
            download_url_with_curl(model_option["url"], temporary_path, curl_path)
        except subprocess.CalledProcessError as curl_error:
            raise_checkpoint_download_error(model_option, temporary_path, curl_error)
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
            except subprocess.CalledProcessError as curl_error:
                if temporary_path.exists() and temporary_path.stat().st_size == 0:
                    temporary_path.unlink()
                raise_checkpoint_download_error(model_option, temporary_path, curl_error)
            except Exception as curl_error:
                if temporary_path.exists() and temporary_path.stat().st_size == 0:
                    temporary_path.unlink()
                raise RuntimeError(
                    f"Failed to download {model_option['checkpoint_name']} with Python urllib "
                    f"or curl fallback: {curl_error}."
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


def resolve_sam2_checkpoint(checkpoint=None, model_name=None, download_checkpoint=False):
    if checkpoint is not None:
        resolved = Path(checkpoint)
        checkpoint_hint = str(resolved)
        model_option = None
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
    if checkpoint is None and model_option is not None and checkpoint_file_looks_incomplete(resolved, model_option):
        if download_checkpoint:
            return download_sam2_checkpoint(model_name)
        raise FileNotFoundError(
            f"SAM2 checkpoint appears incomplete: {resolved}. "
            f"Re-download {checkpoint_hint} into sam2/checkpoints/."
        )
    return resolved


def resolve_sam2_device(device=None):
    if device:
        return str(device)

    torch = import_torch_without_entry_point_scan()

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam2_predictor(
    checkpoint=None,
    config=None,
    device=None,
    model_name=None,
    download_checkpoint=False,
):
    ensure_sam2_on_path()
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
