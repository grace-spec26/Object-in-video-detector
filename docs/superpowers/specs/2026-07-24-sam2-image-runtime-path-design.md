# SAM2 Image Runtime Path Design

## Goal

Make the Gradio SAM image-model download and single-frame SAM preview use the current workspace SAM2 image-model path:

`/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/sam2/checkpoints`

The flow must no longer depend on `MobileSAM-master/sam2/checkpoints` or on importing the SAM2 loader from `MobileSAM-master`.

## Current Behavior

`co-tracker-main/gradio_demo/sam_preview_service.py` owns the UI-side SAM preview state, progress HTML, async preload, and preview actions. It currently defines `MOBILE_SAM_ROOT = PROJECT_ROOT / "MobileSAM-master"` and imports `sam2_coordinate_wrapper` from that directory to resolve and load SAM2 image models.

In the current worktree, `MobileSAM-master/` has been deleted, while the active SAM2 repository and model checkpoints live at workspace root under `sam2/` and `sam2/checkpoints/`. This means the old import dependency can break both:

- `Download SAM Models`
- `Preview SAM on Current Frame`

## Requirements

1. The SAM2 image runtime must define explicit path ownership:
   - `PROJECT_ROOT = Path(__file__).resolve().parents[2]`
   - `SAM2_REPO_ROOT = PROJECT_ROOT / "sam2"`
   - `SAM2_CHECKPOINTS_DIR = SAM2_REPO_ROOT / "checkpoints"`
2. The four supported image-model choices remain:
   - `sam2.1_hiera_tiny.pt`
   - `sam2.1_hiera_small.pt`
   - `sam2.1_hiera_base_plus.pt`
   - `sam2.1_hiera_large.pt`
3. The model configs must resolve from the current SAM2 repo package, using paths under `configs/sam2.1/`.
4. `Download SAM Models` must download and load those four models sequentially through the new SAM2 runtime helper.
5. `Preview SAM on Current Frame` must load the selected dropdown model through the same helper.
6. No SAM2 preview or download path should insert `MobileSAM-master` into `sys.path`.
7. Existing UI behavior and progress/status messages should remain functionally equivalent.

## Design

Create `co-tracker-main/gradio_demo/sam2_image_runtime.py` as the focused owner for SAM2 image-model loading. It will contain:

- SAM2 model registry with labels, checkpoint names, config names, URLs, and expected byte sizes.
- Checkpoint path resolution rooted at `SAM2_CHECKPOINTS_DIR`.
- Download helpers for incomplete, placeholder, and partial downloads.
- Device resolution for `cuda`, `mps`, and `cpu`.
- SAM2 import setup using `SAM2_REPO_ROOT`.
- `load_sam2_predictor(...)`, returning a `SAM2ImagePredictor` and resolved device.

Update `sam_preview_service.py` so it imports from `sam2_image_runtime.py` instead of importing `sam2_coordinate_wrapper` from `MobileSAM-master`. The service remains responsible for UI progress, runtime caching, preload threading, frame prompt extraction, and preview rendering.

## Error Handling

If `sam2/` is missing, model loading should raise a clear `FileNotFoundError` naming `SAM2_REPO_ROOT`.

If a checkpoint is missing, incomplete, or a sparse/cloud placeholder, the helper should either redownload it when `download_checkpoint=True` or raise a clear message telling the user to place the checkpoint in `sam2/checkpoints/`.

Unknown model names should still raise a `ValueError` listing allowed SAM2 image-model filenames.

## Testing

Add regression tests proving:

- Resolving `sam2.1_hiera_small.pt` returns checkpoint path `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/sam2/checkpoints/sam2.1_hiera_small.pt`.
- `get_sam_preview_runtime(...)` uses the new helper and does not add `MobileSAM-master` to `sys.path`.
- `download_all_sam_image_models_with_progress()` uses the same helper path for all four image models.
- `Preview SAM on Current Frame` still forwards the selected SAM model dropdown into the runtime-loading path.

## Non-Goals

This change does not reintroduce MobileSAM, change the processed-video SAM preview behavior, modify YOLO export, or change the visual mask overlay logic.
