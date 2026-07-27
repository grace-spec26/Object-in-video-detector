from __future__ import annotations

import html
import uuid
from pathlib import Path

import cv2
import mediapy
import numpy as np


DEFAULT_YOLO_EVALUATION_OUTPUT_DIR = Path(__file__).resolve().parent / "tmp"


class YoloEvaluationError(ValueError):
    """User-facing validation or runtime error for YOLO video preview."""


def format_yolo_evaluation_progress_html(processed, total, message, state="running"):
    try:
        total_count = max(0, int(total or 0))
    except (TypeError, ValueError):
        total_count = 0
    try:
        processed_count = int(processed or 0)
    except (TypeError, ValueError):
        processed_count = 0
    processed_count = max(0, min(processed_count, total_count)) if total_count else max(0, processed_count)
    percent = int(round((processed_count / total_count) * 100)) if total_count else 0
    color = "#c0392b" if state == "error" else "#2e7d32" if state == "complete" else "#2563eb"
    escaped_message = html.escape(str(message))
    return f"""
<div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 10px 0;">
  <div style="display:flex; justify-content:space-between; gap:12px; font-size:13px; margin-bottom:6px;">
    <span>{escaped_message}</span>
    <span>{processed_count}/{total_count} frame(s) - {percent}%</span>
  </div>
  <div style="height:10px; border-radius:5px; background:#e5e7eb; overflow:hidden;">
    <div style="height:100%; width:{percent}%; background:{color}; transition:width 160ms ease;"></div>
  </div>
</div>
""".strip()


YOLO_EVALUATION_PROGRESS_READY = format_yolo_evaluation_progress_html(
    0,
    0,
    "Upload an evaluation video and trained YOLO .pt model.",
)


def coerce_uploaded_path(uploaded):
    if uploaded is None:
        return None
    if isinstance(uploaded, (str, Path)):
        value = str(uploaded)
    elif isinstance(uploaded, dict):
        value = uploaded.get("name") or uploaded.get("path") or uploaded.get("data")
    else:
        value = getattr(uploaded, "name", None) or getattr(uploaded, "path", None)
    if not value:
        return None
    return Path(value).expanduser()


def validate_yolo_evaluation_inputs(video_path, model_path):
    resolved_video_path = coerce_uploaded_path(video_path)
    resolved_model_path = coerce_uploaded_path(model_path)
    if resolved_video_path is None:
        raise YoloEvaluationError("Upload an evaluation video before previewing YOLO detections.")
    if resolved_model_path is None:
        raise YoloEvaluationError("Upload a trained YOLO .pt model before previewing detections.")
    if not resolved_video_path.exists():
        raise YoloEvaluationError(f"Evaluation video does not exist: {resolved_video_path}")
    if not resolved_model_path.exists():
        raise YoloEvaluationError(f"Trained YOLO model does not exist: {resolved_model_path}")
    if resolved_model_path.suffix.lower() != ".pt":
        raise YoloEvaluationError("Trained YOLO model must be a .pt file.")
    return resolved_video_path, resolved_model_path


def as_uint8_rgb_frame(frame):
    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[-1] not in (3, 4):
        raise YoloEvaluationError("Evaluation video frames must have shape (H, W, 3/4).")
    if frame_array.dtype != np.uint8:
        if np.issubdtype(frame_array.dtype, np.floating) and frame_array.max(initial=0) <= 1.0:
            frame_array = frame_array * 255
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
    if frame_array.shape[-1] == 4:
        frame_array = frame_array[..., :3]
    return np.ascontiguousarray(frame_array)


def read_evaluation_video(video_path, video_reader=None):
    reader = video_reader or mediapy.read_video
    try:
        video = reader(str(video_path))
    except Exception as exc:
        raise YoloEvaluationError(f"Could not read evaluation video: {exc}") from exc

    metadata = getattr(video, "metadata", None)
    try:
        fps = float(getattr(metadata, "fps", 24.0) or 24.0)
    except (TypeError, ValueError):
        fps = 24.0
    if fps <= 0:
        fps = 24.0

    frames = np.asarray(video)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise YoloEvaluationError("Evaluation video must contain RGB or RGBA frames.")
    if frames.shape[0] == 0:
        raise YoloEvaluationError("Evaluation video contains no frames.")
    return np.asarray([as_uint8_rgb_frame(frame) for frame in frames], dtype=np.uint8), fps


def _coerce_positive_fps(fps):
    try:
        parsed_fps = float(fps or 24.0)
    except (TypeError, ValueError):
        parsed_fps = 24.0
    return parsed_fps if parsed_fps > 0 else 24.0


def open_cv2_video_capture(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise YoloEvaluationError(f"Could not open evaluation video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = _coerce_positive_fps(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0:
        capture.release()
        raise YoloEvaluationError("Evaluation video contains no frames.")
    return capture, total_frames, fps


def read_cv2_rgb_frame(capture):
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        return None
    return cv2.cvtColor(as_uint8_rgb_frame(frame_bgr), cv2.COLOR_BGR2RGB)


def to_numpy(value):
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name_for(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, f"class {class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class {class_id}"


def extract_yolo_detections(result, fallback_names=None):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = to_numpy(getattr(boxes, "xyxy", None))
    if xyxy.size == 0:
        return []
    xyxy = xyxy.reshape(-1, 4)
    classes = to_numpy(getattr(boxes, "cls", None)).reshape(-1)
    confidences = to_numpy(getattr(boxes, "conf", None)).reshape(-1)
    names = getattr(result, "names", None) or fallback_names

    detections = []
    for index, coords in enumerate(xyxy):
        class_id = int(classes[index]) if index < len(classes) else 0
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        label = _class_name_for(names, class_id)
        detections.append((coords.astype(float), class_id, confidence, label))
    return detections


def draw_yolo_detections(frame, result, fallback_names=None):
    output = as_uint8_rgb_frame(frame).copy()
    height, width = output.shape[:2]
    thickness = max(1, int(round(min(height, width) / 180)))
    font_scale = max(0.35, min(height, width) / 500)

    for coords, _class_id, confidence, label in extract_yolo_detections(result, fallback_names):
        x1, y1, x2, y2 = coords
        x1 = int(np.clip(round(x1), 0, width - 1))
        y1 = int(np.clip(round(y1), 0, height - 1))
        x2 = int(np.clip(round(x2), 0, width - 1))
        y2 = int(np.clip(round(y2), 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = (255, 140, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, thickness),
        )
        label_bottom = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            output,
            (x1, label_bottom - text_height - baseline - 4),
            (min(width - 1, x1 + text_width + 6), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 3, label_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(1, thickness),
            cv2.LINE_AA,
        )
    return output


def load_yolo_model(model_path, model_loader=None):
    if model_loader is not None:
        return model_loader(str(model_path))
    try:
        from ultralytics import YOLO
    except (ImportError, ModuleNotFoundError) as exc:
        raise YoloEvaluationError(
            "Install ultralytics in the active environment before running YOLO evaluation."
        ) from exc
    try:
        return YOLO(str(model_path))
    except Exception as exc:
        raise YoloEvaluationError(f"Failed to load trained YOLO model: {exc}") from exc


def run_yolo_on_frame(model, frame):
    try:
        results = model(frame, verbose=False)
    except TypeError:
        results = model(frame)
    if isinstance(results, (list, tuple)):
        if not results:
            raise YoloEvaluationError("YOLO model returned no results for a frame.")
        return results[0]
    return results


def write_yolo_evaluation_video(frames, fps, output_dir=None, video_writer=None):
    destination_dir = Path(output_dir).expanduser() if output_dir else DEFAULT_YOLO_EVALUATION_OUTPUT_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{uuid.uuid4().hex}.mp4"
    writer = video_writer or mediapy.write_video
    try:
        writer(str(output_path), np.asarray(frames, dtype=np.uint8), fps=float(fps or 24.0))
    except Exception as exc:
        raise YoloEvaluationError(f"YOLO preview failed while encoding preview video: {exc}") from exc
    return str(output_path)


def open_cv2_video_writer(frame, fps, output_dir=None):
    destination_dir = Path(output_dir).expanduser() if output_dir else DEFAULT_YOLO_EVALUATION_OUTPUT_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{uuid.uuid4().hex}.mp4"
    height, width = as_uint8_rgb_frame(frame).shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _coerce_positive_fps(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise YoloEvaluationError(f"Could not open MP4 writer for: {output_path}")
    return str(output_path), writer


def write_cv2_rgb_frame(writer, frame):
    writer.write(cv2.cvtColor(as_uint8_rgb_frame(frame), cv2.COLOR_RGB2BGR))


def _error_output(message, processed=0, total=0):
    return format_yolo_evaluation_progress_html(processed, total, message, state="error"), None


def _preview_yolo_model_on_video_from_reader(
    resolved_video_path,
    resolved_model_path,
    *,
    model_loader=None,
    video_reader=None,
    video_writer=None,
    output_dir=None,
):
    yield format_yolo_evaluation_progress_html(0, 0, "Loading evaluation video."), None
    try:
        frames, fps = read_evaluation_video(resolved_video_path, video_reader=video_reader)
    except YoloEvaluationError as exc:
        yield _error_output(str(exc))
        return

    total_frames = int(frames.shape[0])
    yield format_yolo_evaluation_progress_html(
        0,
        total_frames,
        f"0/{total_frames} frame(s) - Loading YOLO model.",
    ), None
    try:
        model = load_yolo_model(resolved_model_path, model_loader=model_loader)
    except YoloEvaluationError as exc:
        yield _error_output(str(exc), 0, total_frames)
        return

    output_frames = []
    fallback_names = getattr(model, "names", None)
    for frame_index, frame in enumerate(frames, start=1):
        yield format_yolo_evaluation_progress_html(
            frame_index - 1,
            total_frames,
            f"Processing frame {frame_index}/{total_frames}.",
        ), None
        try:
            result = run_yolo_on_frame(model, frame)
            output_frames.append(draw_yolo_detections(frame, result, fallback_names=fallback_names))
        except Exception as exc:
            message = f"YOLO preview failed while processing frame {frame_index}/{total_frames}: {exc}"
            yield _error_output(message, frame_index - 1, total_frames)
            return
        yield format_yolo_evaluation_progress_html(
            frame_index,
            total_frames,
            f"Processed {frame_index}/{total_frames} frame(s).",
        ), None

    try:
        output_video_path = write_yolo_evaluation_video(
            output_frames,
            fps,
            output_dir=output_dir,
            video_writer=video_writer,
        )
    except YoloEvaluationError as exc:
        yield _error_output(str(exc), total_frames, total_frames)
        return

    yield format_yolo_evaluation_progress_html(
        total_frames,
        total_frames,
        f"YOLO model preview complete for {total_frames}/{total_frames} frame(s).",
        state="complete",
    ), output_video_path


def _preview_yolo_model_on_video_streaming(
    resolved_video_path,
    resolved_model_path,
    *,
    model_loader=None,
    output_dir=None,
):
    try:
        capture, total_frames, fps = open_cv2_video_capture(resolved_video_path)
    except YoloEvaluationError as exc:
        yield _error_output(str(exc))
        return

    yield format_yolo_evaluation_progress_html(
        0,
        total_frames,
        f"0/{total_frames} frame(s) - Loading YOLO model.",
    ), None
    try:
        model = load_yolo_model(resolved_model_path, model_loader=model_loader)
    except YoloEvaluationError as exc:
        capture.release()
        yield _error_output(str(exc), 0, total_frames)
        return

    output_path = None
    writer = None
    processed_count = 0
    fallback_names = getattr(model, "names", None)
    try:
        for frame_index in range(1, total_frames + 1):
            frame = read_cv2_rgb_frame(capture)
            if frame is None:
                break

            yield format_yolo_evaluation_progress_html(
                frame_index - 1,
                total_frames,
                f"Processing frame {frame_index}/{total_frames}.",
            ), None
            try:
                result = run_yolo_on_frame(model, frame)
                output_frame = draw_yolo_detections(frame, result, fallback_names=fallback_names)
                if writer is None:
                    output_path, writer = open_cv2_video_writer(output_frame, fps, output_dir=output_dir)
                write_cv2_rgb_frame(writer, output_frame)
            except Exception as exc:
                message = f"YOLO preview failed while processing frame {frame_index}/{total_frames}: {exc}"
                yield _error_output(message, frame_index - 1, total_frames)
                return

            processed_count = frame_index
            yield format_yolo_evaluation_progress_html(
                processed_count,
                total_frames,
                f"Processed {processed_count}/{total_frames} frame(s).",
            ), None
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processed_count == 0 or output_path is None:
        yield _error_output("Evaluation video contains no readable frames.", 0, total_frames)
        return

    yield format_yolo_evaluation_progress_html(
        processed_count,
        total_frames,
        f"YOLO model preview complete for {processed_count}/{total_frames} frame(s).",
        state="complete",
    ), output_path


def preview_yolo_model_on_video(
    video_path,
    model_path,
    *,
    model_loader=None,
    video_reader=None,
    video_writer=None,
    output_dir=None,
):
    try:
        resolved_video_path, resolved_model_path = validate_yolo_evaluation_inputs(video_path, model_path)
    except YoloEvaluationError as exc:
        yield _error_output(str(exc))
        return

    if video_reader is not None or video_writer is not None:
        yield from _preview_yolo_model_on_video_from_reader(
            resolved_video_path,
            resolved_model_path,
            model_loader=model_loader,
            video_reader=video_reader,
            video_writer=video_writer,
            output_dir=output_dir,
        )
        return

    yield from _preview_yolo_model_on_video_streaming(
        resolved_video_path,
        resolved_model_path,
        model_loader=model_loader,
        output_dir=output_dir,
    )
