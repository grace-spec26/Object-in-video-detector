# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py 
# We updated it to work with CoTracker3 models. We thank authors of LocoTrack
# for such an amazing Gradio demo.

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["GRADIO_SKIP_PYI_GENERATION"] = "1"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")


LOCAL_GRADIO_STARTUP_HOSTS = ("127.0.0.1", "localhost", "::1")


def ensure_localhost_no_proxy():
    for env_name in ("NO_PROXY", "no_proxy"):
        existing_entries = [
            entry.strip()
            for entry in os.environ.get(env_name, "").split(",")
            if entry.strip()
        ]
        for host in LOCAL_GRADIO_STARTUP_HOSTS:
            if host not in existing_entries:
                existing_entries.append(host)
        os.environ[env_name] = ",".join(existing_entries)


import importlib.metadata as importlib_metadata
import inspect
from contextlib import contextmanager
import sys
from pathlib import Path


@contextmanager
def suppress_importlib_entry_points():
    original_entry_points = importlib_metadata.entry_points
    importlib_metadata.entry_points = lambda *args, **kwargs: {}
    try:
        yield
    finally:
        importlib_metadata.entry_points = original_entry_points


with suppress_importlib_entry_points():
    import gradio as gr
    from gradio import data_classes as gradio_data_classes
    from gradio import networking as gradio_networking

import mediapy
import numpy as np

try:
    from .refinement_helpers import (
        append_refinement_point,
        clear_all_refinement_points,
        clear_frame_refinement_points,
        count_frame_points,
        drop_prompt_source,
        empty_frame_points,
        ensure_frame_points,
        flatten_prompt_sources,
        merge_frame_point_lists,
        pending_refinement_points,
        pop_refinement_point,
        remove_prompt_by_source,
        remove_nearest_frame_point,
        remove_nearest_refinement_point,
    )
    from .tracking_helpers import (
        DEFAULT_TRACKING_RESOLUTION,
        TRACKING_RESOLUTION_OPTIONS,
        parse_frame_skip_count,
        parse_max_frame_count,
        resize_video_for_tracking,
        sample_video_for_frame_skip,
        trim_video_to_frame_range,
    )
except ImportError:
    from refinement_helpers import (
        append_refinement_point,
        clear_all_refinement_points,
        clear_frame_refinement_points,
        count_frame_points,
        drop_prompt_source,
        empty_frame_points,
        ensure_frame_points,
        flatten_prompt_sources,
        merge_frame_point_lists,
        pending_refinement_points,
        pop_refinement_point,
        remove_prompt_by_source,
        remove_nearest_frame_point,
        remove_nearest_refinement_point,
    )
    from tracking_helpers import (
        DEFAULT_TRACKING_RESOLUTION,
        TRACKING_RESOLUTION_OPTIONS,
        parse_frame_skip_count,
        parse_max_frame_count,
        resize_video_for_tracking,
        sample_video_for_frame_skip,
        trim_video_to_frame_range,
    )


try:
    from .ui_layout import build_demo_layout, frame_slider_maximum
except ImportError:
    from ui_layout import build_demo_layout, frame_slider_maximum


try:
    from . import sam_preview_service as sam_preview_service
    from .sam_preview_service import (
        DEFAULT_SAM_IMAGE_MODEL,
        POINT_COLORS,
        POINT_PROMPT_RADIUS,
        SAM_IMAGE_MODEL_CHOICES,
        SAM_MODEL_PROGRESS_READY,
        SAM_VIDEO_PROGRESS_READY,
        current_sam_model_progress_html,
        download_all_sam_image_models_with_progress,
        draw_query_point,
        get_sam_preview_runtime,
        get_sam_preview_runtime_if_ready,
        processed_sam_model_switch_preview_with_progress,
        export_selected_sam_frame_as_yolo_train,
        export_selected_sam_frame_as_yolo_val,
        preview_sam_for_selected_frame,
        preview_sam_for_selected_frame_with_progress,
        preview_sam_on_frame,
        preview_sam_on_frame_with_progress,
        preview_sam_prompt_frame_on_model_switch,
        preview_sam_video_for_processed_frames,
        resolve_sam_preview_model_option,
        sam_checkpoint_file_looks_unavailable,
        sam_model_checkpoint_download_progress,
        sam_preview_preload_errors,
        sam_preview_preload_lock,
        sam_preview_preload_started,
        sam_preview_runtime_lock,
        sam_preview_runtimes,
        sam_model_switch_preview_with_progress,
        start_sam_preview_preload,
    )
except ImportError:
    import sam_preview_service as sam_preview_service
    from sam_preview_service import (
        DEFAULT_SAM_IMAGE_MODEL,
        POINT_COLORS,
        POINT_PROMPT_RADIUS,
        SAM_IMAGE_MODEL_CHOICES,
        SAM_MODEL_PROGRESS_READY,
        SAM_VIDEO_PROGRESS_READY,
        current_sam_model_progress_html,
        download_all_sam_image_models_with_progress,
        draw_query_point,
        get_sam_preview_runtime,
        get_sam_preview_runtime_if_ready,
        processed_sam_model_switch_preview_with_progress,
        export_selected_sam_frame_as_yolo_train,
        export_selected_sam_frame_as_yolo_val,
        preview_sam_for_selected_frame,
        preview_sam_for_selected_frame_with_progress,
        preview_sam_on_frame,
        preview_sam_on_frame_with_progress,
        preview_sam_prompt_frame_on_model_switch,
        preview_sam_video_for_processed_frames,
        resolve_sam_preview_model_option,
        sam_checkpoint_file_looks_unavailable,
        sam_model_checkpoint_download_progress,
        sam_preview_preload_errors,
        sam_preview_preload_lock,
        sam_preview_preload_started,
        sam_preview_runtime_lock,
        sam_preview_runtimes,
        sam_model_switch_preview_with_progress,
        start_sam_preview_preload,
    )


try:
    from . import tracking_service as tracking_service
    from .tracking_service import paint_point_track, run_cotracker_tracking
except ImportError:
    import tracking_service as tracking_service
    from tracking_service import paint_point_track, run_cotracker_tracking


try:
    from .yolo_evaluation_service import (
        YOLO_EVALUATION_PROGRESS_READY,
        preview_yolo_model_on_video,
    )
except ImportError:
    from yolo_evaluation_service import (
        YOLO_EVALUATION_PROGRESS_READY,
        preview_yolo_model_on_video,
    )


def patch_gradio_predict_body():
    """Allow Gradio 3.35 request models to run with Pydantic 2."""
    fields = getattr(gradio_data_classes.PredictBody, "model_fields", None)
    if not fields:
        return

    for field_name in ("session_hash", "event_id", "event_data", "fn_index", "request"):
        if field_name in fields:
            fields[field_name].default = None

    gradio_data_classes.PredictBody.model_rebuild(force=True)


patch_gradio_predict_body()
gradio_networking.url_ok = lambda _: True


PREVIEW_WIDTH = 768 # Width of the preview video
DEFAULT_MAX_FRAMES = parse_max_frame_count(os.environ.get("COTRACKER_MAX_FRAMES", "0"))
POSITIVE_POINT_CHOICE = "Positive (+)"
NEGATIVE_POINT_CHOICE = "Negative (-)"
POINT_TYPE_CHOICES = (POSITIVE_POINT_CHOICE, NEGATIVE_POINT_CHOICE)
POINT_LABEL_BY_CHOICE = {
    POSITIVE_POINT_CHOICE: 1,
    NEGATIVE_POINT_CHOICE: 0,
}
POINT_ADD_MODE = "Add"
POINT_DELETE_NEAREST_MODE = "Delete nearest"
POINT_EDIT_MODE_CHOICES = (POINT_ADD_MODE, POINT_DELETE_NEAREST_MODE)
REFINEMENT_ADD_MODE = "Add"
REFINEMENT_DELETE_MODE = "Delete nearest"
REFINEMENT_EDIT_MODE_CHOICES = (REFINEMENT_ADD_MODE, REFINEMENT_DELETE_MODE)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_DATASET_DIR = PROJECT_ROOT / "dataset"
def point_label_from_choice(point_type):
    return POINT_LABEL_BY_CHOICE.get(str(point_type), 1)


def unpack_query_point(point):
    x, y, frame_index = point[:3]
    point_label = int(point[3]) if len(point) >= 4 else 1
    return x, y, frame_index, point_label


def flatten_query_point_labels(query_points):
    labels = []
    for frame_points in query_points:
        for point in frame_points:
            _, _, _, point_label = unpack_query_point(point)
            labels.append(point_label)
    return labels


def redraw_query_frame(frame, frame_points):
    frame_draw = np.asarray(frame).copy()
    for point in frame_points or []:
        x, y, _, point_label = unpack_query_point(point)
        frame_draw = draw_query_point(frame_draw, x, y, point_label)
    return frame_draw


def get_point(
    frame_num,
    point_type,
    point_edit_mode,
    video_preview,
    video_queried_preview,
    query_points,
    query_points_color,
    query_count,
    evt: gr.SelectData,
):
    print(f"You selected {(evt.index[0], evt.index[1], frame_num)}")

    frame_index = int(frame_num)
    x, y = evt.index

    if str(point_edit_mode) == POINT_DELETE_NEAREST_MODE:
        max_distance = max(18.0, POINT_PROMPT_RADIUS * 6.0)
        updated_points, updated_colors, removed = remove_nearest_frame_point(
            query_points,
            query_points_color,
            frame_index=frame_index,
            x=x,
            y=y,
            max_distance=max_distance,
        )
        if removed:
            query_points = updated_points
            query_points_color = updated_colors
            query_count = max(0, int(query_count) - 1)
            current_frame_draw = redraw_query_frame(video_preview[frame_index], query_points[frame_index])
            video_queried_preview[frame_index] = current_frame_draw
        else:
            current_frame_draw = video_queried_preview[frame_index]

        has_points = count_frame_points(query_points) > 0
        return (
            current_frame_draw,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
            gr.update(interactive=has_points),
            gr.update(interactive=has_points),
        )

    # Get the mouse click
    point_label = point_label_from_choice(point_type)
    query_points[frame_index].append((x, y, frame_index, point_label))

    # Choose the color for the point from matplotlib colormap
    color = POINT_COLORS[point_label]
    # print(f"Color: {color}")
    query_points_color[frame_index].append(color)

    # Draw the point on the frame
    current_frame = video_queried_preview[frame_index]
    current_frame_draw = draw_query_point(current_frame, x, y, point_label)

    # Update the frame
    video_queried_preview[frame_index] = current_frame_draw

    # Update the query count
    query_count = int(query_count) + 1
    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count, # Updated query count
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def undo_point(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    if len(query_points[int(frame_num)]) == 0:
        return (
            video_queried_preview[int(frame_num)],
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        )

    # Get the last point
    query_points[int(frame_num)].pop(-1)
    query_points_color[int(frame_num)].pop(-1)

    # Redraw the frame
    current_frame_draw = video_preview[int(frame_num)].copy()
    current_frame_draw = redraw_query_frame(video_preview[int(frame_num)], query_points[int(frame_num)])

    # Update the query count
    query_count -= 1

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw
    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count # Updated query count
    )


def clear_frame_fn(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    query_count -= len(query_points[int(frame_num)])

    query_points[int(frame_num)] = []
    query_points_color[int(frame_num)] = []

    video_queried_preview[int(frame_num)] = video_preview[int(frame_num)].copy()

    return (
        video_preview[int(frame_num)], # Set the preview frame to the original frame
        video_queried_preview, 
        query_points, # Cleared query points
        query_points_color, # Cleared query points color
        query_count # New query count
    )



def clear_all_fn(frame_num, video_preview):
    return (
        video_preview[int(frame_num)],
        video_preview.copy(),
        [[] for _ in range(len(video_preview))],
        [[] for _ in range(len(video_preview))],
        0
    )


def choose_frame(frame_num, video_preview_array):
    return video_preview_array[int(frame_num)]


def draw_refinement_points_on_frame(frame, frame_points):
    if frame is None:
        return None

    frame_draw = np.asarray(frame).copy()
    for point in frame_points or []:
        x, y, _, point_label = unpack_query_point(point)
        frame_draw = draw_query_point(frame_draw, x, y, point_label)
    return frame_draw


def choose_tracked_frame(
    frame_num,
    tracked_video_preview,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    if tracked_video_preview is None:
        return None

    frame_index = int(np.clip(int(frame_num), 0, len(tracked_video_preview) - 1))
    pending_points = pending_refinement_points(
        refinement_query_points,
        tracked_prompt_sources,
        frame_count=len(tracked_video_preview),
    )
    return draw_refinement_points_on_frame(
        tracked_video_preview[frame_index],
        pending_points[frame_index],
    )


def refinement_colors_for_points(refinement_query_points, frame_count):
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    return [
        [POINT_COLORS.get(unpack_query_point(point)[3], POINT_COLORS[1]) for point in frame_points]
        for frame_points in refinement_query_points
    ]


def merge_query_point_colors(query_points_color, refinement_query_points, frame_count):
    merged_colors = [list(frame_colors) for frame_colors in (query_points_color or [])]
    while len(merged_colors) < frame_count:
        merged_colors.append([])
    merged_colors = merged_colors[:frame_count]
    refinement_colors = refinement_colors_for_points(refinement_query_points, frame_count)
    return [
        base_colors + added_colors
        for base_colors, added_colors in zip(merged_colors, refinement_colors)
    ]


def normalized_prompt_sources(prompt_sources):
    return [tuple(source) for source in (prompt_sources or [])]


def prompt_source_exists(query_points, refinement_query_points, source):
    if len(source) < 3:
        return False

    kind, frame_index, point_index = source[:3]
    kind = str(kind)
    try:
        frame_index = int(frame_index)
        point_index = int(point_index)
    except (TypeError, ValueError):
        return False

    if frame_index < 0 or point_index < 0:
        return False

    points_by_frame = refinement_query_points if kind == "refinement" else query_points
    if kind not in ("base", "refinement") or points_by_frame is None:
        return False
    if frame_index >= len(points_by_frame):
        return False
    return point_index < len(points_by_frame[frame_index])


def prompt_sources_match_current_state(
    query_points,
    refinement_query_points,
    sources,
    track_count,
):
    return (
        len(sources) == track_count
        and all(
            prompt_source_exists(query_points, refinement_query_points, source)
            for source in sources
        )
    )


def prompt_sources_for_tracks(query_points, refinement_query_points, selected_tracks, tracked_prompt_sources):
    if selected_tracks is None:
        return []

    track_count = int(np.asarray(selected_tracks).shape[0])
    sources = normalized_prompt_sources(tracked_prompt_sources)
    if prompt_sources_match_current_state(
        query_points,
        refinement_query_points,
        sources,
        track_count,
    ):
        return sources

    base_sources = flatten_prompt_sources(query_points, None)
    if len(base_sources) == track_count:
        return base_sources

    merged_sources = flatten_prompt_sources(query_points, refinement_query_points)
    if len(merged_sources) >= track_count:
        return merged_sources[:track_count]
    return sources


def color_for_prompt_source(query_points_color, refinement_query_points, source):
    kind, frame_index, point_index = source
    kind = str(kind)
    frame_index = int(frame_index)
    point_index = int(point_index)
    if kind == "base":
        if (
            query_points_color is not None
            and 0 <= frame_index < len(query_points_color)
            and 0 <= point_index < len(query_points_color[frame_index])
        ):
            return tuple(query_points_color[frame_index][point_index])
        return POINT_COLORS[1]

    refinement_query_points = ensure_frame_points(
        refinement_query_points,
        max(len(refinement_query_points or []), frame_index + 1),
    )
    if 0 <= frame_index < len(refinement_query_points) and 0 <= point_index < len(refinement_query_points[frame_index]):
        _, _, _, point_label = unpack_query_point(refinement_query_points[frame_index][point_index])
        return POINT_COLORS.get(point_label, POINT_COLORS[1])
    return POINT_COLORS[1]


def colors_for_prompt_sources(query_points_color, refinement_query_points, prompt_sources):
    return np.asarray(
        [
            color_for_prompt_source(query_points_color, refinement_query_points, source)
            for source in normalized_prompt_sources(prompt_sources)
        ],
        dtype=np.uint8,
    )


def nearest_visible_track_index(selected_tracks, selected_visibility, frame_index, x, y, max_distance):
    if selected_tracks is None:
        return None, None

    tracks = np.asarray(selected_tracks)
    if tracks.ndim != 3 or tracks.shape[0] == 0:
        return None, None

    frame_index = int(np.clip(int(frame_index), 0, tracks.shape[1] - 1))
    if selected_visibility is None:
        visible = np.ones((tracks.shape[0],), dtype=bool)
    else:
        visibility = np.asarray(selected_visibility)
        if visibility.shape[:2] != tracks.shape[:2]:
            visible = np.ones((tracks.shape[0],), dtype=bool)
        else:
            visible = visibility[:, frame_index].astype(bool)

    visible_indices = np.flatnonzero(visible)
    if len(visible_indices) == 0:
        return None, None

    frame_tracks = tracks[visible_indices, frame_index, :]
    distances = np.linalg.norm(frame_tracks - np.asarray([x, y], dtype=np.float32), axis=1)
    nearest_visible_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_visible_index])
    if nearest_distance > float(max_distance):
        return None, None
    return int(visible_indices[nearest_visible_index]), nearest_distance


def remove_track_prompt_from_state(
    query_points,
    query_points_color,
    refinement_query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    tracked_prompt_sources,
    track_index,
):
    sources = normalized_prompt_sources(tracked_prompt_sources)
    track_index = int(track_index)
    tracks = np.asarray(selected_tracks)
    if track_index < 0 or track_index >= tracks.shape[0]:
        return (
            query_points,
            query_points_color,
            refinement_query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            sources,
            False,
        )

    updated_query_points = query_points
    updated_query_colors = query_points_color
    updated_refinements = refinement_query_points
    if track_index < len(sources):
        updated_query_points, updated_query_colors, updated_refinements, _ = remove_prompt_by_source(
            query_points,
            query_points_color,
            refinement_query_points,
            sources[track_index],
        )

    updated_tracks = np.delete(tracks, track_index, axis=0)
    updated_visibility = (
        np.delete(np.asarray(selected_visibility), track_index, axis=0)
        if selected_visibility is not None
        else selected_visibility
    )
    updated_labels = list(selected_point_labels or [])
    if track_index < len(updated_labels):
        del updated_labels[track_index]
    updated_sources = drop_prompt_source(sources, track_index)
    return (
        updated_query_points,
        updated_query_colors,
        updated_refinements,
        updated_tracks,
        updated_visibility,
        updated_labels,
        updated_sources,
        True,
    )


def repaint_tracked_video_preview(
    video_preview,
    selected_tracks,
    selected_visibility,
    query_points_color,
    refinement_query_points,
    tracked_prompt_sources,
):
    if video_preview is None:
        return None
    if selected_tracks is None:
        return video_preview

    tracks = np.asarray(selected_tracks)
    if tracks.ndim != 3 or tracks.shape[0] == 0:
        return np.asarray(video_preview).copy()

    colors = colors_for_prompt_sources(query_points_color, refinement_query_points, tracked_prompt_sources)
    visibility = (
        np.asarray(selected_visibility)
        if selected_visibility is not None
        else np.ones(tracks.shape[:2], dtype=bool)
    )
    return paint_point_track(
        np.asarray(video_preview),
        tracks,
        visibility,
        colors,
    )


def edit_refinement_point(
    frame_num,
    refinement_edit_mode,
    refinement_point_type,
    video_preview,
    tracked_video_preview,
    query_points,
    query_points_color,
    query_count,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    tracked_prompt_sources,
    refinement_query_points,
    evt: gr.SelectData,
):
    if tracked_video_preview is None:
        message = "Track a video before editing processed-frame points."
        gr.Warning(message, duration=5)
        return (
            None,
            tracked_video_preview,
            query_points,
            query_points_color,
            query_count,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            refinement_query_points,
            gr.update(interactive=False),
            message,
        )

    frame_count = len(tracked_video_preview)
    frame_index = int(np.clip(int(frame_num), 0, frame_count - 1))
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    tracked_prompt_sources = prompt_sources_for_tracks(
        query_points,
        refinement_query_points,
        selected_tracks,
        tracked_prompt_sources,
    )
    x, y = evt.index

    if str(refinement_edit_mode) == REFINEMENT_DELETE_MODE:
        max_distance = max(18.0, POINT_PROMPT_RADIUS * 6.0)
        track_index, _ = nearest_visible_track_index(
            selected_tracks,
            selected_visibility,
            frame_index=frame_index,
            x=x,
            y=y,
            max_distance=max_distance,
        )
        if track_index is not None:
            (
                updated_query_points,
                updated_query_colors,
                updated_points,
                updated_tracks,
                updated_visibility,
                updated_labels,
                updated_sources,
                removed,
            ) = remove_track_prompt_from_state(
                query_points,
                query_points_color,
                refinement_query_points,
                selected_tracks,
                selected_visibility,
                selected_point_labels,
                tracked_prompt_sources,
                track_index,
            )
            if removed:
                updated_tracked_video = repaint_tracked_video_preview(
                    video_preview,
                    updated_tracks,
                    updated_visibility,
                    updated_query_colors,
                    updated_points,
                    updated_sources,
                )
                editable_prompt_count = count_frame_points(updated_query_points) + count_frame_points(updated_points)
                return (
                    choose_tracked_frame(frame_index, updated_tracked_video, updated_points, updated_sources),
                    updated_tracked_video,
                    updated_query_points,
                    updated_query_colors,
                    count_frame_points(updated_query_points),
                    updated_tracks,
                    updated_visibility,
                    updated_labels,
                    updated_sources,
                    updated_points,
                    gr.update(interactive=editable_prompt_count > 0),
                    f"Removed tracked point prompt on frame {frame_index}; it is deleted from all processed frames.",
                )

        updated_points, removed = remove_nearest_refinement_point(
            refinement_query_points,
            frame_index=frame_index,
            x=x,
            y=y,
            max_distance=max_distance,
        )
        message = f"Removed refinement point on frame {frame_index}." if removed else f"No editable point near click on frame {frame_index}."
    else:
        point_label = point_label_from_choice(refinement_point_type)
        updated_points = append_refinement_point(
            refinement_query_points,
            frame_index=frame_index,
            x=x,
            y=y,
            label=point_label,
        )
        point_name = "positive" if point_label == 1 else "negative"
        message = f"Added {point_name} refinement point on frame {frame_index}."

    return (
        choose_tracked_frame(frame_index, tracked_video_preview, updated_points, tracked_prompt_sources),
        tracked_video_preview,
        query_points,
        query_points_color,
        query_count,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        tracked_prompt_sources,
        updated_points,
        gr.update(interactive=(count_frame_points(query_points) + count_frame_points(updated_points)) > 0),
        message,
    )


def undo_refinement_point(frame_num, tracked_video_preview, refinement_query_points, tracked_prompt_sources=None):
    if tracked_video_preview is None:
        message = "Track a video before editing processed-frame points."
        gr.Warning(message, duration=5)
        return None, refinement_query_points, gr.update(interactive=False), message

    frame_count = len(tracked_video_preview)
    frame_index = int(np.clip(int(frame_num), 0, frame_count - 1))
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    updated_points, removed = pop_refinement_point(refinement_query_points, frame_index)
    message = (
        f"Removed last refinement point on frame {frame_index}."
        if removed
        else f"No refinement points on frame {frame_index}."
    )
    return (
        choose_tracked_frame(frame_index, tracked_video_preview, updated_points, tracked_prompt_sources),
        updated_points,
        gr.update(interactive=count_frame_points(updated_points) > 0),
        message,
    )


def clear_frame_refinement_edits(frame_num, tracked_video_preview, refinement_query_points, tracked_prompt_sources=None):
    if tracked_video_preview is None:
        message = "Track a video before editing processed-frame points."
        gr.Warning(message, duration=5)
        return None, refinement_query_points, gr.update(interactive=False), message

    frame_count = len(tracked_video_preview)
    frame_index = int(np.clip(int(frame_num), 0, frame_count - 1))
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    had_points = len(refinement_query_points[frame_index]) > 0
    updated_points = clear_frame_refinement_points(refinement_query_points, frame_index)
    message = (
        f"Cleared refinement points on frame {frame_index}."
        if had_points
        else f"No refinement points to clear on frame {frame_index}."
    )
    return (
        choose_tracked_frame(frame_index, tracked_video_preview, updated_points, tracked_prompt_sources),
        updated_points,
        gr.update(interactive=count_frame_points(updated_points) > 0),
        message,
    )


def clear_all_refinement_edits(frame_num, tracked_video_preview, refinement_query_points, tracked_prompt_sources=None):
    if tracked_video_preview is None:
        message = "Track a video before editing processed-frame points."
        gr.Warning(message, duration=5)
        return None, refinement_query_points, gr.update(interactive=False), message

    frame_count = len(tracked_video_preview)
    frame_index = int(np.clip(int(frame_num), 0, frame_count - 1))
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    had_points = count_frame_points(refinement_query_points) > 0
    updated_points = clear_all_refinement_points(refinement_query_points)
    message = "Cleared all refinement points." if had_points else "No refinement points to clear."
    return (
        choose_tracked_frame(frame_index, tracked_video_preview, updated_points, tracked_prompt_sources),
        updated_points,
        gr.update(interactive=False),
        message,
    )


def preprocess_video_input(
    video_path,
    tracking_resolution,
    max_frames,
    skip_frames,
    trim_start_frame,
    trim_end_frame,
):
    if video_path is None:
        raise gr.Error("Please upload a video before submitting.")

    try:
        max_frames_to_load = parse_max_frame_count(max_frames)
        frame_skip_count = parse_frame_skip_count(skip_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    start_sam_preview_preload(DEFAULT_SAM_IMAGE_MODEL)

    video_arr = mediapy.read_video(video_path)
    source_video_fps = float(video_arr.metadata.fps)
    video_fps = source_video_fps
    num_frames = video_arr.shape[0]
    original_num_frames = num_frames
    try:
        video_arr, trim_start_index, trim_end_index = trim_video_to_frame_range(
            video_arr,
            start_frame=trim_start_frame,
            end_frame=trim_end_frame,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    num_frames = video_arr.shape[0]
    trimmed_source_frames = num_frames
    loaded_source_frames = num_frames
    if max_frames_to_load and num_frames > max_frames_to_load:
        gr.Warning(
            f"Only the first {max_frames_to_load} of {trimmed_source_frames} trimmed frame(s) will be used.",
            duration=5,
        )
        video_arr = video_arr[:max_frames_to_load]
        num_frames = max_frames_to_load
        loaded_source_frames = num_frames

    sampling_stride = 1
    if frame_skip_count > 0:
        video_arr, video_fps, sampling_stride = sample_video_for_frame_skip(
            video_arr,
            source_fps=source_video_fps,
            skip_count=frame_skip_count,
        )
        num_frames = video_arr.shape[0]
        if sampling_stride > 1:
            gr.Warning(
                (
                    "Sampling point-selection video by keeping 1 frame then skipping "
                    f"{frame_skip_count} frame(s), producing about {video_fps:.2f} FPS."
                ),
                duration=5,
            )

    # Resize to preview size for faster processing, width = PREVIEW_WIDTH
    height, width = video_arr.shape[1:3]
    new_height, new_width = int(PREVIEW_WIDTH * height / width), PREVIEW_WIDTH

    preview_video = mediapy.resize_video(video_arr, (new_height, new_width))
    input_video = resize_video_for_tracking(video_arr, tracking_resolution)

    preview_video = np.array(preview_video)
    input_video = np.array(input_video)

    interactive = True

    load_status_parts = []
    if trim_start_index != 0 or trim_end_index != original_num_frames:
        load_status_parts.append(
            (
                f"Trimmed source video to frames {trim_start_index}-{trim_end_index - 1} "
                f"({trimmed_source_frames} frame(s))."
            )
        )

    if loaded_source_frames != trimmed_source_frames and sampling_stride > 1:
        load_status_parts.append(
            (
                f"Loaded first {loaded_source_frames} of {trimmed_source_frames} trimmed frame(s), "
                "then sampled "
                f"to {num_frames} frame(s) by skipping {frame_skip_count} frame(s) after each loaded frame. "
                "Set Max frames to load to 0 to use the full trimmed video."
            )
        )
    elif loaded_source_frames != trimmed_source_frames:
        load_status_parts.append(
            (
                f"Loaded first {loaded_source_frames} of {trimmed_source_frames} trimmed frame(s) for point selection. "
                "Set Max frames to load to 0 to use the full trimmed video."
            )
        )
    elif sampling_stride > 1:
        load_status_parts.append(
            (
                f"Loaded {num_frames} sampled frame(s) for point selection by skipping "
                f"{frame_skip_count} frame(s) after each loaded frame."
            )
        )
    else:
        load_status_parts.append(f"Loaded {num_frames} frame(s) for point selection.")
    load_status = " ".join(load_status_parts)

    return (
        video_arr, # Original video
        preview_video, # Original preview video, resized for faster processing
        preview_video.copy(), # Copy of preview video for visualization
        input_video, # Resized video input for model
        # None, # video_feature, # Extracted feature
        video_fps, # Set the video FPS
        gr.update(open=False), # Close the video input drawer
        # tracking_mode, # Set the tracking mode
        preview_video[0], # Set the preview frame to the first frame
        gr.update(minimum=0, maximum=frame_slider_maximum(num_frames), value=0, interactive=interactive), # Set slider interactive
        [[] for _ in range(num_frames)], # Set query_points to empty
        [[] for _ in range(num_frames)], # Set query_points_color to empty
        [[] for _ in range(num_frames)], 
        0, # Set query count to 0
        gr.update(interactive=interactive), # Make the buttons interactive
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
        gr.update(interactive=True),
        None,
        None,
        None,
        [],
        gr.update(interactive=True),
        gr.update(interactive=False),
        current_sam_model_progress_html(DEFAULT_SAM_IMAGE_MODEL),
        gr.update(interactive=False),
        None,
        gr.update(interactive=False),
        current_sam_model_progress_html(DEFAULT_SAM_IMAGE_MODEL),
        gr.update(interactive=False),
        None,
        gr.update(interactive=False),
        gr.update(interactive=False),
        None,
        gr.update(minimum=0, maximum=frame_slider_maximum(0), value=0, interactive=False),
        None,
        empty_frame_points(num_frames),
        gr.update(interactive=False),
        load_status,
    )


def track(
    video_preview,
    video_input, 
    video_fps, 
    query_points, 
    query_points_color, 
    query_count, 
):
    result = run_cotracker_tracking(
        video_preview=video_preview,
        video_input=video_input,
        video_fps=video_fps,
        query_points=query_points,
        query_points_color=query_points_color,
        query_count=query_count,
    )
    has_selected_points = result.has_selected_points
    total_frame_count = result.total_frame_count
    painted_video = result.painted_video
    return (
        result.video_file_path,
        result.tracks,
        result.visibility,
        result.selected_point_labels,
        painted_video,
        gr.update(minimum=0, maximum=frame_slider_maximum(total_frame_count), value=0, interactive=True),
        painted_video[0],
        gr.update(interactive=True),
        gr.update(interactive=has_selected_points),
        current_sam_model_progress_html(DEFAULT_SAM_IMAGE_MODEL),
        gr.update(interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        result.export_status,
    )

def track_and_reset_refinements(
    video_preview,
    video_input,
    video_fps,
    query_points,
    query_points_color,
    query_count,
):
    result = track(
        video_preview,
        video_input,
        video_fps,
        query_points,
        query_points_color,
        query_count,
    )
    total_frame_count = video_input.shape[0]
    tracked_prompt_sources = flatten_prompt_sources(query_points, None) if query_count > 0 else []
    return (
        *result[:4],
        tracked_prompt_sources,
        *result[4:7],
        empty_frame_points(total_frame_count),
        gr.update(interactive=False),
        *result[7:],
    )


def _unchanged_reprocess_outputs(message):
    return (*[gr.update() for _ in range(14)], message)


def reprocess_with_refinements(
    video_preview,
    video_input,
    video_fps,
    query_points,
    query_points_color,
    refinement_query_points,
    tracked_frame_num,
    processed_sam_model=DEFAULT_SAM_IMAGE_MODEL,
):
    if video_preview is None or video_input is None:
        message = "Track a video before re-processing refinement points."
        gr.Warning(message, duration=5)
        return _unchanged_reprocess_outputs(message)

    frame_count = int(video_preview.shape[0])
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    merged_query_points = merge_frame_point_lists(query_points, refinement_query_points)
    merged_query_count = count_frame_points(merged_query_points)
    if merged_query_count == 0:
        message = "Add or select at least one point prompt before re-processing."
        gr.Warning(message, duration=5)
        return _unchanged_reprocess_outputs(message)

    merged_query_colors = merge_query_point_colors(
        query_points_color,
        refinement_query_points,
        frame_count,
    )
    result = list(
        track(
            video_preview,
            video_input,
            video_fps,
            merged_query_points,
            merged_query_colors,
            merged_query_count,
        )
    )
    frame_index = int(np.clip(int(tracked_frame_num), 0, frame_count - 1))
    tracked_prompt_sources = flatten_prompt_sources(query_points, refinement_query_points)
    result[6] = choose_tracked_frame(
        frame_index,
        result[4],
        refinement_query_points,
        tracked_prompt_sources,
    )
    result[9] = current_sam_model_progress_html(processed_sam_model)
    result[11] = gr.update(interactive=True)
    result[12] = gr.update(interactive=True)
    result[-1] = (
        f"Re-processing complete with {merged_query_count} point prompt(s). "
        "Query points on video has been replaced."
    )
    return (
        *result[:4],
        tracked_prompt_sources,
        *result[4:],
    )


def export_no_wound_frames_from_state(video_frames):
    if video_frames is None:
        message = "Submit a video before exporting no-wound frames."
        gr.Warning(message, duration=5)
        return message

    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from export_yolo_segmentation_dataset import export_no_wound_frames_to_yolo_dataset

        stats = export_no_wound_frames_to_yolo_dataset(
            frames=video_frames,
            output_dir=DEFAULT_YOLO_DATASET_DIR,
            train_ratio=0.8,
        )
    except Exception as exc:
        message = f"Failed to export no-wound frames: {exc}"
        gr.Warning(message, duration=5)
        return message

    return (
        f"Exported {stats.exported_images} no-wound frame(s) to {DEFAULT_YOLO_DATASET_DIR}: "
        f"{stats.train_images} train and {stats.val_images} val image(s), "
        "each with an empty label."
    )


def configure_demo_callbacks(layout):
    video = layout.video
    video_queried_preview = layout.video_queried_preview
    video_preview = layout.video_preview
    video_input = layout.video_input
    video_fps = layout.video_fps
    query_points = layout.query_points
    query_points_color = layout.query_points_color
    is_tracked_query = layout.is_tracked_query
    query_count = layout.query_count
    selected_tracks = layout.selected_tracks
    selected_visibility = layout.selected_visibility
    selected_point_labels = layout.selected_point_labels
    tracked_prompt_sources = layout.tracked_prompt_sources
    tracked_video_preview = layout.tracked_video_preview
    refinement_query_points = layout.refinement_query_points
    video_in_drawer = layout.video_in_drawer
    video_in = layout.video_in
    tracking_resolution = layout.tracking_resolution
    trim_start_frame_input = layout.trim_start_frame_input
    trim_end_frame_input = layout.trim_end_frame_input
    max_frames_input = layout.max_frames_input
    skip_frames_input = layout.skip_frames_input
    submit = layout.submit
    query_frames = layout.query_frames
    point_type = layout.point_type
    query_point_edit_mode = layout.query_point_edit_mode
    undo = layout.undo
    clear_frame = layout.clear_frame
    clear_all = layout.clear_all
    current_frame = layout.current_frame
    track_button = layout.track_button
    output_video = layout.output_video
    no_wound_export_button = layout.no_wound_export_button
    sam_model_dropdown = layout.sam_model_dropdown
    download_sam_models_button = layout.download_sam_models_button
    sam_model_loading_progress = layout.sam_model_loading_progress
    sam_preview_button = layout.sam_preview_button
    sam_preview_image = layout.sam_preview_image
    tracked_query_frames = layout.tracked_query_frames
    refinement_point_type = layout.refinement_point_type
    refinement_edit_mode = layout.refinement_edit_mode
    refinement_undo = layout.refinement_undo
    refinement_clear_frame = layout.refinement_clear_frame
    refinement_clear_all = layout.refinement_clear_all
    reprocess_button = layout.reprocess_button
    tracked_frame_preview = layout.tracked_frame_preview
    processed_sam_model_dropdown = layout.processed_sam_model_dropdown
    processed_sam_model_loading_progress = layout.processed_sam_model_loading_progress
    processed_sam_preview_button = layout.processed_sam_preview_button
    processed_sam_preview_image = layout.processed_sam_preview_image
    export_status = layout.export_status
    save_sam_frame_train_button = layout.save_sam_frame_train_button
    save_sam_frame_val_button = layout.save_sam_frame_val_button
    yolo_dataset_output_dir = layout.yolo_dataset_output_dir
    evaluation_video_input = layout.evaluation_video_input
    evaluation_yolo_model_input = layout.evaluation_yolo_model_input
    evaluation_preview_button = layout.evaluation_preview_button
    evaluation_progress = layout.evaluation_progress
    evaluation_output_video = layout.evaluation_output_video

    submit.click(
        fn = preprocess_video_input, 
        inputs = [
            video_in,
            tracking_resolution,
            max_frames_input,
            skip_frames_input,
            trim_start_frame_input,
            trim_end_frame_input,
        ],
        outputs = [
            video,
            video_preview,
            video_queried_preview,
            video_input,
            video_fps,
            video_in_drawer,
            current_frame,
            query_frames,
            query_points,
            query_points_color,
            is_tracked_query,
            query_count,
            undo,
            clear_frame,
            clear_all,
            track_button,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            no_wound_export_button,
            sam_model_dropdown,
            sam_model_loading_progress,
            sam_preview_button,
            sam_preview_image,
            processed_sam_model_dropdown,
            processed_sam_model_loading_progress,
            processed_sam_preview_button,
            processed_sam_preview_image,
            save_sam_frame_train_button,
            save_sam_frame_val_button,
            tracked_video_preview,
            tracked_query_frames,
            tracked_frame_preview,
            refinement_query_points,
            reprocess_button,
            export_status,
        ],
        queue = False
    )

    query_frames.change(
        fn = choose_frame,
        inputs = [query_frames, video_queried_preview],
        outputs = [
            current_frame,
        ],
        queue = False
    )

    tracked_query_frames.change(
        fn = choose_tracked_frame,
        inputs = [tracked_query_frames, tracked_video_preview, refinement_query_points, tracked_prompt_sources],
        outputs = [
            tracked_frame_preview,
        ],
        queue = False
    )

    sam_model_dropdown.change(
        fn = sam_model_switch_preview_with_progress,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            sam_model_dropdown,
        ],
        outputs = [
            sam_preview_image,
            sam_model_loading_progress,
            export_status,
        ],
        queue = False,
    )

    processed_sam_model_dropdown.change(
        fn = processed_sam_model_switch_preview_with_progress,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            tracked_query_frames,
            tracked_video_preview,
            processed_sam_model_dropdown,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            processed_sam_preview_image,
            processed_sam_model_loading_progress,
            export_status,
        ],
        queue = False,
    )

    download_sam_models_button.click(
        fn = download_all_sam_image_models_with_progress,
        inputs = [],
        outputs = [
            sam_model_loading_progress,
            processed_sam_model_loading_progress,
            export_status,
        ],
    )

    current_frame.select(
        fn = get_point, 
        inputs = [
            query_frames,
            point_type,
            query_point_edit_mode,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
        ], 
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
            sam_model_dropdown,
            sam_preview_button,
        ], 
        queue = False
    )
    
    undo.click(
        fn = undo_point,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_frame.click(
        fn = clear_frame_fn,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_all.click(
        fn = clear_all_fn,
        inputs = [
            query_frames,
            video_preview,
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    
    track_button.click(
        fn = track_and_reset_refinements,
        inputs = [
            video_preview,
            video_input,
            video_fps,
            query_points,
            query_points_color,
            query_count,
        ],
        outputs = [
            output_video,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            tracked_video_preview,
            tracked_query_frames,
            tracked_frame_preview,
            refinement_query_points,
            reprocess_button,
            no_wound_export_button,
            processed_sam_model_dropdown,
            processed_sam_model_loading_progress,
            processed_sam_preview_button,
            save_sam_frame_train_button,
            save_sam_frame_val_button,
            export_status,
        ],
        queue = False,
    )

    tracked_frame_preview.select(
        fn = edit_refinement_point,
        inputs = [
            tracked_query_frames,
            refinement_edit_mode,
            refinement_point_type,
            video_preview,
            tracked_video_preview,
            query_points,
            query_points_color,
            query_count,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            refinement_query_points,
        ],
        outputs = [
            tracked_frame_preview,
            tracked_video_preview,
            query_points,
            query_points_color,
            query_count,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            refinement_query_points,
            reprocess_button,
            export_status,
        ],
        queue = False,
    )

    refinement_undo.click(
        fn = undo_refinement_point,
        inputs = [
            tracked_query_frames,
            tracked_video_preview,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            tracked_frame_preview,
            refinement_query_points,
            reprocess_button,
            export_status,
        ],
        queue = False,
    )

    refinement_clear_frame.click(
        fn = clear_frame_refinement_edits,
        inputs = [
            tracked_query_frames,
            tracked_video_preview,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            tracked_frame_preview,
            refinement_query_points,
            reprocess_button,
            export_status,
        ],
        queue = False,
    )

    refinement_clear_all.click(
        fn = clear_all_refinement_edits,
        inputs = [
            tracked_query_frames,
            tracked_video_preview,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            tracked_frame_preview,
            refinement_query_points,
            reprocess_button,
            export_status,
        ],
        queue = False,
    )

    reprocess_button.click(
        fn = reprocess_with_refinements,
        inputs = [
            video_preview,
            video_input,
            video_fps,
            query_points,
            query_points_color,
            refinement_query_points,
            tracked_query_frames,
            processed_sam_model_dropdown,
        ],
        outputs = [
            output_video,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_prompt_sources,
            tracked_video_preview,
            tracked_query_frames,
            tracked_frame_preview,
            no_wound_export_button,
            processed_sam_model_dropdown,
            processed_sam_model_loading_progress,
            processed_sam_preview_button,
            save_sam_frame_train_button,
            save_sam_frame_val_button,
            export_status,
        ],
        queue = False,
    )

    no_wound_export_button.click(
        fn = export_no_wound_frames_from_state,
        inputs = [
            video,
        ],
        outputs = [
            export_status,
        ],
        queue = False,
    )

    sam_preview_button.click(
        fn = preview_sam_on_frame_with_progress,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            sam_model_dropdown,
        ],
        outputs = [
            sam_preview_image,
            sam_model_loading_progress,
            export_status,
        ],
        queue = False,
    )

    processed_sam_preview_button.click(
        fn = preview_sam_for_selected_frame_with_progress,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            tracked_query_frames,
            tracked_video_preview,
            processed_sam_model_dropdown,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            processed_sam_preview_image,
            processed_sam_model_loading_progress,
            export_status,
        ],
        queue = False,
    )

    save_sam_frame_train_button.click(
        fn = export_selected_sam_frame_as_yolo_train,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            tracked_query_frames,
            tracked_video_preview,
            processed_sam_model_dropdown,
            refinement_query_points,
            tracked_prompt_sources,
            yolo_dataset_output_dir,
        ],
        outputs = [
            export_status,
        ],
        queue = False,
    )

    save_sam_frame_val_button.click(
        fn = export_selected_sam_frame_as_yolo_val,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            query_frames,
            tracked_query_frames,
            tracked_video_preview,
            processed_sam_model_dropdown,
            refinement_query_points,
            tracked_prompt_sources,
            yolo_dataset_output_dir,
        ],
        outputs = [
            export_status,
        ],
        queue = False,
    )

    evaluation_preview_button.click(
        fn = preview_yolo_model_on_video,
        inputs = [
            evaluation_video_input,
            evaluation_yolo_model_input,
        ],
        outputs = [
            evaluation_progress,
            evaluation_output_video,
        ],
        show_progress = "hidden",
    )



demo_layout = build_demo_layout(
    gr,
    base_dir=Path(__file__).resolve().parent,
    default_tracking_resolution=DEFAULT_TRACKING_RESOLUTION,
    tracking_resolution_options=TRACKING_RESOLUTION_OPTIONS,
    default_max_frames=DEFAULT_MAX_FRAMES,
    point_type_choices=POINT_TYPE_CHOICES,
    positive_point_choice=POSITIVE_POINT_CHOICE,
    point_edit_mode_choices=POINT_EDIT_MODE_CHOICES,
    point_add_mode=POINT_ADD_MODE,
    sam_image_model_choices=SAM_IMAGE_MODEL_CHOICES,
    default_sam_image_model=DEFAULT_SAM_IMAGE_MODEL,
    sam_model_progress_ready=SAM_MODEL_PROGRESS_READY,
    refinement_edit_mode_choices=REFINEMENT_EDIT_MODE_CHOICES,
    refinement_add_mode=REFINEMENT_ADD_MODE,
    default_yolo_dataset_dir=DEFAULT_YOLO_DATASET_DIR,
    yolo_evaluation_progress_ready=YOLO_EVALUATION_PROGRESS_READY,
    configure_callbacks=configure_demo_callbacks,
)
demo = demo_layout.demo

ensure_localhost_no_proxy()

launch_kwargs = {
    "server_name": "127.0.0.1",
    "server_port": int(os.environ.get("PORT", "7860")),
    "show_error": True,
    "share": False,
}
if "show_api" in inspect.signature(demo.launch).parameters:
    launch_kwargs["show_api"] = False

demo.launch(**launch_kwargs)
