# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py 
# We updated it to work with CoTracker3 models. We thank authors of LocoTrack
# for such an amazing Gradio demo.

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["GRADIO_SKIP_PYI_GENERATION"] = "1"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

import hashlib
import sys
import threading
import uuid
from pathlib import Path

import gradio as gr
from gradio import data_classes as gradio_data_classes
from gradio import networking as gradio_networking
import mediapy
import numpy as np
import cv2
import matplotlib
import colorsys
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

_torch_module = None


def get_torch():
    global _torch_module
    if _torch_module is not None:
        return _torch_module

    import importlib.metadata as importlib_metadata

    original_entry_points = importlib_metadata.entry_points
    importlib_metadata.entry_points = lambda *args, **kwargs: {}
    try:
        import torch as torch_module
    finally:
        importlib_metadata.entry_points = original_entry_points

    _torch_module = torch_module
    return _torch_module

try:
    from .export_helpers import (
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        visible_labeled_points_for_frame,
    )
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
        TRACKING_FRAME_STRIDE,
        TRACKING_RESOLUTION_OPTIONS,
        expand_sampled_time_axis,
        get_cached_cotracker_model,
        get_online_chunk_start_indices,
        map_frame_index_to_sampled,
        parse_frame_skip_count,
        parse_max_frame_count,
        resolve_torch_device,
        resize_video_for_tracking,
        sample_video_for_frame_skip,
        save_sam_video_review,
        should_process_frame_for_skip,
        subsample_video_tensor,
    )
except ImportError:
    from export_helpers import (
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        visible_labeled_points_for_frame,
    )
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
        TRACKING_FRAME_STRIDE,
        TRACKING_RESOLUTION_OPTIONS,
        expand_sampled_time_axis,
        get_cached_cotracker_model,
        get_online_chunk_start_indices,
        map_frame_index_to_sampled,
        parse_frame_skip_count,
        parse_max_frame_count,
        resolve_torch_device,
        resize_video_for_tracking,
        sample_video_for_frame_skip,
        save_sam_video_review,
        should_process_frame_for_skip,
        subsample_video_tensor,
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


# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[object] = None,
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    torch = get_torch()
    if device is None:
        device = torch.device("cpu")

    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool
    colormap: colormap for points, each point has a different RGB color.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  num_points, num_frames = point_tracks.shape[0:2]
  if colormap is None:
    colormap = get_colors(num_colors=num_points)
  height, width = frames.shape[1:3]
  dot_size_as_fraction_of_min_edge = 0.015
  radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
  diam = radius * 2 + 1
  quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
  quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
  icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
  sharpness = 0.15
  icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
  icon = 1 - icon[:, :, np.newaxis]
  icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
  icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
  icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
  icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

  video = frames.copy()
  for t in range(num_frames):
    # Pad so that points that extend outside the image frame don't crash us
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      # The icon is centered at the center of a pixel, but the input coordinates
      # are raster coordinates.  Therefore, to render a point at (1,1) (which
      # lies on the corner between four pixels), we need 1/4 of the icon placed
      # centered on the 0'th row, 0'th column, etc.  We need to subtract
      # 0.5 to make the fractional position come out right.
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1

        # bilinear interpolation
        patch = (
            icon1 * (x2 - x) * (y2 - y)
            + icon2 * (x2 - x) * (y - y1)
            + icon3 * (x - x1) * (y2 - y)
            + icon4 * (x - x1) * (y - y1)
        )
        x_ub = x1 + 2 * radius + 2
        y_ub = y1 + 2 * radius + 2
        image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
            y1:y_ub, x1:x_ub, :
        ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

      # Remove the pad
      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video


PREVIEW_WIDTH = 768 # Width of the preview video
POINT_PROMPT_RADIUS = 3
DEFAULT_MAX_FRAMES = parse_max_frame_count(os.environ.get("COTRACKER_MAX_FRAMES", "0"))
POSITIVE_POINT_CHOICE = "Positive (+)"
NEGATIVE_POINT_CHOICE = "Negative (-)"
POINT_TYPE_CHOICES = (POSITIVE_POINT_CHOICE, NEGATIVE_POINT_CHOICE)
POINT_LABEL_BY_CHOICE = {
    POSITIVE_POINT_CHOICE: 1,
    NEGATIVE_POINT_CHOICE: 0,
}
POINT_COLORS = {
    1: (0, 255, 0),
    0: (255, 0, 0),
}
POINT_ADD_MODE = "Add"
POINT_DELETE_NEAREST_MODE = "Delete nearest"
POINT_EDIT_MODE_CHOICES = (POINT_ADD_MODE, POINT_DELETE_NEAREST_MODE)
REFINEMENT_ADD_MODE = "Add"
REFINEMENT_DELETE_MODE = "Delete nearest"
REFINEMENT_EDIT_MODE_CHOICES = (REFINEMENT_ADD_MODE, REFINEMENT_DELETE_MODE)
SAM_IMAGE_MODEL_CHOICES = (
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
)
DEFAULT_SAM_IMAGE_MODEL = "sam2.1_hiera_small.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBILE_SAM_ROOT = PROJECT_ROOT / "MobileSAM-master"
DEFAULT_SAM_VIDEO_SAVE_DIR = PROJECT_ROOT / "sam-video-preview"
DEFAULT_RAW_MASK_ROOT = PROJECT_ROOT / "raw-mask-data"
DEFAULT_YOLO_DATASET_DIR = PROJECT_ROOT / "dataset"
sam_preview_runtime_lock = threading.Lock()
sam_preview_runtimes = {}
sam_preview_preload_lock = threading.Lock()
sam_preview_preload_started = set()
sam_preview_preload_errors = {}


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


def draw_query_point(frame, x, y, point_label):
    point_color = POINT_COLORS.get(int(point_label), POINT_COLORS[1])
    x, y = int(round(x)), int(round(y))
    frame = cv2.circle(frame, (x, y), POINT_PROMPT_RADIUS, point_color, -1)
    frame = cv2.circle(frame, (x, y), POINT_PROMPT_RADIUS, (255, 255, 255), 1)
    return frame


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


def prompt_sources_for_tracks(query_points, refinement_query_points, selected_tracks, tracked_prompt_sources):
    if selected_tracks is None:
        return []

    track_count = int(np.asarray(selected_tracks).shape[0])
    sources = normalized_prompt_sources(tracked_prompt_sources)
    if len(sources) == track_count:
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
    if track_index < 0 or track_index >= len(sources):
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

    updated_query_points, updated_query_colors, updated_refinements, removed = remove_prompt_by_source(
        query_points,
        query_points_color,
        refinement_query_points,
        sources[track_index],
    )
    if not removed:
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

    updated_tracks = np.delete(np.asarray(selected_tracks), track_index, axis=0)
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


def preprocess_video_input(video_path, tracking_resolution, max_frames, skip_frames):
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
    loaded_source_frames = num_frames
    if max_frames_to_load and num_frames > max_frames_to_load:
        gr.Warning(
            f"Only the first {max_frames_to_load} of {original_num_frames} frames will be used.",
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

    load_status = f"Loaded {num_frames} frames for point selection."
    if loaded_source_frames != original_num_frames and sampling_stride > 1:
        load_status = (
            f"Loaded first {loaded_source_frames} of {original_num_frames} source frames, then sampled "
            f"to {num_frames} frames by skipping {frame_skip_count} frame(s) after each loaded frame. "
            "Set Max frames to load to 0 to use the full video."
        )
    elif loaded_source_frames != original_num_frames:
        load_status = (
            f"Loaded first {loaded_source_frames} of {original_num_frames} frames for point selection. "
            "Set Max frames to load to 0 to use the full video."
        )
    elif sampling_stride > 1:
        load_status = (
            f"Loaded {num_frames} sampled frames for point selection by skipping "
            f"{frame_skip_count} frame(s) after each loaded frame."
        )

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
        gr.update(minimum=0, maximum=num_frames - 1, value=0, interactive=interactive), # Set slider interactive
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
        gr.update(interactive=False),
        None,
        gr.update(interactive=False),
        gr.update(interactive=False),
        None,
        gr.update(value=0, interactive=False),
        gr.update(interactive=False),
        None,
        None,
        gr.update(minimum=0, maximum=0, value=0, interactive=False),
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
    has_selected_points = query_count > 0
    tracking_mode = 'selected'
    if not has_selected_points:
        tracking_mode='grid'
    
    torch = get_torch()
    device = resolve_torch_device(torch)
    dtype = torch.float if device == "cuda" else torch.float
    total_frame_count = video_input.shape[0]
    sampled_frame_count = (total_frame_count + TRACKING_FRAME_STRIDE - 1) // TRACKING_FRAME_STRIDE
    input_height, input_width = video_input.shape[1:3]

    # Convert query points to tensor, normalize to input resolution
    if tracking_mode!='grid':
        query_points_tensor = []
        selected_point_labels = []
        for frame_points in query_points:
            for point in frame_points:
                x, y, frame_index, point_label = unpack_query_point(point)
                sampled_frame_index = map_frame_index_to_sampled(
                    frame_index,
                    sampled_frame_count=sampled_frame_count,
                    stride=TRACKING_FRAME_STRIDE,
                )
                query_points_tensor.append((x, y, sampled_frame_index))
                selected_point_labels.append(point_label)
        
        query_points_tensor = torch.tensor(query_points_tensor).float()
        query_points_tensor *= torch.tensor([
            input_width, input_height, 1
        ]) / torch.tensor([
            [video_preview.shape[2], video_preview.shape[1], 1]
        ])
        query_points_tensor = query_points_tensor[None].flip(-1).to(device, dtype) # xyt -> tyx
        query_points_tensor = query_points_tensor[:, :, [0, 2, 1]] # tyx -> txy

    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)
    video_input = subsample_video_tensor(video_input, TRACKING_FRAME_STRIDE)

    model = get_cached_cotracker_model(device)

    video_input = video_input.permute(0, 1, 4, 2, 3)
    if tracking_mode=='grid':
        xy = get_points_on_a_grid(15, video_input.shape[3:], device=device)
        queries = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #
        add_support_grid=False
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        query_points_color = [[]]
        query_count = queries.shape[1]
        for i in range(query_count):
            # Choose the color for the point from matplotlib colormap
            color = cmap(i / float(query_count))
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            query_points_color[0].append(color)
        selected_point_labels = None

    else:
        queries = query_points_tensor
        add_support_grid=True

    model(video_chunk=video_input, is_first_step=True, grid_size=0, queries=queries, add_support_grid=add_support_grid)
    # 
    for ind in get_online_chunk_start_indices(video_input.shape[1], model.step):
        pred_tracks, pred_visibility = model(
            video_chunk=video_input[:, ind : ind + model.step * 2],
            grid_size=0, 
            queries=queries, 
            add_support_grid=add_support_grid
        )  # B T N 2,  B T N 1
    sampled_tracks = (pred_tracks * torch.tensor([video_preview.shape[2], video_preview.shape[1]]).to(device) / torch.tensor([input_width, input_height]).to(device))[0].permute(1, 0, 2).cpu().numpy()
    sampled_occ = pred_visibility[0].permute(1, 0).cpu().numpy()
    tracks = expand_sampled_time_axis(
        sampled_tracks,
        total_frames=total_frame_count,
        stride=TRACKING_FRAME_STRIDE,
        axis=1,
    )
    pred_occ = expand_sampled_time_axis(
        sampled_occ,
        total_frames=total_frame_count,
        stride=TRACKING_FRAME_STRIDE,
        axis=1,
    )

    # make color array
    colors = []
    for frame_colors in query_points_color:
        colors.extend(frame_colors)
    colors = np.array(colors)
    
    painted_video = paint_point_track(video_preview,tracks,pred_occ,colors)

    # save video
    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)

    mediapy.write_video(video_file_path, painted_video, fps=video_fps)

    export_status = (
        "Grid tracking complete."
        if not has_selected_points
        else "Selected-point tracking complete."
    )
    return (
        video_file_path,
        tracks if has_selected_points else None,
        pred_occ if has_selected_points else None,
        selected_point_labels if has_selected_points else None,
        painted_video,
        gr.update(minimum=0, maximum=total_frame_count - 1, value=0, interactive=True),
        painted_video[0],
        gr.update(interactive=True),
        gr.update(interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        gr.update(value=0, interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        None,
        export_status,
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


def reprocess_with_refinements(
    video_preview,
    video_input,
    video_fps,
    query_points,
    query_points_color,
    refinement_query_points,
    tracked_frame_num,
):
    if video_preview is None or video_input is None:
        message = "Track a video before re-processing refinement points."
        gr.Warning(message, duration=5)
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            message,
        )

    frame_count = int(video_preview.shape[0])
    refinement_query_points = ensure_frame_points(refinement_query_points, frame_count)
    merged_query_points = merge_frame_point_lists(query_points, refinement_query_points)
    merged_query_count = count_frame_points(merged_query_points)
    if merged_query_count == 0:
        message = "Add or select at least one point prompt before re-processing."
        gr.Warning(message, duration=5)
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            message,
        )

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
    result[10] = gr.update(interactive=True)
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


def get_sam_preview_runtime(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    if model_name not in SAM_IMAGE_MODEL_CHOICES:
        allowed = ", ".join(SAM_IMAGE_MODEL_CHOICES)
        raise ValueError(f"SAM image model must be one of: {allowed}")

    with sam_preview_runtime_lock:
        runtime = sam_preview_runtimes.get(model_name)
        if runtime is not None:
            return runtime

        if str(MOBILE_SAM_ROOT) not in sys.path:
            sys.path.insert(0, str(MOBILE_SAM_ROOT))

        from sam2_coordinate_wrapper import load_sam2_predictor, resolve_sam2_model_option

        model_option = resolve_sam2_model_option(model_name)
        predictor, device = load_sam2_predictor(
            model_name=model_name,
            download_checkpoint=True,
        )
        runtime = {
            "predictor": predictor,
            "predictor_lock": threading.Lock(),
            "device": device,
            "model_name": model_name,
            "model_label": model_option["label"],
            "image_cache_key": None,
        }
        sam_preview_runtimes[model_name] = runtime
        return runtime


def preload_sam_preview_runtime(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    try:
        runtime = get_sam_preview_runtime(model_name)
        with sam_preview_preload_lock:
            sam_preview_preload_errors.pop(model_name, None)
        print(
            f"SAM preview model preloaded: {runtime['model_label']} on {runtime['device']}",
            flush=True,
        )
    except Exception as exc:
        with sam_preview_preload_lock:
            sam_preview_preload_errors[model_name] = str(exc)
            sam_preview_preload_started.discard(model_name)
        print(f"SAM preview model preload failed for {model_name}: {exc}", flush=True)


def start_sam_preview_preload(sam_model=DEFAULT_SAM_IMAGE_MODEL):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    with sam_preview_preload_lock:
        if model_name in sam_preview_preload_started or model_name in sam_preview_runtimes:
            return
        sam_preview_preload_started.add(model_name)

    thread = threading.Thread(
        target=preload_sam_preview_runtime,
        args=(model_name,),
        daemon=True,
        name=f"sam-preview-preload-{model_name}",
    )
    thread.start()


def get_sam_preview_runtime_if_ready(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    runtime_message = (
        f"SAM preview model {model_name} is still loading or downloading. "
        "Try Preview SAM again in a moment."
    )

    acquired = sam_preview_runtime_lock.acquire(blocking=False)
    if not acquired:
        start_sam_preview_preload(model_name)
        return None, runtime_message

    try:
        runtime = sam_preview_runtimes.get(model_name)
        if runtime is not None:
            return runtime, None
    finally:
        sam_preview_runtime_lock.release()

    with sam_preview_preload_lock:
        previous_error = sam_preview_preload_errors.get(model_name)
    start_sam_preview_preload(model_name)
    if previous_error:
        return (
            None,
            f"SAM preview model {model_name} failed to load previously: {previous_error}. "
            "Retrying in the background.",
        )
    return None, runtime_message


def as_uint8_rgb_frame(frame):
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("Expected an RGB/RGBA frame for SAM preview.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def sam_preview_frame_cache_key(frame):
    image = np.ascontiguousarray(as_uint8_rgb_frame(frame))
    digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    return image.shape, str(image.dtype), digest


def predict_sam_preview_mask(runtime, frame, point_coords, point_labels):
    frame_cache_key = sam_preview_frame_cache_key(frame)
    predictor = runtime["predictor"]
    with runtime["predictor_lock"]:
        if runtime.get("image_cache_key") != frame_cache_key:
            predictor.set_image(frame)
            runtime["image_cache_key"] = frame_cache_key
        return predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            multimask_output=len(point_coords) == 1,
            normalize_coords=True,
        )


def draw_sam_preview(frame, mask, point_coords, point_labels):
    preview = as_uint8_rgb_frame(frame)
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.shape != preview.shape[:2]:
        raise ValueError("SAM mask size does not match frame size.")

    overlay_color = np.asarray([0, 180, 255], dtype=np.float32)
    preview_float = preview.astype(np.float32)
    preview_float[mask_array] = preview_float[mask_array] * 0.55 + overlay_color * 0.45
    preview = np.clip(preview_float, 0, 255).astype(np.uint8)

    height, width = preview.shape[:2]
    for (x, y), label in zip(np.asarray(point_coords), np.asarray(point_labels)):
        x = int(np.clip(round(float(x)), 0, width - 1))
        y = int(np.clip(round(float(y)), 0, height - 1))
        preview = draw_query_point(preview, x, y, int(label))
    return preview


def format_sam_prompt_summary(point_coords, point_labels, max_points=12):
    coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    display_count = min(len(coords), int(max_points))
    display_coords = [
        [round(float(x), 2), round(float(y), 2)]
        for x, y in coords[:display_count]
    ]
    display_labels = [int(label) for label in labels[:display_count]]
    suffix = ""
    if len(coords) > display_count:
        suffix = f", showing first {display_count} of {len(coords)}"
    return f"point_coords={display_coords}, point_labels={display_labels}{suffix}"


def sam_point_prompts_for_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_index,
    prefer_tracked_points=False,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    point_coords = np.empty((0, 2), dtype=np.float32)
    point_labels = np.empty((0,), dtype=np.int32)
    prompt_source = "selected"

    if prefer_tracked_points and selected_tracks is not None and selected_point_labels is not None:
        scaled_tracks = scale_tracks_to_frame_space(
            selected_tracks,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        point_coords, point_labels = visible_labeled_points_for_frame(
            scaled_tracks,
            frame_index,
            point_labels=selected_point_labels,
            visibility=selected_visibility,
        )
        prompt_source = "tracked"

    if len(point_coords) == 0:
        point_coords, point_labels = labeled_query_points_for_frame(
            query_points,
            frame_index,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        prompt_source = "selected"

    if (
        len(point_coords) == 0
        and not prefer_tracked_points
        and selected_tracks is not None
        and selected_point_labels is not None
    ):
        scaled_tracks = scale_tracks_to_frame_space(
            selected_tracks,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        point_coords, point_labels = visible_labeled_points_for_frame(
            scaled_tracks,
            frame_index,
            point_labels=selected_point_labels,
            visibility=selected_visibility,
        )
        prompt_source = "tracked"

    pending_refinement_query_points = pending_refinement_points(
        refinement_query_points,
        tracked_prompt_sources,
        frame_count=len(video_frames),
    )
    refinement_coords, refinement_labels = labeled_query_points_for_frame(
        pending_refinement_query_points,
        frame_index,
        source_hw=video_preview_array.shape[1:3],
        target_hw=video_frames.shape[1:3],
    )
    if len(refinement_coords) > 0:
        if len(point_coords) > 0:
            point_coords = np.concatenate([point_coords, refinement_coords], axis=0)
            point_labels = np.concatenate([point_labels, refinement_labels], axis=0)
            prompt_source = f"{prompt_source} + refinement"
        else:
            point_coords = refinement_coords
            point_labels = refinement_labels
            prompt_source = "refinement"

    return point_coords, point_labels, prompt_source


def preview_sam_on_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_num,
    sam_model,
    prefer_tracked_points=False,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    if video_frames is None or video_preview_array is None:
        message = "Submit a video before previewing SAM."
        gr.Warning(message, duration=5)
        return None, message

    frame_index = int(np.clip(int(frame_num), 0, len(video_frames) - 1))
    point_coords, point_labels, prompt_source = sam_point_prompts_for_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        frame_index,
        prefer_tracked_points=prefer_tracked_points,
        refinement_query_points=refinement_query_points,
        tracked_prompt_sources=tracked_prompt_sources,
    )

    if len(point_coords) == 0:
        message = f"No selected or visible tracked points on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message
    if not np.any(point_labels == 1):
        message = f"SAM needs at least one visible positive point on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message

    frame = as_uint8_rgb_frame(video_frames[frame_index])
    prompt_summary = format_sam_prompt_summary(point_coords, point_labels)
    runtime, loading_message = get_sam_preview_runtime_if_ready(sam_model)
    if runtime is None:
        empty_mask = np.zeros(frame.shape[:2], dtype=bool)
        prompt_preview = draw_sam_preview(frame, empty_mask, point_coords, point_labels)
        return prompt_preview, f"{loading_message} Loaded prompts: {prompt_summary}."

    masks, scores, _ = predict_sam_preview_mask(runtime, frame, point_coords, point_labels)
    best_mask = masks[int(np.argmax(scores))]
    preview = draw_sam_preview(frame, best_mask, point_coords, point_labels)
    positive_count = int(np.sum(point_labels == 1))
    negative_count = int(np.sum(point_labels == 0))
    return (
        preview,
        f"SAM preview frame {frame_index} from {prompt_source} points with "
        f"{runtime['model_label']} on {runtime['device']} "
        f"({positive_count} positive, {negative_count} negative point(s)). "
        f"Loaded prompts: {prompt_summary}.",
    )


def preview_sam_for_selected_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    query_frame_num,
    tracked_frame_num,
    tracked_video_preview,
    sam_model,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    has_processed_selection = tracked_video_preview is not None and selected_tracks is not None
    frame_num = tracked_frame_num if has_processed_selection else query_frame_num
    return preview_sam_on_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        frame_num,
        sam_model,
        prefer_tracked_points=has_processed_selection,
        refinement_query_points=refinement_query_points,
        tracked_prompt_sources=tracked_prompt_sources,
    )


def preview_sam_video_for_processed_frames(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    tracked_video_preview,
    video_fps,
    sam_model,
    sam_video_skip_frames,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    if video_frames is None or video_preview_array is None:
        message = "Submit and track a video before running SAM video review."
        gr.Warning(message, duration=5)
        return None, message
    if tracked_video_preview is None or selected_tracks is None or selected_point_labels is None:
        message = "Track selected points before running SAM video review."
        gr.Warning(message, duration=5)
        return None, message

    try:
        skip_count = parse_frame_skip_count(sam_video_skip_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    runtime = get_sam_preview_runtime(sam_model)
    predictor = runtime["predictor"]
    review_frames = []
    processed_count = 0
    skipped_by_frame_skip = 0
    skipped_no_points = 0
    skipped_no_positive = 0
    frame_count = len(video_frames)

    for frame_index in range(frame_count):
        frame = as_uint8_rgb_frame(video_frames[frame_index])
        if not should_process_frame_for_skip(frame_index, skip_count):
            skipped_by_frame_skip += 1
            review_frames.append(frame)
            continue

        point_coords, point_labels, _ = sam_point_prompts_for_frame(
            video_frames,
            video_preview_array,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            frame_index,
            prefer_tracked_points=True,
            refinement_query_points=refinement_query_points,
            tracked_prompt_sources=tracked_prompt_sources,
        )
        if len(point_coords) == 0:
            skipped_no_points += 1
            review_frames.append(frame)
            continue
        if not np.any(point_labels == 1):
            skipped_no_positive += 1
            review_frames.append(frame)
            continue

        predictor.set_image(frame)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            multimask_output=len(point_coords) == 1,
            normalize_coords=True,
        )
        best_mask = masks[int(np.argmax(scores))]
        review_frames.append(draw_sam_preview(frame, best_mask, point_coords, point_labels))
        processed_count += 1

    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)
    output_fps = float(video_fps or 24)
    if output_fps <= 0:
        output_fps = 24
    mediapy.write_video(video_file_path, np.asarray(review_frames), fps=output_fps)

    skipped_parts = []
    if skipped_by_frame_skip:
        skipped_parts.append(f"{skipped_by_frame_skip} by skip setting")
    if skipped_no_points:
        skipped_parts.append(f"{skipped_no_points} without points")
    if skipped_no_positive:
        skipped_parts.append(f"{skipped_no_positive} without positive points")
    skipped_text = f"; skipped {', '.join(skipped_parts)}" if skipped_parts else ""
    return (
        video_file_path,
        (
            f"SAM video review complete for {processed_count}/{frame_count} frame(s) "
            f"with {runtime['model_label']} on {runtime['device']}{skipped_text}."
        ),
    )


def save_sam_video_review_from_state(sam_video_review, output_dir, video_fps):
    try:
        output_path = save_sam_video_review(
            sam_video_review,
            output_dir or DEFAULT_SAM_VIDEO_SAVE_DIR,
            fps=video_fps or 24,
        )
    except Exception as exc:
        message = f"Failed to save SAM video preview: {exc}"
        gr.Warning(message, duration=5)
        return None, message

    return str(output_path), f"Saved SAM video preview to {output_path}."


def export_sam_preview_as_yolo_custom(raw_mask_root, output_dir):
    raw_mask_path = Path(raw_mask_root).expanduser() if raw_mask_root else DEFAULT_RAW_MASK_ROOT
    dataset_path = Path(output_dir).expanduser() if output_dir else DEFAULT_YOLO_DATASET_DIR

    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from export_yolo_segmentation_dataset import export_yolo_segmentation_dataset

        stats = export_yolo_segmentation_dataset(
            raw_mask_root=raw_mask_path,
            output_dir=dataset_path,
            train_ratio=0.8,
            min_area_px=20,
            approx_epsilon=2.0,
        )
    except Exception as exc:
        message = f"Failed to save preview as YOLO custom: {exc}"
        gr.Warning(message, duration=5)
        return message

    return (
        f"Saved YOLO custom dataset to {dataset_path}: "
        f"{stats.train_images} train image(s), {stats.val_images} val image(s), "
        f"{stats.total_wound_instances} wound polygon(s)."
    )


with gr.Blocks() as demo:
    video = gr.State()
    video_queried_preview = gr.State()
    video_preview = gr.State()
    video_input = gr.State()
    video_fps = gr.State(24)

    query_points = gr.State([])
    query_points_color = gr.State([])
    is_tracked_query = gr.State([])
    query_count = gr.State(0)
    selected_tracks = gr.State(None)
    selected_visibility = gr.State(None)
    selected_point_labels = gr.State(None)
    tracked_prompt_sources = gr.State([])
    tracked_video_preview = gr.State(None)
    refinement_query_points = gr.State([])

    gr.Markdown("# 🎨 CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos")
    gr.Markdown("<div style='text-align: left;'> \
    <p>Welcome to <a href='https://cotracker3.github.io/' target='_blank'>CoTracker</a>! This space demonstrates point (pixel) tracking in videos. \
    The model tracks points on a grid or points selected by you.  </p> \
    <p> To get started, simply upload your <b>.mp4</b> video or click on one of the example videos to load them. The shorter the video, the faster the processing. We recommend submitting short videos of length <b>2-7 seconds</b>.</p> \
    <p> After you uploaded a video, please click \"Submit\" and then click \"Track\" for grid tracking or specify points you want to track before clicking. Enjoy the results! </p>\
    <p style='text-align: left'>For more details, check out our <a href='https://github.com/facebookresearch/co-tracker' target='_blank'>GitHub Repo</a> ⭐. We thank the authors of LocoTrack for their interactive demo.</p> \
    </div>"
    )
    

    gr.Markdown("## First step: upload your video or select an example video, and click submit.")
    with gr.Row():
        

        with gr.Accordion("Your video input", open=True) as video_in_drawer:
            video_in = gr.Video(label="Video Input", format="mp4")
            tracking_resolution = gr.Dropdown(
                choices=list(TRACKING_RESOLUTION_OPTIONS),
                value=DEFAULT_TRACKING_RESOLUTION,
                label="Tracking Resolution",
                interactive=True,
            )
            max_frames_input = gr.Number(
                value=DEFAULT_MAX_FRAMES,
                precision=0,
                label="Max frames to load (0 = full video)",
                interactive=True,
            )
            skip_frames_input = gr.Number(
                value=0,
                precision=0,
                label="Skip frames after each loaded frame (0 = keep all)",
                interactive=True,
            )
            submit = gr.Button("Submit", scale=0)

            import os
            apple = os.path.join(os.path.dirname(__file__), "videos", "apple.mp4")
            bear = os.path.join(os.path.dirname(__file__), "videos", "bear.mp4")
            paragliding_launch = os.path.join(
                os.path.dirname(__file__), "videos", "paragliding-launch.mp4"
            )
            paragliding = os.path.join(os.path.dirname(__file__), "videos", "paragliding.mp4")
            cat = os.path.join(os.path.dirname(__file__), "videos", "cat.mp4")
            pillow = os.path.join(os.path.dirname(__file__), "videos", "pillow.mp4")
            teddy = os.path.join(os.path.dirname(__file__), "videos", "teddy.mp4")
            backpack = os.path.join(os.path.dirname(__file__), "videos", "backpack.mp4")

            if os.environ.get("COTRACKER_DISABLE_EXAMPLES") != "1":
                gr.Examples(examples=[bear, apple, paragliding, paragliding_launch, cat, pillow, teddy, backpack],
                            inputs = [
                                video_in
                            ],
                            )


    gr.Markdown("## Second step: Simply click \"Track\" to track a grid of points or select query points on the video before clicking")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                query_frames = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1, label="Choose Frame", interactive=False)
            with gr.Row():
                point_type = gr.Radio(
                    choices=list(POINT_TYPE_CHOICES),
                    value=POSITIVE_POINT_CHOICE,
                    label="Point Type",
                    interactive=True,
                )
                query_point_edit_mode = gr.Radio(
                    choices=list(POINT_EDIT_MODE_CHOICES),
                    value=POINT_ADD_MODE,
                    label="Mode",
                    interactive=True,
                )
            with gr.Row():
                undo = gr.Button("Undo", interactive=False)
                clear_frame = gr.Button("Clear Frame", interactive=False)
                clear_all = gr.Button("Clear All", interactive=False)

            with gr.Row():
                current_frame = gr.Image(
                    label="Click to add/delete query points",
                    type="numpy",
                    interactive=False
                )
            with gr.Row():
                track_button = gr.Button("Track", interactive=False)
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
                autoplay=True,
                loop=True,
            )
            no_wound_export_button = gr.Button("Export No-Wound Frames to YOLO", interactive=False)
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                lines=3,
            )

        with gr.Column():
            with gr.Row():
                sam_model_dropdown = gr.Dropdown(
                    choices=list(SAM_IMAGE_MODEL_CHOICES),
                    value=DEFAULT_SAM_IMAGE_MODEL,
                    label="SAM Image Model",
                    interactive=False,
                )
                sam_preview_button = gr.Button("Preview SAM on Current Frame", interactive=False)
            sam_preview_image = gr.Image(
                label="SAM point preview",
                type="numpy",
                interactive=False,
            )

    gr.Markdown("## Third step: Fine-tune point adjustment of cotracker and Preview effect of SAM on processed video.")
    with gr.Row():
        with gr.Column():
            tracked_query_frames = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Choose Processed Frame",
                interactive=False,
            )
            with gr.Row():
                refinement_point_type = gr.Radio(
                    choices=list(POINT_TYPE_CHOICES),
                    value=POSITIVE_POINT_CHOICE,
                    label="Refinement Point Type",
                    interactive=True,
                )
                refinement_edit_mode = gr.Radio(
                    choices=list(REFINEMENT_EDIT_MODE_CHOICES),
                    value=REFINEMENT_ADD_MODE,
                    label="Refinement Edit Mode",
                    interactive=True,
                )
            with gr.Row():
                refinement_undo = gr.Button("Undo Frame Edit", interactive=True)
                refinement_clear_frame = gr.Button("Clear Frame Edits", interactive=True)
                refinement_clear_all = gr.Button("Clear All Edits", interactive=True)
            reprocess_button = gr.Button("Re-process", interactive=False)
            tracked_frame_preview = gr.Image(
                label="Query points on video",
                type="numpy",
                interactive=False,
            )
        with gr.Column():
            processed_sam_model_dropdown = gr.Dropdown(
                choices=list(SAM_IMAGE_MODEL_CHOICES),
                value=DEFAULT_SAM_IMAGE_MODEL,
                label="SAM Image Model",
                interactive=False,
            )
            processed_sam_preview_button = gr.Button("Preview SAM on Selected Frame", interactive=False)
            processed_sam_preview_image = gr.Image(
                label="SAM point preview",
                type="numpy",
                interactive=False,
            )
            processed_sam_video_skip_frames = gr.Number(
                value=0,
                precision=0,
                label="Skip frames after each loaded frame (0 = keep all)",
                interactive=False,
            )
            processed_sam_video_button = gr.Button("Preview SAM on Processed Video", interactive=False)
            processed_sam_video = gr.Video(
                label="SAM video review",
                interactive=False,
                autoplay=True,
                loop=True,
            )
            sam_video_save_dir = gr.Textbox(
                value=str(DEFAULT_SAM_VIDEO_SAVE_DIR),
                label="SAM video save directory",
                interactive=True,
            )
            with gr.Row():
                save_sam_video_button = gr.Button("Save SAM Video Preview", interactive=True)
                save_yolo_custom_button = gr.Button("Save Preview as YOLO Custom", interactive=True)
            saved_sam_video_file = gr.File(
                label="Saved SAM preview MP4",
                interactive=False,
            )
            yolo_raw_mask_root = gr.Textbox(
                value=str(DEFAULT_RAW_MASK_ROOT),
                label="YOLO raw-mask root",
                interactive=True,
            )
            yolo_dataset_output_dir = gr.Textbox(
                value=str(DEFAULT_YOLO_DATASET_DIR),
                label="YOLO dataset output directory",
                interactive=True,
            )

    

    submit.click(
        fn = preprocess_video_input, 
        inputs = [video_in, tracking_resolution, max_frames_input, skip_frames_input],
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
            sam_preview_button,
            sam_preview_image,
            processed_sam_model_dropdown,
            processed_sam_preview_button,
            processed_sam_preview_image,
            processed_sam_video_skip_frames,
            processed_sam_video_button,
            processed_sam_video,
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
            processed_sam_preview_button,
            processed_sam_video_skip_frames,
            processed_sam_video_button,
            processed_sam_video,
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
            processed_sam_preview_button,
            processed_sam_video_skip_frames,
            processed_sam_video_button,
            processed_sam_video,
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
        fn = preview_sam_on_frame,
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
            export_status,
        ],
        queue = False,
    )

    processed_sam_preview_button.click(
        fn = preview_sam_for_selected_frame,
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
            export_status,
        ],
        queue = False,
    )

    processed_sam_video_button.click(
        fn = preview_sam_video_for_processed_frames,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_video_preview,
            video_fps,
            processed_sam_model_dropdown,
            processed_sam_video_skip_frames,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            processed_sam_video,
            export_status,
        ],
        queue = False,
    )

    save_sam_video_button.click(
        fn = save_sam_video_review_from_state,
        inputs = [
            processed_sam_video,
            sam_video_save_dir,
            video_fps,
        ],
        outputs = [
            saved_sam_video_file,
            export_status,
        ],
        queue = False,
    )

    save_yolo_custom_button.click(
        fn = export_sam_preview_as_yolo_custom,
        inputs = [
            yolo_raw_mask_root,
            yolo_dataset_output_dir,
        ],
        outputs = [
            export_status,
        ],
        queue = False,
    )

    
demo.launch(
    server_name="127.0.0.1",
    server_port=int(os.environ.get("PORT", "7860")),
    show_api=False,
    show_error=True,
    share=False,
)
