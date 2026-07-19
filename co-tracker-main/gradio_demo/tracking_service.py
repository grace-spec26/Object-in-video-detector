import colorsys
import importlib.metadata as importlib_metadata
import os
import random
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import mediapy
import numpy as np

try:
    from .tracking_helpers import (
        TRACKING_FRAME_STRIDE,
        expand_sampled_time_axis,
        get_cached_cotracker_model,
        get_online_chunk_start_indices,
        map_frame_index_to_sampled,
        resolve_torch_device,
        subsample_video_tensor,
    )
except ImportError:
    from tracking_helpers import (
        TRACKING_FRAME_STRIDE,
        expand_sampled_time_axis,
        get_cached_cotracker_model,
        get_online_chunk_start_indices,
        map_frame_index_to_sampled,
        resolve_torch_device,
        subsample_video_tensor,
    )


@dataclass(frozen=True)
class CoTrackerTrackingResult:
    video_file_path: str
    tracks: Optional[np.ndarray]
    visibility: Optional[np.ndarray]
    selected_point_labels: Optional[List[int]]
    painted_video: np.ndarray
    total_frame_count: int
    has_selected_points: bool
    export_status: str


_torch_module = None


@contextmanager
def suppress_importlib_entry_points():
    original_entry_points = importlib_metadata.entry_points
    importlib_metadata.entry_points = lambda *args, **kwargs: {}
    try:
        yield
    finally:
        importlib_metadata.entry_points = original_entry_points


def get_torch():
    global _torch_module
    if _torch_module is not None:
        return _torch_module

    with suppress_importlib_entry_points():
        import torch as torch_module

    _torch_module = torch_module
    return _torch_module


def _unpack_query_point(point):
    x, y, frame_index = point[:3]
    point_label = int(point[3]) if len(point) >= 4 else 1
    return x, y, frame_index, point_label


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
    r"""Get a grid of points covering a rectangular region.

    Points are returned in row-major `(x, y)` order inside a tensor shaped
    `(1, size * size, 2)`.
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
  """Converts a sequence of points to color code video."""
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
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1
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

      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video


def run_cotracker_tracking(
    video_preview,
    video_input,
    video_fps,
    query_points,
    query_points_color,
    query_count,
    output_dir: Optional[Path] = None,
) -> CoTrackerTrackingResult:
    has_selected_points = int(query_count) > 0
    tracking_mode = "selected" if has_selected_points else "grid"

    torch = get_torch()
    device = resolve_torch_device(torch)
    dtype = torch.float
    total_frame_count = video_input.shape[0]
    sampled_frame_count = (total_frame_count + TRACKING_FRAME_STRIDE - 1) // TRACKING_FRAME_STRIDE
    input_height, input_width = video_input.shape[1:3]

    if tracking_mode != "grid":
        query_points_tensor = []
        selected_point_labels = []
        for frame_points in query_points:
            for point in frame_points:
                x, y, frame_index, point_label = _unpack_query_point(point)
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
        query_points_tensor = query_points_tensor[None].flip(-1).to(device, dtype)
        query_points_tensor = query_points_tensor[:, :, [0, 2, 1]]

    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)
    video_input = subsample_video_tensor(video_input, TRACKING_FRAME_STRIDE)

    model = get_cached_cotracker_model(device)

    video_input = video_input.permute(0, 1, 4, 2, 3)
    if tracking_mode == "grid":
        xy = get_points_on_a_grid(15, video_input.shape[3:], device=device)
        queries = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)
        add_support_grid = False
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        query_points_color = [[]]
        query_count = queries.shape[1]
        for i in range(query_count):
            color = cmap(i / float(query_count))
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            query_points_color[0].append(color)
        selected_point_labels = None
    else:
        queries = query_points_tensor
        add_support_grid = True

    model(video_chunk=video_input, is_first_step=True, grid_size=0, queries=queries, add_support_grid=add_support_grid)
    for ind in get_online_chunk_start_indices(video_input.shape[1], model.step):
        pred_tracks, pred_visibility = model(
            video_chunk=video_input[:, ind : ind + model.step * 2],
            grid_size=0,
            queries=queries,
            add_support_grid=add_support_grid,
        )

    sampled_tracks = (
        pred_tracks
        * torch.tensor([video_preview.shape[2], video_preview.shape[1]]).to(device)
        / torch.tensor([input_width, input_height]).to(device)
    )[0].permute(1, 0, 2).cpu().numpy()
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

    colors = []
    for frame_colors in query_points_color:
        colors.extend(frame_colors)
    colors = np.array(colors)

    painted_video = paint_point_track(video_preview, tracks, pred_occ, colors)

    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "tmp"
    video_file_path = video_path / video_file_name
    os.makedirs(video_path, exist_ok=True)

    mediapy.write_video(str(video_file_path), painted_video, fps=video_fps)

    export_status = (
        "Grid tracking complete."
        if not has_selected_points
        else "Selected-point tracking complete."
    )
    return CoTrackerTrackingResult(
        video_file_path=str(video_file_path),
        tracks=tracks if has_selected_points else None,
        visibility=pred_occ if has_selected_points else None,
        selected_point_labels=selected_point_labels if has_selected_points else None,
        painted_video=painted_video,
        total_frame_count=total_frame_count,
        has_selected_points=has_selected_points,
        export_status=export_status,
    )
