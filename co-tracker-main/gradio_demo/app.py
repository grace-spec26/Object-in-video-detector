# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py 
# We updated it to work with CoTracker3 models. We thank authors of LocoTrack
# for such an amazing Gradio demo.

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["GRADIO_SKIP_PYI_GENERATION"] = "1"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

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
import torch
import colorsys
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from .export_helpers import (
        DEFAULT_COORDINATES_DIR,
        DEFAULT_FRAMES_DIR,
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        store_coordinate_arrays,
        store_original_frames,
        visible_labeled_points_for_frame,
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
        subsample_video_tensor,
    )
except ImportError:
    from export_helpers import (
        DEFAULT_COORDINATES_DIR,
        DEFAULT_FRAMES_DIR,
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        store_coordinate_arrays,
        store_original_frames,
        visible_labeled_points_for_frame,
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
    device: Optional[torch.device] = torch.device("cpu"),
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
SAM_IMAGE_MODEL_CHOICES = (
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
)
DEFAULT_SAM_IMAGE_MODEL = "sam2.1_hiera_small.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBILE_SAM_ROOT = PROJECT_ROOT / "MobileSAM-master"
sam_preview_runtime_lock = threading.Lock()
sam_preview_runtimes = {}


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


def get_point(frame_num, point_type, video_queried_preview, query_points, query_points_color, query_count, evt: gr.SelectData):
    print(f"You selected {(evt.index[0], evt.index[1], frame_num)}")

    current_frame = video_queried_preview[int(frame_num)]

    # Get the mouse click
    point_label = point_label_from_choice(point_type)
    query_points[int(frame_num)].append((evt.index[0], evt.index[1], frame_num, point_label))

    # Choose the color for the point from matplotlib colormap
    color = POINT_COLORS[point_label]
    # print(f"Color: {color}")
    query_points_color[int(frame_num)].append(color)

    # Draw the point on the frame
    x, y = evt.index
    current_frame_draw = draw_query_point(current_frame, x, y, point_label)

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw

    # Update the query count
    query_count += 1
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
    for point, color in zip(query_points[int(frame_num)], query_points_color[int(frame_num)]):
        x, y, _, point_label = unpack_query_point(point)
        current_frame_draw = draw_query_point(current_frame_draw, x, y, point_label)

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


def preprocess_video_input(video_path, tracking_resolution, max_frames, skip_frames):
    if video_path is None:
        raise gr.Error("Please upload a video before submitting.")

    try:
        max_frames_to_load = parse_max_frame_count(max_frames)
        frame_skip_count = parse_frame_skip_count(skip_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

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
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        None,
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
        "Tracking complete. Frames can now be stored."
        if not has_selected_points
        else "Tracking complete. Frames and selected-point coordinates can now be stored."
    )
    return (
        video_file_path,
        tracks if has_selected_points else None,
        pred_occ if has_selected_points else None,
        selected_point_labels if has_selected_points else None,
        gr.update(interactive=True),
        gr.update(interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        gr.update(interactive=has_selected_points),
        export_status,
    )


def store_frames_from_state(video_frames):
    if video_frames is None:
        message = "Submit and track a video before storing frames."
        gr.Warning(message, duration=5)
        return message

    try:
        written_paths = store_original_frames(video_frames, DEFAULT_FRAMES_DIR)
    except Exception as exc:
        message = f"Failed to store frames: {exc}"
        gr.Warning(message, duration=5)
        return message

    return f"Stored {len(written_paths)} original frames in {DEFAULT_FRAMES_DIR}."


def store_coordinates_from_state(video_frames, video_preview_array, selected_tracks, selected_visibility, selected_point_labels):
    if selected_tracks is None:
        message = "Track selected points before storing coordinates."
        gr.Warning(message, duration=5)
        return message
    if video_frames is None or video_preview_array is None:
        message = "Submit and track a video before storing coordinates."
        gr.Warning(message, duration=5)
        return message

    try:
        written_paths = store_coordinate_arrays(
            tracks=selected_tracks,
            output_dir=DEFAULT_COORDINATES_DIR,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
            visibility=selected_visibility,
            point_labels=selected_point_labels,
        )
    except Exception as exc:
        message = f"Failed to store coordinates: {exc}"
        gr.Warning(message, duration=5)
        return message

    frame_files = max(0, len(written_paths) - 1)
    return (
        f"Stored selected-point coordinates for {frame_files} frames in "
        f"{DEFAULT_COORDINATES_DIR}."
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
            "device": device,
            "model_name": model_name,
            "model_label": model_option["label"],
        }
        sam_preview_runtimes[model_name] = runtime
        return runtime


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


def preview_sam_on_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_num,
    sam_model,
):
    if video_frames is None or video_preview_array is None:
        message = "Submit a video before previewing SAM."
        gr.Warning(message, duration=5)
        return None, message

    frame_index = int(np.clip(int(frame_num), 0, len(video_frames) - 1))
    point_coords, point_labels = labeled_query_points_for_frame(
        query_points,
        frame_index,
        source_hw=video_preview_array.shape[1:3],
        target_hw=video_frames.shape[1:3],
    )
    prompt_source = "selected"
    if len(point_coords) == 0 and selected_tracks is not None and selected_point_labels is not None:
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
        message = f"No selected or visible tracked points on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message
    if not np.any(point_labels == 1):
        message = f"SAM needs at least one visible positive point on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message

    runtime = get_sam_preview_runtime(sam_model)
    predictor = runtime["predictor"]
    frame = as_uint8_rgb_frame(video_frames[frame_index])
    predictor.set_image(frame)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.astype(np.float32),
        point_labels=point_labels.astype(np.int32),
        multimask_output=len(point_coords) == 1,
        normalize_coords=True,
    )
    best_mask = masks[int(np.argmax(scores))]
    preview = draw_sam_preview(frame, best_mask, point_coords, point_labels)
    positive_count = int(np.sum(point_labels == 1))
    negative_count = int(np.sum(point_labels == 0))
    return (
        preview,
        f"SAM preview frame {frame_index} from {prompt_source} points with "
        f"{runtime['model_label']} on {runtime['device']} "
        f"({positive_count} positive, {negative_count} negative point(s)).",
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
            with gr.Row():
                undo = gr.Button("Undo", interactive=False)
                clear_frame = gr.Button("Clear Frame", interactive=False)
                clear_all = gr.Button("Clear All", interactive=False)

            with gr.Row():
                current_frame = gr.Image(
                    label="Click to add query points", 
                    type="numpy",
                    interactive=False
                )
            
            with gr.Row():
                track_button = gr.Button("Track", interactive=False)

        with gr.Column():
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
                autoplay=True,
                loop=True,
            )
            with gr.Row():
                store_frames_button = gr.Button("Store Frames", interactive=False)
                store_coordinates_button = gr.Button(
                    "Store Coordinates of Tracked Object",
                    interactive=False,
                )
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                lines=3,
            )
            with gr.Row():
                sam_model_dropdown = gr.Dropdown(
                    choices=list(SAM_IMAGE_MODEL_CHOICES),
                    value=DEFAULT_SAM_IMAGE_MODEL,
                    label="SAM Image Model",
                    interactive=False,
                )
                sam_preview_button = gr.Button("Preview SAM on Current Frame", interactive=False)
            sam_preview_image = gr.Image(
                label="SAM Point Preview",
                type="numpy",
                interactive=False,
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
            store_frames_button,
            store_coordinates_button,
            sam_model_dropdown,
            sam_preview_button,
            sam_preview_image,
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

    current_frame.select(
        fn = get_point, 
        inputs = [
            query_frames,
            point_type,
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
        fn = track,
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
            store_frames_button,
            store_coordinates_button,
            sam_model_dropdown,
            sam_preview_button,
            export_status,
        ],
        queue = False,
    )

    store_frames_button.click(
        fn = store_frames_from_state,
        inputs = [
            video,
        ],
        outputs = [
            export_status,
        ],
        queue = False,
    )

    store_coordinates_button.click(
        fn = store_coordinates_from_state,
        inputs = [
            video,
            video_preview,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
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

    
demo.launch(
    server_name="127.0.0.1",
    server_port=int(os.environ.get("PORT", "7860")),
    show_api=False,
    show_error=True,
    share=False,
)
