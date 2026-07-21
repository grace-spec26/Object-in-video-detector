import os
from pathlib import Path
from types import SimpleNamespace


def _example_video_paths(base_dir):
    video_dir = Path(base_dir) / "videos"
    return [
        str(video_dir / "bear.mp4"),
        str(video_dir / "apple.mp4"),
        str(video_dir / "paragliding.mp4"),
        str(video_dir / "paragliding-launch.mp4"),
        str(video_dir / "cat.mp4"),
        str(video_dir / "pillow.mp4"),
        str(video_dir / "teddy.mp4"),
        str(video_dir / "backpack.mp4"),
    ]


def frame_slider_maximum(frame_count):
    """Return a Gradio-safe slider maximum for a zero-based frame index."""
    try:
        frame_count = int(frame_count)
    except (TypeError, ValueError):
        frame_count = 0
    return max(1, frame_count - 1)


def build_demo_layout(
    gr,
    *,
    base_dir,
    default_tracking_resolution,
    tracking_resolution_options,
    default_max_frames,
    point_type_choices,
    positive_point_choice,
    point_edit_mode_choices,
    point_add_mode,
    sam_image_model_choices,
    default_sam_image_model,
    sam_model_progress_ready,
    refinement_edit_mode_choices,
    refinement_add_mode,
    sam_video_progress_ready,
    default_sam_video_save_dir,
    default_raw_mask_root,
    default_yolo_dataset_dir,
    configure_callbacks=None,
):
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
                    choices=list(tracking_resolution_options),
                    value=default_tracking_resolution,
                    label="Tracking Resolution",
                    interactive=True,
                )
                max_frames_input = gr.Number(
                    value=default_max_frames,
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

                if os.environ.get("COTRACKER_DISABLE_EXAMPLES") != "1":
                    gr.Examples(
                        examples=_example_video_paths(base_dir),
                        inputs=[
                            video_in,
                        ],
                    )

        gr.Markdown("## Second step: Simply click \"Track\" to track a grid of points or select query points on the video before clicking")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    query_frames = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label="Choose Frame",
                        interactive=False,
                    )
                with gr.Row():
                    point_type = gr.Radio(
                        choices=list(point_type_choices),
                        value=positive_point_choice,
                        label="Point Type",
                        interactive=True,
                    )
                    query_point_edit_mode = gr.Radio(
                        choices=list(point_edit_mode_choices),
                        value=point_add_mode,
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
                        interactive=False,
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

            with gr.Column():
                sam_model_dropdown = gr.Dropdown(
                    choices=list(sam_image_model_choices),
                    value=default_sam_image_model,
                    label="SAM Image Model",
                    interactive=False,
                )
                sam_model_loading_progress = gr.HTML(
                    value=sam_model_progress_ready,
                    show_label=False,
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
                    maximum=frame_slider_maximum(0),
                    value=0,
                    step=1,
                    label="Choose Processed Frame",
                    interactive=False,
                )
                with gr.Row():
                    refinement_point_type = gr.Radio(
                        choices=list(point_type_choices),
                        value=positive_point_choice,
                        label="Refinement Point Type",
                        interactive=True,
                    )
                    refinement_edit_mode = gr.Radio(
                        choices=list(refinement_edit_mode_choices),
                        value=refinement_add_mode,
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
                    choices=list(sam_image_model_choices),
                    value=default_sam_image_model,
                    label="SAM Image Model",
                    interactive=False,
                )
                processed_sam_model_loading_progress = gr.HTML(
                    value=sam_model_progress_ready,
                    show_label=False,
                )
                processed_sam_preview_button = gr.Button("Preview SAM on Selected Frame", interactive=False)
                processed_sam_preview_image = gr.Image(
                    label="SAM point preview",
                    type="numpy",
                    interactive=False,
                )
                export_status = gr.Textbox(
                    label="Export Status",
                    interactive=False,
                    lines=3,
                )
                processed_sam_video_skip_frames = gr.Number(
                    value=0,
                    precision=0,
                    label="Skip frames after each loaded frame (0 = keep all)",
                    interactive=False,
                )
                processed_sam_video_button = gr.Button("Preview SAM on Processed Video", interactive=False)
                processed_sam_video_progress = gr.HTML(
                    value=sam_video_progress_ready,
                    show_label=False,
                )
                processed_sam_video = gr.Video(
                    label="SAM video review",
                    interactive=False,
                    autoplay=True,
                    loop=True,
                )
                sam_video_save_dir = gr.Textbox(
                    value=str(default_sam_video_save_dir),
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
                    value=str(default_raw_mask_root),
                    label="YOLO raw-mask root",
                    interactive=True,
                )
                yolo_dataset_output_dir = gr.Textbox(
                    value=str(default_yolo_dataset_dir),
                    label="YOLO dataset output directory",
                    interactive=True,
                )

        components = SimpleNamespace(
            demo=demo,
            video=video,
            video_queried_preview=video_queried_preview,
            video_preview=video_preview,
            video_input=video_input,
            video_fps=video_fps,
            query_points=query_points,
            query_points_color=query_points_color,
            is_tracked_query=is_tracked_query,
            query_count=query_count,
            selected_tracks=selected_tracks,
            selected_visibility=selected_visibility,
            selected_point_labels=selected_point_labels,
            tracked_prompt_sources=tracked_prompt_sources,
            tracked_video_preview=tracked_video_preview,
            refinement_query_points=refinement_query_points,
            video_in_drawer=video_in_drawer,
            video_in=video_in,
            tracking_resolution=tracking_resolution,
            max_frames_input=max_frames_input,
            skip_frames_input=skip_frames_input,
            submit=submit,
            query_frames=query_frames,
            point_type=point_type,
            query_point_edit_mode=query_point_edit_mode,
            undo=undo,
            clear_frame=clear_frame,
            clear_all=clear_all,
            current_frame=current_frame,
            track_button=track_button,
            output_video=output_video,
            no_wound_export_button=no_wound_export_button,
            sam_model_dropdown=sam_model_dropdown,
            sam_model_loading_progress=sam_model_loading_progress,
            sam_preview_button=sam_preview_button,
            sam_preview_image=sam_preview_image,
            tracked_query_frames=tracked_query_frames,
            refinement_point_type=refinement_point_type,
            refinement_edit_mode=refinement_edit_mode,
            refinement_undo=refinement_undo,
            refinement_clear_frame=refinement_clear_frame,
            refinement_clear_all=refinement_clear_all,
            reprocess_button=reprocess_button,
            tracked_frame_preview=tracked_frame_preview,
            processed_sam_model_dropdown=processed_sam_model_dropdown,
            processed_sam_model_loading_progress=processed_sam_model_loading_progress,
            processed_sam_preview_button=processed_sam_preview_button,
            processed_sam_preview_image=processed_sam_preview_image,
            export_status=export_status,
            processed_sam_video_skip_frames=processed_sam_video_skip_frames,
            processed_sam_video_button=processed_sam_video_button,
            processed_sam_video_progress=processed_sam_video_progress,
            processed_sam_video=processed_sam_video,
            sam_video_save_dir=sam_video_save_dir,
            save_sam_video_button=save_sam_video_button,
            save_yolo_custom_button=save_yolo_custom_button,
            saved_sam_video_file=saved_sam_video_file,
            yolo_raw_mask_root=yolo_raw_mask_root,
            yolo_dataset_output_dir=yolo_dataset_output_dir,
        )

        if configure_callbacks is not None:
            configure_callbacks(components)

    return components
