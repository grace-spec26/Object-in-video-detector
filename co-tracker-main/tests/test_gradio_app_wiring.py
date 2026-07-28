import re
import runpy
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "gradio_demo" / "app.py"
SAM_SERVICE_PATH = APP_PATH.parent / "sam_preview_service.py"
UI_LAYOUT_PATH = APP_PATH.parent / "ui_layout.py"


class DummyGradioComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyGradio:
    Accordion = DummyGradioComponent
    Blocks = DummyGradioComponent
    Button = DummyGradioComponent
    Column = DummyGradioComponent
    Dropdown = DummyGradioComponent
    File = DummyGradioComponent
    HTML = DummyGradioComponent
    Image = DummyGradioComponent
    Markdown = DummyGradioComponent
    Number = DummyGradioComponent
    Radio = DummyGradioComponent
    Row = DummyGradioComponent
    Slider = DummyGradioComponent
    State = DummyGradioComponent
    Textbox = DummyGradioComponent
    Video = DummyGradioComponent

    @staticmethod
    def Examples(*args, **kwargs):
        return DummyGradioComponent(*args, **kwargs)


def read_combined_source():
    return "\n".join(
        (
            APP_PATH.read_text(),
            UI_LAYOUT_PATH.read_text(),
            SAM_SERVICE_PATH.read_text(),
        )
    )


class GradioAppWiringTest(unittest.TestCase):
    def test_page_heading_names_object_in_video_detector(self):
        app_source = read_combined_source()

        self.assertIn("# Object-in-Video Detector", app_source)
        self.assertNotIn(
            "# 🎨 CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos",
            app_source,
        )

    def test_intro_descriptor_describes_object_in_video_detector_workflow(self):
        app_source = read_combined_source()

        self.assertIn("Welcome to Object-in-Video Detector!", app_source)
        self.assertIn("SAM/SAM2 mask previews", app_source)
        self.assertIn("YOLO-ready segmentation data", app_source)
        self.assertIn("mark positive points on the object and optional negative points", app_source)
        self.assertIn("MobileSAM/SAM2-based object mask generation", app_source)
        self.assertNotIn("This space demonstrates point (pixel) tracking in videos.", app_source)

    def test_track_button_runs_without_gradio_queue(self):
        app_source = read_combined_source()
        match = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = track", match.group(1))
        self.assertIn("queue = False", match.group(1))

    def test_point_type_selector_is_wired_into_click_handler(self):
        app_source = read_combined_source()

        self.assertIn("point_type = gr.Radio", app_source)
        match = re.search(r"current_frame\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("point_type", match.group(1))
        self.assertIn("fn = get_point", match.group(1))

    def test_second_step_has_add_delete_mode_control_for_query_points(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertIn("query_point_edit_mode = gr.Radio", second_step_block)
        self.assertIn("POINT_EDIT_MODE_CHOICES", second_step_block)
        self.assertIn('label="Mode"', second_step_block)
        self.assertIn("point_add_mode=POINT_ADD_MODE", app_source)
        self.assertIn("value=point_add_mode", second_step_block)

    def test_second_step_click_handler_can_delete_nearest_query_point(self):
        app_source = read_combined_source()
        match = re.search(r"current_frame\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("query_point_edit_mode", match.group(1))
        self.assertRegex(app_source, r"def get_point\([^)]*point_edit_mode")
        self.assertIn("POINT_DELETE_NEAREST_MODE", app_source)
        self.assertIn("remove_nearest_frame_point", app_source)

    def test_second_step_sam_preview_button_uses_current_query_frame(self):
        app_source = read_combined_source()
        match = re.search(r"sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_sam_on_frame", match.group(1))
        self.assertIn("query_points", match.group(1))
        self.assertIn("selected_point_labels", match.group(1))
        self.assertIn("query_frames", match.group(1))
        self.assertIn("sam_model_dropdown", match.group(1))

    def test_selecting_points_enables_sam_preview_before_tracking(self):
        app_source = read_combined_source()
        match = re.search(r"current_frame\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("sam_model_dropdown", match.group(1))
        self.assertIn("sam_preview_button", match.group(1))

    def test_second_step_has_one_atomic_no_wound_export_button(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertIn(
            'no_wound_export_button = gr.Button("Export No-Wound Frames to YOLO", interactive=False)',
            second_step_block,
        )
        self.assertNotIn("store_frames_button = gr.Button", second_step_block)
        self.assertNotIn("store_coordinates_button = gr.Button", second_step_block)

    def test_no_wound_export_uses_clean_video_state_only(self):
        app_source = read_combined_source()
        match = re.search(r"no_wound_export_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = export_no_wound_frames_from_state", match.group(1))
        self.assertRegex(match.group(1), r"inputs\s*=\s*\[\s*video,?\s*\]")
        self.assertNotIn("selected_tracks", match.group(1))
        self.assertNotIn("selected_point_labels", match.group(1))

    def test_submit_track_and_reprocess_keep_no_wound_export_available(self):
        app_source = read_combined_source()
        submit = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        track = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(submit)
        self.assertIsNotNone(track)
        self.assertIsNotNone(reprocess)
        self.assertIn("no_wound_export_button", submit.group(1))
        self.assertIn("no_wound_export_button", track.group(1))
        self.assertIn("no_wound_export_button", reprocess.group(1))
        self.assertNotIn("store_frames_button", app_source)
        self.assertNotIn("store_coordinates_button", app_source)

    def test_clicked_point_marker_uses_compact_prompt_dot(self):
        app_source = read_combined_source()
        service_source = SAM_SERVICE_PATH.read_text()
        match = re.search(r"def draw_query_point\(.*?\n\n", service_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("draw_query_point", app_source)
        self.assertIn("POINT_PROMPT_RADIUS", match.group(0))
        self.assertNotIn("POINT_SIZE + 3", match.group(0))
        self.assertNotIn("cv2.putText", match.group(0))

    def test_submit_uses_frame_skip_input(self):
        app_source = read_combined_source()

        self.assertIn('label="Skip frames after each loaded frame (0 = keep all)"', app_source)
        match = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("skip_frames_input", match.group(1))

    def test_first_step_uses_stable_file_upload_with_server_side_trim_inputs(self):
        app_source = read_combined_source()
        first_step_block = app_source.split("## Second step:", maxsplit=1)[0]
        match = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIn("video_in = gr.File", first_step_block)
        self.assertNotIn("video_in = gr.Video", first_step_block)
        self.assertIn("trim_start_frame_input = gr.Number", first_step_block)
        self.assertIn("trim_end_frame_input = gr.Number", first_step_block)
        self.assertIn('label="Trim start frame (0 = first frame)"', first_step_block)
        self.assertIn('label="Trim end frame, exclusive (0 = video end)"', first_step_block)
        self.assertIsNotNone(match)
        self.assertIn("trim_start_frame_input", match.group(1))
        self.assertIn("trim_end_frame_input", match.group(1))

    def test_second_step_keeps_single_frame_sam_preview_in_query_block(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertIn("sam_model_dropdown = gr.Dropdown", second_step_block)
        self.assertIn('sam_preview_button = gr.Button("Preview SAM on Current Frame"', second_step_block)
        self.assertIn("sam_preview_image = gr.Image", second_step_block)
        self.assertIn('label="SAM point preview"', second_step_block)

    def test_second_step_places_track_output_export_then_sam_preview_controls(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertLess(
            second_step_block.index("current_frame = gr.Image"),
            second_step_block.index("track_button = gr.Button"),
        )
        self.assertLess(
            second_step_block.index("track_button = gr.Button"),
            second_step_block.index("output_video = gr.Video"),
        )
        self.assertLess(
            second_step_block.index("output_video = gr.Video"),
            second_step_block.index("no_wound_export_button = gr.Button"),
        )
        self.assertLess(
            second_step_block.index("no_wound_export_button = gr.Button"),
            second_step_block.index("sam_model_dropdown = gr.Dropdown"),
        )
        self.assertLess(
            second_step_block.index("sam_model_dropdown = gr.Dropdown"),
            second_step_block.index("sam_preview_button = gr.Button"),
        )
        self.assertLess(
            second_step_block.index("sam_preview_button = gr.Button"),
            second_step_block.index("sam_preview_image = gr.Image"),
        )

    def test_cotracker_gradio_requirements_include_sam2_preview_dependencies(self):
        requirements_path = APP_PATH.parent / "requirements.txt"
        requirements = requirements_path.read_text()

        self.assertIn("hydra-core>=1.3.2", requirements)
        self.assertIn("iopath>=0.1.10", requirements)

    def test_cotracker_gradio_requirements_include_ultralytics_for_yolo_evaluation(self):
        requirements_path = APP_PATH.parent / "requirements.txt"
        requirements = requirements_path.read_text()

        self.assertRegex(requirements, r"(?m)^ultralytics\b")

    def test_fourth_step_has_yolo_evaluation_controls(self):
        layout_source = UI_LAYOUT_PATH.read_text()

        self.assertIn("## Fourth step: Evaluation of model.", layout_source)
        fourth_step_block = layout_source.split("## Fourth step: Evaluation of model.", maxsplit=1)[1]
        self.assertIn("evaluation_video_input = gr.File", fourth_step_block)
        self.assertIn('label="Evaluation Video"', fourth_step_block)
        self.assertIn("evaluation_yolo_model_input = gr.File", fourth_step_block)
        self.assertIn('label="Trained YOLO Model"', fourth_step_block)
        self.assertIn('evaluation_preview_button = gr.Button("Preview model on video"', fourth_step_block)
        self.assertIn("evaluation_progress = gr.HTML", fourth_step_block)
        self.assertIn("evaluation_output_video = gr.Video", fourth_step_block)
        self.assertIn('label="YOLO Model Preview"', fourth_step_block)

    def test_fourth_step_components_are_returned_from_layout_namespace(self):
        layout_source = UI_LAYOUT_PATH.read_text()

        for component_name in (
            "evaluation_video_input",
            "evaluation_yolo_model_input",
            "evaluation_preview_button",
            "evaluation_progress",
            "evaluation_output_video",
        ):
            self.assertIn(f"{component_name}={component_name}", layout_source)

    def test_yolo_evaluation_preview_button_is_wired_to_service(self):
        app_source = APP_PATH.read_text()

        self.assertIn("from yolo_evaluation_service import", app_source)
        match = re.search(r"evaluation_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_yolo_model_on_video", match.group(1))
        self.assertIn("evaluation_video_input", match.group(1))
        self.assertIn("evaluation_yolo_model_input", match.group(1))
        self.assertIn("evaluation_progress", match.group(1))
        self.assertIn("evaluation_output_video", match.group(1))

    def test_yolo_evaluation_preview_button_allows_gradio_queue_for_generator_progress(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"evaluation_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_yolo_model_on_video", match.group(1))
        self.assertNotIn("queue = False", match.group(1))

    def test_layout_runtime_exposes_fourth_step_components_to_callbacks(self):
        ui_layout_namespace = runpy.run_path(UI_LAYOUT_PATH)
        captured = {}

        def capture_callbacks(components):
            captured["components"] = components

        layout = ui_layout_namespace["build_demo_layout"](
            DummyGradio,
            base_dir=APP_PATH.parent,
            default_tracking_resolution="512",
            tracking_resolution_options=["512"],
            default_max_frames=0,
            point_type_choices=["Positive (+)", "Negative (-)"],
            positive_point_choice="Positive (+)",
            point_edit_mode_choices=["Add", "Delete nearest"],
            point_add_mode="Add",
            sam_image_model_choices=["sam2.1_hiera_small.pt"],
            default_sam_image_model="sam2.1_hiera_small.pt",
            sam_model_progress_ready="<p>SAM ready</p>",
            refinement_edit_mode_choices=["Add", "Delete nearest"],
            refinement_add_mode="Add",
            default_yolo_dataset_dir=Path("dataset"),
            yolo_evaluation_progress_ready="<p>YOLO ready</p>",
            configure_callbacks=capture_callbacks,
        )

        self.assertIs(layout, captured["components"])
        self.assertEqual(layout.evaluation_video_input.kwargs["label"], "Evaluation Video")
        self.assertEqual(layout.evaluation_video_input.kwargs["type"], "filepath")
        self.assertEqual(layout.evaluation_yolo_model_input.kwargs["label"], "Trained YOLO Model")
        self.assertEqual(layout.evaluation_yolo_model_input.kwargs["file_types"], [".pt"])
        self.assertEqual(layout.evaluation_preview_button.args[0], "Preview model on video")
        self.assertEqual(layout.evaluation_progress.kwargs["value"], "<p>YOLO ready</p>")
        self.assertEqual(layout.evaluation_output_video.kwargs["label"], "YOLO Model Preview")

    def test_frame_slider_maximum_keeps_gradio_slider_range_non_empty(self):
        ui_layout_namespace = runpy.run_path(UI_LAYOUT_PATH)

        self.assertIn("frame_slider_maximum", ui_layout_namespace)
        frame_slider_maximum = ui_layout_namespace["frame_slider_maximum"]
        self.assertEqual(frame_slider_maximum(0), 1)
        self.assertEqual(frame_slider_maximum(1), 1)
        self.assertEqual(frame_slider_maximum(2), 1)
        self.assertEqual(frame_slider_maximum(5), 4)

    def test_frame_slider_updates_use_non_collapsing_ranges(self):
        app_source = APP_PATH.read_text()

        self.assertNotIn("gr.update(minimum=0, maximum=0", app_source)
        self.assertNotIn("maximum=num_frames - 1", app_source)
        self.assertNotIn("maximum=total_frame_count - 1", app_source)

    def test_sam_preview_preloads_default_model_in_background(self):
        app_source = read_combined_source()
        service_source = SAM_SERVICE_PATH.read_text()

        self.assertIn("def start_sam_preview_preload", service_source)
        self.assertIn("threading.Thread", service_source)
        self.assertIn("start_sam_preview_preload(DEFAULT_SAM_IMAGE_MODEL)", app_source)

    def test_sam_preview_reuses_frame_embedding_for_same_frame(self):
        app_source = read_combined_source()
        service_source = SAM_SERVICE_PATH.read_text()
        preview_fn = service_source.split("def preview_sam_on_frame", maxsplit=1)[1]
        preview_fn = preview_fn.split("def preview_sam_for_selected_frame", maxsplit=1)[0]

        self.assertIn("def sam_preview_frame_cache_key", service_source)
        self.assertIn("def predict_sam_preview_mask", service_source)
        self.assertIn('"image_cache_key"', service_source)
        self.assertIn('"predictor_lock"', service_source)
        self.assertIn("predict_sam_preview_mask", preview_fn)
        self.assertNotIn("predictor.set_image", preview_fn)
        self.assertIn("from sam_preview_service import", app_source)

    def test_single_frame_sam_preview_waits_briefly_while_model_preloads(self):
        service_source = SAM_SERVICE_PATH.read_text()
        preview_fn = service_source.split("def preview_sam_on_frame", maxsplit=1)[1]
        preview_fn = preview_fn.split("def preview_sam_for_selected_frame", maxsplit=1)[0]

        self.assertIn("def get_sam_preview_runtime_if_ready", service_source)
        self.assertIn("sam_preview_runtime_lock.acquire(blocking=False)", service_source)
        self.assertIn("SAM_PREVIEW_RUNTIME_READY_WAIT_SECONDS", service_source)
        self.assertIn("runtime, loading_message = get_sam_preview_runtime_if_ready(", preview_fn)
        self.assertIn("wait_for_ready_seconds=SAM_PREVIEW_RUNTIME_READY_WAIT_SECONDS", preview_fn)
        self.assertIn("prompt_preview = draw_sam_preview", preview_fn)
        self.assertIn("Loaded prompts: {prompt_summary}", preview_fn)
        self.assertIn("return prompt_preview", preview_fn)
        self.assertNotIn("runtime = get_sam_preview_runtime(sam_model)", preview_fn)

    def test_sam_image_model_dropdowns_show_loading_progress_above_preview(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("sam_model_loading_progress = gr.HTML", second_step_block)
        self.assertLess(
            second_step_block.index("sam_model_dropdown = gr.Dropdown"),
            second_step_block.index("sam_model_loading_progress = gr.HTML"),
        )
        self.assertLess(
            second_step_block.index("sam_model_loading_progress = gr.HTML"),
            second_step_block.index("sam_preview_image = gr.Image"),
        )

        self.assertIn("processed_sam_model_loading_progress = gr.HTML", third_step_block)
        self.assertLess(
            third_step_block.index("processed_sam_model_dropdown = gr.Dropdown"),
            third_step_block.index("processed_sam_model_loading_progress = gr.HTML"),
        )
        self.assertLess(
            third_step_block.index("processed_sam_model_loading_progress = gr.HTML"),
            third_step_block.index("processed_sam_preview_image = gr.Image"),
        )

    def test_second_step_has_download_all_sam_models_button_in_sam_panel(self):
        app_source = read_combined_source()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertIn("download_sam_models_button = gr.Button", second_step_block)
        self.assertIn('"Download SAM Models"', second_step_block)
        self.assertLess(
            second_step_block.index("sam_model_dropdown = gr.Dropdown"),
            second_step_block.index("download_sam_models_button = gr.Button"),
        )
        self.assertLess(
            second_step_block.index("download_sam_models_button = gr.Button"),
            second_step_block.index("sam_model_loading_progress = gr.HTML"),
        )

    def test_sam_image_model_dropdowns_reset_preview_and_start_preload_on_change(self):
        app_source = read_combined_source()
        single_frame_change = re.search(
            r"sam_model_dropdown\.change\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )
        processed_change = re.search(
            r"processed_sam_model_dropdown\.change\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )

        self.assertIsNotNone(single_frame_change)
        self.assertIsNotNone(processed_change)
        self.assertIn("fn = sam_model_switch_preview_with_progress", single_frame_change.group(1))
        self.assertIn("sam_preview_image", single_frame_change.group(1))
        self.assertIn("sam_model_loading_progress", single_frame_change.group(1))
        self.assertIn("export_status", single_frame_change.group(1))
        self.assertIn("queue = False", single_frame_change.group(1))
        self.assertIn("fn = processed_sam_model_switch_preview_with_progress", processed_change.group(1))
        self.assertIn("processed_sam_preview_image", processed_change.group(1))
        self.assertIn("processed_sam_model_loading_progress", processed_change.group(1))
        self.assertIn("export_status", processed_change.group(1))
        self.assertIn("queue = False", processed_change.group(1))

    def test_download_all_sam_models_button_streams_to_both_progress_bars(self):
        app_source = read_combined_source()
        match = re.search(r"download_sam_models_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = download_all_sam_image_models_with_progress", match.group(1))
        self.assertIn("sam_model_loading_progress", match.group(1))
        self.assertIn("processed_sam_model_loading_progress", match.group(1))
        self.assertIn("export_status", match.group(1))

    def test_submit_track_and_reprocess_update_sam_model_progress_bars(self):
        app_source = read_combined_source()
        submit = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        track = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(submit)
        self.assertIsNotNone(track)
        self.assertIsNotNone(reprocess)
        self.assertIn("sam_model_loading_progress", submit.group(1))
        self.assertIn("processed_sam_model_loading_progress", submit.group(1))
        self.assertIn("processed_sam_model_loading_progress", track.group(1))
        self.assertIn("processed_sam_model_loading_progress", reprocess.group(1))
        self.assertIn("current_sam_model_progress_html(DEFAULT_SAM_IMAGE_MODEL)", app_source)

    def test_sam_preview_clicks_refresh_model_loading_progress_bar(self):
        app_source = read_combined_source()
        single_preview = re.search(r"sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        processed_preview = re.search(
            r"processed_sam_preview_button\.click\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )

        self.assertIsNotNone(single_preview)
        self.assertIsNotNone(processed_preview)
        self.assertIn("fn = preview_sam_on_frame_with_progress", single_preview.group(1))
        self.assertIn("sam_model_loading_progress", single_preview.group(1))
        self.assertIn("export_status", single_preview.group(1))
        self.assertIn("fn = preview_sam_for_selected_frame_with_progress", processed_preview.group(1))
        self.assertIn("processed_sam_model_loading_progress", processed_preview.group(1))
        self.assertIn("export_status", processed_preview.group(1))

    def test_export_status_is_under_processed_sam_preview_on_right_side(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("export_status = gr.Textbox", third_step_block)
        self.assertLess(
            third_step_block.index("processed_sam_preview_image = gr.Image"),
            third_step_block.index("yolo_dataset_output_dir = gr.Textbox"),
        )
        self.assertLess(
            third_step_block.index("yolo_dataset_output_dir = gr.Textbox"),
            third_step_block.index("save_sam_frame_train_button = gr.Button"),
        )
        self.assertLess(
            third_step_block.index("save_sam_frame_val_button = gr.Button"),
            third_step_block.index("export_status = gr.Textbox"),
        )

    def test_lock_busy_sam_preview_prequeues_requested_model(self):
        service_source = SAM_SERVICE_PATH.read_text()
        runtime_fn = service_source.split("def get_sam_preview_runtime_if_ready", maxsplit=1)[1]
        runtime_fn = runtime_fn.split("def as_uint8_rgb_frame", maxsplit=1)[0]

        self.assertIn("while True:", runtime_fn)
        self.assertIn("sam_preview_runtime_lock.acquire(blocking=False)", runtime_fn)
        self.assertIn("start_sam_preview_preload(model_name)", runtime_fn)
        self.assertIn("remaining_wait = deadline - time.time()", runtime_fn)
        self.assertLess(
            runtime_fn.index("start_sam_preview_preload(model_name)"),
            runtime_fn.index("with sam_preview_preload_lock:"),
        )

    def test_third_step_has_processed_frame_and_sam_preview_blocks(self):
        app_source = read_combined_source()

        self.assertIn(
            '## Third step: Fine-tune point adjustment of cotracker and Preview effect of SAM on processed video.',
            app_source,
        )
        self.assertIn("tracked_query_frames = gr.Slider", app_source)
        self.assertIn("processed_sam_model_dropdown = gr.Dropdown", app_source)
        self.assertIn("processed_sam_preview_button = gr.Button", app_source)
        self.assertIn('label="Choose Processed Frame"', app_source)
        self.assertIn('label="Query points on video"', app_source)
        self.assertIn('label="SAM point preview"', app_source)

    def test_third_step_shows_tracked_frame_block_before_processed_sam_block(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertLess(
            third_step_block.index("tracked_query_frames = gr.Slider"),
            third_step_block.index("processed_sam_model_dropdown = gr.Dropdown"),
        )

    def test_track_populates_processed_frame_preview(self):
        app_source = read_combined_source()
        match = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_frame_preview", match.group(1))
        self.assertIn("processed_sam_model_dropdown", match.group(1))
        self.assertIn("processed_sam_preview_button", match.group(1))

    def test_third_step_has_refinement_controls_for_processed_frames(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("refinement_point_type = gr.Radio", third_step_block)
        self.assertIn("refinement_edit_mode = gr.Radio", third_step_block)
        self.assertIn('refinement_undo = gr.Button("Undo Frame Edit"', third_step_block)
        self.assertIn('refinement_clear_frame = gr.Button("Clear Frame Edits"', third_step_block)
        self.assertIn('refinement_clear_all = gr.Button("Clear All Edits"', third_step_block)
        self.assertIn('reprocess_button = gr.Button("Re-process"', third_step_block)
        self.assertIn("tracked_frame_preview = gr.Image", third_step_block)

    def test_processed_frame_click_target_is_display_image_like_query_picker(self):
        app_source = read_combined_source()
        match = re.search(r"tracked_frame_preview = gr\.Image\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn('label="Query points on video"', match.group(1))
        self.assertIn("interactive=False", match.group(1))

    def test_processed_frame_click_adds_or_deletes_refinement_points(self):
        app_source = read_combined_source()
        match = re.search(r"tracked_frame_preview\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = edit_refinement_point", match.group(1))
        self.assertIn("refinement_edit_mode", match.group(1))
        self.assertIn("refinement_point_type", match.group(1))
        self.assertIn("refinement_query_points", match.group(1))
        self.assertIn("reprocess_button", match.group(1))

    def test_processed_frame_delete_can_update_original_tracked_prompt_state(self):
        app_source = read_combined_source()
        match = re.search(r"tracked_frame_preview\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        for state_name in (
            "query_points",
            "query_points_color",
            "query_count",
            "selected_tracks",
            "selected_visibility",
            "selected_point_labels",
            "tracked_prompt_sources",
            "tracked_video_preview",
        ):
            self.assertIn(state_name, match.group(1))

    def test_reprocess_button_uses_refinement_points_and_replaces_processed_video(self):
        app_source = read_combined_source()
        match = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = reprocess_with_refinements", match.group(1))
        self.assertIn("query_points", match.group(1))
        self.assertIn("refinement_query_points", match.group(1))
        self.assertIn("processed_sam_model_dropdown", match.group(1))
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("tracked_frame_preview", match.group(1))
        self.assertIn("tracked_prompt_sources", match.group(1))

    def test_processed_frame_preview_filters_already_tracked_refinement_points(self):
        app_source = read_combined_source()
        frame_change = re.search(r"tracked_query_frames\.change\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = app_source.split("def reprocess_with_refinements", maxsplit=1)[1]
        reprocess = reprocess.split("def store_frames_from_state", maxsplit=1)[0]

        self.assertIsNotNone(frame_change)
        self.assertIn("tracked_prompt_sources", frame_change.group(1))
        self.assertIn("pending_refinement_points", app_source)
        self.assertIn("tracked_prompt_sources,", reprocess)

    def test_refinement_edit_callbacks_enable_reprocess_when_any_edits_remain(self):
        app_source = read_combined_source()

        self.assertIn("gr.update(interactive=count_frame_points(updated_points) > 0)", app_source)
        self.assertIn("gr.update(interactive=False)", app_source)

    def test_sam_preview_uses_processed_frame_selection_after_tracking(self):
        app_source = read_combined_source()
        match = re.search(r"processed_sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_sam_for_selected_frame", match.group(1))
        self.assertIn("query_frames", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("processed_sam_model_dropdown", match.group(1))

    def test_processed_sam_preview_reads_third_step_refinement_points(self):
        app_source = read_combined_source()
        match = re.search(r"processed_sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("refinement_query_points", match.group(1))
        self.assertIn("tracked_prompt_sources", match.group(1))
        self.assertRegex(
            app_source,
            r"def preview_sam_for_selected_frame\([^)]*refinement_query_points",
        )
        self.assertRegex(
            app_source,
            r"def preview_sam_on_frame\([^)]*refinement_query_points",
        )

    def test_third_step_replaces_sam_video_review_with_frame_yolo_export_buttons(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertNotIn("processed_sam_video_skip_frames", app_source)
        self.assertNotIn("processed_sam_video_button", app_source)
        self.assertNotIn("processed_sam_video_progress", app_source)
        self.assertNotIn("processed_sam_video = gr.Video", app_source)
        self.assertNotIn("Save SAM Video Preview", third_step_block)
        self.assertNotIn("Save Preview as YOLO Custom", third_step_block)
        self.assertIn("save_sam_frame_train_button = gr.Button(", third_step_block)
        self.assertIn('"Save Frame Preview as YOLO Custom Train"', third_step_block)
        self.assertIn("save_sam_frame_val_button = gr.Button(", third_step_block)
        self.assertIn('"Save Frame Preview as YOLO Custom Val"', third_step_block)
        self.assertLess(
            third_step_block.index("processed_sam_preview_image = gr.Image"),
            third_step_block.index("save_sam_frame_train_button = gr.Button"),
        )
        self.assertLess(
            third_step_block.index("save_sam_frame_val_button = gr.Button"),
            third_step_block.index("export_status = gr.Textbox"),
        )

    def test_sam_frame_yolo_export_buttons_write_train_and_val_splits(self):
        app_source = read_combined_source()
        train_click = re.search(r"save_sam_frame_train_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        val_click = re.search(r"save_sam_frame_val_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(train_click)
        self.assertIsNotNone(val_click)
        self.assertIn("fn = export_selected_sam_frame_as_yolo_train", train_click.group(1))
        self.assertIn("fn = export_selected_sam_frame_as_yolo_val", val_click.group(1))
        self.assertNotIn("prepare_sam_video_preview", app_source)
        self.assertNotIn("processed_sam_video_start", app_source)
        self.assertNotIn("mark_sam_video_preview_requested", app_source)
        self.assertNotIn("processed_sam_video_button.click", app_source)
        for state_name in (
            "video",
            "video_preview",
            "query_points",
            "selected_tracks",
            "selected_visibility",
            "selected_point_labels",
            "query_frames",
            "tracked_query_frames",
            "tracked_video_preview",
            "processed_sam_model_dropdown",
            "refinement_query_points",
            "tracked_prompt_sources",
            "yolo_dataset_output_dir",
        ):
            self.assertIn(state_name, train_click.group(1))
            self.assertIn(state_name, val_click.group(1))
        self.assertIn("export_status", train_click.group(1))
        self.assertIn("export_status", val_click.group(1))
        self.assertIn("queue = False", train_click.group(1))
        self.assertIn("queue = False", val_click.group(1))

    def test_track_submit_and_reprocess_update_frame_yolo_export_buttons(self):
        app_source = read_combined_source()
        submit = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        track = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(submit)
        self.assertIsNotNone(track)
        self.assertIsNotNone(reprocess)
        self.assertIn("save_sam_frame_train_button", submit.group(1))
        self.assertIn("save_sam_frame_val_button", submit.group(1))
        self.assertIn("save_sam_frame_train_button", track.group(1))
        self.assertIn("save_sam_frame_val_button", track.group(1))
        self.assertIn("save_sam_frame_train_button", reprocess.group(1))
        self.assertIn("save_sam_frame_val_button", reprocess.group(1))

    def test_third_step_has_selected_frame_yolo_export_controls(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("yolo_dataset_output_dir = gr.Textbox", third_step_block)
        self.assertIn('label="YOLO dataset output directory"', third_step_block)
        self.assertIn("save_sam_frame_train_button = gr.Button(", third_step_block)
        self.assertIn('"Save Frame Preview as YOLO Custom Train"', third_step_block)
        self.assertIn("save_sam_frame_val_button = gr.Button(", third_step_block)
        self.assertIn('"Save Frame Preview as YOLO Custom Val"', third_step_block)
        self.assertNotIn("sam_video_save_dir", third_step_block)
        self.assertNotIn("yolo_raw_mask_root", third_step_block)


if __name__ == "__main__":
    unittest.main()
