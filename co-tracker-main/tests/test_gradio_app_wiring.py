import re
import runpy
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "gradio_demo" / "app.py"
SAM_SERVICE_PATH = APP_PATH.parent / "sam_preview_service.py"
UI_LAYOUT_PATH = APP_PATH.parent / "ui_layout.py"


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

    def test_sam_image_model_dropdowns_stream_loading_progress_on_change(self):
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
        self.assertIn("fn = stream_sam_model_loading_progress", single_frame_change.group(1))
        self.assertIn("sam_model_loading_progress", single_frame_change.group(1))
        self.assertIn("queue = True", single_frame_change.group(1))
        self.assertIn("fn = stream_sam_model_loading_progress", processed_change.group(1))
        self.assertIn("processed_sam_model_loading_progress", processed_change.group(1))
        self.assertIn("queue = True", processed_change.group(1))

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
            third_step_block.index("export_status = gr.Textbox"),
        )
        self.assertLess(
            third_step_block.index("export_status = gr.Textbox"),
            third_step_block.index("processed_sam_video_skip_frames = gr.Number"),
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

    def test_third_step_has_sam_video_review_below_point_preview(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("processed_sam_video_skip_frames = gr.Number", third_step_block)
        self.assertIn('label="Skip frames after each loaded frame (0 = keep all)"', third_step_block)
        self.assertIn('processed_sam_video_button = gr.Button("Preview SAM on Processed Video"', third_step_block)
        self.assertIn("processed_sam_video_progress = gr.HTML", third_step_block)
        self.assertIn("processed_sam_video = gr.Video", third_step_block)
        self.assertIn('label="SAM video review"', third_step_block)
        self.assertLess(
            third_step_block.index("processed_sam_preview_image = gr.Image"),
            third_step_block.index("processed_sam_video_skip_frames = gr.Number"),
        )
        self.assertLess(
            third_step_block.index("processed_sam_video_skip_frames = gr.Number"),
            third_step_block.index("processed_sam_video = gr.Video"),
        )
        self.assertLess(
            third_step_block.index("processed_sam_video_button = gr.Button"),
            third_step_block.index("processed_sam_video_progress = gr.HTML"),
        )
        self.assertLess(
            third_step_block.index("processed_sam_video_progress = gr.HTML"),
            third_step_block.index("processed_sam_video = gr.Video"),
        )

    def test_processed_sam_video_button_runs_single_queued_sam_video_review(self):
        app_source = read_combined_source()
        click_match = re.search(
            r"processed_sam_video_button\.click\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )

        self.assertIsNotNone(click_match)
        self.assertNotIn("prepare_sam_video_preview", app_source)
        self.assertNotIn("processed_sam_video_start", app_source)
        self.assertIn("fn = preview_sam_video_for_processed_frames", click_match.group(1))
        for state_name in (
            "video",
            "video_preview",
            "query_points",
            "selected_tracks",
            "selected_visibility",
            "selected_point_labels",
            "tracked_video_preview",
            "video_fps",
            "processed_sam_model_dropdown",
            "processed_sam_video_skip_frames",
            "refinement_query_points",
            "tracked_prompt_sources",
        ):
            self.assertIn(state_name, click_match.group(1))
        self.assertIn("processed_sam_video", click_match.group(1))
        self.assertIn("processed_sam_video_progress", click_match.group(1))
        self.assertIn("export_status", click_match.group(1))
        self.assertIn("queue = True", click_match.group(1))
        self.assertIn('show_progress = "hidden"', click_match.group(1))

    def test_track_and_submit_update_sam_video_review_controls(self):
        app_source = read_combined_source()
        submit = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        track = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(submit)
        self.assertIsNotNone(track)
        self.assertIsNotNone(reprocess)
        self.assertIn("processed_sam_video_button", submit.group(1))
        self.assertIn("processed_sam_video_skip_frames", submit.group(1))
        self.assertIn("processed_sam_video", submit.group(1))
        self.assertIn("processed_sam_video_progress", submit.group(1))
        self.assertIn("processed_sam_video_button", track.group(1))
        self.assertIn("processed_sam_video_skip_frames", track.group(1))
        self.assertIn("processed_sam_video_progress", track.group(1))
        self.assertIn("processed_sam_video_button", reprocess.group(1))
        self.assertIn("processed_sam_video_skip_frames", reprocess.group(1))
        self.assertIn("processed_sam_video_progress", reprocess.group(1))

    def test_third_step_has_sam_video_save_and_yolo_export_controls(self):
        app_source = read_combined_source()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("sam_video_save_dir = gr.Textbox", third_step_block)
        self.assertIn('label="SAM video save directory"', third_step_block)
        self.assertIn('save_sam_video_button = gr.Button("Save SAM Video Preview"', third_step_block)
        self.assertIn("saved_sam_video_file = gr.File", third_step_block)
        self.assertIn("yolo_raw_mask_root = gr.Textbox", third_step_block)
        self.assertIn('label="YOLO raw-mask root"', third_step_block)
        self.assertIn("yolo_dataset_output_dir = gr.Textbox", third_step_block)
        self.assertIn('label="YOLO dataset output directory"', third_step_block)
        self.assertIn('save_yolo_custom_button = gr.Button("Save Preview as YOLO Custom"', third_step_block)

    def test_sam_video_save_button_uses_review_video_and_user_directory(self):
        app_source = read_combined_source()
        match = re.search(r"save_sam_video_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = save_sam_video_review_from_state", match.group(1))
        self.assertIn("processed_sam_video", match.group(1))
        self.assertIn("sam_video_save_dir", match.group(1))
        self.assertIn("video_fps", match.group(1))
        self.assertIn("saved_sam_video_file", match.group(1))
        self.assertIn("export_status", match.group(1))

    def test_yolo_custom_button_runs_existing_segmentation_exporter_defaults(self):
        app_source = read_combined_source()
        match = re.search(r"save_yolo_custom_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = export_sam_preview_as_yolo_custom", match.group(1))
        self.assertIn("yolo_raw_mask_root", match.group(1))
        self.assertIn("yolo_dataset_output_dir", match.group(1))
        self.assertIn("export_status", match.group(1))


if __name__ == "__main__":
    unittest.main()
