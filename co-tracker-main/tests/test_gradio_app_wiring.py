import re
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "gradio_demo" / "app.py"


class GradioAppWiringTest(unittest.TestCase):
    def test_track_button_runs_without_gradio_queue(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = track", match.group(1))
        self.assertIn("queue = False", match.group(1))

    def test_point_type_selector_is_wired_into_click_handler(self):
        app_source = APP_PATH.read_text()

        self.assertIn("point_type = gr.Radio", app_source)
        match = re.search(r"current_frame\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("point_type", match.group(1))
        self.assertIn("fn = get_point", match.group(1))

    def test_second_step_sam_preview_button_uses_current_query_frame(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_sam_on_frame", match.group(1))
        self.assertIn("query_points", match.group(1))
        self.assertIn("selected_point_labels", match.group(1))
        self.assertIn("query_frames", match.group(1))
        self.assertIn("sam_model_dropdown", match.group(1))

    def test_selecting_points_enables_sam_preview_before_tracking(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"current_frame\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("sam_model_dropdown", match.group(1))
        self.assertIn("sam_preview_button", match.group(1))

    def test_store_coordinates_uses_selected_point_labels(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"store_coordinates_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("selected_point_labels", match.group(1))

    def test_clicked_point_marker_uses_compact_prompt_dot(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"def draw_query_point\(.*?\n\n", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("POINT_PROMPT_RADIUS", match.group(0))
        self.assertNotIn("POINT_SIZE + 3", match.group(0))
        self.assertNotIn("cv2.putText", match.group(0))

    def test_submit_uses_frame_skip_input(self):
        app_source = APP_PATH.read_text()

        self.assertIn('label="Skip frames after each loaded frame (0 = keep all)"', app_source)
        match = re.search(r"submit\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("skip_frames_input", match.group(1))

    def test_second_step_keeps_single_frame_sam_preview_in_query_block(self):
        app_source = APP_PATH.read_text()
        second_step_block = app_source.split("## Third step:", maxsplit=1)[0]

        self.assertIn("sam_model_dropdown = gr.Dropdown", second_step_block)
        self.assertIn('sam_preview_button = gr.Button("Preview SAM on Current Frame"', second_step_block)
        self.assertIn("sam_preview_image = gr.Image", second_step_block)
        self.assertIn('label="SAM point preview"', second_step_block)

    def test_third_step_has_processed_frame_and_sam_preview_blocks(self):
        app_source = APP_PATH.read_text()

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
        app_source = APP_PATH.read_text()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertLess(
            third_step_block.index("tracked_query_frames = gr.Slider"),
            third_step_block.index("processed_sam_model_dropdown = gr.Dropdown"),
        )

    def test_track_populates_processed_frame_preview(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_frame_preview", match.group(1))
        self.assertIn("processed_sam_model_dropdown", match.group(1))
        self.assertIn("processed_sam_preview_button", match.group(1))

    def test_third_step_has_refinement_controls_for_processed_frames(self):
        app_source = APP_PATH.read_text()
        third_step_block = app_source.split("## Third step:", maxsplit=1)[1]

        self.assertIn("refinement_point_type = gr.Radio", third_step_block)
        self.assertIn("refinement_edit_mode = gr.Radio", third_step_block)
        self.assertIn('refinement_undo = gr.Button("Undo Frame Edit"', third_step_block)
        self.assertIn('refinement_clear_frame = gr.Button("Clear Frame Edits"', third_step_block)
        self.assertIn('refinement_clear_all = gr.Button("Clear All Edits"', third_step_block)
        self.assertIn('reprocess_button = gr.Button("Re-process"', third_step_block)
        self.assertIn("tracked_frame_preview = gr.Image", third_step_block)

    def test_processed_frame_click_target_is_display_image_like_query_picker(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"tracked_frame_preview = gr\.Image\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn('label="Query points on video"', match.group(1))
        self.assertIn("interactive=False", match.group(1))

    def test_processed_frame_click_adds_or_deletes_refinement_points(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"tracked_frame_preview\.select\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = edit_refinement_point", match.group(1))
        self.assertIn("refinement_edit_mode", match.group(1))
        self.assertIn("refinement_point_type", match.group(1))
        self.assertIn("refinement_query_points", match.group(1))
        self.assertIn("reprocess_button", match.group(1))

    def test_processed_frame_delete_can_update_original_tracked_prompt_state(self):
        app_source = APP_PATH.read_text()
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
        app_source = APP_PATH.read_text()
        match = re.search(r"reprocess_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = reprocess_with_refinements", match.group(1))
        self.assertIn("query_points", match.group(1))
        self.assertIn("refinement_query_points", match.group(1))
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("tracked_frame_preview", match.group(1))
        self.assertIn("tracked_prompt_sources", match.group(1))

    def test_processed_frame_preview_filters_already_tracked_refinement_points(self):
        app_source = APP_PATH.read_text()
        frame_change = re.search(r"tracked_query_frames\.change\((.*?)\n\s*\)", app_source, re.DOTALL)
        reprocess = app_source.split("def reprocess_with_refinements", maxsplit=1)[1]
        reprocess = reprocess.split("def store_frames_from_state", maxsplit=1)[0]

        self.assertIsNotNone(frame_change)
        self.assertIn("tracked_prompt_sources", frame_change.group(1))
        self.assertIn("pending_refinement_points", app_source)
        self.assertIn("tracked_prompt_sources,", reprocess)

    def test_refinement_edit_callbacks_enable_reprocess_when_any_edits_remain(self):
        app_source = APP_PATH.read_text()

        self.assertIn("gr.update(interactive=count_frame_points(updated_points) > 0)", app_source)
        self.assertIn("gr.update(interactive=False)", app_source)

    def test_sam_preview_uses_processed_frame_selection_after_tracking(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"processed_sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_sam_for_selected_frame", match.group(1))
        self.assertIn("query_frames", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("processed_sam_model_dropdown", match.group(1))

    def test_processed_sam_preview_reads_third_step_refinement_points(self):
        app_source = APP_PATH.read_text()
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


if __name__ == "__main__":
    unittest.main()
