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

    def test_sam_preview_button_uses_selected_point_labels(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_sam_for_selected_frame", match.group(1))
        self.assertIn("query_points", match.group(1))
        self.assertIn("selected_point_labels", match.group(1))
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

    def test_third_step_has_processed_frame_and_sam_preview_blocks(self):
        app_source = APP_PATH.read_text()

        self.assertIn(
            '## Third step: Fine-tune point adjustment of cotracker and Preview effect of SAM on processed video.',
            app_source,
        )
        self.assertIn("tracked_query_frames = gr.Slider", app_source)
        self.assertIn('label="Choose Processed Frame"', app_source)
        self.assertIn('label="Query points on video"', app_source)
        self.assertIn('label="SAM point preview"', app_source)

    def test_track_populates_processed_frame_preview(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"track_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("tracked_video_preview", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_frame_preview", match.group(1))

    def test_sam_preview_uses_processed_frame_selection_after_tracking(self):
        app_source = APP_PATH.read_text()
        match = re.search(r"sam_preview_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("query_frames", match.group(1))
        self.assertIn("tracked_query_frames", match.group(1))
        self.assertIn("tracked_video_preview", match.group(1))


if __name__ == "__main__":
    unittest.main()
