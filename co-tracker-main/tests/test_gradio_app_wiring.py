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
        self.assertIn("fn = preview_sam_on_frame", match.group(1))
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


if __name__ == "__main__":
    unittest.main()
