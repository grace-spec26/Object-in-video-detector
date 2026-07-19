import re
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "gradio_demo"
APP_PATH = DEMO_DIR / "app.py"


class GradioRefactorStructureTest(unittest.TestCase):
    def test_app_delegates_sam_preview_logic_to_service_module(self):
        app_source = APP_PATH.read_text()

        self.assertTrue((DEMO_DIR / "sam_preview_service.py").exists())
        self.assertIn("from sam_preview_service import", app_source)
        for function_name in (
            "get_sam_preview_runtime",
            "sam_point_prompts_for_frame",
            "preview_sam_on_frame",
            "preview_sam_video_for_processed_frames",
        ):
            self.assertNotRegex(
                app_source,
                rf"^def {re.escape(function_name)}\(",
                msg=f"{function_name} should live in sam_preview_service.py",
            )

    def test_app_delegates_cotracker_inference_to_tracking_service_module(self):
        app_source = APP_PATH.read_text()

        self.assertTrue((DEMO_DIR / "tracking_service.py").exists())
        self.assertIn("from tracking_service import", app_source)
        self.assertIn("run_cotracker_tracking", app_source)
        self.assertNotIn("model(video_chunk=video_input", app_source)
        self.assertNotIn("get_online_chunk_start_indices(video_input.shape[1]", app_source)

    def test_app_delegates_gradio_component_layout_to_ui_layout_module(self):
        app_source = APP_PATH.read_text()

        self.assertTrue((DEMO_DIR / "ui_layout.py").exists())
        self.assertIn("from ui_layout import", app_source)
        self.assertIn("build_demo_layout", app_source)
        self.assertNotIn("with gr.Blocks() as demo:", app_source)
        self.assertNotIn("video_in = gr.Video", app_source)
        self.assertNotIn("processed_sam_preview_image = gr.Image", app_source)


if __name__ == "__main__":
    unittest.main()
