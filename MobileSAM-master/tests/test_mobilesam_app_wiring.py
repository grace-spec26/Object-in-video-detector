import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


class MobileSAMAppWiringTest(unittest.TestCase):
    def test_gradio_pyi_generation_is_disabled_before_import(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        skip_index = app_source.index('GRADIO_SKIP_PYI_GENERATION')
        import_index = app_source.index("import_gradio_with_fast_metadata_checks()")

        self.assertLess(skip_index, import_index)
        self.assertIn(
            'os.environ.setdefault("GRADIO_SKIP_PYI_GENERATION", "1")',
            app_source,
        )

    def test_coordinate_folder_event_starts_background_worker_outside_gradio_queue(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        click_start = app_source.index("batch_process_btn.click(")
        click_end = app_source.index("    )", click_start)
        click_block = app_source[click_start:click_end]

        self.assertIn("threading.Thread", app_source)
        self.assertIn("coordinate_batch_state", app_source)
        self.assertIn("start_coordinate_folder_batch", click_block)
        self.assertIn("queue=False", click_block)
        self.assertIn("show_progress=\"hidden\"", click_block)

    def test_coordinate_folder_timer_polls_dedicated_progress_html(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        function_start = app_source.index("def _run_coordinate_folder_batch_worker(")
        function_end = app_source.index("\n\npoint_click_js", function_start)
        function_block = app_source[function_start:function_end]
        timer_start = app_source.index("batch_progress_timer.tick(")
        timer_end = app_source.index("    )", timer_start)
        timer_block = app_source[timer_start:timer_end]

        self.assertIn("batch_progress_html = gr.HTML", app_source)
        self.assertIn("batch_progress_timer = gr.Timer", app_source)
        self.assertIn("format_coordinate_progress_html", app_source)
        self.assertIn("iter_sam2_coordinate_prompt_folder_steps", app_source)
        self.assertIn("set_coordinate_batch_state", function_block)
        self.assertIn("poll_coordinate_folder_batch", timer_block)
        self.assertIn("batch_progress_html", timer_block)
        self.assertIn("queue=False", timer_block)

    def test_coordinate_folder_worker_reports_before_setup(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        function_start = app_source.index("def _run_coordinate_folder_batch_worker(")
        function_end = app_source.index("\n\npoint_click_js", function_start)
        function_block = app_source[function_start:function_end]
        first_state_index = function_block.index("set_coordinate_batch_state")
        first_resolve_index = function_block.index("resolve_user_folder_path")

        self.assertLess(first_state_index, first_resolve_index)
        self.assertIn("Initializing SAM2 coordinate folder run", function_block)
        self.assertIn("flush=True", function_block)

    def test_coordinate_folder_ui_uses_frame_step_not_target_fps(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('batch_frame_step = gr.Number(label="frame_step", value=3', app_source)
        self.assertIn("frame_step_value = parse_frame_step_input", app_source)
        self.assertIn("frame_step=frame_step_value", app_source)
        self.assertNotIn('gr.Number(label="target_fps"', app_source)
        self.assertNotIn("source_fps=30.0", app_source)

    def test_coordinate_folder_worker_reuses_cached_sam2_predictor(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        function_start = app_source.index("def _run_coordinate_folder_batch_worker(")
        function_end = app_source.index("\n\npoint_click_js", function_start)
        function_block = app_source[function_start:function_end]

        self.assertIn("load_sam2_predictor", app_source)
        self.assertIn("sam2_coordinate_runtime", app_source)
        self.assertIn("def get_sam2_coordinate_runtime():", app_source)
        self.assertIn('if sam2_coordinate_runtime["predictor"] is not None:', app_source)
        self.assertIn("Loading SAM2 model (first run only)", function_block)

        loading_start = function_block.index('status="Loading SAM2 model (first run only)"')
        loading_end = function_block.index("sam2_runtime = get_sam2_coordinate_runtime()", loading_start)
        loading_block = function_block[loading_start:loading_end]

        self.assertIn("format_coordinate_progress_html", loading_block)
        self.assertIn("1,", loading_block)
        self.assertIn("4,", loading_block)
        self.assertNotIn("0,\n                1,", loading_block)
        self.assertIn("sam2_runtime = get_sam2_coordinate_runtime()", function_block)
        self.assertIn('predictor=sam2_runtime["predictor"]', function_block)
        self.assertIn('device=sam2_runtime["device"]', function_block)


if __name__ == "__main__":
    unittest.main()
