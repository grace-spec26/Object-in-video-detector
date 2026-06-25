import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


class MobileSAMAppWiringTest(unittest.TestCase):
    def test_coordinate_folder_event_uses_queued_minimal_overlay_progress(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        click_start = app_source.index("batch_process_btn.click(")
        click_end = app_source.index("    )", click_start)
        click_block = app_source[click_start:click_end]

        self.assertIn("show_progress=\"minimal\"", click_block)
        self.assertIn("queue=True", click_block)
        self.assertIn("demo.queue(", app_source)

    def test_coordinate_folder_event_streams_dedicated_progress_html(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        function_start = app_source.index("def run_coordinate_folder_batch(")
        function_end = app_source.index("\n\npoint_click_js", function_start)
        function_block = app_source[function_start:function_end]
        click_start = app_source.index("batch_process_btn.click(")
        click_end = app_source.index("    )", click_start)
        click_block = app_source[click_start:click_end]

        self.assertIn("batch_progress_html = gr.HTML", app_source)
        self.assertIn("format_coordinate_progress_html", app_source)
        self.assertIn("iter_coordinate_prompt_folder_steps", app_source)
        self.assertIn("yield", function_block)
        self.assertIn("batch_progress_html", click_block)

    def test_coordinate_folder_event_yields_before_setup_and_avoids_overlay_covering_bar(self):
        app_source = APP_PATH.read_text(encoding="utf-8")

        function_start = app_source.index("def run_coordinate_folder_batch(")
        function_end = app_source.index("\n\npoint_click_js", function_start)
        function_block = app_source[function_start:function_end]
        first_yield_index = function_block.index("yield")
        first_resolve_index = function_block.index("resolve_folder_path")
        click_start = app_source.index("batch_process_btn.click(")
        click_end = app_source.index("    )", click_start)
        click_block = app_source[click_start:click_end]

        self.assertLess(first_yield_index, first_resolve_index)
        self.assertIn("Initializing MobileSAM coordinate folder run", function_block)
        self.assertIn("flush=True", function_block)
        self.assertIn("show_progress=\"minimal\"", click_block)


if __name__ == "__main__":
    unittest.main()
