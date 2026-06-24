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


if __name__ == "__main__":
    unittest.main()
