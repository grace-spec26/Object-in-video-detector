import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

import yolo_evaluation_service as service  # noqa: E402


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.asarray([[2, 2, 12, 12]], dtype=np.float32)
        self.cls = np.asarray([0], dtype=np.float32)
        self.conf = np.asarray([0.91], dtype=np.float32)


class FakeResult:
    names = {0: "wound"}

    def __init__(self):
        self.boxes = FakeBoxes()


class FakeYoloModel:
    names = {0: "wound"}

    def __init__(self):
        self.calls = []

    def __call__(self, frame, verbose=False):
        self.calls.append(np.asarray(frame).copy())
        return [FakeResult()]


class FakeCv2Capture:
    def __init__(self, frames_bgr, fps=30.0):
        self.frames_bgr = [np.asarray(frame).copy() for frame in frames_bgr]
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == service.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames_bgr)
        if prop == service.cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def read(self):
        if self.index >= len(self.frames_bgr):
            return False, None
        frame = self.frames_bgr[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class FakeCv2Writer:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def release(self):
        self.released = True


class YoloEvaluationServiceTest(unittest.TestCase):
    def test_missing_video_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "best.pt"
            model_path.write_bytes(b"weights")

            results = list(service.preview_yolo_model_on_video(None, str(model_path)))

        self.assertEqual(results[-1][1], None)
        self.assertIn("Upload an evaluation video", results[-1][0])

    def test_missing_model_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"video")

            results = list(service.preview_yolo_model_on_video(str(video_path), None))

        self.assertEqual(results[-1][1], None)
        self.assertIn("Upload a trained YOLO .pt model", results[-1][0])

    def test_nonexistent_video_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "best.pt"
            model_path.write_bytes(b"weights")
            missing_video_path = Path(tmp) / "missing.mp4"

            results = list(service.preview_yolo_model_on_video(str(missing_video_path), str(model_path)))

        self.assertEqual(results[-1][1], None)
        self.assertIn("Evaluation video does not exist", results[-1][0])

    def test_non_pt_model_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.onnx"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            results = list(service.preview_yolo_model_on_video(str(video_path), str(model_path)))

        self.assertEqual(results[-1][1], None)
        self.assertIn("must be a .pt file", results[-1][0])

    def test_empty_video_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            results = list(
                service.preview_yolo_model_on_video(
                    str(video_path),
                    str(model_path),
                    video_reader=lambda path: np.zeros((0, 20, 24, 3), dtype=np.uint8),
                )
            )

        self.assertEqual(results[-1][1], None)
        self.assertIn("contains no frames", results[-1][0])

    def test_missing_ultralytics_reports_clear_status_and_no_output(self):
        frames = np.zeros((1, 20, 24, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            with mock.patch.dict(sys.modules, {"ultralytics": None}):
                results = list(
                    service.preview_yolo_model_on_video(
                        str(video_path),
                        str(model_path),
                        video_reader=lambda path: frames,
                    )
                )

        self.assertEqual(results[-1][1], None)
        self.assertIn("Install ultralytics", results[-1][0])

    def test_processes_each_frame_draws_detections_and_writes_preview(self):
        frames = np.zeros((2, 20, 24, 3), dtype=np.uint8)
        fake_model = FakeYoloModel()
        written = {}

        def fake_writer(path, output_frames, fps):
            written["path"] = path
            written["frames"] = np.asarray(output_frames)
            written["fps"] = fps
            Path(path).write_bytes(b"preview")

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            results = list(
                service.preview_yolo_model_on_video(
                    str(video_path),
                    str(model_path),
                    model_loader=lambda path: fake_model,
                    video_reader=lambda path: frames,
                    video_writer=fake_writer,
                    output_dir=tmp,
                )
            )

        progress_html = "\n".join(result[0] for result in results)
        self.assertIn("0/2 frame", progress_html)
        self.assertIn("1/2 frame", progress_html)
        self.assertIn("2/2 frame", progress_html)
        self.assertEqual(len(fake_model.calls), 2)
        self.assertEqual(written["frames"].shape, frames.shape)
        self.assertEqual(written["fps"], 24.0)
        self.assertTrue(np.any(written["frames"] != frames))
        self.assertEqual(results[-1][1], written["path"])
        self.assertIn("complete", results[-1][0].lower())

    def test_default_video_processing_streams_frames_with_cv2(self):
        frames_bgr = np.zeros((2, 20, 24, 3), dtype=np.uint8)
        frames_bgr[0, :, :] = [10, 20, 30]
        frames_bgr[1, :, :] = [40, 50, 60]
        fake_capture = FakeCv2Capture(frames_bgr, fps=12.5)
        fake_model = FakeYoloModel()
        created_writers = []

        def fake_writer_factory(path, fourcc, fps, size):
            writer = FakeCv2Writer(path, fourcc, fps, size)
            created_writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            with mock.patch.object(service.mediapy, "read_video", side_effect=AssertionError("do not load full video")), \
                mock.patch.object(service.cv2, "VideoCapture", return_value=fake_capture), \
                mock.patch.object(service.cv2, "VideoWriter", side_effect=fake_writer_factory), \
                mock.patch.object(service.cv2, "VideoWriter_fourcc", return_value=1234):
                results = list(
                    service.preview_yolo_model_on_video(
                        str(video_path),
                        str(model_path),
                        model_loader=lambda path: fake_model,
                        output_dir=tmp,
                    )
                )

        self.assertEqual(len(fake_model.calls), 2)
        self.assertEqual(fake_model.calls[0][0, 0].tolist(), [30, 20, 10])
        self.assertEqual(fake_model.calls[1][0, 0].tolist(), [60, 50, 40])
        self.assertEqual(len(created_writers), 1)
        self.assertEqual(created_writers[0].fps, 12.5)
        self.assertEqual(created_writers[0].size, (24, 20))
        self.assertEqual(len(created_writers[0].frames), 2)
        self.assertTrue(fake_capture.released)
        self.assertTrue(created_writers[0].released)
        self.assertEqual(results[-1][1], created_writers[0].path)
        self.assertIn("2/2 frame", results[-1][0])

    def test_long_video_preview_throttles_queued_progress_updates(self):
        frames_bgr = np.zeros((300, 8, 10, 3), dtype=np.uint8)
        fake_capture = FakeCv2Capture(frames_bgr, fps=30.0)
        fake_model = FakeYoloModel()
        created_writers = []

        def fake_writer_factory(path, fourcc, fps, size):
            writer = FakeCv2Writer(path, fourcc, fps, size)
            created_writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            with mock.patch.object(service.mediapy, "read_video", side_effect=AssertionError("do not load full video")), \
                mock.patch.object(service.cv2, "VideoCapture", return_value=fake_capture), \
                mock.patch.object(service.cv2, "VideoWriter", side_effect=fake_writer_factory), \
                mock.patch.object(service.cv2, "VideoWriter_fourcc", return_value=1234):
                results = list(
                    service.preview_yolo_model_on_video(
                        str(video_path),
                        str(model_path),
                        model_loader=lambda path: fake_model,
                        output_dir=tmp,
                    )
                )

        progress_html = "\n".join(result[0] for result in results)
        self.assertEqual(len(fake_model.calls), 300)
        self.assertLessEqual(len(results), 120)
        self.assertIn("0/300 frame", progress_html)
        self.assertIn("1/300 frame", progress_html)
        self.assertIn("300/300 frame", progress_html)
        self.assertEqual(len(created_writers[0].frames), 300)
        self.assertEqual(results[-1][1], created_writers[0].path)


if __name__ == "__main__":
    unittest.main()
