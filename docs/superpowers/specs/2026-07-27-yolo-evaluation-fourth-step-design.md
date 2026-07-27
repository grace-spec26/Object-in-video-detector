# YOLO Evaluation Fourth Step Design

## Goal

Add a fourth workflow step to the CoTracker + MSAM Gradio app that lets a user upload an evaluation video, upload a trained YOLO `.pt` model, run the model on the video, and review the processed video in the browser.

The initial feature is preview-focused. Model quality metrics are reserved for a later extension because the current requested UI does not include ground-truth label upload or dataset selection.

## Context

The app currently has three Gradio steps in `co-tracker-main/gradio_demo/ui_layout.py`:

1. Upload and preprocess a video.
2. Add query points and run CoTracker.
3. Refine tracked points, preview SAM masks, and export YOLO training data.

Fourth Step belongs under the current Third Step. It should follow the existing Gradio style: a compact left-side control area and a right-side preview result area, similar to the SAM processed-video preview pattern.

The active local environment does not currently provide `ultralytics`, so the implementation must add it to `co-tracker-main/gradio_demo/requirements.txt` and lazy-import it inside the YOLO evaluation helper so missing dependency errors are clear.

## UI Design

Add this section after the Third Step controls:

`## Fourth step: Evaluation of model.`

Use a two-column layout:

Left column:

- `gr.File` labeled `Evaluation Video`, accepting common video files such as `.mp4`, `.mov`, `.avi`, and `.mkv`.
- `gr.File` labeled `Trained YOLO Model`, accepting `.pt`.
- `gr.Button("Preview model on video")`, initially interactive.
- `gr.HTML` for progress/status so it can show a frame-based progress bar.

Right column:

- `gr.Video` labeled `YOLO Model Preview`, non-interactive, with autoplay and loop enabled.

Expose these components through the existing `SimpleNamespace` returned by `build_demo_layout()`:

- `evaluation_video_input`
- `evaluation_yolo_model_input`
- `evaluation_preview_button`
- `evaluation_progress`
- `evaluation_output_video`

## Runtime Design

Create `co-tracker-main/gradio_demo/yolo_evaluation_service.py` with a focused public generator:

`preview_yolo_model_on_video(video_path, model_path)`

Responsibilities:

- Validate that both inputs exist.
- Validate the model path ends with `.pt`.
- Read the video frames with `mediapy.read_video`.
- Load the YOLO model with `ultralytics.YOLO`.
- Run inference frame by frame.
- Draw bounding boxes, class labels, and confidence scores onto RGB frames.
- Write an MP4 preview to `co-tracker-main/gradio_demo/tmp/<uuid>.mp4`.
- Yield progress after each frame, including `processed / total` and percent.
- Yield final output as `(progress_html, output_video_path, status_text)`.

The service should not depend on Gradio component objects, but it may raise `gr.Error` or return user-readable status strings through the app callback, matching nearby service patterns.

## Data Flow

1. User uploads evaluation video and trained YOLO `.pt`.
2. User clicks `Preview model on video`.
3. `app.py` callback calls `preview_yolo_model_on_video()`.
4. The generator yields an initial progress state before loading the model.
5. For each frame, YOLO returns detection boxes; the helper draws detections onto a copy of the frame.
6. The helper writes the preview MP4 and returns its path to the Gradio `gr.Video` component.

This flow is independent from the first three steps. A user can evaluate a trained YOLO model without rerunning CoTracker or SAM in the same session.

## Error Handling

Show clear errors for:

- Missing evaluation video.
- Missing trained YOLO model.
- Nonexistent uploaded paths.
- Model file not ending in `.pt`.
- Missing `ultralytics` dependency.
- Video files that cannot be read or contain zero frames.
- Model loading or inference failures, including which frame failed when possible.
- MP4 writer failures.

When an error occurs, clear the output video and leave the progress/status area with the failure message.

## Future Metrics Extension

Model quality metrics should be added later only when the UI includes ground-truth labels or a labeled dataset input. That extension can reuse the same helper module and add:

- Ground-truth label upload or dataset directory input.
- Confidence and IoU thresholds.
- Per-class counts.
- Precision, recall, and mAP-style summaries.

The first implementation should avoid pretending to compute metrics from video alone.

## Testing

Add or update tests for:

- UI wiring: Fourth Step text exists, the five components are present, and they are returned in the layout namespace.
- Callback wiring: `evaluation_preview_button.click()` passes the evaluation video and model inputs to the service and writes progress plus output video.
- Service validation: missing video, missing model, nonexistent paths, and non-`.pt` model paths produce clear messages.
- Service processing with a fake YOLO model: frames are processed in order, progress advances after each frame, detections are drawn, and an MP4 path is returned.
- Requirements: `ultralytics` appears in `co-tracker-main/gradio_demo/requirements.txt`.
