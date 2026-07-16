# No-Wound YOLO Export Design

## Goal

Allow a user to mark an uploaded CoTracker video as containing no wound and export its loaded clean frames directly as valid negative examples for Ultralytics YOLO instance segmentation.

## Chosen Approach

Replace the second-step `Store Frames` and `Store Coordinates of Tracked Object` buttons with one atomic `Export No-Wound Frames to YOLO` button. The button writes every image and its matching empty label in one operation, preventing unmatched dataset files.

Alternatives considered:

- Extending the mask-based exporter to accept missing or empty masks would mix confirmed-negative examples with segmentation output and make missing-mask mistakes harder to detect.
- Creating synthetic empty masks in `raw-mask-data` would add unnecessary intermediate files and require running the normal mask conversion stage.

The dedicated exporter is the smallest and clearest path because confirmed-negative frames do not need CoTracker coordinates or SAM masks.

## User Interface

- Show one `Export No-Wound Frames to YOLO` button in the second-step SAM point review block.
- Enable it after a video has been successfully submitted; tracking or selecting query points is not required.
- Remove the old frame and coordinate storage buttons from this block and their callback wiring.
- Display success and error details in the existing `Export Status` textbox.

## Export Behavior

- Read the clean loaded video frames from the existing Gradio video state.
- Append to the repository's existing `dataset/` directory.
- Split the submitted frames deterministically using the existing 80% train and 20% validation rule.
- Keep both splits non-empty, requiring at least two loaded frames.
- Convert each frame to RGB JPEG.
- Continue the existing split-specific numbering without overwriting files:
  - `dataset/images/train/train_img00001.jpg`
  - `dataset/labels/train/train_img00001.txt`
  - `dataset/images/val/val_img00001.jpg`
  - `dataset/labels/val/val_img00001.txt`
- Create each label file as a zero-byte UTF-8 text file. An empty file means the corresponding image contains no wound instances.
- Create or refresh `dataset/dataset.yaml` with class `0: wound` and the existing Ultralytics directory layout.
- Report exported, train, and validation image counts in the UI.

Repeated button presses append another uniquely named copy of the currently loaded frames, matching the existing dataset export behavior.

## Validation

Update dataset validation so that:

- Every JPEG still requires a matching `.txt` file and vice versa.
- Empty `.txt` files are accepted as valid no-object labels.
- Non-empty files retain all current YOLO polygon checks: class `0`, at least three points, paired coordinates, and values normalized to `0-1`.
- Train and validation directories must remain non-empty.

## Code Boundaries

- Put reusable no-wound dataset writing logic beside the existing YOLO dataset exporter in `export_yolo_segmentation_dataset.py`.
- Keep the Gradio callback in `co-tracker-main/gradio_demo/app.py` thin: validate state, call the exporter, and format status/errors.
- Add behavior tests for split, append naming, zero-byte labels, and mixed positive/negative dataset validation.
- Add wiring tests for the new button label, callback, state input, and post-submit availability.

## Error Handling

- If no video is loaded, show a clear instruction to submit a video first.
- If fewer than two frames are loaded, stop before writing and explain that both train and validation splits require at least two frames.
- Do not partially write an image without its corresponding empty label.
- Surface filesystem and image-conversion failures in `Export Status` and a Gradio warning.
