# SAM Processed Video Preview Design

## Goal

Make `Preview SAM on Processed Video` reliably produce a visible Step 3 video containing the chronological subsequence selected by the frame-skip control, with CoTracker point prompts passed to SAM and each retained frame displaying the resulting mask overlay.

## Chosen Approach

Keep the existing queued SAM video generator, but precede it with a lightweight, non-queued start callback. The start callback reads and validates the skip value and immediately changes the custom progress display so every valid button click has visible feedback before model work begins.

The queued generator will continue to own model loading, frame selection, CoTracker-to-SAM prompt conversion, mask prediction, overlay rendering, video encoding, and detailed progress updates.

Alternatives considered:

- Running the whole operation without Gradio's queue would provide immediate callback execution but could block the interface while SAM processes the video.
- Streaming individual preview images would provide finer live feedback but would replace the existing video-review workflow and require substantially broader UI changes.

The staged callback is the smallest change that addresses the silent-click symptom while preserving the existing Step 3 video component and generator architecture.

## Previous Behavior

The older SAM video-review function iterated over every source frame. Frames excluded by the skip setting, frames without visible points, and frames without positive points were appended to the output unchanged. The result retained the full source-frame sequence and showed SAM overlays only intermittently.

The newer implementation correctly builds an explicit selected-frame subsequence and omits frames that did not run SAM. The repair will retain this newer output contract while restoring dependable click feedback and using the shared SAM prediction path.

## Frame Selection

- Read `Skip frames after each loaded frame (0 = keep all)` once when the button is clicked.
- Require a non-negative integer.
- Use a stride of `skip + 1`.
- For `skip = 0`, select frames `0, 1, 2, 3, ...`.
- For `skip = 2`, select frames `0, 3, 6, 9, ...`.
- Preserve the selected frames' chronological order.
- Encode the reduced sequence at the original video FPS. Skipping two frames therefore makes the preview play approximately three times faster.

## SAM Processing

For every selected frame:

1. Convert the source frame to an RGB `uint8` image.
2. Obtain visible CoTracker coordinates and their positive or negative labels for that exact original frame index.
3. Include pending Step 3 refinement prompts using the existing prompt-source mapping.
4. Require at least one visible positive point before running SAM.
5. Run prediction through the existing shared SAM prediction helper so predictor locking and image-cache handling are consistent with single-frame preview.
6. Select the best candidate mask using the existing mask-selection logic.
7. Draw the mask overlay and prompt points on the frame.
8. Append only successfully masked frames to the output sequence.

Frames without usable coordinates or without a positive point are reported and excluded because they cannot satisfy the requirement that every output frame display a SAM mask.

## User Interface Flow

1. The Step 3 button click invokes a non-queued preparation callback.
2. The preparation callback validates the skip value and immediately displays `Preparing SAM video preview` with `0/N` selected frames.
3. The queued generator loads or waits for the selected SAM model.
4. Progress changes as each selected frame is checked and processed.
5. After successful encoding, the generated MP4 path is returned to the existing `SAM video review` component.
6. The final status reports total source frames, selected frames, masked frames, and skipped-frame reasons.

The operation does not write raw masks or alter the existing YOLO dataset export path.

## Error Handling

- An invalid skip value produces a visible Gradio error before SAM processing starts.
- Missing video or CoTracker state produces a visible progress and status message rather than a silent no-op.
- Missing visible coordinates or positive prompts are counted and reported per selected frame.
- A SAM model-loading failure is displayed in the progress and status outputs.
- If no selected frame can be masked, no empty MP4 is written and the preview component remains empty.
- Video-encoding or prediction exceptions are surfaced in the existing Export Status area and Gradio error display.

## Testing

Add or update tests that verify:

- The preparation callback reads the skip value and reports the correct selected-frame count.
- `skip = 2` selects original frame indices `0, 3, 6, 9, ...`.
- Selected frames are passed to SAM in chronological order.
- CoTracker coordinates and labels for each original frame index are passed to SAM.
- Every output frame contains the SAM mask overlay and prompt rendering.
- Unselected and unmasked frames are absent from the encoded output.
- The encoded MP4 uses the original FPS.
- Invalid input and missing tracking state produce visible status output.
- The Gradio button wiring runs the preparation stage before the queued video generator.

## Scope

Changes are limited to the Step 3 SAM processed-video preview service, its Gradio callback wiring, and focused tests. CoTracker tracking, single-frame SAM preview, raw-mask generation, and YOLO dataset export remain unchanged.
