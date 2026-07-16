# PySceneDetect Camera Switch Detection

This helper detects camera/shot switches in a video with PySceneDetect adaptive
detection and reports each switch timestamp in `MM:SS:MMM` format.

Install the local dependency:

```bash
python -m pip install -r video-frame-extractor-master/requirements.txt
```

Print detected switch times:

```bash
python video-frame-extractor-master/scene_frame_cutter.py input.mp4
```

Save one frame at each switch:

```bash
python video-frame-extractor-master/scene_frame_cutter.py input.mp4 \
  --output-dir switch_frames \
  --json switch_frames/switches.json \
  --csv switch_frames/switches.csv
```

Use it from Python:

```python
from scene_frame_cutter import detect_camera_switch_times

times = detect_camera_switch_times("input.mp4")
print(times)  # ["00:12:345", "01:04:008", ...]
```

Tune adaptive detection if needed:

```bash
python video-frame-extractor-master/scene_frame_cutter.py input.mp4 \
  --adaptive-threshold 2.0 \
  --min-scene-len 0.5s \
  --window-width 2 \
  --min-content-val 10.0
```
