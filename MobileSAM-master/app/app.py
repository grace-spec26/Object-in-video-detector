import os
import json
import sys
import threading
from pathlib import Path

os.environ.setdefault("GRADIO_SKIP_PYI_GENERATION", "1")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

def import_fsspec_without_entry_point_scan():
    import importlib.metadata as importlib_metadata

    original_entry_points = importlib_metadata.entry_points

    class EmptyEntryPoints(list):
        def select(self, *args, **kwargs):
            return []

    importlib_metadata.entry_points = lambda *args, **kwargs: EmptyEntryPoints()
    try:
        import fsspec  # noqa: F401
    finally:
        importlib_metadata.entry_points = original_entry_points


import_fsspec_without_entry_point_scan()

def import_gradio_with_fast_metadata_checks():
    import importlib.metadata as importlib_metadata
    import importlib.util
    import pkgutil

    original_version = importlib_metadata.version
    original_get_data = pkgutil.get_data

    # huggingface_hub checks these package versions at import time for telemetry
    # and optional-feature flags. On this local environment, reading distribution
    # metadata is very slow, so answer that narrow availability check without
    # walking every .dist-info/METADATA file. Normal metadata behavior is restored
    # immediately after Gradio imports.
    package_modules = {
        "aiohttp": "aiohttp",
        "fastai": "fastai",
        "fastapi": "fastapi",
        "fastcore": "fastcore",
        "gradio": "gradio",
        "graphviz": "graphviz",
        "hf_transfer": "hf_transfer",
        "Jinja2": "jinja2",
        "keras": "keras",
        "minijinja": "minijinja",
        "numpy": "numpy",
        "Pillow": "PIL",
        "pydantic": "pydantic",
        "pydot": "pydot",
        "safetensors": "safetensors",
        "tensorboardX": "tensorboardX",
        "tensorflow": "tensorflow",
        "tensorflow-cpu": "tensorflow",
        "tensorflow-gpu": "tensorflow",
        "tf-nightly": "tensorflow",
        "tf-nightly-cpu": "tensorflow",
        "tf-nightly-gpu": "tensorflow",
        "intel-tensorflow": "tensorflow",
        "intel-tensorflow-avx512": "tensorflow",
        "tensorflow-rocm": "tensorflow",
        "tensorflow-macos": "tensorflow",
        "torch": "torch",
    }

    def fast_version(package_name):
        module_name = package_modules.get(package_name)
        if module_name is None:
            return original_version(package_name)
        if importlib.util.find_spec(module_name) is None:
            raise importlib_metadata.PackageNotFoundError(package_name)
        return "0.0.0"

    def fast_get_data(package, resource):
        if str(package).startswith("gradio") and resource == "package.json":
            return b'{"version": "0.0.0"}'
        return original_get_data(package, resource)

    importlib_metadata.version = fast_version
    pkgutil.get_data = fast_get_data
    try:
        import gradio as imported_gradio
        from gradio import data_classes as imported_data_classes
        from gradio import networking as imported_networking
    finally:
        importlib_metadata.version = original_version
        pkgutil.get_data = original_get_data

    return imported_gradio, imported_data_classes, imported_networking


gr, gradio_data_classes, gradio_networking = import_gradio_with_fast_metadata_checks()
import numpy as np

MOBILE_SAM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MOBILE_SAM_ROOT.parent
RUNTIME_CACHE_DIR = Path(os.environ.get("COTRACKER_MSAM_CACHE_DIR", PROJECT_ROOT / ".runtime_cache"))
MPL_CACHE_DIR = RUNTIME_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = RUNTIME_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
if str(MOBILE_SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_ROOT))

from PIL import Image, ImageDraw
from mobilesam_coordinate_wrapper import (
    DEFAULT_RAW_MASK_DATA_DIR,
    format_coordinate_progress_html,
)
from sam2_coordinate_wrapper import (
    DEFAULT_SOURCE_COORDINATES_DIR,
    DEFAULT_SOURCE_FRAMES_DIR,
    iter_sam2_coordinate_prompt_folder_steps,
)

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.


def patch_gradio_predict_body():
    """Allow Gradio 3.35 request models to run with Pydantic 2."""
    fields = getattr(gradio_data_classes.PredictBody, "model_fields", None)
    if not fields:
        return

    for field_name in ("session_hash", "event_id", "event_data", "fn_index", "request"):
        if field_name in fields:
            fields[field_name].default = None

    gradio_data_classes.PredictBody.model_rebuild(force=True)


patch_gradio_predict_body()
gradio_networking.url_ok = lambda _: True

mobile_sam_runtime_lock = threading.Lock()
mobile_sam_runtime = {
    "torch": None,
    "device": None,
    "mask_generator": None,
    "predictor": None,
}


def get_mobile_sam_runtime():
    with mobile_sam_runtime_lock:
        if mobile_sam_runtime["predictor"] is not None:
            return mobile_sam_runtime

        import torch
        from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_checkpoint = str(MOBILE_SAM_ROOT / "weights" / "mobile_sam.pt")
        model_type = "vit_t"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam = mobile_sam.to(device=device)
        mobile_sam.eval()

        mobile_sam_runtime.update(
            {
                "torch": torch,
                "device": device,
                "mask_generator": SamAutomaticMaskGenerator(mobile_sam),
                "predictor": SamPredictor(mobile_sam),
            }
        )
        return mobile_sam_runtime

# Description
title = "<center><strong><font size='8'>Faster Segment Anything(MobileSAM)<font></strong></center>"

description_e = """This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).

                   We will provide box mode soon. 

                   Enjoy!
                
              """

description_p = """ # Instructions for point mode

                0. Restart by click the Restart button
                1. Select a point with Add Mask for the foreground (Must)
                2. Select a point with Remove Area for the background (Optional)
                3. Click the Start Segmenting.

              """

examples = [
    ["assets/picture3.jpg"],
    ["assets/picture4.jpg"],
    ["assets/picture5.jpg"],
    ["assets/picture6.jpg"],
    ["assets/picture1.jpg"],
    ["assets/picture2.jpg"],
]

default_example = examples[0]

css = """
h1 { text-align: center }
.about { text-align: justify; padding-left: 10%; padding-right: 10%; }
#point-image img.selectable { cursor: crosshair; }
#point-click-payload, #point-click-button { display: none !important; }
"""


def render_annotations_with_fast_process(**kwargs):
    from utils.tools_gradio import fast_process

    return fast_process(**kwargs)


def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    runtime = get_mobile_sam_runtime()
    torch = runtime["torch"]
    device = runtime["device"]
    mask_generator = runtime["mask_generator"]

    with torch.no_grad():
        input_size = int(input_size)
        w, h = image.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))

        nd_image = np.array(image)
        annotations = mask_generator.generate(nd_image)

        fig = render_annotations_with_fast_process(
            annotations=annotations,
            image=image,
            device=device,
            scale=(1024 // input_size),
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            withContours=withContours,
        )
        return fig


def segment_with_points(
    image,
    original_image=None,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    image = ensure_pil_image(original_image) or ensure_pil_image(image)
    if image is None:
        return None, None, "please upload an image first"

    point_coords = np.array(global_points, dtype=np.float32)
    point_labels = np.array(global_point_label, dtype=np.int32)

    if point_coords.size == 0 and point_labels.size == 0:
        print("No points added")
        return image, image, "no points added"

    runtime = get_mobile_sam_runtime()
    torch = runtime["torch"]
    device = runtime["device"]
    predictor = runtime["predictor"]

    print(point_coords, point_coords is not None)
    print(point_labels, point_labels is not None)

    with torch.no_grad():
        nd_image = np.array(image)
        predictor.set_image(nd_image)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=len(point_coords) == 1,
        )
        annotations = np.array([masks[np.argmax(scores)]])

        fig = render_annotations_with_fast_process(
            annotations=annotations,
            image=image,
            device=device,
            scale=1,
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            withContours=withContours,
        )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, image, ""


def ensure_pil_image(image):
    if image is None:
        return None
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def draw_prompt_point(image, x, y, label):
    image = ensure_pil_image(image)
    if image is None:
        return None

    point_radius = 15
    is_positive = label == "Add Mask"
    point_color = (255, 255, 0) if is_positive else (255, 0, 255)
    text_color = (0, 0, 0) if is_positive else (255, 255, 255)
    point_text = "+" if is_positive else "-"

    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    text_bbox = draw.textbbox((0, 0), point_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        (x - text_width / 2, y - text_height / 2 - 1),
        point_text,
        fill=text_color,
    )
    return image


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    if image is None:
        return None, ""

    x, y = evt.index[0], evt.index[1]
    is_positive = label == "Add Mask"
    global_points.append([x, y])
    global_point_label.append(1 if is_positive else 0)

    print(x, y, is_positive)

    return draw_prompt_point(image, x, y, label), ""


def add_point_from_payload(image, label, payload):
    global global_points
    global global_point_label

    if image is None:
        return None, "please upload an image first"

    try:
        point = json.loads(payload or "{}")
        x, y = int(point["x"]), int(point["y"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return image, ""

    image = ensure_pil_image(image)
    if x < 0 or y < 0 or x >= image.width or y >= image.height:
        return image, ""

    is_positive = label == "Add Mask"
    global_points.append([x, y])
    global_point_label.append(1 if is_positive else 0)

    print(x, y, is_positive)

    return draw_prompt_point(image, x, y, label), ""


def reset_points_on_upload(image):
    global global_points
    global global_point_label

    global_points = []
    global_point_label = []
    image = ensure_pil_image(image)
    return image, image, None, ""


def resolve_folder_path(folder_value, default_path):
    if not folder_value:
        return default_path

    path = Path(str(folder_value)).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            PROJECT_ROOT / path,
            MOBILE_SAM_ROOT / path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in candidates:
        if candidate.name in {"frame", "coordinate"}:
            plural_fallback = candidate.with_name(f"{candidate.name}s")
            if plural_fallback.exists():
                return plural_fallback

    return candidates[0]


def resolve_user_folder_path(folder_value, default_path):
    raw_value = str(folder_value or "").strip()
    if not raw_value:
        return Path(default_path)

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        resolved_path = path
    else:
        resolved_path = PROJECT_ROOT / path

    if resolved_path.exists():
        return resolved_path

    folder_aliases = {
        "frame": "frames",
        "frames": "frame",
        "coordinate": "coordinates",
        "coordinates": "coordinate",
    }
    alias_name = folder_aliases.get(resolved_path.name)
    if alias_name:
        alias_path = resolved_path.with_name(alias_name)
        if alias_path.exists():
            return alias_path

    return resolved_path


def parse_float_input(value, field_name):
    raw_value = str(value).strip()
    if "=" in raw_value:
        raw_value = raw_value.split("=", 1)[1].strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc


def parse_frame_step_input(value):
    frame_step = parse_float_input(value, "frame_step")
    if not frame_step.is_integer():
        raise ValueError(f"frame_step must be a whole number, got {value!r}")
    frame_step = int(frame_step)
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    return frame_step


coordinate_batch_state_lock = threading.Lock()
coordinate_batch_state = {
    "running": False,
    "status": "Ready",
    "progress_html": format_coordinate_progress_html(0, 1, "Ready"),
    "frames_zip": None,
    "masks_zip": None,
    "previews_zip": None,
}


def set_coordinate_batch_state(**updates):
    with coordinate_batch_state_lock:
        coordinate_batch_state.update(updates)


def get_coordinate_batch_outputs():
    with coordinate_batch_state_lock:
        state = dict(coordinate_batch_state)
    return (
        state.get("status") or "Ready",
        state.get("progress_html") or format_coordinate_progress_html(0, 1, "Ready"),
        state.get("frames_zip"),
        state.get("masks_zip"),
        state.get("previews_zip"),
    )


def _run_coordinate_folder_batch_worker(
    frame_folder,
    coordinate_folder,
    frame_step,
    padding_ratio,
):
    print("[SAM2 coordinate folders] request received", flush=True)
    set_coordinate_batch_state(
        running=True,
        status="Initializing SAM2 coordinate folder run",
        progress_html=format_coordinate_progress_html(
            0,
            1,
            "Initializing SAM2 coordinate folder run",
        ),
        frames_zip=None,
        masks_zip=None,
        previews_zip=None,
    )
    try:
        frames_dir = resolve_user_folder_path(frame_folder, DEFAULT_SOURCE_FRAMES_DIR)
        coordinates_dir = resolve_user_folder_path(
            coordinate_folder,
            DEFAULT_SOURCE_COORDINATES_DIR,
        )
        print(
            "[SAM2 coordinate folders] "
            f"frames_dir={frames_dir} coordinates_dir={coordinates_dir} "
            f"frame_step={frame_step} padding_ratio={padding_ratio}",
            flush=True,
        )
        frame_step_value = parse_frame_step_input(frame_step)
        padding_ratio_value = parse_float_input(padding_ratio, "padding_ratio")
        set_coordinate_batch_state(
            status="Scanning input folders",
            progress_html=format_coordinate_progress_html(0, 1, "Scanning input folders"),
            frames_zip=None,
            masks_zip=None,
            previews_zip=None,
        )
        final_result = None
        for update in iter_sam2_coordinate_prompt_folder_steps(
            frames_dir=frames_dir,
            coordinates_dir=coordinates_dir,
            output_root=DEFAULT_RAW_MASK_DATA_DIR,
            frame_step=frame_step_value,
            padding_ratio=padding_ratio_value,
        ):
            result = update.get("result")
            progress_html = format_coordinate_progress_html(
                completed=update["completed"],
                total=update["total"],
                message=update["message"],
            )
            print(
                "[SAM2 coordinate folders] "
                f"{update['stage']} {update['completed']}/{update['total']}: "
                f"{update['message']}",
                flush=True,
            )
            if result:
                final_result = result
                status = (
                    "Processed with SAM2 "
                    f"{result['processed_frames']} frame(s). "
                    f"Source frames: {result['source_frames_dir']} | "
                    f"Downloaded frames: {result['frames_dir']} | "
                    f"Source prompts: {result['source_coordinates_dir']} | "
                    f"Processed prompts: {result['coordinates_dir']} | "
                    f"Raw masks: {result['masks_dir']} | "
                    f"Preview frames: {result['previews_dir']}"
                )
                set_coordinate_batch_state(
                    running=False,
                    status=status,
                    progress_html=progress_html,
                    frames_zip=str(result["frames_zip"]),
                    masks_zip=str(result["masks_zip"]),
                    previews_zip=str(result["previews_zip"]),
                )
            else:
                set_coordinate_batch_state(
                    status=update["message"],
                    progress_html=progress_html,
                    frames_zip=None,
                    masks_zip=None,
                    previews_zip=None,
                )

        if final_result is None:
            raise RuntimeError("SAM2 did not produce an output result.")
    except Exception as exc:
        print(f"[SAM2 coordinate folders] failed: {exc}", flush=True)
        set_coordinate_batch_state(
            running=False,
            status=f"SAM2 batch failed: {exc}",
            progress_html=format_coordinate_progress_html(0, 1, "Failed"),
            frames_zip=None,
            masks_zip=None,
            previews_zip=None,
        )


def start_coordinate_folder_batch(
    frame_folder,
    coordinate_folder,
    frame_step,
    padding_ratio,
):
    with coordinate_batch_state_lock:
        if coordinate_batch_state.get("running"):
            return (
                "A SAM2 coordinate folder run is already active.",
                coordinate_batch_state["progress_html"],
                coordinate_batch_state.get("frames_zip"),
                coordinate_batch_state.get("masks_zip"),
                coordinate_batch_state.get("previews_zip"),
            )
        coordinate_batch_state.update(
            running=True,
            status="Starting SAM2 coordinate folder worker",
            progress_html=format_coordinate_progress_html(
                0,
                1,
                "Starting SAM2 coordinate folder worker",
            ),
            frames_zip=None,
            masks_zip=None,
            previews_zip=None,
        )

    worker = threading.Thread(
        target=_run_coordinate_folder_batch_worker,
        args=(frame_folder, coordinate_folder, frame_step, padding_ratio),
        daemon=True,
    )
    worker.start()
    return get_coordinate_batch_outputs()


def poll_coordinate_folder_batch():
    return get_coordinate_batch_outputs()


point_click_js = """
() => {
  const setNativeValue = (element, value) => {
    const prototype = Object.getPrototypeOf(element);
    const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
    if (descriptor && descriptor.set) {
      descriptor.set.call(element, value);
    } else {
      element.value = value;
    }
  };

  const imagePoint = (event, image) => {
    const rect = image.getBoundingClientRect();
    const widthRatio = image.naturalWidth / rect.width;
    const heightRatio = image.naturalHeight / rect.height;
    let x;
    let y;

    if (!image.naturalWidth || !image.naturalHeight) {
      return null;
    }

    if (widthRatio > heightRatio) {
      const drawnHeight = image.naturalHeight / widthRatio;
      const offsetY = (rect.height - drawnHeight) / 2;
      x = Math.round((event.clientX - rect.left) * widthRatio);
      y = Math.round((event.clientY - rect.top - offsetY) * widthRatio);
    } else {
      const drawnWidth = image.naturalWidth / heightRatio;
      const offsetX = (rect.width - drawnWidth) / 2;
      x = Math.round((event.clientX - rect.left - offsetX) * heightRatio);
      y = Math.round((event.clientY - rect.top) * heightRatio);
    }

    if (x < 0 || y < 0 || x >= image.naturalWidth || y >= image.naturalHeight) {
      return null;
    }

    return { x, y, nonce: Date.now() };
  };

  const bindPointImage = () => {
    const image = document.querySelector("#point-image img");
    const payload = document.querySelector(
      "#point-click-payload textarea, #point-click-payload input"
    );
    const button =
      document.querySelector("#point-click-button button") ||
      document.querySelector("#point-click-button");

    if (!image || !payload || !button || image.dataset.mobileSamPointBridge) {
      return;
    }

    image.dataset.mobileSamPointBridge = "true";
    image.style.cursor = "crosshair";
    image.addEventListener("click", (event) => {
      const point = imagePoint(event, image);
      if (!point) {
        return;
      }

      setNativeValue(payload, JSON.stringify(point));
      payload.dispatchEvent(new Event("input", { bubbles: true }));
      payload.dispatchEvent(new Event("change", { bubbles: true }));
      button.click();
    });
  };

  bindPointImage();
  new MutationObserver(bindPointImage).observe(document.body, {
    childList: true,
    subtree: true,
  });
}
"""


cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
upload_img_p = gr.Image(
    label="Upload image",
    value=default_example[0],
    type="pil",
    interactive=True,
)
cond_img_p = gr.Image(
    label="Click image to add + / - points",
    value=default_example[0],
    type="pil",
    interactive=False,
    elem_id="point-image",
)

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")
segm_img_p = gr.Image(
    label="Segmented Image with points", interactive=False, type="pil"
)
status_text_p = gr.Textbox(
    label="Status",
    interactive=False,
    show_label=False,
)
batch_frame_folder = gr.Textbox(
    label="Frame folder",
    value=str(DEFAULT_SOURCE_FRAMES_DIR),
)
batch_coordinate_folder = gr.Textbox(
    label="Coordinate folder",
    value=str(DEFAULT_SOURCE_COORDINATES_DIR),
)
batch_frame_step = gr.Number(label="frame_step", value=3)
batch_padding_ratio = gr.Number(label="Negative padding ratio for positive-only JSON", value=0.15)
batch_status = gr.Textbox(label="Status", interactive=False)
batch_progress_html = gr.HTML(
    value=format_coordinate_progress_html(0, 1, "Ready"),
    label="Progress",
)
batch_frames_download = gr.File(label="Download raw-mask-data/frames")
batch_masks_download = gr.File(label="Download raw-mask-data/mask class-ID PNGs")
batch_previews_download = gr.File(label="Download raw-mask-data/masked_frames previews")
point_click_payload = gr.Textbox(
    label="Point click payload",
    interactive=False,
    show_label=False,
    elem_id="point-click-payload",
)
point_click_button = gr.Button(
    "Add point from image click",
    elem_id="point-click-button",
)

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

with gr.Blocks(
    css=css,
    title="Faster Segment Anything(MobileSAM)",
    analytics_enabled=False,
) as demo:
    original_img_p = gr.State(value=default_example[0])

    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    # with gr.Tab("Everything mode"):
    #     # Images
    #     with gr.Row(variant="panel"):
    #         with gr.Column(scale=1):
    #             cond_img_e.render()
    #
    #         with gr.Column(scale=1):
    #             segm_img_e.render()
    #
    #     # Submit & Clear
    #     with gr.Row():
    #         with gr.Column():
    #             input_size_slider.render()
    #
    #             with gr.Row():
    #                 contour_check = gr.Checkbox(
    #                     value=True,
    #                     label="withContours",
    #                     info="draw the edges of the masks",
    #                 )
    #
    #                 with gr.Column():
    #                     segment_btn_e = gr.Button(
    #                         "Segment Everything", variant="primary"
    #                     )
    #                     clear_btn_e = gr.Button("Clear", variant="secondary")
    #
    #             gr.Markdown("Try some of the examples below ⬇️")
    #             gr.Examples(
    #                 examples=examples,
    #                 inputs=[cond_img_e],
    #                 outputs=segm_img_e,
    #                 fn=segment_everything,
    #                 cache_examples=True,
    #                 examples_per_page=4,
    #             )
    #
    #         with gr.Column():
    #             with gr.Accordion("Advanced options", open=False):
    #                 # text_box = gr.Textbox(label="text prompt")
    #                 with gr.Row():
    #                     mor_check = gr.Checkbox(
    #                         value=False,
    #                         label="better_visual_quality",
    #                         info="better quality using morphologyEx",
    #                     )
    #                     with gr.Column():
    #                         retina_check = gr.Checkbox(
    #                             value=True,
    #                             label="use_retina",
    #                             info="draw high-resolution segmentation masks",
    #                         )
    #             # Description
    #             gr.Markdown(description_e)
    #
    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                upload_img_p.render()
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Add Mask", "Remove Area"],
                        value="Add Mask",
                    )

                    with gr.Column():
                        segment_btn_p = gr.Button(
                            "Start segmenting!", variant="primary"
                        )
                        clear_btn_p = gr.Button("Restart", variant="secondary")
                        status_text_p.render()
                        point_click_payload.render()
                        point_click_button.render()

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    with gr.Tab("Coordinate folders"):
        with gr.Row():
            with gr.Column():
                batch_frame_folder.render()
                batch_coordinate_folder.render()
            with gr.Column():
                batch_frame_step.render()
                batch_padding_ratio.render()
                batch_process_btn = gr.Button(
                    "Run SAM2 from coordinate folders",
                    variant="primary",
                )

        batch_status.render()
        batch_progress_html.render()
        batch_progress_timer = gr.Timer(0.5)
        with gr.Row():
            batch_frames_download.render()
            batch_masks_download.render()
            batch_previews_download.render()

    upload_img_p.upload(
        reset_points_on_upload,
        inputs=upload_img_p,
        outputs=[original_img_p, cond_img_p, segm_img_p, status_text_p],
        queue=False,
    )
    upload_img_p.change(
        reset_points_on_upload,
        inputs=upload_img_p,
        outputs=[original_img_p, cond_img_p, segm_img_p, status_text_p],
        queue=False,
    )
    point_click_button.click(
        add_point_from_payload,
        [cond_img_p, add_or_remove, point_click_payload],
        [cond_img_p, status_text_p],
        queue=False,
    )

    # segment_btn_e.click(
    #     segment_everything,
    #     inputs=[
    #         cond_img_e,
    #         input_size_slider,
    #         mor_check,
    #         contour_check,
    #         retina_check,
    #     ],
    #     outputs=segm_img_e,
    # )

    segment_btn_p.click(
        segment_with_points,
        inputs=[cond_img_p, original_img_p],
        outputs=[segm_img_p, cond_img_p, status_text_p],
    )
    batch_process_btn.click(
        start_coordinate_folder_batch,
        inputs=[
            batch_frame_folder,
            batch_coordinate_folder,
            batch_frame_step,
            batch_padding_ratio,
        ],
        outputs=[
            batch_status,
            batch_progress_html,
            batch_frames_download,
            batch_masks_download,
            batch_previews_download,
        ],
        queue=False,
        show_progress="hidden",
    )
    batch_progress_timer.tick(
        poll_coordinate_folder_batch,
        outputs=[
            batch_status,
            batch_progress_html,
            batch_frames_download,
            batch_masks_download,
            batch_previews_download,
        ],
        queue=False,
        show_progress="hidden",
    )

    def clear():
        global global_points
        global global_point_label

        global_points = []
        global_point_label = []
        return None, None, None, None, ""

    def clear_text():
        return None, None, None

    # clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(
        clear,
        outputs=[upload_img_p, original_img_p, cond_img_p, segm_img_p, status_text_p],
    )
    demo.load(None, None, None, queue=False, js=point_click_js)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "8080")))
