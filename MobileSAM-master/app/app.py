import os
import json

import gradio as gr
from gradio import data_classes as gradio_data_classes
from gradio import networking as gradio_networking
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image, ImageDraw
from utils.tools_gradio import fast_process

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
sam_checkpoint = "../weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

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


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
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

    print(point_coords, point_coords is not None)
    print(point_labels, point_labels is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=len(point_coords) == 1,
    )
    annotations = np.array([masks[np.argmax(scores)]])

    fig = fast_process(
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
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "8080")))
