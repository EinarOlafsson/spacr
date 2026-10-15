"""Dedicated N698 Gradio service UI; no server or model starts on import.

This is a source implementation, not a deployed service or scientific parity
receipt. The operator supplies trusted local model paths. Remaining adapters
and the browser WebGPU route must be implemented before whole-item acceptance.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import zipfile
import io

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

from spacr.qt.i18n import tr

from .inputs import describe_upload
from .registry import model_families
from .service import PlaqueService


FAMILY_LABELS = {
    "toxoplasma_pv": "Parasite vacuoles", "toxoplasma_plaque": "Plaques",
    "toxoplasma_well_detector": "Plaque wells", "cell_from_hoechst": "Cells from Hoechst",
    "nuclei_from_cellmask": "Nuclei from cell images", "toxoplasma_from_cellmask": "Vacuoles from cell images",
    "toxoplasma_from_hoechst": "Vacuoles from Hoechst", "live_cell": "Live cells",
}


def _selection_index(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise ValueError(tr("{name} must be a finite whole number at least {minimum}.", name=name, minimum=minimum))
    if isinstance(value, (int, np.integer)):
        index = int(value)
    elif isinstance(value, (float, np.floating)) and math.isfinite(value) and value.is_integer():
        index = int(value)
    else:
        raise ValueError(tr("{name} must be a finite whole number at least {minimum}.", name=name, minimum=minimum))
    if index < minimum:
        raise ValueError(tr("{name} must be a finite whole number at least {minimum}.", name=name, minimum=minimum))
    return index


def _card(metadata):
    rows = [tr("Model: {name}", name=metadata["display_name"]), metadata["trained_on"],
            metadata["versus_stock"], tr("Source: {url}", url=metadata["source_url"])]
    return "\n\n".join(str(row) for row in rows if row)


def _preview(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        image = tifffile.imread(io.BytesIO(archive.read("image.tif")))
        mask = tifffile.imread(io.BytesIO(archive.read("mask.tif")))
    grey = image.mean(axis=0) if image.ndim == 3 else image.astype(float)
    low, high = np.percentile(grey, [1, 99])
    grey = np.zeros_like(grey) if high <= low else np.clip((grey - low) / (high - low), 0, 1)
    rgb = np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=-1)
    rgb[find_boundaries(mask, mode="inner")] = [255, 70, 70]
    return rgb


def build_app(provisioned_models, *, preload=False, zero_gpu=False, job_parent=None):
    """Construct actual Gradio components without launching a listening server."""
    import gradio as gr
    from gradio.processing_utils import save_file_to_cache

    controller = PlaqueService(provisioned_models, job_parent=job_parent)
    if preload:
        controller.preload_default()
    catalogue = model_families()
    with gr.Blocks(title=tr("spaCR mask generator"), analytics_enabled=False,
                   delete_cache=(300, 3600)) as demo:
        gr.Markdown(tr("# spaCR mask generator"))
        gr.Markdown(tr("Research use only. The hosted route uploads selected files. Downloaded ZIPs include the original upload. Uploads and downloads use a temporary cache configured for deletion after one hour, checked every five minutes."))
        gr.Markdown(tr("Hosted GPU availability and visitor quotas depend on the configured service. Browser WebGPU is not implemented by this service."))
        registry_output = gr.JSON(visible=False)
        registry_button = gr.Button(visible=False)
        registry_button.click(model_families, outputs=registry_output, api_name="models", queue=False)
        with gr.Tabs():
            for family in catalogue:
                with gr.Tab(tr(FAMILY_LABELS.get(family["family"], family["family"]))):
                    metadata = {record["key"]: record for record in family["versions"]}
                    selected = gr.Dropdown(
                        [(record["display_name"] + " (" + record["status"] + ")", record["key"]) for record in family["versions"]],
                        value=family["default"], label=tr("Model version"))
                    card = gr.Markdown(_card(metadata[family["default"]]))
                    def show_card(key, entries=metadata):
                        return _card(entries[key])
                    selected.change(show_card, inputs=selected, outputs=card, api_visibility="private", queue=False)
                    if family["family"] != "toxoplasma_plaque":
                        gr.Markdown(tr("This model class is listed for provenance; its hosted inference adapter is not implemented yet."))
                        continue
                    consent = gr.Checkbox(False, label=tr("I agree to upload this image for processing and understand that the ZIP includes the original file."))
                    upload = gr.File(label=tr("Image or PDF"), type="filepath", interactive=False,
                                     file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".pdf", ".czi", ".nd2"])
                    consent.change(lambda value: gr.File(interactive=bool(value)), inputs=consent, outputs=upload,
                                   api_visibility="private", queue=False)
                    dimensions = gr.Markdown()
                    with gr.Row():
                        series = gr.Number(0, precision=None, minimum=0, label=tr("Image series / position (zero-based)"))
                        time = gr.Number(0, precision=None, minimum=0, label=tr("Time frame (zero-based)"))
                        depth = gr.Number(0, precision=None, minimum=0, label=tr("Z plane (zero-based)"))
                    channels = gr.Dropdown([], multiselect=True, max_choices=3, value=[], label=tr("Input channels"))
                    with gr.Row():
                        page = gr.Number(0, precision=None, minimum=0, label=tr("PDF page (zero-based)"))
                        dpi = gr.Number(150, precision=None, minimum=1, maximum=600, label=tr("PDF rendering DPI"))
                    with gr.Row():
                        diameter = gr.Number(None, minimum=0, label=tr("Resolved diameter in pixels"),
                                             info=tr("Use the same effective diameter as your desktop run."))
                        native = gr.Checkbox(False, label=tr("Use native scale explicitly"))
                        flow = gr.Number(0.4, minimum=0, label=tr("Flow threshold"))
                        score = gr.Number(0.0, label=tr("Cell-score threshold (logits)"))
                    run_button = gr.Button(tr("Generate masks"), variant="primary")
                    cancel = gr.Button(tr("Cancel pending request"))
                    preview = gr.Image(label=tr("Mask boundary preview"), interactive=False)
                    gr.Markdown(tr("Preview: mean of selected channels with a display-only percentile stretch. Red boundaries show labels; original data and measurements are unchanged in the ZIP."))
                    result = gr.File(label=tr("Download result ZIP"), interactive=False)

                    def choices(path, selected_series):
                        if not path:
                            return gr.Dropdown(choices=[], value=[], multiselect=True, max_choices=3), ""
                        description = describe_upload(path)
                        if description["format"] == "PDF":
                            count = 3
                            text = tr("{count} PDF pages; choose the page and rendering DPI.", count=len(description["pages"]))
                        else:
                            index = _selection_index(selected_series, tr("Image series"))
                            if index < 0 or index >= len(description["series"]):
                                raise gr.Error(tr("Image series is out of range."))
                            info = description["series"][index]
                            sizes = dict(zip(info["axes"].replace("S", "C"), info["shape"]))
                            count = sizes.get("C", 1)
                            text = tr("{width} × {height} pixels; {channels} channels; {series} image series.", width=sizes["X"], height=sizes["Y"], channels=count, series=len(description["series"]))
                        return gr.Dropdown(choices=[(tr("Channel {number}", number=i + 1), i) for i in range(count)],
                                           value=[0] if count == 1 else [], multiselect=True, max_choices=3), text
                    upload.change(choices, inputs=[upload, series], outputs=[channels, dimensions], api_visibility="private", queue=False)
                    series.change(choices, inputs=[upload, series], outputs=[channels, dimensions], api_visibility="private", queue=False)

                    def execute(path, request):
                        with controller.job(path, request) as artifact:
                            cached = save_file_to_cache(artifact["result_zip"], demo.GRADIO_CACHE)
                            overlay = _preview(cached)
                        return cached, overlay
                    gpu_execute = execute
                    if zero_gpu:
                        import spaces
                        gpu_execute = spaces.GPU(duration=120)(execute)

                    def infer(path, model_key, accepted, selected_series, selected_t, selected_z, chosen_channels,
                              selected_page, selected_dpi, chosen_diameter, native_scale, flow_threshold, score_threshold):
                        if not path:
                            raise gr.Error(tr("Choose an image or PDF first."))
                        if accepted is not True:
                            raise gr.Error(tr("Confirm upload consent before processing."))
                        selected_series = _selection_index(selected_series, tr("Image series"))
                        selected_t = _selection_index(selected_t, tr("Time frame"))
                        selected_z = _selection_index(selected_z, tr("Z plane"))
                        selected_page = _selection_index(selected_page, tr("PDF page"))
                        selected_dpi = _selection_index(selected_dpi, tr("PDF rendering DPI"), minimum=1)
                        if not isinstance(chosen_channels, (list, tuple)) or not chosen_channels:
                            raise gr.Error(tr("Choose at least one image channel."))
                        chosen_channels = [_selection_index(value, tr("Image channel")) for value in chosen_channels]
                        header = describe_upload(path)
                        indices = {}
                        if header["format"] != "PDF":
                            if selected_series < 0 or selected_series >= len(header["series"]):
                                raise gr.Error(tr("Image series is out of range."))
                            info = header["series"][selected_series]
                            if "T" in info["axes"]:
                                indices["T"] = selected_t
                            if "Z" in info["axes"]:
                                indices["Z"] = selected_z
                        request = {"family": "toxoplasma_plaque", "model_key": model_key,
                                   "consent_upload": accepted,
                                   "settings": {"diameter": None if native_scale else chosen_diameter,
                                                "flow_threshold": flow_threshold, "CP_prob": score_threshold},
                                   "selection": {"series": selected_series, "indices": indices,
                                                 "channels": chosen_channels, "page": selected_page, "pdf_dpi": selected_dpi}}
                        if chosen_diameter is None and not native_scale:
                            raise gr.Error(tr("Enter the resolved diameter or explicitly choose native scale."))
                        return gpu_execute(path, request)
                    event = run_button.click(
                        infer, inputs=[upload, selected, consent, series, time, depth, channels, page, dpi, diameter, native, flow, score],
                        outputs=[result, preview], api_name="segment_plaque", concurrency_limit=1, concurrency_id="spacr-model")
                    cancel.click(fn=None, cancels=[event], api_visibility="private", queue=False)
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


def main():
    paths = json.loads(os.environ.get("SPACR_WEB_MODELS_JSON", "{}"))
    zero_gpu = os.environ.get("SPACR_WEB_ZERO_GPU") == "1"
    demo = build_app(paths, preload=zero_gpu, zero_gpu=zero_gpu)
    demo.launch(share=False, server_name="0.0.0.0", max_file_size="64mb", show_error=False,
                theme="base", strict_cors=True, run_history=False)


if __name__ == "__main__":
    main()
