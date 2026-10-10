"""Synthetic validation of the figure integrity guard (item 572).

Builds figures with planted manipulations and clean controls, runs the same
report save_figure builds, and counts detections and false positives.
CPU only.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage as ndi

from spacr import plot

OUT = "/mnt/wd4tb/scratch/f572/validation"
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(572)


def field(size=128, cells=None):
    """A fluorescence-like uint16 field: blobs on background + noise."""
    img = np.zeros((size, size))
    n = cells or rng.integers(5, 15)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n):
        cy, cx = rng.uniform(0, size, 2)
        r = rng.uniform(4, 12)
        a = rng.uniform(0.4, 1.0)
        img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    img = ndi.gaussian_filter(img, 1) * 2000 + 300
    img = rng.poisson(img).astype(np.float64)
    img += rng.normal(0, 30, img.shape)
    return np.clip(img, 0, 65534).astype(np.uint16)


def cell_crop(size=64):
    """A single centred cell, similar shape every time (hard negative)."""
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 2, 2)
    ry, rx = rng.uniform(10, 16, 2)
    ang = rng.uniform(0, np.pi)
    y, x = yy - cy, xx - cx
    u = x * np.cos(ang) + y * np.sin(ang)
    v = -x * np.sin(ang) + y * np.cos(ang)
    body = np.exp(-((u / rx) ** 2 + (v / ry) ** 2) ** 2)
    nucleus = np.exp(-(((u - rng.normal(0, 2)) / 5) ** 2
                       + ((v - rng.normal(0, 2)) / 5) ** 2))
    img = (body * rng.uniform(800, 1500) + nucleus * rng.uniform(500, 1200)
           + 200)
    img = rng.poisson(img).astype(float) + rng.normal(0, 40, img.shape)
    return np.clip(img, 0, 65534).astype(np.uint16)


def figure(panels, ranges, cmap="gray"):
    n = len(panels)
    cols = min(n, 4)
    rows = -(-n // cols)
    fig = Figure(figsize=(2 * cols, 2 * rows), dpi=100)
    for i, (img, rg) in enumerate(zip(panels, ranges)):
        ax = fig.add_subplot(rows, cols, i + 1)
        kw = {} if rg is None else {"vmin": rg[0], "vmax": rg[1]}
        ax.imshow(img, cmap=cmap, **kw)
        ax.set_axis_off()
    return fig


def shared_range(panels):
    pooled = np.concatenate([p.ravel() for p in panels])
    lo, hi = np.percentile(pooled, [0.5, 99.9])
    return (float(lo), float(hi))


def report(fig, fmt="png", requested=None):
    return plot._integrity_report(fig, fmt=fmt, requested_fmt=requested,
                                  dpi=300)


def warnings_of(rep, check=None):
    return [f for f in rep["integrity"]["findings"]
            if f["severity"] == "warning"
            and (check is None or f["check"] == check)]


def dup_hit(rep, a, b):
    return any(set(f["panels"]) == {a, b}
               for f in warnings_of(rep, "duplicate")
               + warnings_of(rep, "region_reuse"))


def gamma(img, g):
    x = img.astype(float) / 65535
    return (x ** g * 65535).astype(np.uint16)


def resize_to(img, shape):
    from PIL import Image
    return np.asarray(Image.fromarray(img.astype(np.float32), mode="F")
                      .resize(shape[::-1], Image.BILINEAR)).astype(np.uint16)


conditions = {}


def run(name, build):
    hits = 0
    per_check = {}
    for _ in range(N):
        ok, rep = build()
        hits += bool(ok)
        for f in warnings_of(rep):
            per_check[f["check"]] = per_check.get(f["check"], 0) + 1
        plt.close("all")
    conditions[name] = {"n": N, "flagged": hits, "rate": hits / N,
                        "warnings_by_check": per_check}
    print(f"{name:32s} {hits:4d}/{N} = {hits / N:6.1%}  {per_check}",
          flush=True)


def clean_fields():
    panels = [field() for _ in range(6)]
    rg = shared_range(panels)
    rep = report(figure(panels, [rg] * 6))
    return bool(warnings_of(rep)), rep


def clean_cells():
    panels = [cell_crop() for _ in range(12)]
    rg = shared_range(panels)
    rep = report(figure(panels, [rg] * 12))
    return bool(warnings_of(rep)), rep


def clean_channels():
    """Two channels of the same field, shown side by side (legitimate)."""
    a = field()
    b = np.clip(ndi.gaussian_filter(a.astype(float), 3) * 0.6
                + field().astype(float) * 0.5, 0, 65534).astype(np.uint16)
    fig = figure([a, b], [shared_range([a])] * 1 + [shared_range([b])],
                 cmap="gray")
    fig.axes[0].images[0].set_cmap("Greens")
    fig.axes[1].images[0].set_cmap("Reds")
    rep = report(fig)
    return bool(warnings_of(rep)), rep


def planted_dup(transform):
    def build():
        panels = [field() for _ in range(6)]
        j, k = rng.choice(6, 2, replace=False)
        panels[k] = transform(panels[j])
        rg = shared_range(panels)
        rep = report(figure(panels, [rg] * 6))
        return dup_hit(rep, int(j), int(k)), rep
    return build


def planted_dup_cells(transform):
    def build():
        panels = [cell_crop() for _ in range(12)]
        j, k = rng.choice(12, 2, replace=False)
        panels[k] = transform(panels[j])
        rg = shared_range(panels)
        rep = report(figure(panels, [rg] * 12))
        return dup_hit(rep, int(j), int(k)), rep
    return build


def noisy(sigma_frac):
    def t(img):
        span = float(np.percentile(img, 99.9) - np.percentile(img, 0.5))
        out = img.astype(float) + rng.normal(0, sigma_frac * span, img.shape)
        return np.clip(out, 0, 65534).astype(np.uint16)
    return t


def crop_reuse(img):
    h, w = img.shape
    y0, x0 = rng.integers(0, h // 4, 2)
    sub = img[y0:y0 + 3 * h // 4, x0:x0 + 3 * w // 4]
    return resize_to(sub, img.shape)


def planted_range(factor, autoscale=False):
    def build():
        panels = [field() for _ in range(6)]
        rg = shared_range(panels)
        k = int(rng.integers(6))
        ranges = [rg] * 6
        if autoscale:
            ranges = [None] * 6
        else:
            ranges[k] = (rg[0], rg[0] + (rg[1] - rg[0]) * factor)
        rep = report(figure(panels, ranges))
        return bool(warnings_of(rep, "display_range")), rep
    return build


TRUE_CLIP = []


def planted_saturation(gain):
    def build():
        panels = [field() for _ in range(6)]
        rg = shared_range(panels)
        k = int(rng.integers(6))
        panels[k] = np.clip(panels[k].astype(float) * gain, 0,
                            65534).astype(np.uint16)
        rep = report(figure(panels, [rg] * 6))
        hit = any(k in f["panels"] for f in warnings_of(rep, "saturation"))
        TRUE_CLIP.append((float(np.mean(panels[k] > rg[1])), hit))
        return hit, rep
    return build


def planted_sensor(frac):
    def build():
        panels = [field() for _ in range(6)]
        k = int(rng.integers(6))
        flat = panels[k].ravel().copy()
        idx = rng.choice(flat.size, int(frac * flat.size), replace=False)
        flat[idx] = 65535
        panels[k] = flat.reshape(panels[k].shape)
        rg = (0.0, 65535.0)
        rep = report(figure(panels, [rg] * 6))
        return any(k in f["panels"] and "acquisition" in f["message"]
                   for f in warnings_of(rep, "saturation")), rep
    return build


def planted_lossy():
    panels = [field() for _ in range(4)]
    rg = shared_range(panels)
    rep = report(figure(panels, [rg] * 4), fmt="jpg")
    return bool(warnings_of(rep, "lossy_format")), rep


print("controls (any warning = false positive)")
run("control_fields_shared_range", clean_fields)
run("control_single_cell_montage", clean_cells)
run("control_two_channels_one_field", clean_channels)
print("planted duplicates (6 fields)")
run("dup_exact", planted_dup(lambda x: x.copy()))
run("dup_flip", planted_dup(lambda x: x[:, ::-1].copy()))
run("dup_rot90", planted_dup(lambda x: np.rot90(x).copy()))
run("dup_linear_contrast", planted_dup(
    lambda x: np.clip(x.astype(float) * 0.7 + 200, 0, 65534)
    .astype(np.uint16)))
run("dup_gamma_0.6", planted_dup(lambda x: gamma(x, 0.6)))
run("dup_noise_5pct", planted_dup(noisy(0.05)))
run("dup_noise_15pct", planted_dup(noisy(0.15)))
run("dup_crop_75pct_rezoomed", planted_dup(crop_reuse))
print("planted duplicates (12 single-cell crops)")
run("dup_cells_exact", planted_dup_cells(lambda x: x.copy()))
run("dup_cells_flip", planted_dup_cells(lambda x: x[::-1].copy()))
run("dup_cells_noise_5pct", planted_dup_cells(noisy(0.05)))

print("planted region reuse and splices")
def crop_frac(frac):
    def t(img):
        h, w = img.shape
        c = int(round(h * frac))
        y0, x0 = rng.integers(0, h - c + 1, 2)
        return resize_to(img[y0:y0 + c, x0:x0 + c], img.shape)
    return t
run("region_crop_50pct_rezoomed", planted_dup(crop_frac(0.5)))
run("region_crop_90pct_rezoomed", planted_dup(crop_frac(0.9)))
run("region_crop_75pct_noise_5pct", planted_dup(lambda x: noisy(0.05)(crop_reuse(x))))


def planted_splice(kind):
    def build():
        panels = [field(256) for _ in range(4)]
        k = int(rng.integers(4))
        a = panels[k].astype(float)
        if kind == "noise":
            b = field(256).astype(float)
            b = b + rng.normal(0, 150, b.shape)
        elif kind == "offset":
            b = field(256).astype(float) + 400
        cut = int(rng.integers(80, 176))
        if kind == "clone":
            y0, x0 = rng.integers(0, 90, 2)
            y1, x1 = rng.integers(130, 210, 2)
            a[y1:y1 + 40, x1:x1 + 40] = a[y0:y0 + 40, x0:x0 + 40]
        elif rng.random() < 0.5:
            a[:, cut:] = b[:, cut:]
        else:
            a[cut:, :] = b[cut:, :]
        panels[k] = np.clip(a, 0, 65534).astype(np.uint16)
        rg = shared_range(panels)
        rep = report(figure(panels, [rg] * 4))
        return any(k in f["panels"] for f in warnings_of(rep, "splice")), rep
    return build
run("splice_noise_level", planted_splice("noise"))
run("splice_background_offset", planted_splice("offset"))
run("splice_clone_40px", planted_splice("clone"))
def clean_large():
    panels = [field(256) for _ in range(4)]
    rg = shared_range(panels)
    rep = report(figure(panels, [rg] * 4))
    return bool(warnings_of(rep)), rep
run("control_fields_256", clean_large)
print("planted display-range mismatch")
run("range_one_panel_upper_x0.6", planted_range(0.6))
run("range_one_panel_upper_x0.9", planted_range(0.9))
run("range_one_panel_upper_x0.97", planted_range(0.97))
run("range_per_panel_autoscale", planted_range(1, autoscale=True))
print("planted saturation")
run("clip_gain_x1.5", planted_saturation(1.5))
run("clip_gain_x2", planted_saturation(2.0))
run("clip_gain_x3", planted_saturation(3.0))
run("sensor_1pct_at_65535", planted_sensor(0.01))
run("sensor_0.2pct_at_65535", planted_sensor(0.002))
bins = {}
for frac, hit in TRUE_CLIP:
    key = ("<2%" if frac < 0.02 else "2-5%" if frac <= 0.05 else
           "5-10%" if frac <= 0.10 else ">10%")
    n, h = bins.get(key, (0, 0))
    bins[key] = (n + 1, h + hit)
print("saturation detection by true clipped fraction of the planted panel:",
      {k: f"{h}/{n}" for k, (n, h) in bins.items()})
conditions["clip_by_true_fraction"] = {k: [h, n] for k, (n, h) in bins.items()}


def grid_roundtrip():
    from PIL import Image
    paths = []
    for i in range(9):
        tile = field(64)
        p = f"{OUT}/tile_{i}.png"
        Image.fromarray(tile).save(p)
        paths.append(p)
    fig = plot.plot_image_grid(paths, (2, 98))
    out = plot.save_figure(fig, f"{OUT}/grid.png", fmt="png", integrity=True)
    sc = plot._provenance_sidecar_path(out)
    rep = json.load(open(sc))
    ok = sum(plot._reproduce_panel(sc, p["panel"])[1] for p in rep["panels"])
    return ok, len(rep["panels"]), rep


ok_total = n_total = 0
for _ in range(max(1, N // 20)):
    ok, n, rep = grid_roundtrip()
    ok_total += ok
    n_total += n
print(f"plot_image_grid tiles rebuilt bit-identical from sidecar: "
      f"{ok_total}/{n_total}")
conditions["grid_reproduction"] = [ok_total, n_total]

print("lossy format")
run("lossy_jpg_written", planted_lossy)

timing = {"off": [], "on": [], "report_only": []}
for i in range(30):
    panels = [field(512) for _ in range(6)]
    fig = figure(panels, [shared_range(panels)] * 6)
    for label, on in ((("off", False), ("on", True)) if i % 2 else
                      (("on", True), ("off", False))):
        t0 = time.perf_counter()
        plot.save_figure(fig, f"{OUT}/timing_{label}_{i}.png", fmt="png",
                         dpi=300, integrity=on)
        timing[label].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    plot._integrity_report(fig, fmt="png", dpi=300)
    timing["report_only"].append(time.perf_counter() - t0)
    plt.close("all")
timing = {k: {"median_s": round(float(np.median(v)), 4),
              "p90_s": round(float(np.percentile(v, 90)), 4)}
          for k, v in timing.items()}
print("timing (6 x 512x512 uint16 panels, PNG 300 dpi):", timing)

json.dump({"conditions": conditions, "timing": timing}, open(
    f"{OUT}/results.json", "w"), indent=2)
