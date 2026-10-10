import sys, time
import numpy as np
import cv2
sys.path.insert(0, "/mnt/wd4tb/spacr-worktrees/claude-f572-20261010")
from spacr import plot as P


def field(seed, size=256, cells=12, noise=4.0, background=20.0, smooth=False):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), background, np.float32)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(cells):
        cy, cx = rng.uniform(0, size, 2)
        r = rng.uniform(8, 22)
        amp = rng.uniform(60, 160)
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    if not smooth:
        tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.5)
        img *= 1 + 0.25 * tex
        img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def cell(seed, size=96, noise=4.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r = rng.uniform(20, 30)
    img = 15 + 140 * np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * r * r)))
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.2)
    img = img * (1 + 0.2 * tex) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def rec(i, a):
    return {"panel": i, "shape": list(a.shape), "source": []}


def run(arrays):
    panels = [rec(i, a) for i, a in enumerate(arrays)]
    t = time.monotonic()
    regions, stats = P._region_reuse_findings(panels, arrays)
    t1 = time.monotonic() - t
    t = time.monotonic()
    spl = P._splice_findings(panels, arrays)
    t2 = time.monotonic() - t
    return regions, stats, spl, t1, t2


def zoom_crop(a, frac, rng, noise=0.0):
    s = a.shape[0]
    c = int(round(s * frac))
    y, x = rng.integers(0, s - c + 1, 2)
    out = cv2.resize(a[y:y + c, x:x + c], (s, s), interpolation=cv2.INTER_LINEAR)
    return out + rng.normal(0, noise, out.shape).astype(np.float32)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = {}
    def tally(name, got):
        res.setdefault(name, [0, 0])
        res[name][0] += bool(got)
        res[name][1] += 1
    times = []
    for k in range(N):
        rng = np.random.default_rng(1000 + k)
        fields = [field(k * 10 + j) for j in range(6)]
        r, st, sp, t1, t2 = run(fields)
        times.append((t1, t2, st))
        tally("clean6 region FP", r); tally("clean6 splice FP", sp)
        cells = [cell(k * 20 + j) for j in range(12)]
        r, st, sp, t1, t2 = run(cells)
        tally("cells12 region FP", r); tally("cells12 splice FP", sp)
        smooth = [field(k * 10 + j, smooth=True) for j in range(4)]
        r, st, sp, *_ = run(smooth)
        tally("smooth region FP", r); tally("smooth splice FP", sp)
        small = cv2.resize(field(k + 500, size=48), (240, 240), interpolation=cv2.INTER_NEAREST)
        r, st, sp, *_ = run([small, field(k + 501)])
        tally("nearest-up splice FP", sp)
        for frac in (0.5, 0.75, 0.9):
            planted = fields[:5] + [zoom_crop(fields[0], frac, rng)]
            r, *_ = run(planted)
            tally(f"crop {frac} region TP", any(set(f["panels"]) == {0, 5} for f in r))
        planted = fields[:5] + [zoom_crop(fields[0], 0.75, rng, noise=4.0)]
        r, *_ = run(planted)
        tally("crop 0.75+noise region TP", any(set(f["panels"]) == {0, 5} for f in r))
        a, b = field(k + 700, noise=3), field(k + 701, noise=9)
        sp_img = a.copy(); sp_img[:, 128:] = b[:, 128:]
        r, st, sp, *_ = run([sp_img])
        tally("splice noise TP", any(f["kind"] == "seam" for f in sp))
        a, b = field(k + 700), field(k + 701, background=40)
        sp_img = a.copy(); sp_img[100:, :] = b[100:, :]
        r, st, sp, *_ = run([sp_img])
        tally("splice offset TP", any(f["kind"] == "seam" for f in sp))
        c = field(k + 800).copy()
        y0, x0 = rng.integers(10, 80, 2); y1, x1 = rng.integers(140, 200, 2)
        c[y1:y1 + 40, x1:x1 + 40] = c[y0:y0 + 40, x0:x0 + 40]
        r, st, sp, *_ = run([c])
        tally("clone 40px TP", any(f["kind"] == "clone" for f in sp))
        pad = field(k + 900).copy(); pad[:, :60] = 0
        r, st, sp, *_ = run([pad])
        tally("padding splice FP", sp)
    for name, (hit, n) in res.items():
        print(f"{name:28s} {hit}/{n}")
    t1 = np.median([t[0] for t in times]); t2 = np.median([t[1] for t in times])
    print("median region s", round(t1, 3), "splice s", round(t2, 3), times[0][2])
