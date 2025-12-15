import os, re, csv, math, time, hashlib, threading, shutil, cv2, tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple, Dict, Union, List, Pattern, Any
import matplotlib.pyplot as plt
import numpy as np

class _DiskFeatureStore:
    """
    Disk-backed feature cache with an in-RAM LRU of limited size.

    Saves one NPZ per image containing:
      - ds8  : uint8 downsampled plane
      - pts  : float32 (N,2)
      - desc : uint8 for ORB or float32 for SIFT
      - Hds, Wds, H, W : int32 scalars
    """
    def __init__(self, root_dir: str, max_ram_items: int = 256, verbose: bool = False):
        self.root = os.path.abspath(root_dir)
        os.makedirs(self.root, exist_ok=True)
        self.max_ram = int(max_ram_items)
        self.verbose = bool(verbose)
        self._ram: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._lru_lock = threading.Lock()  # NEW

    @staticmethod
    def _key_for_path(path: str) -> str:
        h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()
        return h[:16]

    def _npz_path(self, path: str) -> str:
        return os.path.join(self.root, f"{self._key_for_path(path)}.npz")

    def get(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        # LRU hit
        with self._lru_lock:
            if path in self._ram:
                v = self._ram.pop(path)
                self._ram[path] = v
                return v
        # Disk hit (no lock while reading disk)
        pz = self._npz_path(path)
        if os.path.exists(pz):
            with np.load(pz, allow_pickle=False) as Z:
                feat = dict(
                    ds8=Z["ds8"],
                    pts=Z["pts"].astype(np.float32),
                    desc=Z["desc"],
                    Hds=int(Z["Hds"]),
                    Wds=int(Z["Wds"]),
                    H=int(Z["H"]),
                    W=int(Z["W"]),
                )
            # insert into LRU
            with self._lru_lock:
                self._ram[path] = feat
                if len(self._ram) > self.max_ram:
                    self._ram.popitem(last=False)
            return feat
        return None

    def put(self, path: str, feat: Dict[str, np.ndarray]) -> None:
        # Save to disk
        np.savez_compressed(self._npz_path(path),
                            ds8=feat["ds8"],
                            pts=feat["pts"],
                            desc=feat["desc"],
                            Hds=np.int32(feat["Hds"]),
                            Wds=np.int32(feat["Wds"]),
                            H=np.int32(feat["H"]),
                            W=np.int32(feat["W"]))
        # Insert in RAM LRU
        with self._lru_lock:
            self._ram[path] = feat
            if len(self._ram) > self.max_ram:
                self._ram.popitem(last=False)


class spacrStitcher:
    """
    Pairwise stitcher with downsampled scoring and whole-well mosaic assembly.

    Robustness features for very large datasets:
      - feature_cache_mode: "disk" (default) writes DS features to disk with an LRU RAM cap
      - pair_batch_size:    limit number of concurrent pair futures
      - stream_csv:         write pairwise rows immediately (low RAM)
      - opencv_threads:     limit OpenCV internal threading to avoid oversubscription

    Transform control
    -----------------
      allow_scale:    if True  → use RANSAC affine (rotation+scale+translation)
                      if False → enforce unit scale (see allow_rotation)
      allow_rotation: if True  → rotation+translation
                      if False → translation-only

    Mosaic
    ------
      Uses topology-aware pruning + maximum spanning tree on pairwise scores
      to build a cohesive grid. Exports mosaic image and a mosaic.csv manifest.

    Axis & Z handling
    -----------------
      arr_axes : str in {"AUTO"} or a string over {T,C,Z,Y,X} (e.g. "CZYX", "CYX", "ZYX").
                 Determines how to interpret multi-dimensional TIFFs.
      mip      : bool. If True and Z exists, perform max-intensity projection over Z.
      z_index  : int  (if mip=False) choose Z slice index.
      t_index  : int  choose time index if T exists.
    """

    # ----------------------------- init ---------------------------------
    def __init__(self,
                 detector: str = "ORB",
                 nfeatures: int = 6000,
                 max_keypoints: Optional[int] = 2000,
                 downsample: float = 0.5,
                 ransac_thresh_px: float = 3.0,
                 allow_scale: bool = False,
                 allow_rotation: bool = False,
                 # QC outlines (DS)
                 outline_source: str = "otsu",
                 canny: Tuple[int, int] = (40, 120),
                 blur_sigma: float = 0.0,
                 dilate_ksize: int = 0,
                 line_thickness: int = 1,
                 outline_alpha: float = 1.0,
                 # IO
                 outdir: str = "./sbs_out",
                 save_qc: bool = True,
                 save_stitched_default: bool = True,
                 # scoring & control
                 all_scores: bool = False,
                 score_threshold: Optional[float] = None,
                 verbose: bool = False,
                 # robustness / scaling controls
                 feature_cache_mode: str = "disk",          # "ram" | "disk"
                 feature_cache_dir: Optional[str] = None,   # where to store DS features if disk
                 max_ram_features: int = 256,               # LRU size if disk mode
                 n_workers_features: Optional[int] = None,  # feature threads
                 pair_batch_size: int = 8000,               # max pairs processed per batch
                 stream_csv: bool = True,                   # write rows as we go
                 opencv_threads: int = 1,                   # avoid thread oversubscription
                 # axis/Z/time handling
                 arr_axes: str = "AUTO",
                 mip: bool = False,
                 z_index: int = 0,
                 t_index: int = 0,
                 squeeze_singleton: bool = True,
                 ):
        self.detector = detector.upper()
        self.nfeatures = int(nfeatures)
        self.max_keypoints = None if max_keypoints is None else int(max_keypoints)
        self.downsample = float(downsample)
        self.ransac_thresh_px = float(ransac_thresh_px)
        self.allow_scale = bool(allow_scale)
        self.allow_rotation = bool(allow_rotation)
    
        self.outline_source = outline_source.lower()
        self.canny = tuple(canny)
        self.blur_sigma = float(blur_sigma)
        self.dilate_ksize = int(dilate_ksize)
        self.line_thickness = int(line_thickness)
        self.outline_alpha = float(outline_alpha)
    
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.save_qc = bool(save_qc)
        self.save_stitched_default = bool(save_stitched_default)
    
        self.all_scores = bool(all_scores)
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.verbose = bool(verbose)
    
        # detector init
        if self.detector == "ORB":
            self._det = cv2.ORB_create(nfeatures=self.nfeatures, fastThreshold=5)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._use_flann = False
        elif self.detector == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT requested but opencv-contrib build not found.")
            self._det = cv2.SIFT_create(nfeatures=self.nfeatures)
            self._use_flann = True
            self._flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        else:
            raise ValueError("detector must be 'ORB' or 'SIFT'")
    
        # OpenCV threads
        try:
            cv2.setNumThreads(int(opencv_threads))
        except Exception:
            pass
        self._opencv_threads = int(opencv_threads)
    
        # Feature cache
        self.feature_cache_mode = feature_cache_mode.lower()
        if self.feature_cache_mode not in ("ram", "disk"):
            raise ValueError("feature_cache_mode must be 'ram' or 'disk'")
    
        self._feat_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._feat_lock = threading.Lock()
    
        self._store = None
        if self.feature_cache_mode == "disk":
            cache_dir = feature_cache_dir or os.path.join(self.outdir, "feat_cache")
            self.feature_cache_dir = cache_dir
            self._store = _DiskFeatureStore(cache_dir, max_ram_items=int(max_ram_features), verbose=self.verbose)
    
        self.n_workers_features = int(n_workers_features) if n_workers_features is not None else max(1, os.cpu_count() // 2)
        self.pair_batch_size = int(pair_batch_size)
        self.stream_csv = bool(stream_csv)
    
        # default metadata regex (10X_c1_A1_..._Site-5.tif)
        self._meta_re = re.compile(
            r'(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)\.(?:tif|tiff)$',
            re.IGNORECASE
        )
    
        # Axis/Z/time handling
        self.arr_axes = str(arr_axes).upper()
        self.mip = bool(mip)
        self.z_index = int(z_index)
        self.t_index = int(t_index)
        self.squeeze_singleton = bool(squeeze_singleton)
        if self.arr_axes != "AUTO":
            if "Y" not in self.arr_axes or "X" not in self.arr_axes:
                raise ValueError("arr_axes must include 'Y' and 'X' (or use 'AUTO').")

    # ------------------------- utilities (static) ------------------------
    @staticmethod
    def _ensure_dir(p: str) -> str:
        p = os.path.abspath(p)
        os.makedirs(p, exist_ok=True)
        return p

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
        if mx <= mn + 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    @staticmethod
    def _edge_zncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Zero-mean normalized cross-correlation of Sobel gradient energy between a and b.
        If `mask` is provided, the ZNCC is computed only over mask==True.
        """
        # ensure float32
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
    
        # gradient energy images
        ea = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)**2
        eb = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)**2
    
        if mask is not None:
            idx = mask.astype(bool)
            # require some overlap
            if idx.sum() < 25:
                return 0.0
            ea = ea[idx]
            eb = eb[idx]
    
        # ZNCC on gradient energy
        ea = ea - ea.mean()
        eb = eb - eb.mean()
        den = (ea.std() * eb.std()) + 1e-9
        return float((ea * eb).mean() / den)


    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        m, M = float(np.nanmin(img)), float(np.nanmax(img))
        if M <= m + 1e-12:
            return np.zeros_like(img, dtype=np.uint8)
        return np.clip(255.0 * (img - m) / (M - m), 0, 255).astype(np.uint8)

    @staticmethod
    def _affine_to_3x3(M2x3: np.ndarray) -> np.ndarray:
        A = np.eye(3, dtype=np.float32); A[:2, :3] = M2x3.astype(np.float32)
        return A

    # --------------------------- masks (Otsu/Cellpose) -------------------
    def _foreground_mask(self, img_u8: np.ndarray) -> np.ndarray:
        """Binary foreground mask; used at DS or full-res depending on caller."""
        if self.outline_source == "none":
            return np.zeros_like(img_u8, dtype=bool)

        if self.outline_source == "cellpose":
            try:
                from cellpose import models
            except Exception:
                raise RuntimeError("outline_source='cellpose' requires `cellpose` installed.")
            x = img_u8.astype(np.float32) / 255.0
            model = models.Cellpose(model_type="nuclei")
            masks, _, _, _ = model.eval(x, diameter=None, channels=[0,0],
                                        flow_threshold=0.4, cellprob_threshold=0.0)
            return (masks.astype(np.int32) > 0)

        # Otsu (default)
        I = img_u8
        if self.blur_sigma and self.blur_sigma > 0:
            ksz = max(1, int(2 * round(3 * self.blur_sigma) + 1))
            I = cv2.GaussianBlur(I, (ksz, ksz), self.blur_sigma)
        _, th = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
        mask = (I >= th)
        if self.dilate_ksize and self.dilate_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_ksize, self.dilate_ksize))
            mask = cv2.dilate(mask.astype(np.uint8), k) > 0
        return mask

    def _outline_mask(self, img_u8: np.ndarray) -> np.ndarray:
        """Edge pixels (bool) for DS QC overlays."""
        if self.outline_source == "none":
            return np.zeros_like(img_u8, dtype=bool)
    
        if self.outline_source == "cellpose":
            try:
                from cellpose import models
            except Exception:
                raise RuntimeError("outline_source='cellpose' requires `cellpose` installed.")
            x = img_u8.astype(np.float32) / 255.0
            model = models.Cellpose(model_type="nuclei")
            masks, _, _, _ = model.eval(x, diameter=None, channels=[0, 0],
                                        flow_threshold=0.4, cellprob_threshold=0.0)
            mask = (masks.astype(np.int32) > 0).astype(np.uint8)
        else:
            I = img_u8.copy()
            if self.blur_sigma and self.blur_sigma > 0:
                ksz = max(1, int(2 * round(3 * self.blur_sigma) + 1))
                I = cv2.GaussianBlur(I, (ksz, ksz), self.blur_sigma)
            _, th = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
            mask = (I >= th).astype(np.uint8)
    
        # NOTE: use dilate_ksize here (bugfix). line_thickness is for edge thickening later.
        if self.dilate_ksize and self.dilate_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_ksize, self.dilate_ksize))
            mask = cv2.dilate(mask, k)
    
        edges = cv2.Canny((mask * 255).astype(np.uint8), self.canny[0], self.canny[1])
    
        if self.line_thickness > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_thickness, self.line_thickness))
            edges = cv2.dilate(edges, k)
    
        return edges > 0

    # --------------------------- Axis helpers ----------------------------
    @staticmethod
    def _is_large_dim(n: int) -> bool:
        return n >= 128

    @staticmethod
    def _guess_axes_from_shape(shape: Tuple[int, ...]) -> str:
        nd = len(shape)
        if nd == 2:
            return "YX"
    
        if nd == 3:
            if spacrStitcher._is_large_dim(shape[-1]) and spacrStitcher._is_large_dim(shape[-2]):
                a = shape[0]
                return "CYX" if a <= 8 else "ZYX"
            if spacrStitcher._is_large_dim(shape[0]) and spacrStitcher._is_large_dim(shape[1]):
                a = shape[2]
                return "YXC" if a <= 8 else "YXZ"
            return "CYX"
    
        if nd == 4:
            if spacrStitcher._is_large_dim(shape[-1]) and spacrStitcher._is_large_dim(shape[-2]):
                a, b = shape[0], shape[1]
                if a <= 8 and b > 8:
                    return "CZYX"
                if b <= 8 and a > 8:
                    return "ZCYX"
                if a <= 8 and b <= 8:
                    return "CZYX"
                return "CZYX"
            return "TCYX"
    
        if nd == 5:
            if spacrStitcher._is_large_dim(shape[-1]) and spacrStitcher._is_large_dim(shape[-2]):
                b = shape[1]
                return "TCZYX" if b <= 8 else "TZCYX"
            return "TZCYX"
    
        return "CZYX" if nd >= 4 else "CYX"

    def _normalize_to_yx(self, arr: np.ndarray, ch: int, axes_hint: Optional[str] = None) -> np.ndarray:
        # Choose base axes
        if self.arr_axes and self.arr_axes.upper() != "AUTO":
            axes = "".join(a for a in self.arr_axes.upper() if a in "TCZYX")
        elif axes_hint:
            axes = "".join(a for a in axes_hint.upper() if a in "TCZYX")
        else:
            axes = self._guess_axes_from_shape(arr.shape)
    
        # Align axes length to array rank
        ax = list(axes)
        # If too many labels, drop T/C first
        while len(ax) > arr.ndim:
            for d in ("T", "C"):
                if d in ax and len(ax) > arr.ndim:
                    ax.remove(d)
        # If still too many, drop from the left (safest for unexpected leading dims)
        while len(ax) > arr.ndim:
            ax.pop(0)
        # If too few labels, pad (prefer adding missing T then C at the front)
        while len(ax) < arr.ndim:
            if "T" not in ax:
                ax.insert(0, "T")
            elif "C" not in ax:
                ax.insert(0, "C")
            else:
                ax.insert(0, "Z")
        axes = "".join(ax)
    
        # Build slicers
        slicers = []
        for a in axes:
            if a == "T": slicers.append(self.t_index)
            elif a == "C": slicers.append(ch)
            elif a == "Z": slicers.append(slice(None) if self.mip else self.z_index)
            elif a in ("Y", "X"): slicers.append(slice(None))
            else: slicers.append(0)
    
        sub = arr[tuple(slicers)]
    
        # Max-project if Z kept
        if self.mip and sub.ndim == 3:
            z_axis = 0 if (self._is_large_dim(sub.shape[-1]) and self._is_large_dim(sub.shape[-2])) else int(np.argmin(sub.shape))
            sub = sub.max(axis=z_axis)
    
        if self.squeeze_singleton:
            sub = np.squeeze(sub)
    
        if sub.ndim == 3:  # defensively drop a small stray axis
            small = [i for i, n in enumerate(sub.shape) if n <= 8]
            if small:
                sub = sub.take(indices=0, axis=small[0])
    
        if sub.ndim != 2:
            raise ValueError(f"Expected 2D YX after axis handling, got {sub.shape} (axes='{axes}')")
    
        return sub.astype(np.float32, copy=False)

    # --------------------------- IO & metadata ---------------------------
    def _read_plane(self, path: str, ch: int = 0) -> np.ndarray:
        """Load a single 2D YX plane from a possibly multi-axis TIFF."""
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            axes_hint = getattr(series, "axes", None)
            if axes_hint:
                # Keep only T/C/Z/Y/X symbols
                axes_hint = "".join(a for a in axes_hint.upper() if a in "TCZYX")
            arr = series.asarray()
    
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
    
        # If no axes hint and 3-D, pick CYX vs ZYX sensibly
        if axes_hint is None and arr.ndim == 3:
            fn = os.path.basename(path).lower()
            # If filename contains a channel token like _c1_/_c2_, treat first axis as C
            if re.search(r'(^|[_\-])c\d+([_\-]|$)', fn):
                axes_hint = "CYX"
            else:
                axes_hint = "ZYX" if self.mip else "CYX"
    
        return self._normalize_to_yx(arr, ch=ch, axes_hint=axes_hint)


    def set_meta_regex(self, pattern: Union[str, re.Pattern]):
        self._meta_re = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern

    def _parse_meta(self, path: str) -> Dict[str, Union[str, int, None]]:
        fn = os.path.basename(path)
        out = {"well": None, "site": None, "chan": None, "mag": None}
        m = self._meta_re.search(fn)
        if m:
            out["well"] = (m.group("well") or "").upper() if "well" in m.groupdict() else None
            out["site"] = int(m.group("site")) if "site" in m.groupdict() else None
            out["chan"] = int(m.group("chan")) if "chan" in m.groupdict() else None
            out["mag"]  = m.group("mag") if "mag" in m.groupdict() else None
            return out
        # lenient fallbacks
        mw = re.search(r"([A-H]\d{1,2})", fn, re.IGNORECASE)
        ms = re.search(r"Site[-_](\d+)", fn, re.IGNORECASE)
        mc = re.search(r"_c(\d+)_", fn, re.IGNORECASE)
        if mw: out["well"] = mw.group(1).upper()
        if ms: out["site"] = int(ms.group(1))
        if mc: out["chan"] = int(mc.group(1))
        return out

    # ----------------------- feature extraction/cache --------------------
    def _detect_and_describe(self, I8: np.ndarray):
        kp, desc = self._det.detectAndCompute(I8, None)
        if kp is None or desc is None or len(kp) < 4:
            pts = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 32), np.uint8) if self.detector == "ORB" else np.zeros((0, 128), np.float32)
            return pts, desc
        # Top-K by response
        if self.max_keypoints is not None and len(kp) > self.max_keypoints:
            idx = np.argsort([-k.response for k in kp])[:self.max_keypoints]
            kp = [kp[i] for i in idx]
            desc = desc[idx]
        pts = np.float32([k.pt for k in kp])
        return pts, desc

    def _compute_features_one(self, path: str, channel_index: int) -> Dict[str, np.ndarray]:
        I = self._read_plane(path, ch=channel_index)
        H, W = I.shape
        s = self.downsample if self.downsample > 0 else 1.0
        # ensure at least 1 px after DS (robust to extreme s)
        Hds = max(1, int(round(H * s)))
        Wds = max(1, int(round(W * s)))
        I_ds = cv2.resize(I, (Wds, Hds), interpolation=cv2.INTER_LINEAR)
        I8 = self._to_uint8(I_ds)
        pts, desc = self._detect_and_describe(I8)
        # post-cap (if requested)
        if self.max_keypoints is not None and pts.shape[0] > self.max_keypoints:
            idx = np.argsort(-np.linalg.norm(pts - pts.mean(0), axis=1))[:self.max_keypoints]
            pts = pts[idx]
            desc = desc[idx]
        return dict(ds8=I8, Hds=np.int32(Hds), Wds=np.int32(Wds),
                    pts=pts.astype(np.float32), desc=desc,
                    H=np.int32(H), W=np.int32(W))


    def prepare_features(self, paths: List[str], channel_index: int, num_workers: Optional[int] = None):
        """
        Precompute features. In 'disk' mode, every computed feature is flushed to disk
        and only an LRU-sized subset is kept in RAM. In 'ram' mode, behaves like before.
        """
        if num_workers is None:
            num_workers = self.n_workers_features

        if self.feature_cache_mode == "disk":
            # only compute for items missing on disk
            todo = []
            for p in paths:
                if self._store.get(p) is None:
                    todo.append(p)
        else:
            todo = [p for p in paths if p not in self._feat_cache]

        if not todo:
            if self.verbose:
                print("[features] nothing to compute (all cached or on disk)", flush=True)
            return

        if self.verbose:
            print(f"[features] computing features for {len(todo)} images at downsample {self.downsample}", flush=True)

        start = time.time()

        def _job(p):
            return p, self._compute_features_one(p, channel_index)

        total = len(todo)
        done = 0
        step = max(1, total // 20)

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
            for fut in as_completed([ex.submit(_job, p) for p in todo]):
                p, feat = fut.result()
                if self.feature_cache_mode == "disk":
                    self._store.put(p, feat)
                else:
                    with self._feat_lock:
                        self._feat_cache[p] = feat
                done += 1
                if self.verbose and (done % step == 0 or done == total):
                    dt = time.time() - start
                    print(f"[features] {done}/{total} ({done/total:.0%}) in {dt:.1f}s", flush=True)

    def _get_features(self, path: str, channel_index: int) -> Dict[str, np.ndarray]:
        """
        Return features for 'path' from RAM or disk; compute if missing.
        """
        if self.feature_cache_mode == "disk":
            feat = self._store.get(path)
            if feat is None:
                feat = self._compute_features_one(path, channel_index)
                self._store.put(path, feat)
            return feat
        else:
            with self._feat_lock:
                f = self._feat_cache.get(path)
            if f is None:
                f = self._compute_features_one(path, channel_index)
                with self._feat_lock:
                    self._feat_cache[path] = f
            return f

    # ---------------------------- matching/RANSAC ------------------------
    def _match(self, fA: Dict[str, np.ndarray], fB: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if fA["pts"].shape[0] < 4 or fB["pts"].shape[0] < 4:
            return np.zeros((0,2), np.float32), np.zeros((0,2), np.float32)
        if self.detector == "ORB":
            matches = self._bf.match(fA["desc"], fB["desc"])
            matches = list(matches); matches.sort(key=lambda m: m.distance)
            idxA = [m.queryIdx for m in matches]
            idxB = [m.trainIdx for m in matches]
        else:
            raw = self._flann.knnMatch(fA["desc"], fB["desc"], k=2)
            good = []
            for pair in raw:
                if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                    good.append(pair[0])
            good.sort(key=lambda m: m.distance)
            idxA = [m.queryIdx for m in good]
            idxB = [m.trainIdx for m in good]
        if not idxA:
            return np.zeros((0,2), np.float32), np.zeros((0,2), np.float32)
        return fA["pts"][idxA].astype(np.float32), fB["pts"][idxB].astype(np.float32)

    @staticmethod
    def _affine_from_pts(ptsA: np.ndarray, ptsB: np.ndarray, ransac_thresh_px: float):
        """
        Robustly estimate a partial affine with RANSAC (B -> A).
        Returns:
          M (2x3) | None,
          inlier_mask (bool array with shape (N,)) | None,
          inlier_ratio (float)
        """
        if ptsA.shape[0] < 4 or ptsB.shape[0] < 4:
            return None, None, 0.0
    
        M, inliers = cv2.estimateAffinePartial2D(
            ptsB.reshape(-1, 1, 2), ptsA.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh_px),
            maxIters=5000,
            confidence=0.999
        )
        if M is None:
            return None, None, 0.0
    
        if inliers is None:
            inlier_mask = None
            inlier_ratio = 0.0
        else:
            inlier_mask = inliers.ravel().astype(bool)
            inlier_ratio = float(inlier_mask.mean())
    
        return M.astype(np.float32), inlier_mask, inlier_ratio

    # ------------------------------ helpers ------------------------------
    @staticmethod
    def _closest_rotation(A: np.ndarray) -> np.ndarray:
        """Project 2x2 matrix A to the nearest proper rotation (det=+1)."""
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R.astype(np.float32)

    # ------------------------------ stitch one ---------------------------
    def stitch_pair(self,
                    pathA: str,
                    pathB: str,
                    channel_index: int = 0,
                    score_threshold: Optional[float] = None,
                    save_stitched: Optional[bool] = None,
                    # NEW: QC gating (safe defaults preserve current behavior)
                    force_no_qc: bool = False,
                    qc_only_if_score_ge: Optional[float] = None) -> Optional[Dict]:
        t0 = time.time()
    
        # ---- helpers for dtype preservation ----
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
        def _common_dtype(*dts: np.dtype) -> np.dtype:
            return np.result_type(*dts)
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # DS features (8-bit only for keypoints)
        fA = self._get_features(pathA, channel_index)
        fB = self._get_features(pathB, channel_index)
        A_ds8, B_ds8 = fA["ds8"], fB["ds8"]
        Hds, Wds = int(fA["Hds"]), int(fA["Wds"])
        s = self.downsample if self.downsample > 0 else 1.0
    
        # match & model @ DS
        ptsA, ptsB = self._match(fA, fB)
        if ptsA.shape[0] < 4:
            if self.verbose:
                print(f"[stitch_pair] {os.path.basename(pathA)} vs {os.path.basename(pathB)}: <4 matches → skip")
            return None
        M_ds, inlier_mask, inlier_ratio = self._affine_from_pts(ptsA, ptsB, self.ransac_thresh_px)
        if M_ds is None:
            if self.verbose:
                print(f"[stitch_pair] {os.path.basename(pathA)} vs {os.path.basename(pathB)}: RANSAC failed → skip")
            return None
    
        # inliers for constrained recompute
        if inlier_mask is not None and inlier_mask.any():
            pA = ptsA[inlier_mask]; pB = ptsB[inlier_mask]
        else:
            pA = ptsA; pB = ptsB
    
        A_lin = M_ds[:, :2].astype(np.float32)
    
        if not self.allow_scale and not self.allow_rotation:
            A_lin = np.eye(2, dtype=np.float32)
            t_mean = (pA - pB).mean(axis=0).astype(np.float32)
            M_ds = np.zeros((2, 3), dtype=np.float32); M_ds[:, :2] = A_lin; M_ds[:, 2] = t_mean
        elif (not self.allow_scale) and self.allow_rotation:
            A_rot = self._closest_rotation(A_lin)
            t_mean = (pA - (pB @ A_rot.T)).mean(axis=0).astype(np.float32)
            M_ds = np.zeros((2, 3), dtype=np.float32); M_ds[:, :2] = A_rot; M_ds[:, 2] = t_mean
    
        # DS masks & score
        mA_ds = self._foreground_mask(A_ds8)
        mB_ds = self._foreground_mask(B_ds8)
        B_ds_warp = cv2.warpAffine(B_ds8, M_ds, (Wds, Hds), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mB_ds_warp = cv2.warpAffine((mB_ds.astype(np.uint8) * 255), M_ds, (Wds, Hds),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        m_int_ds = (mA_ds & mB_ds_warp)
        edge_zncc_fg = self._edge_zncc(A_ds8.astype(np.float32), B_ds_warp.astype(np.float32), mask=m_int_ds)
        score = float(edge_zncc_fg * inlier_ratio)
    
        # lift DS → full-res
        M_full = M_ds.astype(np.float32).copy()
        if s != 0:
            M_full[0, 2] /= s
            M_full[1, 2] /= s
    
        a, b, tx = float(M_full[0, 0]), float(M_full[0, 1]), float(M_full[0, 2])
        c, d, ty = float(M_full[1, 0]), float(M_full[1, 1]), float(M_full[1, 2])
        scale = float(np.sqrt(max(1e-12, (a * d - b * c))))
        theta = float(np.degrees(np.arctan2(c, a)))
    
        # ---- QC overlay (DS) with gating ----
        qc_paths = {}
        do_qc = (self.save_qc and not bool(force_no_qc))
        if do_qc and (qc_only_if_score_ge is not None):
            try:
                do_qc = do_qc and (score >= float(qc_only_if_score_ge))
            except Exception:
                # if threshold is malformed, fall back to current do_qc
                pass
    
        if do_qc:
            edgesA = self._outline_mask(A_ds8)
            edgesB = self._outline_mask(B_ds8)
            edgesB_warp = cv2.warpAffine((edgesB.astype(np.uint8) * 255), M_ds, (Wds, Hds),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
            bg = self._norm01(A_ds8)
            qc_rgb = np.dstack([bg, bg, bg]).astype(np.float32)
            colA = np.array([0.0000, 0.4470, 0.6980], np.float32)  # blue
            colB = np.array([0.8350, 0.3650, 0.0000], np.float32)  # orange
            qc_rgb = (1 - self.outline_alpha) * qc_rgb + self.outline_alpha * np.where(edgesA[..., None], colA, qc_rgb)
            qc_rgb = (1 - self.outline_alpha) * qc_rgb + self.outline_alpha * np.where(edgesB_warp[..., None], colB, qc_rgb)
    
            stem = f"{os.path.splitext(os.path.basename(pathA))[0]}__{os.path.splitext(os.path.basename(pathB))[0]}"
            p_outline = os.path.join(self.outdir, f"{stem}__qc_outlines.png")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(np.clip(qc_rgb, 0, 1)); ax.set_axis_off()
            ax.set_title(f"score={score:.3f} (edge_zncc_fg={edge_zncc_fg:.3f} · inliers={inlier_ratio:.3f})")
            fig.savefig(p_outline, dpi=200, bbox_inches="tight"); plt.close(fig)
            qc_paths["qc_outline_png"] = p_outline
    
        # decide whether to stitch now
        if save_stitched is None:
            save_stitched = self.save_stitched_default
        save_stitched = bool(save_stitched)
    
        stitched_paths = {}
        Hc = Wc = 0
        if save_stitched:
            thr = score_threshold if score_threshold is not None else self.score_threshold
            if (thr is not None) and (score >= float(thr)):
                # Read full-res (float32 workspace), but keep input dtypes for final cast
                A_full = self._read_plane(pathA, ch=channel_index)
                B_full = self._read_plane(pathB, ch=channel_index)
                H, W = A_full.shape
    
                dtypeA = _series_dtype(pathA)
                dtypeB = _series_dtype(pathB)
                out_dtype = _common_dtype(dtypeA, dtypeB)
    
                # Canvas geometry
                corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
                B_c = cv2.transform(corners, M_full).reshape(-1, 2)
                all_x = np.concatenate([corners.reshape(-1, 2)[:, 0], B_c[:, 0]])
                all_y = np.concatenate([corners.reshape(-1, 2)[:, 1], B_c[:, 1]])
                x_min, y_min = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
                x_max, y_max = float(np.ceil(all_x.max())), float(np.ceil(all_y.max()))
                Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
                off_x, off_y = int(-x_min), int(-y_min)
                T = np.array([[1, 0, off_x], [0, 1, off_y]], np.float32)
    
                # Blend in native intensity space (no normalization)
                canvas = np.zeros((Hc, Wc), np.float32)
                wgt = np.zeros_like(canvas)
                # A contribution
                canvas[off_y:off_y + H, off_x:off_x + W] += A_full
                wgt[off_y:off_y + H, off_x:off_x + W] += 1.0
                # B contribution (warp image + warp 1-mask to get coverage)
                M_canvas = (T @ self._affine_to_3x3(M_full))[:2, :]
                B_can = cv2.warpAffine(B_full, M_canvas, (Wc, Hc),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                maskB = cv2.warpAffine(np.ones_like(B_full, dtype=np.float32), M_canvas, (Wc, Hc),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                canvas += B_can
                wgt += maskB
    
                stitched = np.divide(canvas, np.maximum(wgt, 1e-6))
    
                stem = f"{os.path.splitext(os.path.basename(pathA))[0]}__{os.path.splitext(os.path.basename(pathB))[0]}"
                p_tif = os.path.join(self.outdir, f"{stem}__stitched_full.tif")
                p_png = os.path.join(self.outdir, f"{stem}__stitched_full.png")
                tifffile.imwrite(p_tif, _cast(stitched, out_dtype))
                plt.imsave(p_png, (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-12), cmap="gray")
                stitched_paths["stitched_full_tif"] = p_tif
                stitched_paths["stitched_full_png"] = p_png
            else:
                if self.verbose and save_stitched:
                    msg_thr = self.score_threshold if score_threshold is None else score_threshold
                    print(f"[stitch_pair] score {score:.3f} < threshold {msg_thr} → no stitch")
    
        metaA = self._parse_meta(pathA); metaB = self._parse_meta(pathB)
    
        # optional full-res metrics (unchanged)
        edge_zncc_full = ""
        fg_corr = ""
        fg_iou = ""
        fg_dice = ""
        fg_xor_frac = ""
        fg_xor_entropy = ""
    
        if self.all_scores:
            A_full = self._read_plane(pathA, ch=channel_index)
            B_full = self._read_plane(pathB, ch=channel_index)
            H, W = A_full.shape
            B_in_A = cv2.warpAffine(B_full, M_full, (W, H),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            edge_zncc_full = float(self._edge_zncc(A_full, B_in_A))
    
            A_u8 = self._to_uint8(A_full)
            B_u8 = self._to_uint8(B_full)
            mA = self._foreground_mask(A_u8)
            mB = self._foreground_mask(B_u8)
            mB_warp = cv2.warpAffine((mB.astype(np.uint8) * 255), M_full, (W, H),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
            m_int = (mA & mB_warp)
            m_union = (mA | mB_warp)
            n_int = int(m_int.sum()); n_union = int(m_union.sum())
    
            if n_union == 0:
                fg_iou_v = 0.0; fg_dice_v = 0.0; fg_xor_frac_v = 1.0; fg_xor_entropy_v = 0.0
            else:
                inter = float(n_int); union = float(n_union)
                fg_iou_v = inter / union
                fg_dice_v = (2.0 * inter) / (float(mA.sum()) + float(mB_warp.sum()) + 1e-9)
                xor_frac = (m_union ^ m_int).sum() / union
                p = float(xor_frac)
                fg_xor_entropy_v = 0.0 if (p <= 0.0 or p >= 1.0) else float(-(p*math.log(p,2) + (1-p)*math.log(1-p,2)))
                fg_xor_frac_v = float(xor_frac)
    
            if n_int >= 25:
                a_vals = A_full[m_int].astype(np.float64)
                b_vals = B_in_A[m_int].astype(np.float64)
                a_vals -= a_vals.mean(); b_vals -= b_vals.mean()
                denom = (a_vals.std() * b_vals.std()) + 1e-12
                fg_corr_v = float((a_vals*b_vals).mean() / denom) if denom > 0 else 0.0
            else:
                fg_corr_v = 0.0
    
            fg_corr = float(fg_corr_v)
            fg_iou = float(fg_iou_v)
            fg_dice = float(fg_dice_v)
            fg_xor_frac = float(fg_xor_frac_v)
            fg_xor_entropy = float(fg_xor_entropy_v)
    
        # choose canvas dims for CSV row
        if (Hc > 0) and (Wc > 0):
            canvas_H_out, canvas_W_out = int(Hc), int(Wc)
        else:
            canvas_H_out, canvas_W_out = int(Hds), int(Wds)
    
        return dict(
            pathA=pathA, pathB=pathB, channel_index=channel_index,
            dy_px_full=ty, dx_px_full=tx, theta_deg=theta, scale=scale,
            inlier_ratio=inlier_ratio,
            edge_zncc_fg=edge_zncc_fg,
            edge_zncc_full=edge_zncc_full,
            fg_corr=fg_corr, fg_iou=fg_iou, fg_dice=fg_dice,
            fg_xor_frac=fg_xor_frac, fg_xor_entropy=fg_xor_entropy,
            score=score, weight=score,
            canvas_H=canvas_H_out, canvas_W=canvas_W_out,
            qc_outline_png=qc_paths.get("qc_outline_png",""),
            stitched_full_tif=stitched_paths.get("stitched_full_tif",""),
            stitched_full_png=stitched_paths.get("stitched_full_png",""),
            seconds=time.time() - t0,
            well=self._parse_meta(pathA).get("well"),
            siteA=self._parse_meta(pathA).get("site"),
            siteB=self._parse_meta(pathB).get("site"),
        )

    
    @staticmethod
    def _get_channel_count_tif(path: str) -> int:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            axes = getattr(series, "axes", None)
            shape = series.shape
        if axes:
            axes = "".join(a for a in axes.upper() if a in "TCZYX")
            return int(shape[axes.index("C")]) if "C" in axes else 1
        gh = spacrStitcher._guess_axes_from_shape(shape)
        return int(shape[gh.index("C")]) if "C" in gh else 1
    
    def _read_all_channels_cyx(self, path: str) -> np.ndarray:
        nC = self._get_channel_count_tif(path)
        planes = []
        for c in range(nC):
            planes.append(self._read_plane(path, ch=c).astype(np.float32, copy=False))
        return np.stack(planes, axis=0)  # (C,H,W)

    # ------------------------ auto-knee + plotting -----------------------
    @staticmethod
    def _auto_elbow_threshold(scores: List[float]) -> float:
        """
        Simple 'knee' on sorted scores (ascending).
        Finds the index with the maximum perpendicular distance to the line
        connecting first and last points.
        """
        if not scores:
            return 0.0
        s = np.array(sorted(scores), dtype=np.float64)
        n = s.size
        if n < 3:
            return float(s[-1] if n == 1 else s[int(n/2)])

        x = np.arange(n, dtype=np.float64)
        p1 = np.array([0.0, s[0]])
        p2 = np.array([float(n-1), s[-1]])
        v = p2 - p1
        v_norm = v / (np.linalg.norm(v) + 1e-12)
        pts = np.stack([x, s], axis=1)
        w = pts - p1[None, :]
        proj = (w @ v_norm)[:, None] * v_norm[None, :]
        perp = w - proj
        d = np.linalg.norm(perp, axis=1)
        k = int(np.argmax(d))
        return float(s[k])

    def _plot_sorted_scores(self, scores: List[float], thr: float, out_png: str):
        s = np.array(sorted(scores), dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(np.arange(len(s)), s, lw=1)
        ax.axhline(thr, linestyle='--')
        ax.set_title(f"Sorted pairwise scores (n={len(s)}), threshold={thr:.3f}")
        ax.set_xlabel("pair index (sorted)")
        ax.set_ylabel("score = edge_zncc_fg(DS) × inlier_ratio")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------ pairing ------------------------------
    @staticmethod
    def _list_tifs(folder: str, recursive: bool, exts: Tuple[str,...]) -> List[str]:
        exts = tuple(e.lower() for e in exts)
        out = []
        if recursive:
            for root, _, files in os.walk(folder):
                for fn in files:
                    if fn.lower().endswith(exts):
                        out.append(os.path.join(root, fn))
        else:
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(exts):
                    out.append(os.path.join(folder, fn))
        return out

    def _group_by_well(self, paths: List[str]) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for p in paths:
            w = self._parse_meta(p).get("well") or "UNK"
            buckets.setdefault(w, []).append(p)
        for w, lst in buckets.items():
            lst.sort(key=lambda x: (self._parse_meta(x).get("site") or 10**9, x))
        return buckets

    def _pairs_by_site_window(self, files: List[str], max_site_gap: int) -> List[Tuple[str,str]]:
        n = len(files)
        site = [self._parse_meta(p).get("site") for p in files]
        idx_by_site = {}
        for i, s in enumerate(site):
            if s is not None:
                idx_by_site[s] = i
        cand = set()
        for i, p in enumerate(files):
            si = site[i]
            if si is None:
                for j in range(i+1, min(i+1+max_site_gap, n)):
                    cand.add((files[i], files[j]))
                continue
            for k in range(1, max_site_gap+1):
                for s_adj in (si + k, si - k):
                    j = idx_by_site.get(s_adj)
                    if j is not None and j > i:
                        cand.add((files[i], files[j]))
        return sorted(list(cand))

    # ------------------------------ driver -------------------------------
    def run_folder(self,
                   folder: str,
                   csv_path: str,
                   *,
                   channel_index: int = 0,
                   exts: Tuple[str, ...] = (".tif", ".tiff"),
                   recursive: bool = False,
                   same_well_only: bool = True,
                   max_site_gap: int = 3,
                   n_workers: int = 8,
                   stitch: bool = True,
                   score_threshold: Optional[float] = None,
                   meta_regex: Optional[Union[str, re.Pattern]] = None,
                   # mosaic controls
                   mosaic: bool = False,
                   mosaic_out: Optional[str] = None,
                   mosaic_min_score: Optional[float] = None,
                   mosaic_csv_out: Optional[str] = None,
                   # NEW: multi-channel mosaic controls
                   mosaic_all_channels: bool = False,
                   mosaic_channel_count: Optional[int] = None,
                   mosaic_channel_index_order: Optional[List[int]] = None,
                   # NEW: QC gating controls (do not break existing calls)
                   qc_pairs_threshold: int = 1000,
                   qc_only_above_threshold_when_many: bool = True) -> str:
        """
        When number of candidate pairs exceeds `qc_pairs_threshold`, QC plotting is suppressed
        during the first (threaded) scoring pass and, once a threshold is known, QC is produced
        only for pairs with score >= threshold (in the second pass).
        """
        if meta_regex is not None:
            self.set_meta_regex(meta_regex)
    
        paths = self._list_tifs(folder, recursive, exts)
        if self.verbose:
            print(f"[run_folder] scanning {folder}; found {len(paths)} files (recursive={recursive})", flush=True)
    
        header = [
            "pathA","pathB","channel_index",
            "score","inlier_ratio","edge_zncc_fg","edge_zncc_full","fg_corr","fg_iou","fg_dice","fg_xor_frac","fg_xor_entropy",
            "dy_px_full","dx_px_full","theta_deg","scale",
            "canvas_H","canvas_W",
            "qc_outline_png","stitched_full_tif","stitched_full_png",
            "seconds","well","siteA","siteB"
        ]
    
        # Handle no files
        if not paths:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
            if self.verbose:
                print("[run_folder] No files found; wrote empty CSV.", flush=True)
            return csv_path
    
        if same_well_only:
            groups = self._group_by_well(paths)
            pairs = []
            for _, files in groups.items():
                pairs.extend(self._pairs_by_site_window(files, max_site_gap=max_site_gap))
            if self.verbose:
                print(f"[run_folder] same_well_only=True; wells={len(groups)}; candidate pairs={len(pairs)}; site_window={max_site_gap}", flush=True)
        else:
            files = sorted(paths)
            pairs = [(files[i], files[j]) for i in range(len(files)) for j in range(i + 1, len(files))]
            if self.verbose:
                print(f"[run_folder] same_well_only=False; candidate pairs={len(pairs)}", flush=True)
    
        # Precompute DS features (disk-backed)
        self.prepare_features(list(set([p for pair in pairs for p in pair])), channel_index, num_workers=self.n_workers_features)
        if self.verbose:
            print("[run_folder] feature cache ready; scoring pairs…", flush=True)
    
        scores: List[float] = []
    
        outdir_csv = os.path.dirname(os.path.abspath(csv_path))
        if outdir_csv and not os.path.isdir(outdir_csv):
            os.makedirs(outdir_csv, exist_ok=True)
        f_csv = open(csv_path, "w", newline="")
        w_csv = csv.DictWriter(f_csv, fieldnames=header)
        w_csv.writeheader()
    
        total_pairs = len(pairs)
        too_many_pairs = (total_pairs > int(qc_pairs_threshold))
    
        def _job(pair):
            A, B = pair
            try:
                thr_now = score_threshold if score_threshold is not None else self.score_threshold
                do_stitch_now = (stitch is True) and (thr_now is not None)
    
                # QC gating for the first (threaded) pass
                # - If too many pairs and threshold unknown: suppress QC now
                # - If too many pairs and threshold known: only QC when score >= threshold
                if too_many_pairs:
                    if thr_now is None:
                        force_no_qc = True
                        qc_only_if_score_ge = None
                    else:
                        force_no_qc = False
                        qc_only_if_score_ge = float(thr_now) if qc_only_above_threshold_when_many else None
                else:
                    force_no_qc = False
                    qc_only_if_score_ge = None
    
                return self.stitch_pair(
                    A, B,
                    channel_index=channel_index,
                    score_threshold=thr_now,
                    save_stitched=do_stitch_now,
                    # NEW controls (default keep behavior)
                    force_no_qc=force_no_qc,
                    qc_only_if_score_ge=qc_only_if_score_ge
                )
            except Exception as e:
                if self.verbose:
                    print(f"[run_folder] Pair {os.path.basename(A)} vs {os.path.basename(B)} failed: {e}", flush=True)
                return None
    
        done = 0
        step = max(1, total_pairs // 20)
        start_score = time.time()
    
        batch_size = max(1024, self.pair_batch_size)
        for start_idx in range(0, total_pairs, batch_size):
            batch = pairs[start_idx:start_idx + batch_size]
            with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
                futs = [ex.submit(_job, p) for p in batch]
                for fut in as_completed(futs):
                    row = fut.result()
                    if row is not None:
                        w_csv.writerow({k: row.get(k, "") for k in header})
                        if row.get("score", "") != "":
                            try:
                                scores.append(float(row["score"]))
                            except Exception:
                                pass
                    done += 1
                    if self.verbose and (done % step == 0 or done == total_pairs):
                        dt = time.time() - start_score
                        print(f"[run_folder] scored {done}/{total_pairs} pairs ({done/total_pairs:.0%}) in {dt:.1f}s", flush=True)
    
        f_csv.flush()
        f_csv.close()
    
        # threshold
        thr = score_threshold if score_threshold is not None else self.score_threshold
        if thr is None:
            if self.verbose:
                print(f"[run_folder] computing auto threshold from {len(scores)} scores…", flush=True)
            thr = self._auto_elbow_threshold(scores)
            plot_png = os.path.join(self.outdir, "score_sorted_line.png")
            self._plot_sorted_scores(scores, thr, plot_png)
            if self.verbose:
                print(f"[run_folder] Auto threshold = {thr:.4f}; plot → {plot_png}", flush=True)
    
        # optional second pass (stitch winners and, if many pairs, generate QC only for winners)
        if stitch and (score_threshold is None and self.score_threshold is None):
            winners = []
            with open(csv_path, "r", newline="") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        sc = float(r["score"])
                    except Exception:
                        continue
                    if sc >= float(thr):
                        winners.append((r["pathA"], r["pathB"]))
            if self.verbose:
                print(f"[run_folder] stitching winners second pass: {len(winners)} (score ≥ {thr:.3f})", flush=True)
    
            for i, (A, B) in enumerate(winners, 1):
                _ = self.stitch_pair(
                    A, B,
                    channel_index=channel_index,
                    score_threshold=thr,
                    save_stitched=True,
                    # If too many pairs, now allow QC but only for score≥thr (these are winners anyway)
                    force_no_qc=False,
                    qc_only_if_score_ge=thr if (too_many_pairs and qc_only_above_threshold_when_many) else None
                )
                if self.verbose and (i % max(1, len(winners) // 10) == 0 or i == len(winners)):
                    print(f"[run_folder] stitched {i}/{len(winners)}", flush=True)
    
        if self.verbose:
            print(f"[run_folder] Done. CSV → {csv_path}", flush=True)
    
        # ---- mosaic output(s) ----
        if mosaic:
            if mosaic_out is None:
                if self.verbose:
                    print(f"[run_folder] mosaic_out: {mosaic_out}, skipping masaic tif, only generating csv", flush=True)
            #    mosaic_out = os.path.join(self.outdir, "mosaic_full.tif")
            min_sc = mosaic_min_score if mosaic_min_score is not None else float(thr)
    
            if mosaic_all_channels:
                if self.verbose:
                    print(f"[run_folder] rendering multi-channel mosaic (min_score={min_sc:.4f}) → {mosaic_out}", flush=True)
                self.mosaic_all_channels_from_csv(
                    csv_path, mosaic_out,
                    min_score=min_sc,
                    channel_count=mosaic_channel_count,
                    channel_index_order=mosaic_channel_index_order,
                    out_csv=mosaic_csv_out
                )
            else:
                mosaic_png = os.path.splitext(mosaic_out)[0] + ".png"
                if self.verbose:
                    print(f"[run_folder] rendering single-channel mosaic (min_score={min_sc:.4f}) → {mosaic_out}", flush=True)
                self.render_mosaic_from_csv(
                    csv_path, mosaic_out, mosaic_png,
                    channel_index=channel_index, min_score=min_sc,
                    out_csv=mosaic_csv_out
                )
    
        return csv_path

    def build_multichannel_mosaic_from_manifest(
        self,
        manifest_csv: str,
        out_tif: str,
        out_png: Optional[str] = None,
        channel_indices: Optional[List[int]] = None,
        blend: str = "max",           # "max" or "overwrite"
        tmp_dir: Optional[str] = None,
        preview_downsample: int = 8
    ):
        """
        Build a CYX BigTIFF mosaic from a 'mosaic.csv' manifest written by
        spacrStitcher.render_mosaic_from_csv(...).
    
        Expected CSV columns (per tile row):
          - path, H, W,
            M00, M01, M02,
            M10, M11, M12,
            canvas_x, canvas_y, best_pair_score
    
        Notes
        -----
        * Uses the 2x3 affine in the manifest (already includes global offset).
        * If `channel_indices` is None, infers the channel count from the first valid tile.
        * `blend="max"` takes per-pixel max across overlapping tiles;
          `blend="overwrite"` writes the latest tile over earlier ones where coverage>0.
        * If `tmp_dir` is provided (or available from self.feature_cache_dir), the output
          workspace uses a disk memmap to reduce RAM.
        """
        # --- Default tmp_dir: reuse feature cache root if available ---
        if tmp_dir is None:
            base_cache = getattr(self, "feature_cache_dir", None)
            root = base_cache if base_cache else (os.path.dirname(out_tif) or ".")
            tmp_dir = os.path.join(root, "mosaic_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
    
        # --- Read manifest rows and basic integrity checks ---
        rows = []
        with open(manifest_csv, newline="") as f:
            rdr = csv.DictReader(f)
            required = ["path","H","W","M00","M01","M02","M10","M11","M12","canvas_x","canvas_y"]
            missing = [c for c in required if c not in (rdr.fieldnames or [])]
            if missing:
                raise RuntimeError(f"{manifest_csv} missing columns required by mosaic builder: {missing}")
            for r in rdr:
                p = (r["path"] or "").strip()
                if not p or not os.path.exists(p):
                    continue
                try:
                    H = int(float(r["H"])); W = int(float(r["W"]))
                    M = np.array([[float(r["M00"]), float(r["M01"]), float(r["M02"])],
                                  [float(r["M10"]), float(r["M11"]), float(r["M12"])]],
                                 dtype=np.float32)
                except Exception:
                    continue
                rows.append({"path": p, "H": H, "W": W, "M": M})
        if not rows:
            raise RuntimeError("build_multichannel_mosaic_from_manifest: no usable rows in manifest.")
    
        # --- Helpers to read channels from TIFFs (local, minimal axis handling) ---
        def _get_channel_count_tif_local(path: str) -> int:
            with tifffile.TiffFile(path) as tf:
                series = tf.series[0]
                axes = getattr(series, "axes", None)
                shape = series.shape
            if axes:
                axes = "".join(a for a in axes.upper() if a in "TCZYX")
                return int(shape[axes.index("C")]) if "C" in axes else 1
            if len(shape) == 3:
                return shape[0] if shape[0] <= 8 else 1
            return 1
    
        def _read_plane_local(path: str, ch: int = 0) -> np.ndarray:
            with tifffile.TiffFile(path) as tf:
                series = tf.series[0]
                axes = getattr(series, "axes", None)
                arr = series.asarray()
            if arr.ndim == 2:
                return arr.astype(np.float32, copy=False)
            if axes:
                axes = "".join(a for a in axes.upper() if a in "TCZYX")
                if "C" in axes:
                    cidx = axes.index("C")
                    slicers = [slice(None)] * arr.ndim
                    slicers[cidx] = ch
                    arr = arr[tuple(slicers)]
                    arr = np.squeeze(arr)
                    if arr.ndim == 3 and "Z" in axes:
                        zidx = axes.index("Z")
                        arr = arr.max(axis=zidx if zidx < arr.ndim else 0)
            else:
                if arr.ndim == 3 and arr.shape[0] <= 8:
                    arr = arr[ch]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D plane from {os.path.basename(path)}, got shape {arr.shape}")
            return arr.astype(np.float32, copy=False)
    
        # --- Decide which channels to mosaic ---
        if channel_indices is None:
            nC = _get_channel_count_tif_local(rows[0]["path"])
            ch_list = list(range(nC))
        else:
            ch_list = [int(c) for c in channel_indices]
    
        # --- Compute canvas bounds from affines ---
        xs, ys = [], []
        for r in rows:
            H, W = r["H"], r["W"]
            corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=np.float32).reshape(-1,1,2)
            C = cv2.transform(corners, r["M"]).reshape(-1,2)
            xs.append(C[:,0]); ys.append(C[:,1])
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        x_min, y_min = float(np.floor(xs.min())), float(np.floor(ys.min()))
        x_max, y_max = float(np.ceil(xs.max())),  float(np.ceil(ys.max()))
        Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
        off = np.array([[1,0,-x_min],[0,1,-y_min]], dtype=np.float32)  # canvas origin at (0,0)
    
        # --- Allocate output stack (float32 workspace); pick output dtype later ---
        # If tmp_dir is set, use an on-disk memmap to reduce RAM.
        mmap_path = os.path.join(tmp_dir, f"_mosaic_{os.path.splitext(os.path.basename(out_tif))[0]}.mmap")
        try:
            out_stack = np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=(len(ch_list), Hc, Wc))
            use_memmap = True
        except Exception:
            out_stack = np.zeros((len(ch_list), Hc, Wc), np.float32)
            use_memmap = False
    
        if blend == "max":
            out_stack[:] = -np.inf  # sentinel for max blending
    
        # track dtypes across input tiles (to pick safe output dtype)
        in_dtypes: List[np.dtype] = []
    
        # --- Composite ---
        for r in rows:
            p, H, W, M = r["path"], r["H"], r["W"], r["M"]
            M_can = (off @ np.vstack([M, [0,0,1]]) )[:2,:]
            cov = cv2.warpAffine(np.ones((H, W), np.float32), M_can, (Wc, Hc),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            with tifffile.TiffFile(p) as tf:
                in_dtypes.append(np.dtype(tf.series[0].dtype))
            for ci, ch in enumerate(ch_list):
                I = _read_plane_local(p, ch=ch)
                warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                m = cov > 0
                if blend == "max":
                    out_slice = out_stack[ci]
                    out_slice[m] = np.maximum(out_slice[m], warped[m])
                elif blend == "overwrite":
                    out_stack[ci][m] = warped[m]
                else:
                    raise ValueError("blend must be 'max' or 'overwrite'")
    
        if blend == "max":
            out_stack[np.isneginf(out_stack)] = 0.0
    
        # --- Save BigTIFF with axes metadata; choose common dtype over inputs ---
        out_dtype = np.result_type(*in_dtypes) if in_dtypes else np.float32
        tifffile.imwrite(out_tif, np.asarray(out_stack, dtype=out_dtype, order="C"), metadata={"axes": "CYX"})
    
        # --- Optional preview (channel 0 min-max normalized, downsampled) ---
        if out_png:
            c0 = np.asarray(out_stack[0])
            if preview_downsample and preview_downsample > 1:
                pw = max(1, Wc // preview_downsample)
                ph = max(1, Hc // preview_downsample)
                c0_prev = cv2.resize(c0, (pw, ph), interpolation=cv2.INTER_AREA)
            else:
                c0_prev = c0
            mn, mx = float(c0_prev.min()), float(c0_prev.max())
            prev = np.zeros_like(c0_prev, dtype=np.float32) if mx <= mn else (c0_prev - mn) / (mx - mn + 1e-12)
            plt.imsave(out_png, prev, cmap="gray")
    
        # ensure memmap data hits disk
        if use_memmap and hasattr(out_stack, "flush"):
            out_stack.flush()
    
        return out_tif

    # --------------------------- mosaic helpers --------------------------
    @staticmethod
    def _invert_affine(M: np.ndarray) -> np.ndarray:
        """
        Invert a 2x3 affine matrix.
        """
        A = M[:,:2]
        t = M[:,2:]
        Ai = np.linalg.inv(A + 1e-12*np.eye(2, dtype=np.float32))
        ti = -Ai @ t
        Mi = np.zeros((2,3), dtype=np.float32)
        Mi[:,:2] = Ai.astype(np.float32)
        Mi[:,2:] = ti.astype(np.float32)
        return Mi

    @staticmethod
    def _affine_from_row(row: Dict[str, Union[str,float,int]]) -> np.ndarray:
        """
        Reconstruct 2x3 affine from CSV fields for B→A:
          dx_px_full (tx), dy_px_full (ty), theta_deg, scale
        """
        tx = float(row["dx_px_full"])
        ty = float(row["dy_px_full"])
        theta = float(row["theta_deg"])
        scale = float(row["scale"])
        rad = np.deg2rad(theta)
        c, s = np.cos(rad), np.sin(rad)
        A = np.array([[c, -s],
                      [s,  c]], dtype=np.float32) * float(scale)
        M = np.zeros((2,3), dtype=np.float32)
        M[:,:2] = A
        M[:,2]  = [tx, ty]
        return M

    @staticmethod
    def _direction_bin(tx: float, ty: float, angle_tol_deg: float = 30.0) -> Optional[str]:
        """
        Classify a translation vector (tx, ty) into one of four bins:
        'R' (0°), 'U' (90°), 'L' (±180°), 'D' (-90°).
        Return None if the vector is too diagonal (outside tolerance).
        """
        ang = np.degrees(np.arctan2(ty, tx))  # [-180,180]
        candidates = {'R': 0.0, 'U': 90.0, 'L': 180.0, 'D': -90.0}
        best_dir, best_err = None, 1e9
        for k, a0 in candidates.items():
            err = abs(((ang - a0 + 180.0) % 360.0) - 180.0)
            if err < best_err:
                best_err, best_dir = err, k
        return best_dir if best_err <= float(angle_tol_deg) else None

    def _estimate_grid_steps(self,
                             rows: List[Dict],
                             min_score: float,
                             angle_tol_deg: float = 35.0) -> Tuple[float, float]:
        """
        Robustly estimate step size along X (horizontal neighbors) and Y (vertical neighbors)
        from high-scoring pairs.
        """
        xs, ys = [], []
        for r in rows:
            sc = float(r["score"]) if r["score"] != "" else -np.inf
            if not np.isfinite(sc) or sc < float(min_score):
                continue
            tx = float(r["dx_px_full"]); ty = float(r["dy_px_full"])
            db = self._direction_bin(tx, ty, angle_tol_deg=angle_tol_deg)
            if db in ("R", "L"):
                xs.append(abs(tx))
            elif db in ("U", "D"):
                ys.append(abs(ty))
        step_x = float(np.median(xs)) if len(xs) else 0.0
        step_y = float(np.median(ys)) if len(ys) else 0.0
        if self.verbose:
            print(f"[mosaic] estimated steps: step_x≈{step_x:.1f}, step_y≈{step_y:.1f}", flush=True)
        return step_x, step_y

    # ---------------------- mosaic: transforms & render -------------------
    def _compute_mosaic_transforms(self,
                                   rows: List[Dict],
                                   min_score: float,
                                   *,
                                   angle_tol_deg: float = 30.0,
                                   step_tol_frac: float = 0.25,
                                   rot_tol_deg: float = 5.0,
                                   scale_tol: float = 0.03,
                                   cap_one_per_dir: bool = True
                                   ) -> Tuple[Dict[str, np.ndarray], List[Tuple[str,str,float]]]:
        """
        Build transforms using a topology-aware pruning:
          1) Keep only edges with score ≥ min_score.
          2) Estimate grid steps (X,Y); require |dx|≈step_x for R/L and |dy|≈step_y for U/D (within step_tol_frac).
          3) Respect allow_rotation/allow_scale by enforcing small |theta| and |scale-1| if disallowed.
          4) For each tile, keep at most one best edge per direction bin (R/L/U/D) if cap_one_per_dir.
          5) Run a Kruskal maximum-spanning tree on the pruned edges to ensure connectivity.

        Returns:
          - T2: dict path -> 2x3 affine mapping that path's pixels into root coordinates
          - used_edges: list of (src, dst, score) in the MST
        """
        # Nodes present
        nodes = set()
        for r in rows:
            nodes.add(r["pathA"]); nodes.add(r["pathB"])
        nodes = sorted(nodes)
        if not nodes:
            return {}, []

        # Step estimates from high-score pairs
        step_x, step_y = self._estimate_grid_steps(rows, min_score, angle_tol_deg=max(30.0, angle_tol_deg))

        # Helper to check geometry/tolerances for one directed edge (src->dst)
        def edge_ok(tx, ty, theta, scale, dbin):
            if dbin is None:
                return False
            # rotation/scale limits (if disallowed)
            if not self.allow_rotation and abs(theta) > float(rot_tol_deg):
                return False
            if not self.allow_scale and abs(scale - 1.0) > float(scale_tol):
                return False
            # step gating
            if dbin in ("R", "L"):
                if step_x <= 0:
                    return False
                if abs(abs(tx) - step_x) > step_tol_frac * step_x:
                    return False
            else:  # U/D
                if step_y <= 0:
                    return False
                if abs(abs(ty) - step_y) > step_tol_frac * step_y:
                    return False
            return True

        # Gather per-direction best candidate edges
        best_per_node_dir: Dict[Tuple[str,str], Tuple[float, str, str, np.ndarray]] = {}
        all_cand: List[Tuple[str, str, float, np.ndarray, str]] = []

        for r in rows:
            sc = float(r["score"]) if r["score"] != "" else -np.inf
            if not np.isfinite(sc) or sc < float(min_score):
                continue
            A, B = r["pathA"], r["pathB"]

            # B->A from CSV row; also add A->B via inverse
            for (src, dst, M_src_to_dst) in (
                (B, A, self._affine_from_row(r)),  # B->A
                (A, B, None)                       # A->B (inverse later)
            ):
                if M_src_to_dst is None:
                    M_src_to_dst = self._invert_affine(self._affine_from_row(r))

                tx = float(M_src_to_dst[0,2]); ty = float(M_src_to_dst[1,2])

                # Recover theta, scale of this directed transform (approx)
                a,b = float(M_src_to_dst[0,0]), float(M_src_to_dst[0,1])
                c,d = float(M_src_to_dst[1,0]), float(M_src_to_dst[1,1])
                theta = np.degrees(np.arctan2(c, a))
                scale = float(np.sqrt(max(1e-12, (a*d - b*c))))

                dbin = self._direction_bin(tx, ty, angle_tol_deg=angle_tol_deg)
                if not edge_ok(tx, ty, theta, scale, dbin):
                    continue

                key = (src, dbin)
                prev = best_per_node_dir.get(key)
                if (prev is None) or (sc > prev[0]):
                    best_per_node_dir[key] = (sc, src, dst, M_src_to_dst)

        cand: List[Tuple[str,str,float,np.ndarray]] = []
        for (_src, _dir), (sc, src, dst, M) in best_per_node_dir.items():
            cand.append((src, dst, sc, M))

        if self.verbose:
            deg = {p:0 for p in nodes}
            for src, dst, _, _ in cand:
                deg[src] += 1
                deg[dst] += 1
            hist = {}
            for v in deg.values():
                hist[v] = hist.get(v, 0) + 1
            print(f"[mosaic] candidate edges after gating: {len(cand)}; degree histogram {hist}", flush=True)

        # Kruskal MST on pruned edges (max spanning)
        idx = {p:i for i,p in enumerate(nodes)}
        N = len(nodes)
        parent = list(range(N))
        rank = [0]*N
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        cand.sort(key=lambda t: t[2], reverse=True)

        # Build adjacency for traversal using the selected MST edges
        adj: Dict[str, List[Tuple[str, np.ndarray, float]]] = {p:[] for p in nodes}
        used_edges: List[Tuple[str,str,float]] = []
        for src, dst, sc, M in cand:
            isrc, idst = idx[src], idx[dst]
            if union(isrc, idst):
                adj[src].append((dst, M, sc))
                adj[dst].append((src, self._invert_affine(M), sc))
                used_edges.append((src, dst, sc))
            if len(used_edges) >= N-1:
                break

        # Choose root = node with max degree in MST
        root = max(nodes, key=lambda p: len(adj[p])) if nodes else None

        # BFS to compute transforms to root (homogeneous 3x3 to avoid shape bugs)
        T3: Dict[str, np.ndarray] = {}
        if root is None:
            return {}, used_edges
        T3[root] = np.eye(3, dtype=np.float32)
        stack = [root]
        visited = set([root])
        while stack:
            u = stack.pop()
            Tu = T3[u]
            for v, M_v_to_u_2x3, _ in adj[u]:
                if v in visited:
                    continue
                M_v_to_u_3x3 = np.eye(3, dtype=np.float32)
                M_v_to_u_3x3[:2,:3] = M_v_to_u_2x3
                T3[v] = Tu @ M_v_to_u_3x3
                visited.add(v)
                stack.append(v)

        # Convert to 2x3 for rendering
        T2: Dict[str, np.ndarray] = {k: v[:2,:] for k,v in T3.items()}
        return T2, used_edges

    def mosaic_all_channels_from_csv_v1(self,
                                     csv_path: str,
                                     out_tif: str,
                                     *,
                                     min_score: Optional[float] = None,
                                     channel_count: Optional[int] = None,
                                     channel_index_order: Optional[List[int]] = None,
                                     angle_tol_deg: float = 30.0,
                                     step_tol_frac: float = 0.25,
                                     rot_tol_deg: float = 5.0,
                                     scale_tol: float = 0.03,
                                     cap_one_per_dir: bool = True,
                                     out_csv: Optional[str] = None) -> str:
        """
        Build a CYX mosaic by reusing pairwise transforms computed on (typically) the nuclei channel.
    
        Parameters
        ----------
        csv_path : str
            Pairwise results CSV produced by `run_folder(...)`.
        out_tif : str
            Output TIFF path (saved with axes='CYX').
        min_score : Optional[float]
            Minimum pairwise score to keep when building the mosaic graph. If None, auto-knee.
        channel_count : Optional[int]
            If provided, force this many channels to be mosaicked (0..channel_count-1).
            Otherwise inferred as the minimum channel count across nodes.
        channel_index_order : Optional[List[int]]
            Optional explicit channel indices to mosaic (e.g., [0,3,1]). Overrides `channel_count` if given.
        angle_tol_deg, step_tol_frac, rot_tol_deg, scale_tol, cap_one_per_dir
            Same gating params as single-channel mosaic.
        out_csv : Optional[str]
            If provided, write a manifest with per-node canvas transforms.
    
        Returns
        -------
        str
            Path to the saved multi-channel mosaic TIFF (CYX).
        """
        # ---- helpers for dtype preservation ----
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # ---- load rows ----
        rows: List[Dict] = []
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["score"] == "" or r["dx_px_full"] == "" or r["dy_px_full"] == "" or r["theta_deg"] == "" or r["scale"] == "":
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError("mosaic_all_channels_from_csv: CSV has no usable rows.")
    
        # ---- threshold (auto-knee if needed) ----
        if min_score is None:
            scores = [float(r["score"]) for r in rows if r["score"] != ""]
            min_score = self._auto_elbow_threshold(scores)
            if self.verbose:
                print(f"[mosaic-all] auto min_score = {min_score:.4f}", flush=True)
    
        # ---- compute transforms and nodes ----
        T, used_edges = self._compute_mosaic_transforms(
            rows, float(min_score),
            angle_tol_deg=angle_tol_deg,
            step_tol_frac=step_tol_frac,
            rot_tol_deg=rot_tol_deg,
            scale_tol=scale_tol,
            cap_one_per_dir=cap_one_per_dir
        )
        kept_nodes = sorted(T.keys())
        if self.verbose:
            print(f"[mosaic-all] nodes in mosaic: {len(kept_nodes)}; edges used: {len(used_edges)}", flush=True)
        if not kept_nodes:
            raise RuntimeError("mosaic_all_channels_from_csv: no nodes remained after pruning.")
    
        # ---- per-node shape and dtype; also infer channel counts ----
        shapes: Dict[str, Tuple[int,int]] = {}
        node_dtype: Dict[str, np.dtype] = {}
        node_channels: Dict[str, int] = {}
        for p in kept_nodes:
            I0 = self._read_plane(p, ch=0)
            H, W = I0.shape
            shapes[p] = (H, W)
            node_dtype[p] = _series_dtype(p)
            node_channels[p] = self._get_channel_count_tif(p)
    
        # decide which channels to mosaic
        if channel_index_order is not None and len(channel_index_order) > 0:
            ch_list = [int(c) for c in channel_index_order]
        else:
            nC = min(node_channels.values()) if channel_count is None else int(channel_count)
            if nC <= 0:
                raise ValueError("mosaic_all_channels_from_csv: channel_count resolved to 0.")
            ch_list = list(range(nC))
        if self.verbose:
            ch_info = {p: node_channels[p] for p in kept_nodes}
            print(f"[mosaic-all] channel plan: {ch_list} ; per-node channel counts: {ch_info}", flush=True)
    
        # ---- determine canvas bounds (from transforms on geometry) ----
        all_x, all_y = [], []
        for p in kept_nodes:
            H, W = shapes[p]
            corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=np.float32).reshape(-1,1,2)
            M = T[p]
            C = cv2.transform(corners, M).reshape(-1,2)
            all_x.append(C[:,0]); all_y.append(C[:,1])
        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        x_min, y_min = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
        x_max, y_max = float(np.ceil(all_x.max())),  float(np.ceil(all_y.max()))
        Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
        off_x, off_y = int(-x_min), int(-y_min)
        if self.verbose:
            print(f"[mosaic-all] canvas = {Wc} x {Hc}", flush=True)
    
        T_off3 = np.array([[1,0,off_x],
                           [0,1,off_y],
                           [0,0,   1 ]], dtype=np.float32)
    
        # ---- compose per-channel canvases, then stack to CYX ----
        out_dtype = np.result_type(*[node_dtype[p] for p in kept_nodes])
        out_stack = np.zeros((len(ch_list), Hc, Wc), np.float32)
    
        for ci, ch in enumerate(ch_list):
            canvas = np.zeros((Hc, Wc), np.float32)
            wgt    = np.zeros((Hc, Wc), np.float32)
            for p in kept_nodes:
                H, W = shapes[p]
                M3 = np.eye(3, dtype=np.float32); M3[:2,:] = T[p]
                M_can = (T_off3 @ M3)[:2,:]
    
                I = self._read_plane(p, ch=ch).astype(np.float32, copy=False)
                warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cov    = cv2.warpAffine(np.ones((H, W), np.float32), M_can, (Wc, Hc),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                canvas += warped
                wgt    += cov
    
            out_stack[ci] = np.divide(canvas, np.maximum(wgt, 1e-6))
    
        tifffile.imwrite(out_tif, _cast(out_stack, out_dtype), metadata={"axes": "CYX"})
    
        # ---- optional manifest (same fields as single-channel) ----
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            best_edge_score: Dict[str, float] = {}
            for a,b,sc in used_edges:
                best_edge_score[a] = max(best_edge_score.get(a, float("-inf")), float(sc))
                best_edge_score[b] = max(best_edge_score.get(b, float("-inf")), float(sc))
    
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "path","H","W","M00","M01","M02","M10","M11","M12","canvas_x","canvas_y","best_pair_score"
                ])
                w.writeheader()
                for p in kept_nodes:
                    H, W = shapes[p]
                    M3 = np.eye(3, dtype=np.float32); M3[:2,:] = T[p]
                    M_can = (T_off3 @ M3)[:2,:]
                    corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=np.float32).reshape(-1,1,2)
                    C = cv2.transform(corners, M_can).reshape(-1,2)
                    x0, y0 = float(np.min(C[:,0])), float(np.min(C[:,1]))
                    w.writerow(dict(
                        path=p, H=int(H), W=int(W),
                        M00=float(M_can[0,0]), M01=float(M_can[0,1]), M02=float(M_can[0,2]),
                        M10=float(M_can[1,0]), M11=float(M_can[1,1]), M12=float(M_can[1,2]),
                        canvas_x=float(x0), canvas_y=float(y0),
                        best_pair_score=float(best_edge_score.get(p, float("nan")))
                    ))
    
        return out_tif

    def render_mosaic_from_csv_v1(self,
                               csv_path: str,
                               out_tif: str,
                               out_png: Optional[str] = None,
                               channel_index: int = 0,
                               min_score: Optional[float] = None,
                               *,
                               angle_tol_deg: float = 30.0,
                               step_tol_frac: float = 0.25,
                               rot_tol_deg: float = 5.0,
                               scale_tol: float = 0.03,
                               cap_one_per_dir: bool = True,
                               out_csv: Optional[str] = None) -> Tuple[str, Optional[str]]:
        # dtype helpers
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # Load rows
        rows: List[Dict] = []
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["score"] == "" or r["dx_px_full"] == "" or r["dy_px_full"] == "" or r["theta_deg"] == "" or r["scale"] == "":
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError("render_mosaic_from_csv: CSV has no usable rows.")
    
        # Threshold
        if min_score is None:
            scores = [float(r["score"]) for r in rows if r["score"] != ""]
            min_score = self._auto_elbow_threshold(scores)
            if self.verbose:
                print(f"[mosaic] auto min_score = {min_score:.4f}", flush=True)
    
        # Transforms to a common root using pruned graph
        T, used_edges = self._compute_mosaic_transforms(
            rows, float(min_score),
            angle_tol_deg=angle_tol_deg,
            step_tol_frac=step_tol_frac,
            rot_tol_deg=rot_tol_deg,
            scale_tol=scale_tol,
            cap_one_per_dir=cap_one_per_dir
        )
        kept_nodes = sorted(T.keys())
        if self.verbose:
            print(f"[mosaic] nodes in mosaic: {len(kept_nodes)}; edges used: {len(used_edges)}", flush=True)
        if not kept_nodes:
            raise RuntimeError("render_mosaic_from_csv: no nodes remained after pruning.")
    
        # Determine canvas bounds + remember per-node dtype
        all_x, all_y = [], []
        shapes: Dict[str, Tuple[int,int]] = {}
        node_dtype: Dict[str, np.dtype] = {}
        for p in kept_nodes:
            I = self._read_plane(p, ch=channel_index)
            H, W = I.shape
            shapes[p] = (H, W)
            node_dtype[p] = _series_dtype(p)
            corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=np.float32).reshape(-1,1,2)
            M = T[p]
            C = cv2.transform(corners, M).reshape(-1,2)
            all_x.append(C[:,0]); all_y.append(C[:,1])
    
        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        x_min, y_min = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
        x_max, y_max = float(np.ceil(all_x.max())),  float(np.ceil(all_y.max()))
        Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
        off_x, off_y = int(-x_min), int(-y_min)
        if self.verbose:
            print(f"[mosaic] canvas = {Wc} x {Hc}", flush=True)
    
        # Compose canvas in raw intensity space
        canvas = np.zeros((Hc, Wc), np.float32)
        wgt    = np.zeros((Hc, Wc), np.float32)
        T_off3 = np.array([[1,0,off_x],
                           [0,1,off_y],
                           [0,0,   1 ]], dtype=np.float32)
    
        manifest_rows: List[Dict[str, Union[str, float, int]]] = []
        best_edge_score: Dict[str, float] = {}
        for a,b,sc in used_edges:
            best_edge_score[a] = max(best_edge_score.get(a, float("-inf")), float(sc))
            best_edge_score[b] = max(best_edge_score.get(b, float("-inf")), float(sc))
    
        for i, p in enumerate(kept_nodes, 1):
            I = self._read_plane(p, ch=channel_index).astype(np.float32)
            H, W = shapes[p]
    
            M3 = np.eye(3, dtype=np.float32); M3[:2,:] = T[p]
            M_can = (T_off3 @ M3)[:2,:]
    
            # warp image and a 1-mask (coverage)
            warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cov    = cv2.warpAffine(np.ones((H, W), np.float32), M_can, (Wc, Hc),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    
            canvas += warped
            wgt    += cov
    
            # Top-left of warped bbox (for manifest)
            corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=np.float32).reshape(-1,1,2)
            C = cv2.transform(corners, M_can).reshape(-1,2)
            x0, y0 = float(np.min(C[:,0])), float(np.min(C[:,1]))
    
            manifest_rows.append(dict(
                path=p,
                H=int(H), W=int(W),
                M00=float(M_can[0,0]), M01=float(M_can[0,1]), M02=float(M_can[0,2]),
                M10=float(M_can[1,0]), M11=float(M_can[1,1]), M12=float(M_can[1,2]),
                canvas_x=float(x0), canvas_y=float(y0),
                best_pair_score=float(best_edge_score.get(p, float("nan")))
            ))
    
            if self.verbose and (i % max(1, len(kept_nodes)//10) == 0 or i == len(kept_nodes)):
                print(f"[mosaic] placed {i}/{len(kept_nodes)} images", flush=True)
    
        out = np.divide(canvas, np.maximum(wgt, 1e-6))
    
        # choose output dtype across all inputs
        out_dtype = np.result_type(*[node_dtype[p] for p in kept_nodes])
        tifffile.imwrite(out_tif, _cast(out, out_dtype))
        if out_png:
            prev = (out - out.min()) / (out.max() - out.min() + 1e-12)
            plt.imsave(out_png, prev, cmap="gray")
    
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "path","H","W","M00","M01","M02","M10","M11","M12","canvas_x","canvas_y","best_pair_score"
                ])
                w.writeheader()
                for r in manifest_rows:
                    w.writerow(r)
    
        return out_tif, out_png

    def render_mosaic_from_csv(self,
                               csv_path: str,
                               out_tif: str,
                               out_png: Optional[str] = None,
                               channel_index: int = 0,
                               min_score: Optional[float] = None,
                               *,
                               angle_tol_deg: float = 30.0,
                               step_tol_frac: float = 0.25,
                               rot_tol_deg: float = 5.0,
                               scale_tol: float = 0.03,
                               cap_one_per_dir: bool = True,
                               out_csv: Optional[str] = None) -> Tuple[str, Optional[str]]:
        # dtype helpers
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
    
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # NEW: manifest-only mode (no mosaic rendering)
        manifest_only = (out_csv is not None) and (out_tif is None)
        if manifest_only:
            # No point accepting a PNG target if we aren't rendering
            out_png = None
    
        # Load rows
        rows: List[Dict] = []
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["score"] == "" or r["dx_px_full"] == "" or r["dy_px_full"] == "" or r["theta_deg"] == "" or r["scale"] == "":
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError("render_mosaic_from_csv: CSV has no usable rows.")
    
        # Threshold
        if min_score is None:
            scores = [float(r["score"]) for r in rows if r["score"] != ""]
            min_score = self._auto_elbow_threshold(scores)
            if self.verbose:
                print(f"[mosaic] auto min_score = {min_score:.4f}", flush=True)
    
        # Transforms to a common root using pruned graph
        T, used_edges = self._compute_mosaic_transforms(
            rows, float(min_score),
            angle_tol_deg=angle_tol_deg,
            step_tol_frac=step_tol_frac,
            rot_tol_deg=rot_tol_deg,
            scale_tol=scale_tol,
            cap_one_per_dir=cap_one_per_dir
        )
        kept_nodes = sorted(T.keys())
        if self.verbose:
            print(f"[mosaic] nodes in mosaic: {len(kept_nodes)}; edges used: {len(used_edges)}", flush=True)
        if not kept_nodes:
            raise RuntimeError("render_mosaic_from_csv: no nodes remained after pruning.")
    
        # Determine canvas bounds (+ optionally remember per-node dtype)
        all_x, all_y = [], []
        shapes: Dict[str, Tuple[int, int]] = {}
        node_dtype: Dict[str, np.dtype] = {}  # only used when writing out_tif
    
        for p in kept_nodes:
            I = self._read_plane(p, ch=channel_index)
            H, W = I.shape
            shapes[p] = (H, W)
            if not manifest_only:
                node_dtype[p] = _series_dtype(p)
    
            corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
            M = T[p]
            C = cv2.transform(corners, M).reshape(-1, 2)
            all_x.append(C[:, 0])
            all_y.append(C[:, 1])
    
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        x_min, y_min = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
        x_max, y_max = float(np.ceil(all_x.max())), float(np.ceil(all_y.max()))
        Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
        off_x, off_y = int(-x_min), int(-y_min)
        if self.verbose:
            print(f"[mosaic] canvas = {Wc} x {Hc}", flush=True)
    
        T_off3 = np.array([[1, 0, off_x],
                           [0, 1, off_y],
                           [0, 0,   1 ]], dtype=np.float32)
    
        # Only allocate + blend if we are actually rendering
        if not manifest_only:
            canvas = np.zeros((Hc, Wc), np.float32)
            wgt    = np.zeros((Hc, Wc), np.float32)
    
        manifest_rows: List[Dict[str, Union[str, float, int]]] = []
        best_edge_score: Dict[str, float] = {}
        for a, b, sc in used_edges:
            best_edge_score[a] = max(best_edge_score.get(a, float("-inf")), float(sc))
            best_edge_score[b] = max(best_edge_score.get(b, float("-inf")), float(sc))
    
        for i, p in enumerate(kept_nodes, 1):
            H, W = shapes[p]
    
            M3 = np.eye(3, dtype=np.float32)
            M3[:2, :] = T[p]
            M_can = (T_off3 @ M3)[:2, :]
    
            if not manifest_only:
                I = self._read_plane(p, ch=channel_index).astype(np.float32)
    
                # warp image and a 1-mask (coverage)
                warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cov    = cv2.warpAffine(np.ones((H, W), np.float32), M_can, (Wc, Hc),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    
                canvas += warped
                wgt    += cov
    
            # Top-left of warped bbox (for manifest)
            corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
            C = cv2.transform(corners, M_can).reshape(-1, 2)
            x0, y0 = float(np.min(C[:, 0])), float(np.min(C[:, 1]))
    
            manifest_rows.append(dict(
                path=p,
                H=int(H), W=int(W),
                M00=float(M_can[0, 0]), M01=float(M_can[0, 1]), M02=float(M_can[0, 2]),
                M10=float(M_can[1, 0]), M11=float(M_can[1, 1]), M12=float(M_can[1, 2]),
                canvas_x=float(x0), canvas_y=float(y0),
                best_pair_score=float(best_edge_score.get(p, float("nan")))
            ))
    
            if self.verbose and (i % max(1, len(kept_nodes) // 10) == 0 or i == len(kept_nodes)):
                print(f"[mosaic] placed {i}/{len(kept_nodes)} images", flush=True)
    
        # Write manifest CSV (works in both modes)
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "path", "H", "W",
                    "M00", "M01", "M02",
                    "M10", "M11", "M12",
                    "canvas_x", "canvas_y",
                    "best_pair_score"
                ])
                w.writeheader()
                for r in manifest_rows:
                    w.writerow(r)
    
        # If manifest-only, stop here (no mosaic image)
        if manifest_only:
            return out_tif, out_png  # both None in this mode
    
        # Otherwise, build and save mosaic image(s)
        out = np.divide(canvas, np.maximum(wgt, 1e-6))
    
        out_dtype = np.result_type(*[node_dtype[p] for p in kept_nodes])
        tifffile.imwrite(out_tif, _cast(out, out_dtype))
        if out_png:
            prev = (out - out.min()) / (out.max() - out.min() + 1e-12)
            plt.imsave(out_png, prev, cmap="gray")
    
        return out_tif, out_png

    def mosaic_all_channels_from_csv(self,
                                       csv_path: str,
                                       out_tif: Optional[str],
                                       *,
                                       min_score: Optional[float] = None,
                                       channel_count: Optional[int] = None,
                                       channel_index_order: Optional[List[int]] = None,
                                       angle_tol_deg: float = 30.0,
                                       step_tol_frac: float = 0.25,
                                       rot_tol_deg: float = 5.0,
                                       scale_tol: float = 0.03,
                                       cap_one_per_dir: bool = True,
                                       out_csv: Optional[str] = None) -> Optional[str]:
        """
        Build a CYX mosaic by reusing pairwise transforms computed on (typically) the nuclei channel.
    
        If out_csv is not None and out_tif is None, run in "manifest-only" mode:
        compute transforms + canvas geometry + per-node canvas transforms and write the manifest CSV,
        but DO NOT render/write the mosaic TIFF.
        """
        # ---- helpers for dtype preservation ----
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
    
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # NEW: manifest-only mode (no mosaic rendering)
        manifest_only = (out_csv is not None) and (out_tif is None)
    
        # ---- load rows ----
        rows: List[Dict] = []
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["score"] == "" or r["dx_px_full"] == "" or r["dy_px_full"] == "" or r["theta_deg"] == "" or r["scale"] == "":
                    continue
                rows.append(r)
        if not rows:
            raise RuntimeError("mosaic_all_channels_from_csv: CSV has no usable rows.")
    
        # ---- threshold (auto-knee if needed) ----
        if min_score is None:
            scores = [float(r["score"]) for r in rows if r["score"] != ""]
            min_score = self._auto_elbow_threshold(scores)
            if self.verbose:
                print(f"[mosaic-all] auto min_score = {min_score:.4f}", flush=True)
    
        # ---- compute transforms and nodes ----
        T, used_edges = self._compute_mosaic_transforms(
            rows, float(min_score),
            angle_tol_deg=angle_tol_deg,
            step_tol_frac=step_tol_frac,
            rot_tol_deg=rot_tol_deg,
            scale_tol=scale_tol,
            cap_one_per_dir=cap_one_per_dir
        )
        kept_nodes = sorted(T.keys())
        if self.verbose:
            print(f"[mosaic-all] nodes in mosaic: {len(kept_nodes)}; edges used: {len(used_edges)}", flush=True)
        if not kept_nodes:
            raise RuntimeError("mosaic_all_channels_from_csv: no nodes remained after pruning.")
    
        # ---- per-node shape (and dtype/channel info only if rendering) ----
        shapes: Dict[str, Tuple[int, int]] = {}
        node_dtype: Dict[str, np.dtype] = {}
        node_channels: Dict[str, int] = {}
    
        for p in kept_nodes:
            I0 = self._read_plane(p, ch=0)
            H, W = I0.shape
            shapes[p] = (H, W)
            if not manifest_only:
                node_dtype[p] = _series_dtype(p)
                node_channels[p] = self._get_channel_count_tif(p)
    
        # decide which channels to mosaic (only if rendering)
        if not manifest_only:
            if channel_index_order is not None and len(channel_index_order) > 0:
                ch_list = [int(c) for c in channel_index_order]
            else:
                nC = min(node_channels.values()) if channel_count is None else int(channel_count)
                if nC <= 0:
                    raise ValueError("mosaic_all_channels_from_csv: channel_count resolved to 0.")
                ch_list = list(range(nC))
            if self.verbose:
                ch_info = {p: node_channels[p] for p in kept_nodes}
                print(f"[mosaic-all] channel plan: {ch_list} ; per-node channel counts: {ch_info}", flush=True)
    
        # ---- determine canvas bounds (from transforms on geometry) ----
        all_x, all_y = [], []
        for p in kept_nodes:
            H, W = shapes[p]
            corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
            M = T[p]
            C = cv2.transform(corners, M).reshape(-1, 2)
            all_x.append(C[:, 0])
            all_y.append(C[:, 1])
    
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        x_min, y_min = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
        x_max, y_max = float(np.ceil(all_x.max())),  float(np.ceil(all_y.max()))
        Wc, Hc = int(max(1, x_max - x_min)), int(max(1, y_max - y_min))
        off_x, off_y = int(-x_min), int(-y_min)
        if self.verbose:
            print(f"[mosaic-all] canvas = {Wc} x {Hc}", flush=True)
    
        T_off3 = np.array([[1, 0, off_x],
                           [0, 1, off_y],
                           [0, 0,   1 ]], dtype=np.float32)
    
        # ---- optional manifest (works in both modes) ----
        if out_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            best_edge_score: Dict[str, float] = {}
            for a, b, sc in used_edges:
                best_edge_score[a] = max(best_edge_score.get(a, float("-inf")), float(sc))
                best_edge_score[b] = max(best_edge_score.get(b, float("-inf")), float(sc))
    
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "path", "H", "W",
                    "M00", "M01", "M02",
                    "M10", "M11", "M12",
                    "canvas_x", "canvas_y",
                    "best_pair_score"
                ])
                w.writeheader()
                for p in kept_nodes:
                    H, W = shapes[p]
                    M3 = np.eye(3, dtype=np.float32)
                    M3[:2, :] = T[p]
                    M_can = (T_off3 @ M3)[:2, :]
    
                    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
                    C = cv2.transform(corners, M_can).reshape(-1, 2)
                    x0, y0 = float(np.min(C[:, 0])), float(np.min(C[:, 1]))
    
                    w.writerow(dict(
                        path=p, H=int(H), W=int(W),
                        M00=float(M_can[0, 0]), M01=float(M_can[0, 1]), M02=float(M_can[0, 2]),
                        M10=float(M_can[1, 0]), M11=float(M_can[1, 1]), M12=float(M_can[1, 2]),
                        canvas_x=float(x0), canvas_y=float(y0),
                        best_pair_score=float(best_edge_score.get(p, float("nan")))
                    ))
    
        # ---- manifest-only: stop here ----
        if manifest_only:
            return out_tif  # None in this mode
    
        # ---- otherwise, render and save mosaic TIFF ----
        if out_tif is None:
            raise ValueError("mosaic_all_channels_from_csv: out_tif is None but manifest_only is False.")
    
        out_dtype = np.result_type(*[node_dtype[p] for p in kept_nodes])
        out_stack = np.zeros((len(ch_list), Hc, Wc), np.float32)
    
        for ci, ch in enumerate(ch_list):
            canvas = np.zeros((Hc, Wc), np.float32)
            wgt    = np.zeros((Hc, Wc), np.float32)
            for p in kept_nodes:
                H, W = shapes[p]
                M3 = np.eye(3, dtype=np.float32)
                M3[:2, :] = T[p]
                M_can = (T_off3 @ M3)[:2, :]
    
                I = self._read_plane(p, ch=ch).astype(np.float32, copy=False)
                warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cov    = cv2.warpAffine(np.ones((H, W), np.float32), M_can, (Wc, Hc),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                canvas += warped
                wgt    += cov
    
            out_stack[ci] = np.divide(canvas, np.maximum(wgt, 1e-6))
    
        tifffile.imwrite(out_tif, _cast(out_stack, out_dtype), metadata={"axes": "CYX"})
        return out_tif

class StitchedMultiAligner:
    """
    Align an arbitrary number of stitched mosaics (multi-channel, arbitrary axes) to a common reference
    using the nuclei/ Hoechst channel. Saves a channel-concatenated aligned stack and a CSV manifest
    describing the mapping from input channels to output channels and the estimated transforms/scores.

    Output image axes: CYX (channels stacked in the order inputs are provided).

    Args
    ----
    detector : {"ORB","SIFT"}
        Feature detector for keypoint matching.
    nfeatures : int
        Feature budget for detector.
    max_keypoints : Optional[int]
        Hard cap on kept keypoints after detection (by detector’s internal ranking).
    downsample : float in (0,1]
        Downsample factor for feature/score pass.
    ransac_thresh_px : float
        Reprojection threshold (pixels) for affine estimation (downsampled space).
    allow_scale : bool
        If False, constrain to rotation+translation (or translation only if allow_rotation=False).
    allow_rotation : bool
        If False, constrain to translation only.
    outdir : str
        Output directory for images/csv.
    opencv_threads : int
        Limit OpenCV internal threading (avoid oversubscription).
    # Axis/Z/time handling (for TIFF reading)
    arr_axes : "AUTO" or a string over {T,C,Z,Y,X}
    mip : bool
        If True and Z exists, max-project Z.
    z_index : int
        If mip=False, choose Z slice.
    t_index : int
        Choose T index if T exists.
    squeeze_singleton : bool
        Squeeze 1-length axes after slicing.

    Notes
    -----
    - Alignment is done to the first image in `paths` (reference).
    - For each image, you can provide a per-image nuclei channel index via `nuclei_channel_indices`.
      If None, defaults to 0 for all.
    """

    def __init__(self,
                 detector: str = "ORB",
                 nfeatures: int = 6000,
                 max_keypoints: Optional[int] = 2000,
                 downsample: float = 0.5,
                 ransac_thresh_px: float = 3.0,
                 allow_scale: bool = False,
                 allow_rotation: bool = False,
                 outdir: str = "./align_out",
                 opencv_threads: int = 1,
                 # axes/time/Z
                 arr_axes: str = "AUTO",
                 mip: bool = False,
                 z_index: int = 0,
                 t_index: int = 0,
                 squeeze_singleton: bool = True):
        self.detector = detector.upper()
        self.nfeatures = int(nfeatures)
        self.max_keypoints = None if max_keypoints is None else int(max_keypoints)
        self.downsample = float(downsample)
        self.ransac_thresh_px = float(ransac_thresh_px)
        self.allow_scale = bool(allow_scale)
        self.allow_rotation = bool(allow_rotation)

        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)

        try:
            cv2.setNumThreads(int(opencv_threads))
        except Exception:
            pass

        # axes
        self.arr_axes = str(arr_axes).upper()
        self.mip = bool(mip)
        self.z_index = int(z_index)
        self.t_index = int(t_index)
        self.squeeze_singleton = bool(squeeze_singleton)

        # detector init
        if self.detector == "ORB":
            self._det = cv2.ORB_create(nfeatures=self.nfeatures, fastThreshold=5)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._use_flann = False
        elif self.detector == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT requested but opencv-contrib build not found.")
            self._det = cv2.SIFT_create(nfeatures=self.nfeatures)
            self._use_flann = True
            self._flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        else:
            raise ValueError("detector must be 'ORB' or 'SIFT'")

    # ---------------------- basic IO / axis helpers ----------------------
    @staticmethod
    def _is_large_dim(n: int) -> bool:
        return n >= 128

    @staticmethod
    def _guess_axes_from_shape(shape: Tuple[int, ...]) -> str:
        nd = len(shape)
        if nd == 2:
            return "YX"
        if nd == 3:
            if StitchedMultiAligner._is_large_dim(shape[-1]) and StitchedMultiAligner._is_large_dim(shape[-2]):
                a = shape[0]
                return "CYX" if a <= 8 else "ZYX"
            if StitchedMultiAligner._is_large_dim(shape[0]) and StitchedMultiAligner._is_large_dim(shape[1]):
                a = shape[2]
                return "YXC" if a <= 8 else "YXZ"
            return "CYX"
        if nd == 4:
            if StitchedMultiAligner._is_large_dim(shape[-1]) and StitchedMultiAligner._is_large_dim(shape[-2]):
                a, b = shape[0], shape[1]
                if a <= 8 and b > 8:
                    return "CZYX"
                if b <= 8 and a > 8:
                    return "ZCYX"
                if a <= 8 and b <= 8:
                    return "CZYX"
                return "CZYX"
            return "TCYX"
        if nd == 5:
            if StitchedMultiAligner._is_large_dim(shape[-1]) and StitchedMultiAligner._is_large_dim(shape[-2]):
                b = shape[1]
                return "TCZYX" if b <= 8 else "TZCYX"
            return "TZCYX"
        return "CZYX" if nd >= 4 else "CYX"

    def _normalize_to_yx(self, arr: np.ndarray, ch: int, axes_hint: Optional[str] = None) -> np.ndarray:
        if self.arr_axes and self.arr_axes != "AUTO":
            axes = "".join(a for a in self.arr_axes if a in "TCZYX")
        elif axes_hint:
            axes = "".join(a for a in axes_hint if a in "TCZYX")
        else:
            axes = self._guess_axes_from_shape(arr.shape)

        ax = list(axes)
        while len(ax) > arr.ndim:
            for d in ("T", "C"):
                if d in ax and len(ax) > arr.ndim:
                    ax.remove(d)
        while len(ax) > arr.ndim:
            ax.pop(0)
        while len(ax) < arr.ndim:
            if "T" not in ax:
                ax.insert(0, "T")
            elif "C" not in ax:
                ax.insert(0, "C")
            else:
                ax.insert(0, "Z")
        axes = "".join(ax)

        slicers = []
        for a in axes:
            if a == "T": slicers.append(self.t_index)
            elif a == "C": slicers.append(ch)
            elif a == "Z": slicers.append(slice(None) if self.mip else self.z_index)
            elif a in ("Y", "X"): slicers.append(slice(None))
            else: slicers.append(0)

        sub = arr[tuple(slicers)]
        if self.mip and sub.ndim == 3:
            z_axis = 0 if (self._is_large_dim(sub.shape[-1]) and self._is_large_dim(sub.shape[-2])) else int(np.argmin(sub.shape))
            sub = sub.max(axis=z_axis)
        if self.squeeze_singleton:
            sub = np.squeeze(sub)
        if sub.ndim == 3:
            small = [i for i, n in enumerate(sub.shape) if n <= 8]
            if small:
                sub = sub.take(indices=0, axis=small[0])
        if sub.ndim != 2:
            raise ValueError(f"Expected 2D YX after axis handling, got {sub.shape} (axes='{axes}')")
        return sub.astype(np.float32, copy=False)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        m, M = float(np.nanmin(img)), float(np.nanmax(img))
        if M <= m + 1e-12:
            return np.zeros_like(img, dtype=np.uint8)
        return np.clip(255.0 * (img - m) / (M - m), 0, 255).astype(np.uint8)

    @staticmethod
    def _edge_zncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        ea = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)**2
        eb = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)**2
        if mask is not None:
            idx = mask.astype(bool)
            if idx.sum() < 25:
                return 0.0
            ea = ea[idx]; eb = eb[idx]
        ea = ea - ea.mean(); eb = eb - eb.mean()
        den = (ea.std() * eb.std()) + 1e-9
        return float((ea * eb).mean() / den)

    def _read_plane(self, path: str, ch: int = 0) -> np.ndarray:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            axes_hint = getattr(series, "axes", None)
            if axes_hint:
                axes_hint = "".join(a for a in axes_hint.upper() if a in "TCZYX")
            arr = series.asarray()
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
        if axes_hint is None and arr.ndim == 3:
            fn = os.path.basename(path).lower()
            if re.search(r'(^|[_\-])c\d+([_\-]|$)', fn):
                axes_hint = "CYX"
            else:
                axes_hint = "ZYX" if self.mip else "CYX"
        return self._normalize_to_yx(arr, ch=ch, axes_hint=axes_hint)

    @staticmethod
    def _get_channel_count_tif(path: str) -> int:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            axes = getattr(series, "axes", None)
            shape = series.shape
        if axes:
            axes = "".join(a for a in axes.upper() if a in "TCZYX")
            return int(shape[axes.index("C")]) if "C" in axes else 1
        gh = StitchedMultiAligner._guess_axes_from_shape(shape)
        return int(shape[gh.index("C")]) if "C" in gh else 1

    def _read_all_channels_cyx(self, path: str) -> np.ndarray:
        nC = self._get_channel_count_tif(path)
        planes = [self._read_plane(path, ch=c).astype(np.float32, copy=False) for c in range(nC)]
        return np.stack(planes, axis=0)  # (C,H,W)

    # --------------------------- feature/matching ------------------------
    def _detect_and_describe(self, I8: np.ndarray):
        kp, desc = self._det.detectAndCompute(I8, None)
        if kp is None or desc is None or len(kp) < 4:
            pts = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 32), np.uint8) if self.detector == "ORB" else np.zeros((0, 128), np.float32)
            return pts, desc
        if self.max_keypoints is not None and len(kp) > self.max_keypoints:
            idx = np.argsort([-k.response for k in kp])[:self.max_keypoints]
            kp = [kp[i] for i in idx]
            desc = desc[idx]
        pts = np.float32([k.pt for k in kp])
        return pts, desc

    def _match(self, fA: Dict[str, np.ndarray], fB: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if fA["pts"].shape[0] < 4 or fB["pts"].shape[0] < 4:
            return np.zeros((0,2), np.float32), np.zeros((0,2), np.float32)
        if self.detector == "ORB":
            matches = self._bf.match(fA["desc"], fB["desc"])
            matches = list(matches); matches.sort(key=lambda m: m.distance)
            idxA = [m.queryIdx for m in matches]
            idxB = [m.trainIdx for m in matches]
        else:
            raw = self._flann.knnMatch(fA["desc"], fB["desc"], k=2)
            good = []
            for pair in raw:
                if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                    good.append(pair[0])
            good.sort(key=lambda m: m.distance)
            idxA = [m.queryIdx for m in good]
            idxB = [m.trainIdx for m in good]
        if not idxA:
            return np.zeros((0,2), np.float32), np.zeros((0,2), np.float32)
        return fA["pts"][idxA].astype(np.float32), fB["pts"][idxB].astype(np.float32)

    @staticmethod
    def _affine_from_pts(ptsA: np.ndarray, ptsB: np.ndarray, ransac_thresh_px: float):
        if ptsA.shape[0] < 4 or ptsB.shape[0] < 4:
            return None, None, 0.0
        M, inliers = cv2.estimateAffinePartial2D(
            ptsB.reshape(-1, 1, 2), ptsA.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh_px),
            maxIters=5000,
            confidence=0.999
        )
        if M is None:
            return None, None, 0.0
        if inliers is None:
            inlier_mask = None
            inlier_ratio = 0.0
        else:
            inlier_mask = inliers.ravel().astype(bool)
            inlier_ratio = float(inlier_mask.mean())
        return M.astype(np.float32), inlier_mask, inlier_ratio

    @staticmethod
    def _closest_rotation(A: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R.astype(np.float32)

    # ------------------------------- driver ------------------------------
    def align(self,
              paths: List[str],
              nuclei_channel_indices: Optional[List[int]] = None,
              out_tif: Optional[str] = None,
              out_png_preview: Optional[str] = None,
              csv_path: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
        assert len(paths) >= 1, "Provide at least one stitched image."
        if nuclei_channel_indices is None:
            nuclei_channel_indices = [0] * len(paths)
        assert len(nuclei_channel_indices) == len(paths), "nuclei_channel_indices length must match paths."
    
        if out_tif is None:
            out_tif = os.path.join(self.outdir, "aligned_allc.tif")
        if csv_path is None:
            csv_path = os.path.join(self.outdir, "aligned_manifest.csv")
    
        # dtype helpers (local)
        def _series_dtype(p: str) -> np.dtype:
            with tifffile.TiffFile(p) as tf:
                return np.dtype(tf.series[0].dtype)
        def _common_dtype(dts: List[np.dtype]) -> np.dtype:
            return np.result_type(*dts)
        def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)
            return arr.astype(dtype, copy=False)
    
        # Reference
        ref_path = paths[0]
        ref_ch = int(nuclei_channel_indices[0])
        Iref = self._read_plane(ref_path, ch=ref_ch)
        H, W = Iref.shape
    
        # DS ref for features (8-bit only here)
        s = float(self.downsample)
        Hds = max(1, int(round(H * s))); Wds = max(1, int(round(W * s)))
        Iref_ds = cv2.resize(Iref, (Wds, Hds), interpolation=cv2.INTER_LINEAR)
        Iref_u8 = self._to_uint8(Iref_ds)
        ref_kp, ref_desc = self._detect_and_describe(Iref_u8)
        Fref = {"pts": ref_kp, "desc": ref_desc}
    
        # Output buffer
        all_arrays: List[np.ndarray] = []
        manifest_rows: List[Dict[str, Union[str, int, float]]] = []
    
        # Keep track of input dtypes to choose a common output dtype
        input_dtypes: List[np.dtype] = [_series_dtype(ref_path)]
    
        # Reference channels (no warp)
        A0 = self._read_all_channels_cyx(ref_path)
        C0 = A0.shape[0]
        all_arrays.append(A0)
        for c in range(C0):
            manifest_rows.append(dict(
                input_path=ref_path,
                input_channel=c,
                output_channel=len(manifest_rows),
                ref=True,
                tx=0.0, ty=0.0, theta_deg=0.0, scale=1.0,
                score=1.0, inlier_ratio=1.0
            ))
    
        # Others: estimate M (B -> ref), warp all channels at full-res
        for k in range(1, len(paths)):
            p = paths[k]
            input_dtypes.append(_series_dtype(p))
    
            ch_nuc = int(nuclei_channel_indices[k])
            I = self._read_plane(p, ch=ch_nuc)
            Ih, Iw = I.shape
    
            Ids = cv2.resize(I, (max(1, int(round(Iw * s))), max(1, int(round(Ih * s)))), interpolation=cv2.INTER_LINEAR)
            Iu8 = self._to_uint8(Ids)
            kp, desc = self._detect_and_describe(Iu8)
            Fb = {"pts": kp, "desc": desc}
    
            ptsA, ptsB = self._match(Fref, Fb)  # A=ref, B=curr
            score = 0.0
            inlier_ratio = 0.0
            if ptsA.shape[0] >= 4:
                M_ds, inmask, inlier_ratio = self._affine_from_pts(ptsA, ptsB, self.ransac_thresh_px)
            else:
                M_ds = None
            if M_ds is None:
                continue
    
            # constraints
            A_lin = M_ds[:, :2].astype(np.float32)
            pA = ptsA[inmask] if (inmask is not None and inmask.any()) else ptsA
            pB = ptsB[inmask] if (inmask is not None and inmask.any()) else ptsB
            if not self.allow_scale and not self.allow_rotation:
                A_lin = np.eye(2, dtype=np.float32)
                t_mean = (pA - pB).mean(axis=0).astype(np.float32)
                M_ds = np.zeros((2, 3), dtype=np.float32); M_ds[:, :2] = A_lin; M_ds[:, 2] = t_mean
            elif (not self.allow_scale) and self.allow_rotation:
                A_rot = self._closest_rotation(A_lin)
                t_mean = (pA - (pB @ A_rot.T)).mean(axis=0).astype(np.float32)
                M_ds = np.zeros((2, 3), dtype=np.float32); M_ds[:, :2] = A_rot; M_ds[:, 2] = t_mean
    
            # lift to full res
            M_full = M_ds.copy()
            if s != 0:
                M_full[0, 2] /= s
                M_full[1, 2] /= s
    
            a, b, tx = float(M_full[0, 0]), float(M_full[0, 1]), float(M_full[0, 2])
            c, d, ty = float(M_full[1, 0]), float(M_full[1, 1]), float(M_full[1, 2])
            scale = float(np.sqrt(max(1e-12, (a * d - b * c))))
            theta = float(np.degrees(np.arctan2(c, a)))
    
            # score on DS (foreground of ref)
            B_warp_ds = cv2.warpAffine(Iu8, M_ds, (Wds, Hds), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            _, th = cv2.threshold(Iref_u8, 0, 255, cv2.THRESH_OTSU)
            mA = (Iref_u8 >= th)
            score = float(self._edge_zncc(Iref_u8.astype(np.float32), B_warp_ds.astype(np.float32), mask=mA)) * float(inlier_ratio)
    
            # warp all channels
            B_all = self._read_all_channels_cyx(p)  # float32 workspace
            outC = np.zeros((B_all.shape[0], H, W), np.float32)
            for cidx in range(B_all.shape[0]):
                outC[cidx] = cv2.warpAffine(B_all[cidx], M_full, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
            base_out_ch = sum(arr.shape[0] for arr in all_arrays)
            all_arrays.append(outC)
            for cidx in range(B_all.shape[0]):
                manifest_rows.append(dict(
                    input_path=p,
                    input_channel=cidx,
                    output_channel=base_out_ch + cidx,
                    ref=False,
                    tx=tx, ty=ty, theta_deg=theta, scale=scale,
                    score=score, inlier_ratio=float(inlier_ratio)
                ))
    
        # concatenate (float32 workspace) and SAVE using common input dtype
        out = np.concatenate(all_arrays, axis=0)
        out_dtype = _common_dtype(input_dtypes)
        tifffile.imwrite(out_tif, _cast(out, out_dtype), metadata={"axes": "CYX"})
    
        if out_png_preview:
            ref_idx = int(nuclei_channel_indices[0])
            img = out[ref_idx]
            prev = (img - img.min()) / (img.max() - img.min() + 1e-12)
            plt.imsave(out_png_preview, prev, cmap="gray")
    
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "input_path", "input_channel", "output_channel", "ref",
                "tx", "ty", "theta_deg", "scale", "score", "inlier_ratio"
            ])
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)
    
        return out_tif, out_png_preview, csv_path

def stitch_cycle_wells(settings):

    # ---- Apply defaults (single flat dict) ----
    settings = get_preprocess_ops_settings(settings)

    vprint = print if settings.get("verbose", False) else (lambda *a, **k: None)

    # ---- Required/assumed inputs ----
    src = settings.get("src")
    if not src or not os.path.isdir(src):
        raise ValueError("settings['src'] must point to an existing directory")

    dst_root = settings.get("dst_root") or src
    os.makedirs(dst_root, exist_ok=True)

    meta_regex = settings.get(
        "meta_regex",
        r"(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)\.(?:tif|tiff)$",
    )
    well_group = settings.get("well_group", "well")
    recursive = bool(settings.get("recursive", True))
    exts = tuple(x.lower() for x in settings.get("exts", (".tif", ".tiff")))
    dry_run = bool(settings.get("dry_run", False))
    collision = settings.get("collision", "rename")       # {'rename','skip','overwrite'}
    on_missing = settings.get("on_missing", "error")      # {'error','skip'}
    do_organize = bool(settings.get("do_organize", True))
    do_nuc_stitch = bool(settings.get("do_nuc_stitch", True))
    do_multichannel = bool(settings.get("do_multichannel", True))
    nuc_channel_index = int(settings.get("channel_index", 0))

    # plate id for filenames (plate + well metadata in all CSV names)
    plate_id = settings.get("plate") or settings.get("plate_id") or settings.get("experiment") or os.path.basename(os.path.normpath(dst_root))
    plate_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(plate_id)).strip("_") or "plate"

    # Compile metadata regex
    meta_re = re.compile(meta_regex, re.IGNORECASE)

    # ---- Scan files ----
    def _iter_files(root: str, recursive_flag: bool, _exts: tuple):
        if recursive_flag:
            for r, _, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(_exts):
                        yield os.path.join(r, fn)
        else:
            for fn in sorted(os.listdir(root)):
                p = os.path.join(root, fn)
                if os.path.isfile(p) and fn.lower().endswith(_exts):
                    yield p

    files = list(_iter_files(src, recursive, exts))
    vprint(f"[organize] scanned {len(files)} files from {src}")

    # ---- Group by well using regex ----
    grouped: Dict[str, List[str]] = {}
    skipped_missing = 0
    for p in files:
        m = meta_re.search(os.path.basename(p))
        if not m or (well_group not in m.groupdict()) or not m.group(well_group):
            if on_missing == "error":
                raise ValueError(f"Missing '{well_group}' in filename: {os.path.basename(p)}")
            skipped_missing += 1
            continue
        well = (m.group(well_group) or "").upper()
        grouped.setdefault(well, []).append(p)

    if not grouped:
        vprint("[organize] no wells found after grouping.")
        return {
            "organized": {
                "moved": 0,
                "skipped": skipped_missing,
                "linked": 0,
                "by_well": {},
            },
            "wells": {},
        }

    # ---- Organize into per-well folders (or create symlinks if not organizing) ----
    moved = 0
    skipped = skipped_missing
    linked = 0
    by_well_outpaths: Dict[str, List[str]] = {}

    link_root = os.path.join(dst_root, "_links")  # used when do_organize=False
    if not do_organize:
        os.makedirs(link_root, exist_ok=True)

    def _resolve_collision(dst_path: str) -> Optional[str]:
        if not os.path.exists(dst_path):
            return dst_path
        if collision == "skip":
            return None
        if collision == "overwrite":
            return dst_path
        # rename: add numeric suffix
        base, ext = os.path.splitext(dst_path)
        for k in range(1, 10000):
            cand = f"{base}_{k:03d}{ext}"
            if not os.path.exists(cand):
                return cand
        raise RuntimeError(f"Could not resolve collision for {dst_path}")

    for well, src_list in grouped.items():
        out_well_dir = os.path.join(dst_root, well)
        link_well_dir = os.path.join(link_root, well)
        if do_organize:
            os.makedirs(out_well_dir, exist_ok=True)
        else:
            os.makedirs(link_well_dir, exist_ok=True)

        out_paths: List[str] = []
        for sp in src_list:
            fn = os.path.basename(sp)
            if do_organize:
                dp = os.path.join(out_well_dir, fn)
                rp = _resolve_collision(dp)
                if rp is None:
                    skipped += 1
                    continue
                if not dry_run:
                    if collision == "overwrite" and os.path.exists(dp) and rp == dp:
                        try:
                            os.remove(dp)
                        except FileNotFoundError:
                            pass
                    if sp != rp:
                        shutil.move(sp, rp)
                    moved += 1
                out_paths.append(rp if rp is not None else dp)
            else:
                # create a symlink into link_well_dir
                dp = os.path.join(link_well_dir, fn)
                rp = _resolve_collision(dp)
                if rp is None:
                    skipped += 1
                    continue
                if not dry_run:
                    try:
                        if os.path.lexists(rp):
                            os.remove(rp)
                        os.symlink(sp, rp)
                    except FileExistsError:
                        pass
                    linked += 1
                out_paths.append(rp if rp is not None else dp)

        # sort by site number if present
        def _site_key(pth: str) -> int:
            m = re.search(r"Site[-_](\d+)", os.path.basename(pth), re.IGNORECASE)
            return int(m.group(1)) if m else 10**9

        out_paths.sort(key=lambda pth: (_site_key(pth), pth))
        by_well_outpaths[well] = out_paths

    organized_summary = {
        "moved": moved,
        "skipped": skipped,
        "linked": linked,
        "by_well": by_well_outpaths,
    }
    vprint(f"[organize] moved={moved}, linked={linked}, skipped={skipped}; wells={len(by_well_outpaths)}")

    if not do_nuc_stitch:
        return {"organized": organized_summary, "wells": {}}

    # ---- Per-well stitch + mosaic ----
    results_by_well: Dict[str, Dict[str, Any]] = {}

    for well, well_files in by_well_outpaths.items():
        if not well_files:
            continue

        # Where run_folder will scan (keep your behavior)
        scan_dir = os.path.dirname(well_files[0])

        # Final per-well root (always under dst_root/{well})
        well_root = os.path.join(dst_root, well)
        os.makedirs(well_root, exist_ok=True)

        # ---- Desired layout ----
        # images moved to:     {dst_root}/{well}/{well}
        # qc images saved in:  {dst_root}/{well}/qc/pairs
        # stitch outputs in:   {dst_root}/{well}/{well}/stitch
        # csv files in:        {dst_root}/{well}/results
        orig_outdir = os.path.join(well_root, well)                  # {src}/{well}/{well}
        qc_pairs_dir = os.path.join(well_root, "qc", "pairs")         # {src}/{well}/qc/pairs
        stitch_outdir = os.path.join(well_root, "stitch")           # {src}/{well}/{well}/stitch
        results_outdir = os.path.join(well_root, "results")           # {src}/{well}/results
        feat_cache_dir = os.path.join(well_root, "cache")             # tidy cache

        os.makedirs(orig_outdir, exist_ok=True)
        os.makedirs(qc_pairs_dir, exist_ok=True)
        os.makedirs(stitch_outdir, exist_ok=True)
        os.makedirs(results_outdir, exist_ok=True)
        os.makedirs(feat_cache_dir, exist_ok=True)

        prefix = f"{plate_id}_{well}"

        # Pairwise CSV + mosaic outputs
        pairwise_csv = os.path.join(results_outdir, f"{prefix}_pairs.csv")
        mosaic_csv = os.path.join(results_outdir, f"{prefix}_mosaic.csv")

        # Single-channel mosaic (if multichannel=False)
        mosaic_tif_sc = os.path.join(stitch_outdir, f"{prefix}_mosaic_full.tif")
        mosaic_png_sc = os.path.splitext(mosaic_tif_sc)[0] + ".png"

        # Multi-channel mosaic (if multichannel=True)
        mosaic_tif_mc = os.path.join(stitch_outdir, f"{prefix}_mosaic_allc.tif")

        do_mc = bool(do_multichannel)

        if settings.get("write_mosaic", False):
            mosaic_out = None
        else:
            mosaic_out = mosaic_tif_mc if do_mc else mosaic_tif_sc

        # Instantiate stitcher with your settings
        # NOTE: outdir now points at qc/pairs (so qc is not mixed with tiles)
        stitcher = spacrStitcher(
            detector=settings.get("detector", "ORB"),
            nfeatures=int(settings.get("nfeatures", 8000)),
            max_keypoints=settings.get("max_keypoints", 4000),
            downsample=float(settings.get("downsample", 0.5)),
            ransac_thresh_px=float(settings.get("ransac_thresh_px", 3.0)),
            allow_scale=bool(settings.get("allow_scale", False)),
            allow_rotation=bool(settings.get("allow_rotation", False)),
            outline_source=str(settings.get("outline_source", "otsu")),
            canny=tuple(settings.get("canny", (40, 120))),
            blur_sigma=float(settings.get("blur_sigma", 0.0)),
            dilate_ksize=int(settings.get("dilate_ksize", 0)),
            line_thickness=int(settings.get("line_thickness", 1)),
            outline_alpha=float(settings.get("outline_alpha", 1.0)),
            outdir=qc_pairs_dir,
            save_qc=bool(settings.get("save_qc", False)),
            save_stitched_default=bool(settings.get("save_stitched_default", False)),
            all_scores=bool(settings.get("all_scores", False)),
            score_threshold=settings.get("score_threshold", None),
            verbose=bool(settings.get("verbose", True)),
            feature_cache_mode=str(settings.get("feature_cache_mode", "disk")),
            feature_cache_dir=feat_cache_dir,
            max_ram_features=int(settings.get("max_ram_features", 256)),
            n_workers_features=settings.get("n_workers_features", None),
            pair_batch_size=int(settings.get("pair_batch_size", 8192)),
            stream_csv=bool(settings.get("stream_csv", True)),
            opencv_threads=int(settings.get("opencv_threads", 1)),
            arr_axes=str(settings.get("arr_axes", "AUTO")),
            mip=bool(settings.get("mip", True)),
            z_index=int(settings.get("z_index", 0)),
            t_index=int(settings.get("t_index", 0)),
            squeeze_singleton=bool(settings.get("squeeze_singleton", True)),
        )

        # Run per-well; enable mosaic here (single- or multi-channel)
        
        ch_order = settings.get("channel_indices", None)  # None → infer from tiles
        mosaic_min_score = settings.get("mosaic_min_score", None)

        csv_out = stitcher.run_folder(
            folder=scan_dir,
            csv_path=pairwise_csv,
            channel_index=nuc_channel_index,
            exts=exts,
            recursive=False,
            same_well_only=True,
            max_site_gap=int(settings.get("max_site_gap", 64)),
            n_workers=int(settings.get("n_workers", max(1, (os.cpu_count() or 8) // 2))),
            stitch=settings.get("stitch", False),
            score_threshold=settings.get("score_threshold", None),
            meta_regex=meta_re,  # use compiled regex here
            mosaic=settings.get("mosaic", False),
            mosaic_out=mosaic_out,
            mosaic_min_score=mosaic_min_score,
            mosaic_csv_out=mosaic_csv,
            mosaic_all_channels=do_mc,
            mosaic_channel_count=None,  # infer min across tiles unless order provided
            mosaic_channel_index_order=ch_order,
        )

        # ---- Move (or mirror) tiles into {well_root}/{well}/{...} after stitching ----
        moved_tiles: List[str] = []
        for sp in well_files:
            fn = os.path.basename(sp)
            dp = os.path.join(orig_outdir, fn)
            rp = _resolve_collision(dp)
            if rp is None:
                continue

            if not dry_run:
                if do_organize:
                    # move file into orig_outdir
                    if collision == "overwrite" and os.path.exists(dp) and rp == dp:
                        try:
                            os.remove(dp)
                        except FileNotFoundError:
                            pass
                    if os.path.abspath(sp) != os.path.abspath(rp):
                        shutil.move(sp, rp)
                else:
                    # create a symlink into orig_outdir (keep source untouched)
                    target = os.path.realpath(sp)
                    try:
                        if os.path.lexists(rp):
                            os.remove(rp)
                        os.symlink(target, rp)
                    except FileExistsError:
                        pass

            moved_tiles.append(rp if rp is not None else dp)

        # keep return metadata consistent with the new layout
        by_well_outpaths[well] = moved_tiles

        results_by_well[well] = {
            "plate": plate_id,
            "well": well,
            "scan_dir": scan_dir,
            "well_root": well_root,
            "tiles_dir": orig_outdir,
            "tiles": moved_tiles,
            "qc_pairs_dir": qc_pairs_dir,
            "stitch_dir": stitch_outdir,
            "results_dir": results_outdir,
            "cache_dir": feat_cache_dir,
            "pairwise_csv": csv_out,
            "mosaic_csv": mosaic_csv if os.path.exists(mosaic_csv) else None,
            "mosaic_tif": None if do_mc else (mosaic_tif_sc if os.path.exists(mosaic_tif_sc) else None),
            "mosaic_cyx": (mosaic_tif_mc if do_mc and os.path.exists(mosaic_tif_mc) else None),
            "preview_png": (mosaic_png_sc if (not do_mc and os.path.exists(mosaic_png_sc)) else None),
        }
        vprint(f"[stitch] well {well}: pairs→{csv_out}")

    # update organized_summary to reflect the final tile locations
    organized_summary["by_well"] = by_well_outpaths

    return {"organized": organized_summary, "wells": results_by_well}


def get_preprocess_ops_settings(settings):
    
    # high-level sources
    settings.setdefault("phenotype_source", "path")
    settings.setdefault("genotype_source", "path")

    # IO / basic parsing
    settings.setdefault("src", None)
    settings.setdefault("dst_root", None)
    #settings.setdefault("meta_regex",r"(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)\.(?:tif|tiff)$")
    settings.setdefault("meta_regex",r'(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)(?:_[0-9]+)?\.(?:tif|tiff)$')

    
    settings.setdefault("well_group", "well")
    settings.setdefault("exts", [".tif", ".tiff"])

    settings.setdefault("recursive", True)
    settings.setdefault("collision", "rename")   # {'rename','skip','overwrite'}
    settings.setdefault("on_missing", "error")   # {'error','skip'}
    settings.setdefault("dry_run", False)
    settings.setdefault("verbose", True)

    # pipeline toggles
    settings.setdefault("do_organize", True)
    settings.setdefault("do_nuc_stitch", True)
    settings.setdefault("do_multichannel", True)

    # alignment / nuclei channel
    settings.setdefault("channel_index", 0)
    settings.setdefault("relative_scale", 2.0)
    settings.setdefault("qc_outlines", True)

    # --- spacrStitcher(...) core parameters ---
    settings.setdefault("detector", "ORB")
    settings.setdefault("nfeatures", 8000)
    settings.setdefault("max_keypoints", 4000)
    settings.setdefault("downsample", 0.5)
    settings.setdefault("ransac_thresh_px", 3.0)
    settings.setdefault("allow_scale", False)
    settings.setdefault("allow_rotation", False)
    settings.setdefault("score_threshold", 0.001)
    settings.setdefault("all_scores", False)
    settings.setdefault("outline_source", "otsu")
    settings.setdefault("save_qc", False)
    settings.setdefault("save_stitched_default", False)
    settings.setdefault("canny", (40, 120))
    settings.setdefault("blur_sigma", 0.0)
    settings.setdefault("dilate_ksize", 0)
    settings.setdefault("line_thickness", 1)
    settings.setdefault("outline_alpha", 1.0)
    settings.setdefault("feature_cache_mode", "disk")
    settings.setdefault("max_qc_plots_total", 1000)   # hard cap across the whole run
    settings.setdefault("plot_only_above_threshold", True)
    settings.setdefault("feature_cache_dir", None)     # per well
    settings.setdefault("max_ram_features", 256)
    settings.setdefault("n_workers_features", None)
    settings.setdefault("pair_batch_size", 8192)
    settings.setdefault("stream_csv", True)
    settings.setdefault("opencv_threads", 1)
    settings.setdefault("arr_axes", "AUTO")
    settings.setdefault("mip", True)
    settings.setdefault("z_index", 0)
    settings.setdefault("t_index", 0)
    settings.setdefault("squeeze_singleton", True)
    settings.setdefault("write_mosaic", False)

    # --- st.run_folder(...) ---
    settings.setdefault("n_workers", 26)
    settings.setdefault("max_site_gap", 64)
    settings.setdefault("mosaic_min_score", None)      # auto elbow

    # per-well outputs (filled by caller, if desired)
    settings.setdefault("mosaic_out", None)
    settings.setdefault("mosaic_csv_out", None)

    # --- multichannel (CYX mosaic build) ---
    settings.setdefault("channel_indices", None)       # infer from first tile
    settings.setdefault("blend", "max")
    settings.setdefault("preview_downsample", 8)

    # per-well outputs (filled by caller)
    settings.setdefault("tmp_dir", None)
    settings.setdefault("out_tif", None)
    settings.setdefault("out_png", None)

    return settings

class FOVAlignAndCropper:
    """
    Align each image in a folder (arbitrary channels) to a stitched mosaic (arbitrary channels) using
    the Hoechst/nuclei channel, then extract the FOV region from the mosaic at the aligned location.
    For each input FOV, saves:
      - a .npy array with shape (C_fov + C_mosaic, H_fov, W_fov):
          [FOV channels stacked; mosaic channels warped into FOV frame and stacked]
      - a CSV row with file paths, transform, score, and the mosaic-space top-left of the aligned FOV bbox.

    Args
    ----
    detector, nfeatures, max_keypoints, downsample, ransac_thresh_px, allow_scale, allow_rotation, outdir, opencv_threads
        Same semantics as in StitchedMultiAligner.
    arr_axes, mip, z_index, t_index, squeeze_singleton
        Same TIFF axis handling semantics as in StitchedMultiAligner.

    Notes
    -----
    - Alignment is (FOV -> mosaic). Extraction uses the inverse transform to warp the mosaic into the FOV frame,
      guaranteeing the combined array has the FOV’s native (H_fov, W_fov).
    """

    def __init__(self,
                 detector: str = "ORB",
                 nfeatures: int = 6000,
                 max_keypoints: Optional[int] = 2000,
                 downsample: float = 0.5,
                 ransac_thresh_px: float = 3.0,
                 allow_scale: bool = False,
                 allow_rotation: bool = False,
                 outdir: str = "./fov_out",
                 opencv_threads: int = 1,
                 # axes/time/Z
                 arr_axes: str = "AUTO",
                 mip: bool = False,
                 z_index: int = 0,
                 t_index: int = 0,
                 squeeze_singleton: bool = True,
                 folder_image_scale: float = 1.0):
        """
        Parameters
        ----------
        ...
        folder_image_scale : float
            Default known FOV→mosaic scale used by `run()` when not explicitly provided there.
            Examples:
              mosaic 10×, FOV 20×  -> 0.5
              mosaic 20×, FOV 10×  -> 2.0
        """
        self._aligner = StitchedMultiAligner(detector=detector, nfeatures=nfeatures,
                                             max_keypoints=max_keypoints, downsample=downsample,
                                             ransac_thresh_px=ransac_thresh_px,
                                             allow_scale=allow_scale, allow_rotation=allow_rotation,
                                             outdir=outdir, opencv_threads=opencv_threads,
                                             arr_axes=arr_axes, mip=mip, z_index=z_index,
                                             t_index=t_index, squeeze_singleton=squeeze_singleton)
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
    
        # New: default scale to use if run(folder_image_scale=None)
        self.folder_image_scale = float(folder_image_scale) if folder_image_scale and folder_image_scale > 0 else 1.0


    # small proxies for IO helpers
    def _read_plane(self, *a, **k): return self._aligner._read_plane(*a, **k)
    def _read_all_channels_cyx(self, *a, **k): return self._aligner._read_all_channels_cyx(*a, **k)
    def _to_uint8(self, *a, **k): return self._aligner._to_uint8(*a, **k)
    def _detect_and_describe(self, *a, **k): return self._aligner._detect_and_describe(*a, **k)
    def _match(self, *a, **k): return self._aligner._match(*a, **k)
    def _affine_from_pts(self, *a, **k): return self._aligner._affine_from_pts(*a, **k)
    def _closest_rotation(self, *a, **k): return self._aligner._closest_rotation(*a, **k)
    def _edge_zncc(self, *a, **k): return self._aligner._edge_zncc(*a, **k)

    @staticmethod
    def _list_tifs(folder: str, recursive: bool, exts: Tuple[str, ...]) -> List[str]:
        exts = tuple(e.lower() for e in exts)
        out = []
        if recursive:
            for root, _, files in os.walk(folder):
                for fn in files:
                    if fn.lower().endswith(exts):
                        out.append(os.path.join(root, fn))
        else:
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(exts):
                    out.append(os.path.join(folder, fn))
        return out

    @staticmethod
    def _affine_to_3x3(M2x3: np.ndarray) -> np.ndarray:
        A = np.eye(3, dtype=np.float32); A[:2, :3] = M2x3.astype(np.float32)
        return A

    @staticmethod
    def _invert_affine(M: np.ndarray) -> np.ndarray:
        A = M[:,:2]
        t = M[:,2:]
        Ai = np.linalg.inv(A + 1e-12*np.eye(2, dtype=np.float32))
        ti = -Ai @ t
        Mi = np.zeros((2,3), dtype=np.float32)
        Mi[:,:2] = Ai.astype(np.float32)
        Mi[:,2:] = ti.astype(np.float32)
        return Mi

    def run(self,
            stitched_path: str,
            folder: str,
            *,
            stitched_nuclei_idx: int = 0,
            fov_nuclei_idx: int = 0,
            exts: Tuple[str, ...] = (".tif", ".tiff"),
            recursive: bool = False,
            csv_path: Optional[str] = None,
            npy_dir: Optional[str] = None,
            folder_image_scale: Optional[float] = None) -> str:
        """
        Align each FOV in `folder` to the stitched mosaic at `stitched_path` using the nuclei channel,
        then save a stacked array [FOV channels; mosaic channels warped into FOV] to .npy and a CSV row.
    
        Known scale handling:
          - If the FOV magnification differs from the mosaic, set `folder_image_scale` to the
            FOV→mosaic pixel-scale factor (e.g., mosaic 10× vs FOV 20× -> 0.5; mosaic 20× vs FOV 10× -> 2.0).
    
        Returns
        -------
        str
            Path to the CSV manifest.
        """
        # Outputs
        if csv_path is None:
            csv_path = os.path.join(self.outdir, "fov_align_manifest.csv")
        if npy_dir is None:
            npy_dir = os.path.join(self.outdir, "npy")
        os.makedirs(npy_dir, exist_ok=True)
    
        # Load mosaic (all channels) and nuclei for features
        mosa_all = self._read_all_channels_cyx(stitched_path)   # (C_m, Hm, Wm)
        mosa_nuc = mosa_all[int(stitched_nuclei_idx)]
        Hm, Wm = mosa_nuc.shape
    
        # Feature DS for mosaic
        s = float(self._aligner.downsample)
        Hmds = max(1, int(round(Hm * s)))
        Wmds = max(1, int(round(Wm * s)))
        mosa_ds = cv2.resize(mosa_nuc, (Wmds, Hmds), interpolation=cv2.INTER_LINEAR)
        mosa_u8 = self._to_uint8(mosa_ds)
        ref_kp, ref_desc = self._detect_and_describe(mosa_u8)
        Fref = {"pts": ref_kp, "desc": ref_desc}
    
        # Known FOV→mosaic scale
        s_known = (float(folder_image_scale) if folder_image_scale is not None
                   else float(getattr(self, "folder_image_scale", 1.0)))
        if not (s_known > 0):
            s_known = 1.0
    
        files = self._list_tifs(folder, recursive, exts)
    
        with open(csv_path, "w", newline="") as fcsv:
            fieldnames = [
                "fov_path", "stitched_path",
                "tx", "ty", "theta_deg", "scale",
                "score", "inlier_ratio",
                "mosaic_x0", "mosaic_y0",
                "npy_path"
            ]
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader()
    
            for p in files:
                try:
                    # --- FOV nuclei (full-res) ---
                    fov_nuc = self._read_plane(p, ch=int(fov_nuclei_idx))
                    Hf, Wf = fov_nuc.shape
    
                    # DS for FOV features *including known scale*
                    Wfds = max(1, int(round(Wf * s * s_known)))   # FIXED name
                    Hfds = max(1, int(round(Hf * s * s_known)))   # FIXED name
                    fov_ds = cv2.resize(fov_nuc, (Wfds, Hfds), interpolation=cv2.INTER_LINEAR)  # FIXED usage
                    fov_u8 = self._to_uint8(fov_ds)
    
                    kp, desc = self._detect_and_describe(fov_u8)
                    Fb = {"pts": kp, "desc": desc}
    
                    # Match (A=mosaic_ds, B=fov_ds)
                    ptsA, ptsB = self._match(Fref, Fb)
                    if ptsA.shape[0] < 4:
                        continue
    
                    M_ds, inmask, inlier_ratio = self._affine_from_pts(ptsA, ptsB, self._aligner.ransac_thresh_px)
                    if M_ds is None:
                        continue
    
                    # Optional constraints (keep known scale separate from "disallowed scale")
                    A_lin = M_ds[:, :2].astype(np.float32)
                    pA = ptsA[inmask] if (inmask is not None and inmask.any()) else ptsA
                    pB = ptsB[inmask] if (inmask is not None and inmask.any()) else ptsB
    
                    if not self._aligner.allow_scale and not self._aligner.allow_rotation:
                        R = np.eye(2, dtype=np.float32)
                        t_mean = (pA - pB).mean(axis=0).astype(np.float32)
                        M_ds = np.zeros((2, 3), np.float32); M_ds[:, :2] = R; M_ds[:, 2] = t_mean
                    elif (not self._aligner.allow_scale) and self._aligner.allow_rotation:
                        R = self._closest_rotation(A_lin)
                        t_mean = (pA - (pB @ R.T)).mean(axis=0).astype(np.float32)
                        M_ds = np.zeros((2, 3), np.float32); M_ds[:, :2] = R; M_ds[:, 2] = t_mean
    
                    # ---- Lift DS → full-res with known scale ----
                    # DS relation:
                    #   x_mosa_ds = A_ds * x_fov_ds + t_ds,
                    #   x_fov_ds  = s*s_known*x_fov_full,  x_mosa_ds = s*x_mosa_full
                    # ⇒ x_mosa_full = A_ds*(s_known)*x_fov_full + t_ds/s
                    # ⇒ A_full = s_known * A_ds ; t_full = t_ds / s
                    M_full = M_ds.astype(np.float32).copy()
                    M_full[:2, :2] *= float(s_known)
                    if s != 0:
                        M_full[0, 2] /= float(s)
                        M_full[1, 2] /= float(s)
    
                    # Decompose
                    a, b, tx = float(M_full[0, 0]), float(M_full[0, 1]), float(M_full[0, 2])
                    c, d, ty = float(M_full[1, 0]), float(M_full[1, 1]), float(M_full[1, 2])
                    scale = float(np.sqrt(max(1e-12, (a * d - b * c))))
                    theta = float(np.degrees(np.arctan2(c, a)))
    
                    # Score on DS (foreground of mosaic nuclei)
                    B_warp_ds = cv2.warpAffine(fov_u8, M_ds, (Wmds, Hmds),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    _, th = cv2.threshold(mosa_u8, 0, 255, cv2.THRESH_OTSU)
                    mA = (mosa_u8 >= th)
                    score = float(self._edge_zncc(mosa_u8.astype(np.float32),
                                                  B_warp_ds.astype(np.float32),
                                                  mask=mA)) * float(inlier_ratio)
    
                    # Compute FOV bbox top-left in mosaic coords (using M_full)
                    corners = np.array([[0, 0], [Wf, 0], [0, Hf], [Wf, Hf]], dtype=np.float32).reshape(-1, 1, 2)
                    C = cv2.transform(corners, M_full).reshape(-1, 2)
                    mosaic_x0 = float(np.min(C[:, 0]))
                    mosaic_y0 = float(np.min(C[:, 1]))
    
                    # Read full FOV & Mosaic channels and build stacked output:
                    #   [FOV channels; mosaic channels warped into FOV frame]
                    fov_all = self._read_all_channels_cyx(p)         # (C_f, Hf, Wf), float32
                    mosa_all_full = mosa_all                         # (C_m, Hm, Wm), already loaded
    
                    # Inverse transform (mosaic -> FOV)
                    M_inv = self._invert_affine(M_full)
    
                    # Warp mosaic channels into FOV frame
                    C_f = fov_all.shape[0]
                    C_m = mosa_all_full.shape[0]
                    out = np.zeros((C_f + C_m, Hf, Wf), np.float32)
                    out[:C_f] = fov_all
    
                    for ci in range(C_m):
                        warped = cv2.warpAffine(mosa_all_full[ci], M_inv, (Wf, Hf),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        out[C_f + ci] = warped
    
                    # Save npy
                    base = os.path.splitext(os.path.basename(p))[0]
                    npy_path = os.path.join(npy_dir, f"{base}__with_mosaic.npy")
                    np.save(npy_path, out)
    
                    # Write CSV row
                    w.writerow(dict(
                        fov_path=p,
                        stitched_path=stitched_path,
                        tx=tx, ty=ty, theta_deg=theta, scale=scale,
                        score=score, inlier_ratio=float(inlier_ratio),
                        mosaic_x0=mosaic_x0, mosaic_y0=mosaic_y0,
                        npy_path=npy_path
                    ))
    
                except Exception as e:
                    # Silent skip unless you want verbose logging:
                    # print(f"[FOVAlignAndCropper.run] Skipping {os.path.basename(p)}: {e}")
                    continue
    
        return csv_path

def align_image_to_stitch(
    stitch_dst_root: str,
    align_src: str,
    *,
    meta_regex: str = r"(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)\.(?:tif|tiff)$",
    well_group: str = "well",
    channel_index: int = 0,
    relative_scale: float = 2.0,   # 20× vs 10× → ~2.0; adjust as needed
    downsample: float = 0.5,
    nfeatures: int = 4000,
    ransac_thresh_px: float = 3.0,
    allow_scale: bool = False,
    allow_rotation: bool = False,
    qc_outlines: bool = False,
    recursive_align_src: bool = True,
    exts: tuple = (".tif", ".tiff")
) -> Dict[str, Dict[str, str]]:
    """
    For each WELL that already has a stitched mosaic under stitch_dst_root/<WELL>/_stitch/mosaic_allc.tif,
    align all 20× images from align_src (grouped by WELL via meta_regex) to that mosaic using
    FOVAlignAndCropper, and write per-well crop manifests and .npy crops.

    Returns:
      { WELL: {"mosaic": <path>, "align_folder": <per-well link>, "manifest_csv": <path>} }
    """
    import os, re, shutil
    from typing import Dict, List, Optional

    # ---------- helpers ----------
    def _scan_tifs(root: str, recursive: bool, exts: tuple) -> List[str]:
        out = []
        if recursive:
            for r, _, fs in os.walk(root):
                for fn in fs:
                    if fn.lower().endswith(exts):
                        out.append(os.path.join(r, fn))
        else:
            for fn in sorted(os.listdir(root)):
                p = os.path.join(root, fn)
                if os.path.isfile(p) and fn.lower().endswith(exts):
                    out.append(p)
        return out

    def _group_by_well(paths: List[str], meta_re: re.Pattern, well_group: str) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for p in paths:
            m = meta_re.search(os.path.basename(p))
            if not m:
                continue
            w = (m.groupdict().get(well_group) or "").upper()
            if not w:
                continue
            buckets.setdefault(w, []).append(p)
        # sort per site if present
        def _site_key(p):
            m = re.search(r"(?:Site|Field|FOV)[-_]?(\d+)", os.path.basename(p), re.IGNORECASE)
            return int(m.group(1)) if m else 10**9
        for w in list(buckets.keys()):
            buckets[w].sort(key=lambda p: (_site_key(p), p))
        return buckets

    def _symlink_list(files: List[str], target_dir: str) -> List[str]:
        os.makedirs(target_dir, exist_ok=True)
        made = []
        for sp in files:
            fn = os.path.basename(sp)
            dp = os.path.join(target_dir, fn)
            base, ext = os.path.splitext(dp)
            k = 1
            while os.path.lexists(dp):
                k += 1
                dp = f"{base}_{k:03d}{ext}"
            try:
                os.symlink(sp, dp)
            except OSError:
                shutil.copy2(sp, dp)
            made.append(dp)
        return made

    # ---------- 1) find per-well mosaics built by stitch_cycle_wells ----------
    # Expected location from your pipeline: <stitch_dst_root>/<WELL>/_stitch/mosaic_allc.tif
    wells_with_mosaic: Dict[str, str] = {}
    if not os.path.isdir(stitch_dst_root):
        raise ValueError(f"stitch_dst_root does not exist: {stitch_dst_root}")
    for entry in sorted(os.listdir(stitch_dst_root)):
        well_dir = os.path.join(stitch_dst_root, entry)
        if not os.path.isdir(well_dir):
            continue
        mpath = os.path.join(well_dir, "_stitch", "mosaic_allc.tif")
        if os.path.isfile(mpath):
            wells_with_mosaic[entry.upper()] = mpath

    # ---------- 2) group 20× (align) images by well ----------
    meta_re = re.compile(meta_regex, re.IGNORECASE)
    align_files = _scan_tifs(align_src, recursive_align_src, exts)
    by_well_align = _group_by_well(align_files, meta_re, well_group)

    # ---------- 3) per-well FOV→mosaic alignment via FOVAlignAndCropper ----------
    results: Dict[str, Dict[str, str]] = {}
    links_root = os.path.join(stitch_dst_root, "_links", "align20x")
    os.makedirs(links_root, exist_ok=True)

    # Instantiate aligner (matches your class' default knobs)
    aligner = FOVAlignAndCropper(
        relative_scale=relative_scale,
        downsample=downsample,
        nfeatures=nfeatures,
        ransac_thresh_px=ransac_thresh_px,
        allow_scale=allow_scale,
        allow_rotation=allow_rotation,
    )

    for well, mosaic_path in sorted(wells_with_mosaic.items()):
        if well not in by_well_align:
            continue  # no 20× images for this well
        well_align_srcs = by_well_align[well]
        # make a light per-well link folder so paths are clean/reproducible
        link_well = os.path.join(links_root, well)
        link_paths = _symlink_list(well_align_srcs, link_well)

        crops_dir = os.path.join(os.path.dirname(mosaic_path), "crops_20x")
        os.makedirs(crops_dir, exist_ok=True)

        manifest_csv = aligner.run(
            folder=link_well,
            mosaic_tif=mosaic_path,
            outdir=crops_dir,
            channel_index=channel_index,
            qc_outlines=qc_outlines,
            meta_regex=meta_regex
        )

        results[well] = {
            "mosaic": mosaic_path,
            "align_folder": link_well,
            "manifest_csv": manifest_csv
        }

    return results

def ops_preprocess(settings):

    import os
    import numpy as np  # noqa: F401  (likely used when you add npy writing)
    import pandas as pd  # noqa: F401  (keep if you use it later)
    from tifffile import imread  # noqa: F401

    # Fill in defaults for all stitching / alignment-related keys
    settings = get_preprocess_ops_settings(settings)

    phenotype_src = settings["phenotype_source"]
    genotype_src = settings["genotype_source"]

    # ---- Normalize phenotype_src ----
    if not isinstance(phenotype_src, (str, os.PathLike)):
        raise ValueError("settings['phenotype_source'] must be a path to a folder.")
    phenotype_src = str(phenotype_src)

    # Where to store npy outputs (you can change this if you like)
    npy_out_root = os.path.join(phenotype_src, "output")
    os.makedirs(npy_out_root, exist_ok=True)

    # ---- Normalize genotype_src into a list of folders ----
    if isinstance(genotype_src, (str, os.PathLike)):
        genotype_src = str(genotype_src)
        # List subdirectories; if none, treat the folder itself as one genotype
        subdirs = [
            os.path.join(genotype_src, d)
            for d in os.listdir(genotype_src)
            if os.path.isdir(os.path.join(genotype_src, d))
        ]
        if subdirs:
            genotype_folders = subdirs
        else:
            genotype_folders = [genotype_src]
    elif isinstance(genotype_src, (list, tuple)):
        genotype_folders = [str(g) for g in genotype_src]
    else:
        raise ValueError(
            "settings['genotype_source'] must be a path or a list/tuple of paths."
        )

    stitch_summaries = []
    align_results = []

    for geno_fldr in genotype_folders:
        # ---- 1) per-genotype stitching ----
        stitch_settings = dict(settings)  # shallow copy is fine for simple values
        stitch_settings["src"] = geno_fldr
        if stitch_settings.get("dst_root") is None:
            stitch_settings["dst_root"] = geno_fldr

        summary = stitch_cycle_wells(stitch_settings)
        stitch_summaries.append({
            "genotype_folder": geno_fldr,
            "summary": summary,
        })

        # ---- 2) alignment of phenotype images to stitched mosaics ----
        # Only run if align_image_to_stitch is available in this module.
        if "align_image_to_stitch" in globals():
            dst_root = stitch_settings["dst_root"]
            ar = align_image_to_stitch(
                stitch_dst_root=dst_root,
                align_src=phenotype_src,
                meta_regex=stitch_settings["meta_regex"],
                channel_index=stitch_settings["channel_index"],
                relative_scale=stitch_settings["relative_scale"],
                qc_outlines=stitch_settings["qc_outlines"],
            )
            align_results.append({
                "genotype_folder": geno_fldr,
                "align": ar,
            })
            
    return {
        "stitch": stitch_summaries,
        "align": align_results,
        "npy_out_root": npy_out_root,
    }
