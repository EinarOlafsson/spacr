# spacrcluster.py
# End-to-end: SQL -> CNN embeddings -> (optional Graph Transformer refinement) ->
# Supervised phenotype scoring and/or Unsupervised clustering ->
# Gene-by-phenotype/cluster enrichment & gene clustering ->
# UMAP coords + cluster sprites (image tiles) linked back to png_list.

from __future__ import annotations
import os, io, math, json, sqlite3, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# ------------------------- Config -------------------------

@dataclass
class PCConfig:
    db_path: str = "/path/to/your.sqlite"
    table: str = "png_list"
    img_col: str = "img_path"
    gene_col: str = "gene"
    grna_col: str = "grna"          # optional
    plate_col: str = "plate"        # strongly recommended for stratification
    pheno_cols: List[str] = None    # e.g. ["phenotype_A","phenotype_B"]; set [] or None to skip supervised

    out_dir: str = "./pspacrcluster_out"

    # Featurization
    img_size: int = 224
    batch_size_img: int = 256
    num_workers: int = 8
    seed: int = 1337

    # Supervised head
    epochs_supervised: int = 1
    lr_supervised: float = 2e-3
    head_hidden: int = 1024
    head_dropout: float = 0.2

    # Unsupervised
    unsup_k: int = 50               # clusters for cells
    kmeans_batch: int = 10000

    # Graph refiner (optional, needs torch_geometric)
    use_graph_refiner: bool = False
    knn_k: int = 15
    gt_hidden: int = 512
    gt_emb: int = 256
    gt_heads: int = 4
    gt_layers: int = 4
    gt_dropout: float = 0.2
    gt_lr: float = 3e-4
    gt_wd: float = 0.05
    gt_epochs: int = 15

    # UMAP/TSNE sampling for 2D viz
    sample_cells_2d: int = 200_000  # subsample for cell-level UMAP
    use_umap: bool = True

    # Output sprites
    sprites_per_cluster: int = 100  # max images per cluster sprite
    sprite_thumb: int = 64          # px per tile

# ------------------------- Utils -------------------------

def set_seed(s: int = 1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def table_columns(db_path: str, table: str) -> List[str]:
    with sqlite3.connect(db_path, timeout=60) as conn:
        return pd.read_sql_query(f'PRAGMA table_info({table})', conn)['name'].tolist()

def stream_sql(db_path: str, table: str, columns: List[str], chunksize: int = 50_000) -> Iterable[pd.DataFrame]:
    q = f'SELECT {", ".join(columns)} FROM {table}'
    with sqlite3.connect(db_path, timeout=120) as conn:
        for chunk in pd.read_sql_query(q, conn, chunksize=chunksize):
            yield chunk

def filter_existing_paths(df: pd.DataFrame, img_col: str) -> pd.DataFrame:
    m = df[img_col].apply(lambda p: Path(p).is_file())
    if (~m).any():
        print(f"[warn] dropping {int((~m).sum())} rows with missing files")
    return df[m].reset_index(drop=True)

def reservoir_sample_indices(total_n: int, stream_counts: List[int], k: int, seed: int=1337) -> List[int]:
    rng = np.random.default_rng(seed)
    res, seen, offset = [], 0, 0
    for n in stream_counts:
        for i in range(n):
            seen += 1
            if len(res) < k:
                res.append(offset + i)
            else:
                j = rng.integers(0, seen)
                if j < k: res[j] = offset + i
        offset += n
    return sorted(res)

# ------------------------- Data -------------------------

class ImageDataset(Dataset):
    """Loads RGB PNGs by path (fast, minimal)."""
    def __init__(self, paths: List[str], img_size: int):
        self.paths = paths; self.img_size = img_size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.img_size, self.img_size))
        x = torch.from_numpy(np.asarray(img)).permute(2,0,1).float()/255.0
        return x, i

class ResNet50Featurizer(nn.Module):
    """ResNet50 backbone -> 2048-d global pooled features (L2-normalized outside)."""
    def __init__(self):
        super().__init__()
        import torchvision
        try:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            weights = None
        m = torchvision.models.resnet50(weights=weights)
        self.body = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = m.fc.in_features  # 2048
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)                      # (B, 2048, 1, 1)
        return f.view(f.size(0), -1)          # (B, 2048)

def compute_embeddings(paths: List[str], img_size: int, batch_size: int, num_workers: int, device: torch.device) -> np.ndarray:
    ds = ImageDataset(paths, img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    net = ResNet50Featurizer().to(device).eval()
    X = np.zeros((len(paths), net.out_dim), dtype=np.float32)
    with torch.no_grad():
        for xb, idx in dl:
            fb = net(xb.to(device, non_blocking=True)).detach().cpu().numpy().astype(np.float32)
            X[idx.numpy()] = fb
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X

# ------------------------- Graph Transformer (optional) -------------------------

def knn_graph(feats: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")
    nn.fit(feats)
    dists, nbrs = nn.kneighbors(feats, return_distance=True)
    nbrs = nbrs[:,1:]; dists = dists[:,1:]
    sims = 1.0 - dists
    src = np.repeat(np.arange(feats.shape[0]), k); dst = nbrs.reshape(-1)
    sim = sims.reshape(-1)
    e = np.stack([src, dst], 1); er = np.stack([dst, src], 1)
    es = np.concatenate([e, er], 0); ws = np.concatenate([sim, sim], 0)
    order = np.lexsort((es[:,1], es[:,0])); es, ws = es[order], ws[order]
    keep = np.ones(len(es), dtype=bool); keep[1:] = np.any(es[1:] != es[:-1], axis=1)
    return es[keep], ws[keep]

class GraphRefiner:
    """Refine embeddings via a Graph Transformer trained to predict gene labels on a kNN graph."""
    def __init__(self, hidden=512, out_emb=256, heads=4, layers=4, dropout=0.2):
        try:
            from torch_geometric.nn import TransformerConv, LayerNorm  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
        self.hidden = hidden; self.out_emb = out_emb
        self.heads = heads; self.layers = layers; self.dropout = dropout

    def fit_transform(self, feats: np.ndarray, labels: np.ndarray, edge_index: np.ndarray,
                      edge_weight: np.ndarray, epochs: int, lr: float, wd: float, device: torch.device) -> np.ndarray:
        if not self.available:
            print("[info] torch_geometric not installed; skipping graph refinement.")
            return feats

        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import TransformerConv, LayerNorm

        x = torch.from_numpy(feats).float()
        y = torch.from_numpy(labels).long()
        ei = torch.from_numpy(edge_index.T).long()
        ea = torch.from_numpy(edge_weight.reshape(-1,1)).float()

        data = Data(x=x, y=y, edge_index=ei, edge_attr=ea)
        n_in = feats.shape[1]; n_classes = int(labels.max()+1)

        class GT(nn.Module):
            def __init__(self, in_dim, hidden, out_emb, heads, layers, edge_dim, dropout, n_classes):
                super().__init__()
                self.proj = nn.Linear(in_dim, hidden)
                self.blocks = nn.ModuleList()
                self.norms  = nn.ModuleList()
                for _ in range(layers):
                    self.blocks.append(
                        TransformerConv(hidden, hidden // heads, heads=heads,
                                        dropout=dropout, edge_dim=edge_dim)
                    )
                    self.norms.append(LayerNorm(hidden, affine=True))
                self.drop = nn.Dropout(dropout)
                self.head_emb = nn.Linear(hidden, out_emb)
                self.head_cls = nn.Linear(out_emb, n_classes)
            def forward(self, x, ei, ea):
                h = F.gelu(self.proj(x))
                for conv, ln in zip(self.blocks, self.norms):
                    h = ln(F.gelu(conv(h, ei, ea))) + h
                    h = self.drop(h)
                z = F.normalize(self.head_emb(h), dim=1)
                logits = self.head_cls(z)
                return logits, z

        model = GT(n_in, self.hidden, self.out_emb, self.heads, self.layers, edge_dim=1,
                   dropout=self.dropout, n_classes=n_classes).to(device)
        w = class_weights(labels, n_classes).to(device)
        ce = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        model.train()
        for ep in range(1, epochs+1):
            logits, _ = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
            loss = ce(logits, data.y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            if ep % 5 == 0 or ep == 1:
                with torch.no_grad():
                    preds = logits.argmax(1).cpu().numpy()
                    acc = (preds == labels).mean()
                print(f"[graph] epoch {ep}/{epochs}  loss={loss.item():.4f} acc={acc:.3f}")

        model.eval()
        with torch.no_grad():
            _, z = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
        return z.cpu().numpy()

# ------------------------- Supervised phenotype head -------------------------

class PhenoHead(nn.Module):
    """Two-layer MLP for multi-label phenotype prediction from embeddings."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int=1024, dropout: float=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    beta = 0.9999
    eff = 1.0 - np.power(beta, cnt)
    w = (1.0 - beta) / np.maximum(eff, 1e-8)
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)

# ------------------------- Enrichment & Clustering -------------------------

def cmh_enrichment_soft(df: pd.DataFrame, gene_col: str, plate_col: str,
                        score_col: str) -> pd.DataFrame:
    """CMH across plates for gene vs all-others using soft counts in score_col (0..1)."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    from statsmodels.stats.multitest import multipletests
    res = []
    genes = df[gene_col].unique().tolist()
    for g in genes:
        tables = []
        for plate, d in df.groupby(plate_col):
            a = d.loc[d[gene_col]==g, score_col].sum()
            n_g = (d[gene_col]==g).sum()
            c = d.loc[d[gene_col]!=g, score_col].sum()
            n_not = (d[gene_col]!=g).sum()
            b = n_g - a; d_not = n_not - c
            if (n_g>0) and (n_not>0):
                tables.append(np.array([[a,b],[c,d_not]], float))
        if not tables: continue
        st = StratifiedTable(tables)
        or_mh = st.oddsratio_pooled
        lcl,ucl = st.oddsratio_pooled_confint()
        pval = st.test_null_odds().pvalue
        delta = (df.loc[df[gene_col]==g, score_col].mean()
                 - df.loc[df[gene_col]!=g, score_col].mean())
        res.append((g, or_mh, lcl, ucl, delta, pval))
    out = pd.DataFrame(res, columns=["gene","OR_MH","CI_low","CI_high","Delta","pval"])
    out["qval"] = multipletests(out["pval"], method="fdr_bh")[1]
    return out.sort_values(["qval","OR_MH"], ascending=[True,False])

def gene_prototypes_from_cells(emb: np.ndarray, genes: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    uniq = np.sort(pd.unique(genes))
    g2i = {g:i for i,g in enumerate(uniq)}
    P = np.zeros((len(uniq), emb.shape[1]), dtype=np.float32)
    c = np.zeros(len(uniq), dtype=np.int64)
    for gi, e in zip(genes, emb):
        i = g2i[gi]; P[i] += e; c[i] += 1
    P /= (c.reshape(-1,1) + 1e-9)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    return P, list(uniq)

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T

def cluster_genes_from_profiles(M: np.ndarray, n_clusters: int=20) -> np.ndarray:
    """Cluster genes by their phenotype/cluster enrichment profiles."""
    Mz = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    lab = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average").fit_predict(Mz)
    return lab

# ------------------------- Visualization -------------------------

def reduce_2d(X: np.ndarray, use_umap: bool=True, seed: int=1337) -> np.ndarray:
    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=seed)
            return reducer.fit_transform(X)
        except Exception as e:
            print(f"[info] umap not available ({e}); falling back to TSNE")
    return TSNE(n_components=2, init="random", perplexity=30, metric="cosine", random_state=seed).fit_transform(X)

def scatter_png(xy: np.ndarray, color: np.ndarray, out_png: str, title: Optional[str]=None, s: float=1.0, alpha: float=0.7):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,8), dpi=200)
    sc = plt.scatter(xy[:,0], xy[:,1], c=color, s=s, alpha=alpha)
    if title: plt.title(title)
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def make_sprite(paths: List[str], out_png: str, thumb: int=64, max_imgs: int=100):
    """Save a grid sprite of up to max_imgs images."""
    sel = paths[:max_imgs]
    n = len(sel)
    cols = int(math.ceil(math.sqrt(n))); rows = int(math.ceil(n/cols))
    canvas = Image.new("RGB", (cols*thumb, rows*thumb), (0,0,0))
    for i,p in enumerate(sel):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
            canvas.paste(im, ((i%cols)*thumb, (i//cols)*thumb))
        except Exception:
            continue
    canvas.save(out_png)

# ------------------------- Pipeline -------------------------

class PhenoCluster:
    """Main orchestrator. Call run_supervised/run_unsupervised and then umap/sprites as needed."""
    def __init__(self, cfg: PCConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        ensure_dir(cfg.out_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Embeddings (streamed) ----------

    def _iter_stream_with_paths(self, cols: List[str], chunksize: int=50_000):
        for chunk in stream_sql(self.cfg.db_path, self.cfg.table, cols, chunksize):
            chunk = filter_existing_paths(chunk, self.cfg.img_col)
            if len(chunk): yield chunk

    def embed_all(self, save_parquet: bool=False) -> Tuple[np.ndarray, pd.DataFrame]:
        """Embed all cells. Returns (embeddings, metadata_df) in RAM; can be huge.
        Prefer using streamed methods for kmeans/heads. Use when you really need all embeddings."""
        cols = [self.cfg.img_col, self.cfg.gene_col, self.cfg.plate_col]
        if self.cfg.grna_col in table_columns(self.cfg.db_path, self.cfg.table):
            cols.append(self.cfg.grna_col)
        paths, metas = [], []
        for chunk in self._iter_stream_with_paths(cols):
            paths.extend(chunk[self.cfg.img_col].tolist())
            metas.append(chunk.drop(columns=[self.cfg.img_col]).copy())
        if not paths:
            raise RuntimeError("No images found.")
        X = compute_embeddings(paths, self.cfg.img_size, self.cfg.batch_size_img, self.cfg.num_workers, self.device)
        meta = pd.concat(metas, ignore_index=True)
        meta.insert(0, self.cfg.img_col, paths)
        if save_parquet:
            meta.to_parquet(Path(self.cfg.out_dir, "all_cells_meta.parquet"), index=False)
            np.save(Path(self.cfg.out_dir, "all_cells_embeddings.npy"), X)
        return X, meta

    # ---------- Graph refinement (optional) ----------

    def maybe_refine_with_graph(self, X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        if not self.cfg.use_graph_refiner:
            return X
        print("[graph] building kNN graph...")
        E, W = knn_graph(X, self.cfg.knn_k)
        labels = meta[self.cfg.gene_col].astype("category").cat.codes.to_numpy()
        refiner = GraphRefiner(self.cfg.gt_hidden, self.cfg.gt_emb, self.cfg.gt_heads,
                               self.cfg.gt_layers, self.cfg.gt_dropout)
        Z = refiner.fit_transform(X, labels, E, W, epochs=self.cfg.gt_epochs,
                                  lr=self.cfg.gt_lr, wd=self.cfg.gt_wd, device=self.device)
        return Z

    # ---------- Supervised path ----------

    def run_supervised(self) -> Dict[str, str]:
        """Train multi-label phenotype head on embeddings (streamed), score all cells,
        run CMH per (gene, phenotype), and save outputs. Returns dict of output paths."""
        if not self.cfg.pheno_cols:
            raise ValueError("pheno_cols is empty; set Config.pheno_cols for supervised mode.")

        out = {}
        ph_cols = self.cfg.pheno_cols
        cols_train = [self.cfg.img_col, self.cfg.gene_col, self.cfg.plate_col] + ph_cols

        # 1) Estimate phenotype prevalence for pos_weight
        pos_sum = np.zeros(len(ph_cols), float); n_seen = 0
        for chunk in self._iter_stream_with_paths(cols_train):
            y = chunk[ph_cols].astype(float).clip(0,1).values
            pos_sum += y.sum(axis=0); n_seen += y.shape[0]
        pos_rate = np.maximum(pos_sum / max(n_seen,1), 1e-6)
        pos_weight = torch.from_numpy((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6)).float().to(self.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 2) Init head
        # Embed dim from a small probe
        probe_paths = []
        for chunk in self._iter_stream_with_paths([self.cfg.img_col]): 
            probe_paths = chunk[self.cfg.img_col].tolist()[:8]; break
        in_dim = compute_embeddings(probe_paths, self.cfg.img_size, 8, self.cfg.num_workers, self.device).shape[1]
        head = PhenoHead(in_dim, len(ph_cols), self.cfg.head_hidden, self.cfg.head_dropout).to(self.device)
        opt = torch.optim.AdamW(head.parameters(), lr=self.cfg.lr_supervised, weight_decay=0.05)

        # 3) Epochs over stream
        for ep in range(1, self.cfg.epochs_supervised+1):
            rows = 0
            for chunk in self._iter_stream_with_paths(cols_train):
                X = compute_embeddings(chunk[self.cfg.img_col].tolist(), self.cfg.img_size,
                                       self.cfg.batch_size_img, self.cfg.num_workers, self.device)
                y = chunk[ph_cols].astype(float).clip(0,1).values
                bs = 4096
                for s in range(0, X.shape[0], bs):
                    e = min(X.shape[0], s+bs)
                    xb = torch.from_numpy(X[s:e]).to(self.device, non_blocking=True)
                    yb = torch.from_numpy(y[s:e]).to(self.device, non_blocking=True).float()
                    logits = head(xb); loss = bce(logits, yb)
                    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 3.0); opt.step()
                rows += X.shape[0]
            print(f"[supervised] epoch {ep} seen={rows}")

        # 4) Score all cells, collect skinny DF
        scored_parts = []
        cols_skinny = [self.cfg.img_col, self.cfg.gene_col, self.cfg.plate_col]
        if self.cfg.grna_col in table_columns(self.cfg.db_path, self.cfg.table):
            cols_skinny.append(self.cfg.grna_col)
        for chunk in self._iter_stream_with_paths(cols_skinny):
            X = compute_embeddings(chunk[self.cfg.img_col].tolist(), self.cfg.img_size,
                                   self.cfg.batch_size_img, self.cfg.num_workers, self.device)
            with torch.no_grad():
                pr = torch.sigmoid(head(torch.from_numpy(X).to(self.device))).cpu().numpy()
            base = chunk.copy()
            for j, ph in enumerate(ph_cols):
                base[ph] = pr[:, j]
            scored_parts.append(base)
        scored = pd.concat(scored_parts, ignore_index=True)
        p_scored = str(Path(self.cfg.out_dir, "supervised_cell_scores.parquet"))
        scored.to_parquet(p_scored, index=False); out["supervised_cell_scores"] = p_scored

        # 5) CMH per phenotype
        cmh_all = []
        for ph in ph_cols:
            cmh = cmh_enrichment_soft(scored[[self.cfg.gene_col, self.cfg.plate_col, ph]].rename(columns={ph:"score"}),
                                      self.cfg.gene_col, self.cfg.plate_col, "score")
            cmh.insert(1, "phenotype", ph); cmh_all.append(cmh)
        cmh_df = pd.concat(cmh_all, ignore_index=True)
        p_cmh = str(Path(self.cfg.out_dir, "gene_phenotype_cmh_supervised.csv"))
        cmh_df.to_csv(p_cmh, index=False); out["gene_phenotype_cmh_supervised"] = p_cmh

        # 6) Gene-by-phenotype profile & clustering
        gp = (scored.groupby(self.cfg.gene_col)[ph_cols].mean()).reset_index()
        gp_m = gp[ph_cols].to_numpy()
        gp["gene_cluster"] = cluster_genes_from_profiles(gp_m, n_clusters=min(20, max(2, gp_m.shape[1]//2)))
        p_gp = str(Path(self.cfg.out_dir, "gene_by_phenotype_profile.csv"))
        p_gc = str(Path(self.cfg.out_dir, "gene_clusters_by_phenotype.csv"))
        gp.to_csv(p_gp, index=False); gp[["gene", "gene_cluster"]].to_csv(p_gc, index=False)
        out["gene_by_phenotype_profile"] = p_gp
        out["gene_clusters_by_phenotype"] = p_gc

        return out

    # ---------- Unsupervised path ----------

    def run_unsupervised(self) -> Dict[str, str]:
        """Cluster cells with MiniBatchKMeans, save per-cell clusters, gene enrichment,
        gene prototypes/similarity, and gene clustering by cluster profile."""
        out = {}
        cols = [self.cfg.img_col, self.cfg.gene_col, self.cfg.plate_col]
        if self.cfg.grna_col in table_columns(self.cfg.db_path, self.cfg.table):
            cols.append(self.cfg.grna_col)

        # Fit KMeans incremental
        km = MiniBatchKMeans(n_clusters=self.cfg.unsup_k, batch_size=self.cfg.kmeans_batch,
                             reassignment_ratio=0.01, random_state=self.cfg.seed)
        n_fit = 0
        for chunk in self._iter_stream_with_paths(cols):
            X = compute_embeddings(chunk[self.cfg.img_col].tolist(), self.cfg.img_size,
                                   self.cfg.batch_size_img, self.cfg.num_workers, self.device)
            km.partial_fit(X); n_fit += X.shape[0]
        print(f"[unsup] fitted k-means on {n_fit} cells -> {self.cfg.unsup_k} clusters")

        # Assign all + save skinny parquet
        parts = []
        for chunk in self._iter_stream_with_paths(cols):
            X = compute_embeddings(chunk[self.cfg.img_col].tolist(), self.cfg.img_size,
                                   self.cfg.batch_size_img, self.cfg.num_workers, self.device)
            lab = km.predict(X)
            base = chunk.copy(); base["cluster"] = lab
            parts.append(base)
        cells = pd.concat(parts, ignore_index=True)
        p_cells = str(Path(self.cfg.out_dir, "unsupervised_cells.parquet"))
        cells.to_parquet(p_cells, index=False); out["unsupervised_cells"] = p_cells

        # CMH per (gene, cluster)
        cmh_all = []
        for cl in sorted(cells["cluster"].unique().tolist()):
            d = cells.assign(score=(cells["cluster"]==cl).astype(float))
            cmh = cmh_enrichment_soft(d[[self.cfg.gene_col, self.cfg.plate_col, "score"]],
                                      self.cfg.gene_col, self.cfg.plate_col, "score")
            cmh.insert(1, "cluster", int(cl)); cmh_all.append(cmh)
        cmh_df = pd.concat(cmh_all, ignore_index=True)
        p_cmh = str(Path(self.cfg.out_dir, "gene_cluster_cmh_unsupervised.csv"))
        cmh_df.to_csv(p_cmh, index=False); out["gene_cluster_cmh_unsupervised"] = p_cmh

        # Gene cluster profiles (fraction of a gene's cells in each cluster)
        pivot = (cells.groupby([self.cfg.gene_col, "cluster"]).size().unstack(fill_value=0).astype(float))
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        gprof = pivot.reset_index()
        p_gprof = str(Path(self.cfg.out_dir, "gene_by_unsup_cluster_profile.csv"))
        gprof.to_csv(p_gprof, index=False); out["gene_by_unsup_cluster_profile"] = p_gprof

        # Cluster genes by these profiles
        lab = cluster_genes_from_profiles(pivot.to_numpy(),
                                          n_clusters=min(20, max(2, pivot.shape[1]//2)))
        gcl = pd.DataFrame({self.cfg.gene_col: pivot.index.tolist(), "gene_cluster": lab})
        p_genecl = str(Path(self.cfg.out_dir, "gene_clusters_by_unsup_profile.csv"))
        gcl.to_csv(p_genecl, index=False); out["gene_clusters_by_unsup_profile"] = p_genecl

        # Optional: compute prototypes/similarity from embeddings for a sample (for sanity)
        # (If you want exact prototypes over all cells, re-embed everything once and call gene_prototypes_from_cells)
        return out

    # ---------- UMAP + Sprites (linked to png_list) ----------

    def umap_cells_with_images(self):
        """Subsample cells, compute embeddings, cluster ids (kmeans), 2D coords, and save:
           - cells_2d.csv (img_path, gene, plate, cluster, x, y)
           - cells_2d.png (colored by cluster)
           - cluster_X_sprite.png (tiles of example images)"""
        ensure_dir(self.cfg.out_dir)
        cols = [self.cfg.img_col, self.cfg.gene_col, self.cfg.plate_col]
        counts, chunks = [], []
        for chunk in stream_sql(self.cfg.db_path, self.cfg.table, cols, 200_000):
            chunk = filter_existing_paths(chunk, self.cfg.img_col)
            if len(chunk): counts.append(len(chunk)); chunks.append(chunk)
        if not counts: raise RuntimeError("No images found for UMAP.")
        k = min(self.cfg.sample_cells_2d, sum(counts))
        picks = set(reservoir_sample_indices(sum(counts), counts, k, self.cfg.seed))

        # collect sampled rows
        rows = []
        start = 0
        for ch in chunks:
            end = start + len(ch)
            take = [i for i in range(start, end) if i in picks]
            if take:
                rows.append(ch.iloc[np.array(take)-start])
            start = end
        df = pd.concat(rows, ignore_index=True)

        # embeddings & kmeans for coloring
        X = compute_embeddings(df[self.cfg.img_col].tolist(), self.cfg.img_size,
                               self.cfg.batch_size_img, self.cfg.num_workers, self.device)
        km = MiniBatchKMeans(n_clusters=min(self.cfg.unsup_k, max(5, int(np.sqrt(len(df)/50)))), 
                             batch_size=self.cfg.kmeans_batch, random_state=self.cfg.seed).fit(X)
        lab = km.labels_

        xy = reduce_2d(X, use_umap=self.cfg.use_umap, seed=self.cfg.seed)
        df2 = df.copy(); df2["x"]=xy[:,0]; df2["y"]=xy[:,1]; df2["cluster"]=lab

        p_csv = str(Path(self.cfg.out_dir, "cells_2d.csv"))
        df2.to_csv(p_csv, index=False)
        p_png = str(Path(self.cfg.out_dir, "cells_2d.png"))
        scatter_png(xy, lab, p_png, title=f"Cells UMAP (n={len(df2)})", s=1.0, alpha=0.7)

        # sprites per cluster
        for c in sorted(np.unique(lab).tolist()):
            paths = df2.loc[df2["cluster"]==c, self.cfg.img_col].tolist()
            outp = str(Path(self.cfg.out_dir, f"cluster_{c}_sprite.png"))
            make_sprite(paths, outp, thumb=self.cfg.sprite_thumb, max_imgs=self.cfg.sprites_per_cluster)
        print(f"[umap] saved {p_csv} and {p_png}; sprites in {self.cfg.out_dir}")

# ------------------------- Example CLI -------------------------

def run_supervised(cfg: PCConfig):
    pc = PhenoCluster(cfg)
    return pc.run_supervised()

def run_unsupervised(cfg: PCConfig):
    pc = PhenoCluster(cfg)
    return pc.run_unsupervised()

def run_umap(cfg: PCConfig):
    pc = PhenoCluster(cfg)
    pc.umap_cells_with_images()

if __name__ == "__main__":
    # Quick demo config (EDIT paths/columns)
    cfg = PCConfig(
        db_path="/path/to/your.sqlite",
        table="png_list",
        img_col="img_path",
        gene_col="gene",
        grna_col="grna",
        plate_col="plate",
        pheno_cols=["phenotype_A","phenotype_B"],  # set [] to skip supervised
        out_dir="./spacrcluster_out",
        use_graph_refiner=False,  # set True if torch_geometric installed
    )
    ensure_dir(cfg.out_dir)
    print(json.dumps(asdict(cfg), indent=2))
    # Choose what to run:
    # run_supervised(cfg)
    # run_unsupervised(cfg)
    # run_umap(cfg)
    
#from spacr.spacrcluster import PCConfig, run_supervised
#cfg = PCConfig(db_path=".../your.sqlite", pheno_cols=["mito_bundle","pvm_tsg101"], out_dir="./out")
#run_supervised(cfg)

#from spacr.spacrcluster import PCConfig, run_unsupervised
#cfg = PCConfig(db_path=".../your.sqlite", pheno_cols=[], unsup_k=50, out_dir="./out")
#run_unsupervised(cfg)

#from spacr.spacrcluster import PCConfig, run_umap
#cfg = PCConfig(db_path=".../your.sqlite", out_dir="./out", sample_cells_2d=200000)
#run_umap(cfg)

#cfg.use_graph_refiner = True

