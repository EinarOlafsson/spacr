// Builds spacr_lab_meeting.pptx from images/ and the content below.
// Run from the presentation folder:  node scripts/build.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.join(__dirname, "..");
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, "sizes.json")));
const IMG = (rel) => path.join(ROOT, "images", rel);

// palette: the app's own dark ink with teal and magenta
const BG = "0E1217", PANEL = "171E26", PANEL2 = "212A34", LINE = "2C3642";
const TXT = "E8EDF2", MUTED = "B8C2CC", DIM = "8C98A5";
const TEAL = "2EC4B6", MAG = "D946A8", AMBER = "F2B84B", BLUE = "5B9CF5", WHITE = "FFFFFF", NEUT = "C9D1D9";
// Open Sans: SemiBold for headings and anything bold, Regular/Italic for body, Light for captions and muted text.
const HF = "Open Sans", BF = "Open Sans Light", LF = "Open Sans Light", MONO = "Courier New";
const STAGE = { stable: TEAL, beta: AMBER, alpha: MAG };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Einar Olafsson";
pres.title = "spaCR: spatial phenotype analysis of CRISPR screens";
const W = 13.333, H = 7.5, M = 0.5;
// PUBLIC=1 builds the general overview for GitHub: no talk references, no demo badges, no speaker notes.
const PUBLIC = process.env.PUBLIC === "1";
const L = (talk, pub) => (PUBLIC ? pub : talk);

function size(rel) {
  const s = SIZES["images/" + rel];
  if (!s) throw new Error("no size for " + rel);
  return s;
}

// place an image inside a box, keeping its aspect ratio, centred
function fit(s, rel, x, y, w, h, opt = {}) {
  const [iw, ih] = size(rel);
  const pad = opt.card ? 0.1 : 0;
  const bw = w - 2 * pad, bh = h - 2 * pad - (opt.caption ? 0.32 : 0);
  const r = Math.min(bw / iw, bh / ih);
  const dw = iw * r, dh = ih * r;
  const dx = x + pad + (bw - dw) / 2, dy = y + pad + (bh - dh) / 2;
  if (opt.card) {
    const cx = opt.tight ? dx - pad : x, cw = opt.tight ? dw + 2 * pad : w;
    const cy = opt.tight ? dy - pad : y, ch = opt.tight ? dh + 2 * pad : h - (opt.caption ? 0.32 : 0);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: ch, fill: { color: opt.cardColor || WHITE }, line: { color: opt.cardColor || WHITE }, rectRadius: 0.06 });
  } else if (opt.frame !== false) {
    s.addShape(pres.shapes.RECTANGLE, { x: dx - 0.02, y: dy - 0.02, w: dw + 0.04, h: dh + 0.04, fill: { color: LINE }, line: { color: LINE } });
  }
  s.addImage({ path: IMG(rel), x: dx, y: dy, w: dw, h: dh });
  if (opt.caption) {
    s.addText(opt.caption, { x, y: y + h - 0.3, w, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", valign: "top", margin: 0, isTextBox: true });
  }
  return { x: dx, y: dy, w: dw, h: dh };
}

function iconTile(s, icon, x, y, sz, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: sz, h: sz, fill: { color: fill || TEAL }, line: { color: fill || TEAL }, rectRadius: sz * 0.2 });
  const p = sz * 0.17;
  s.addImage({ path: IMG("icons/" + icon + ".png"), x: x + p, y: y + p, w: sz - 2 * p, h: sz - 2 * p });
}

function pill(s, text, x, y, color, w) {
  const pw = w || 0.2 + text.length * 0.085;
  s.addText(text, { x, y, w: pw, h: 0.3, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.15, fill: { color: PANEL2 }, line: { color, width: 1 }, fontFace: HF, fontSize: 10, bold: true, color, align: "center", valign: "middle", margin: 0, isTextBox: true });
  return pw;
}

// map bold -> SemiBold face, muted body text -> Light face (Open Sans has no synthetic weights to lean on)
function fontFix(o, dflt) {
  if (!o) return o;
  const face = o.fontFace || dflt;
  if (o.bold && (face === BF || face === HF || face === LF)) { o.fontFace = HF; o.bold = false; }
  else if (face === HF) { o.bold = false; }
  else if ((face === BF) && (o.color === MUTED || o.color === DIM) && !o.bold) { o.fontFace = LF; }
  return o;
}
function wrapSlide(s) {
  const at = s.addText.bind(s), tb = s.addTable.bind(s);
  s.addText = (t, o) => {
    if (Array.isArray(t)) t.forEach((r) => { r.options = r.options || {}; const base = (o && o.fontFace) || BF; if (!r.options.fontFace) r.options.fontFace = base; fontFix(r.options, base); if (!r.options.color && o && o.color) {} });
    fontFix(o, BF);
    return at(t, o);
  };
  s.addTable = (rows, o) => {
    rows.forEach((row) => row.forEach((c) => { c.options = c.options || {}; if (!c.options.fontFace) c.options.fontFace = (o && o.fontFace) || BF; fontFix(c.options, BF); }));
    return tb(rows, o);
  };
  return s;
}

function base() {
  const s = wrapSlide(pres.addSlide());
  s.background = { color: BG };
  if (PUBLIC) s.addNotes = () => {};
  return s;
}

function header(s, o) {
  let tx = M;
  if (o.icon) { iconTile(s, o.icon, M, 0.38, 0.78, o.iconColor); tx = M + 1.0; }
  if (o.kicker) s.addText(o.kicker.toUpperCase(), { x: tx, y: 0.36, w: 8, h: 0.28, fontFace: HF, fontSize: 11, bold: true, color: o.kickerColor || NEUT, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText(o.title, { x: tx, y: 0.62, w: W - tx - M - (o.stage || o.demo ? 2.6 : 0), h: 0.6, fontFace: HF, fontSize: 30, bold: true, color: TXT, margin: 0, valign: "middle", isTextBox: true });
  let px = W - M;
  if (o.demo && !PUBLIC) { const w = 1.55; px -= w; pill(s, "▶  LIVE DEMO NEXT", px, 0.5, MAG, w); px -= 0.12; }
  if (o.stage) { const t = o.stage.toUpperCase(); const w = 0.95; px -= w; pill(s, t, px, 0.5, STAGE[o.stage], w); }
}

function heading(s, text, x, y, w, color) {
  s.addText(text.toUpperCase(), { x, y, w, h: 0.26, fontFace: HF, fontSize: 11, bold: true, color: color || NEUT, charSpacing: 1.5, margin: 0, isTextBox: true });
}

function bullets(s, items, x, y, w, h, fs) {
  s.addText(items.map((t, i) => {
    const runs = [];
    if (Array.isArray(t)) { runs.push({ text: t[0], options: { bold: true, color: TXT, bullet: { indent: 14 }, breakLine: false } }); runs.push({ text: " " + t[1], options: { color: MUTED, breakLine: i < items.length - 1 } }); return runs; }
    return [{ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1 } }];
  }).flat(), { x, y, w, h, fontFace: BF, fontSize: fs || 13, color: TXT, paraSpaceAfter: 5, valign: "top", margin: 0, isTextBox: true });
}

// rough height of a bullet list: Calibri averages ~0.5 em per character
function estH(items, w, fs) {
  const perLine = Math.max(10, Math.floor(w / (fs * 0.5 / 72 * 1.02)));
  let lines = 0;
  items.forEach((t) => { const n = Array.isArray(t) ? t.join(" ").length : t.length; lines += Math.ceil((n + 2) / perLine); });
  return lines * fs * 1.2 / 72 + items.length * 5 / 72 + 0.08;
}

// left text column used by module slides
function leftColumn(s, m, x, y, w, h) {
  let cy = y;
  if (m.purpose) {
    s.addText(m.purpose, { x, y: cy, w, h: 0.75, fontFace: BF, fontSize: 15, italic: true, color: NEUT, valign: "top", margin: 0, isTextBox: true });
    cy += 0.85;
  }
  if (m.can) {
    heading(s, "What you can do", x, cy, w);
    const ch = estH(m.can, w, m.fs || 13);
    bullets(s, m.can, x, cy + 0.32, w, ch, m.fs || 13);
    cy += 0.32 + ch + 0.12;
  }
  if (m.out) {
    heading(s, "Output", x, cy, w, MAG);
    bullets(s, m.out, x, cy + 0.32, w, y + h - cy - 0.32, m.fs || 13);
  }
}

// sub-tool chips along the bottom
function subtools(s, list, y) {
  if (!list || !list.length) return;
  const perRow = Math.min(list.length, 4);
  const rows = Math.ceil(list.length / perRow);
  const gap = 0.15, cw = (W - 2 * M - gap * (perRow - 1)) / perRow;
  const top = y || (H - 0.3 - rows * 0.78 - 0.3);
  heading(s, "Sub-tools inside this module", M, top, 6, MUTED);
  list.forEach(([name, desc], i) => {
    const r = Math.floor(i / perRow), c = i % perRow;
    const x = M + c * (cw + gap), yy = top + 0.3 + r * 0.78;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w: cw, h: 0.68, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.08 });
    s.addText([{ text: name, options: { bold: true, color: TXT, breakLine: true } }, { text: desc, options: { color: MUTED } }],
      { x: x + 0.14, y: yy + 0.05, w: cw - 0.24, h: 0.6, fontFace: BF, fontSize: 11, valign: "middle", margin: 0, isTextBox: true });
  });
  return top;
}

function notes(s, t) { s.addNotes(t); }

// ---------------------------------------------------------------- intro
{
  const s = base();
  s.addImage({ path: IMG("icons/spacr_logo_color.png"), x: 5.833, y: 0, w: 7.5, h: 7.5, altText: "spaCR logo as a photomosaic of microscopy tiles with four parasites invading" });
  s.addText("spaCR", { x: M + 0.2, y: 1.55, w: 4.9, h: 1.2, fontFace: HF, fontSize: 66, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Spatial phenotype analysis of CRISPR screens", { x: M + 0.2, y: 2.85, w: 4.9, h: 1.1, fontFace: BF, fontSize: 26, color: NEUT, margin: 0, isTextBox: true });
  if (!PUBLIC) s.addText("A tour of every module, with live demos", { x: M + 0.2, y: 3.8, w: 4.9, h: 0.5, fontFace: BF, fontSize: 18, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addText(L("Einar Olafsson  ·  Lab meeting  ·  22 September 2026  ·  spaCR 1.5.1.0", "spaCR 1.5.1.0  ·  github.com/EinarOlafsson/spacr"), { x: M + 0.2, y: 6.55, w: 10, h: 0.4, fontFace: BF, fontSize: 14, color: DIM, margin: 0, isTextBox: true });
  notes(s, "Plan: what spaCR is and how to install it, then each module (slides, then a live demo), then the models I have trained.");
}

{
  const s = base();
  header(s, { kicker: "Introduction", title: "What is spaCR?" });
  fit(s, "manuscript/fig1A_concept.png", M, 1.45, W - 2 * M, 2.1, { card: true, caption: "Pooled screen: every well holds a mix of knockouts. Image it, sequence it, and regress. (Preprint, Fig. 1A)" });
  const cols = [
    ["mask", "Image every well", "Segment, measure and classify every cell. The phenotype becomes a number per well."],
    ["map_barcodes", "Sequence every well", "Read the gRNA barcodes to learn which knockouts are in each well, and in what proportion."],
    ["regression", "Link the two", "Regress the phenotype on guide composition. Out comes an effect per gene: a ranked hit list."],
  ];
  const gap = 0.3, cw = (W - 2 * M - 2 * gap) / 3;
  cols.forEach(([ic, h, d], i) => {
    const x = M + i * (cw + gap), y = 3.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: 2.0, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
    iconTile(s, ic, x + 0.25, y + 0.25, 0.7, [TEAL, MAG, BLUE][i]);
    s.addText(h, { x: x + 1.1, y: y + 0.25, w: cw - 1.3, h: 0.7, fontFace: HF, fontSize: 18, bold: true, color: TXT, valign: "middle", margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: y + 1.05, w: cw - 0.5, h: 0.85, fontFace: BF, fontSize: 13, color: MUTED, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText([{ text: "No sequencing? ", options: { bold: true, color: AMBER } }, { text: "Segmentation, measurement, annotation, classification and the assays work on any high-content imaging experiment.", options: { color: TXT } }],
    { x: M, y: 6.25, w: W - 2 * M, h: 0.5, fontFace: BF, fontSize: 14, margin: 0, valign: "middle", isTextBox: true });
  notes(s, "The core idea. A pooled library means each well contains several knockouts. Phenotype per well comes from images, genotype per well from sequencing; regression disentangles which gene drives the phenotype.");
}

// ---------------------------------------------------------------- landscape: spaCR next to other tools
{
  const s = base();
  header(s, { kicker: "Introduction", title: "Where spaCR fits among existing tools" });
  const tools = ["CellProfiler", "Fiji / ImageJ", "napari", "Cellpose", "ilastik", "QuPath", "DL4Mic", "MAGeCK", "spaCR"];
  const best = ["cell measurement", "image analysis", "image viewing", "segmentation", "pixel / object ML", "tissue slides", "no-code DL", "sequencing screens", "image-based screens"];
  const Y = "✓", N = "✗", PL = "plugin", LI = "limited";
  // one row per feature; values in tool order, spaCR last
  // 2026-09-26 (item 589): general features first; the niche rows (OPS, screen
  // power, parasite assays, gRNA linking, gene-level calls) folded into one row.
  // 2026-09-28 (item 590): the third entry is the feature's category, shown
  // uncoloured in its own column right of spaCR.
  const rows = [
    ["Point-and-click interface", [Y, Y, Y, Y, Y, Y, LI, N, Y], "Usability"],
    ["Batch and headless runs (server, cluster)", [Y, LI, N, Y, Y, Y, LI, Y, Y], "Automation"],
    ["Reads most microscope file formats", [Y, Y, PL, LI, LI, Y, LI, N, Y], "Input"],
    ["Deep-learning segmentation", [PL, PL, PL, Y, LI, PL, Y, N, Y], "Segmentation"],
    ["Train your own segmentation model", [N, PL, PL, Y, Y, LI, Y, N, Y], "Segmentation"],
    ["Live assisted segmentation", [N, PL, PL, LI, Y, Y, N, N, Y], "Segmentation"],
    ["Correct masks by hand", [LI, Y, Y, Y, Y, Y, N, N, Y], "Segmentation"],
    ["Per-cell measurements", [Y, Y, PL, N, Y, Y, N, N, Y], "Measurement"],
    ["3-D (z-stack) images", [Y, Y, Y, Y, Y, LI, LI, N, "beta"], "Image types"],
    ["Time-lapse tracking", [Y, PL, PL, N, Y, N, N, N, Y], "Image types"],
    ["Whole plates in one run", [Y, LI, N, LI, LI, LI, LI, N, Y], "Screening"],
    ["Plate maps, heat maps and QC", [LI, N, N, N, N, N, N, N, Y], "Screening"],
    ["Classify cells from images (deep learning)", [N, N, N, N, N, N, N, N, Y], "Classification"],
    ["Classify cells from measurements", [LI, N, PL, N, Y, Y, N, N, Y], "Classification"],
    ["Explore cells: UMAP with images, gating", [LI, N, PL, N, N, N, N, N, Y], "Analysis"],
    ["Plots and statistics without code", [LI, LI, PL, N, N, LI, N, LI, Y], "Analysis"],
    ["GPU acceleration", [N, PL, PL, LI, LI, LI, Y, N, Y], "Performance"],
    ["Pooled CRISPR screens: images to genes", [N, N, N, N, N, N, N, LI, Y], "Screening"],
    ["Pixel classifier from brush strokes", [N, PL, PL, N, Y, Y, N, N, N], "Segmentation"],
    ["Whole-slide / tissue images", [N, LI, N, N, N, Y, N, N, N], "Image types"],
    ["Large plugin ecosystem", [LI, Y, Y, N, N, LI, N, N, N], "Extensibility"],
  ];
  const RED = "8E2F3A", GREEN = "1F7A55", BLUEC = "2F5DA8";
  const CAT = { text: "Category", options: { bold: true, color: TXT, fill: { color: BG }, align: "center", fontSize: 10 } };
  const hdr = [{ text: "", options: { fill: { color: BG } } }].concat(tools.map((t, i) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: i === tools.length - 1 ? TEAL : PANEL2 }, align: "center", fontSize: 10 } })), [CAT]);
  const bestRow = [{ text: "Best for", options: { italic: true, color: MUTED, fill: { color: BG }, fontSize: 9 } }].concat(best.map((t, i) => ({ text: t, options: { italic: true, color: i === best.length - 1 ? WHITE : MUTED, fill: { color: i === best.length - 1 ? "17463F" : PANEL }, align: "center", fontSize: 7.5 } })), [{ text: "", options: { fill: { color: BG } } }]);
  const body = rows.map(([f, vals, cat]) => {
    const sp = vals[vals.length - 1] !== N;
    return [{ text: f, options: { color: TXT, fill: { color: PANEL }, fontSize: 8.5 } }].concat(vals.map((v, i) => {
      const last = i === vals.length - 1;
      const fill = v === N ? RED : (last || sp ? GREEN : BLUEC);
      return { text: v, options: { color: WHITE, fill: { color: fill }, align: "center", fontSize: v.length > 1 ? 7.5 : 9, bold: last } };
    }), [{ text: cat, options: { color: TXT, fill: { color: PANEL }, align: "center", fontSize: 8.5 } }]);
  });
  const fw = 2.7, catw = 1.05, cw = (W - 2 * M - fw - catw) / tools.length;
  s.addTable([hdr, bestRow].concat(body), { x: M, y: 1.28, w: W - 2 * M, colW: [fw].concat(tools.map(() => cw), [catw]), rowH: [0.28, 0.34].concat(rows.map(() => 0.215)), fontFace: BF, valign: "middle", border: { type: "solid", pt: 1, color: BG }, margin: 0.02 });
  [["tool lacks it", RED], ["tool has it: spaCR has it too", GREEN], ["tool has it: spaCR does not", BLUEC]].forEach(([t, c], i) => {
    const ly = 0.36 + i * 0.27, lx = 9.9;
    s.addShape(pres.shapes.RECTANGLE, { x: lx, y: ly + 0.04, w: 0.28, h: 0.17, fill: { color: c }, line: { color: c } });
    s.addText(t, { x: lx + 0.38, y: ly, w: 2.6, h: 0.25, fontFace: BF, fontSize: 10.5, color: TXT, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Core features; 'plugin' = needs an add-on, 'limited' = partly. Simplified: check each tool for details.", { x: M, y: 7.12, w: W - 2 * M, h: 0.28, fontFace: BF, fontSize: 9, italic: true, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  notes(s, "spaCR's unique part is the chain from images to gRNAs to genes, and deep-learning classification of cells in whole plates. Where others are stronger: interactive pixel classification (ilastik), whole-slide tissue (QuPath), plugin ecosystems (Fiji, napari), and screens with no images (MAGeCK). spaCR can hand masks to napari (Napari Bridge) and import results from other tools (Import).");
}

// ---------------------------------------------------------------- alpha features (item 590, 2026-09-28)
// One line per entry of ALPHA_FEATURES in spacr/settings.py (51 items on
// 2026-09-28), grouped by category. Hidden in the app until Preferences ->
// Show alpha features is on.
{
  const s = base();
  header(s, { kicker: "Introduction", title: "Alpha features: switch them on in Preferences", stage: "alpha" });
  const left = [
    ["Segmentation", [
      "StarDist segmentation backend",
      "InstanSeg segmentation backend",
      "Omnipose segmentation backend",
      "Spotiflow spot detection",
      "Click- and box-prompted segmentation (micro-SAM)",
      "Segmentation uncertainty maps",
      "Parameter-robustness report",
      "Run CellProfiler .cppipe pipelines",
      "Masks and ROIs in and out: QuPath, ImageJ, COCO",
      "Virtual staining from brightfield",
      "Mask batches across several GPUs",
    ]],
    ["Image preprocessing", [
      "Spectral unmixing (bleed-through correction)",
      "Photobleaching correction",
      "Vendor flat-field profiles (Harmony, ZEN)",
      "Intensity calibration across sessions",
      "Deep-learning image QC: focus, debris, empty fields",
    ]],
    ["Measurements and assays", [
      "Cell-cycle phase classification",
      "Live/dead viability and cytotoxicity index",
      "Confluency for any channel",
      "Scratch / wound-closure assay",
      "Colony / CFU counting",
      "Image-based profiling: normalise, select, mAP",
      "GPU measurement (cuCIM, CuPy)",
    ]],
    ["Reproducibility", [
      "Figure-integrity check",
      "Pre-registered analysis lock",
      "Metadata and archive packages (REMBI, IDR)",
      "One-click Zenodo archive with a DOI",
    ]],
  ];
  const right = [
    ["Time-lapse", [
      "SAM2 propagation: segment once, follow each object",
      "Timeflows: a temporal Cellpose",
      "Lineage trees from tracks",
      "Event detection: mitosis, egress, invasion, death",
      "Time-to-event: Kaplan-Meier and Cox",
    ]],
    ["Cells and machine learning", [
      "Foundation-model embeddings (SubCell, OpenPhenom, …)",
      "Well labels to responding cells (multiple-instance)",
      "Anomaly detection against negative controls",
      "Find cells like this one (similarity search)",
      "Blind scoring",
      "Counterfactuals: morph a control cell toward a hit",
    ]],
    ["Screens and statistics", [
      "Arrayed-screen hits: SSMD, robust z, B-score",
      "Chemistry-aware hits: SMILES, clusters, SAR",
      "Sample-size planner for arrayed assays",
      "Spatial transcriptomics: Visium and Xenium",
      "LIMS / plate-barcode linkage",
    ]],
    ["Automation and data", [
      "Watch a folder, analyse images as they arrive",
      "Smart microscopy: send hits back for re-imaging",
      "Cloud storage (S3 and others)",
      "Export a run as a Nextflow / Snakemake workflow",
      "Run-finished notifications (email, Slack, ntfy)",
      "Export for R (SingleCellExperiment), tidy Parquet",
      "Plugin and recipe catalogue",
      "Postgres or DuckDB / Parquet measurement store",
    ]],
  ];
  const RH = 0.195, gap = 0.3, tw = (W - 2 * M - gap) / 2, catw = 1.55;
  const table = (groups, x) => {
    const hdr = [
      { text: "Category", options: { bold: true, color: WHITE, fill: { color: PANEL2 }, fontSize: 10 } },
      { text: "Alpha feature", options: { bold: true, color: WHITE, fill: { color: PANEL2 }, fontSize: 10 } },
    ];
    const body = [];
    groups.forEach(([cat, feats]) => feats.forEach((f, i) => {
      const row = [];
      if (i === 0) row.push({ text: cat, options: { color: TXT, fill: { color: PANEL }, fontSize: 8.5, valign: "middle", rowspan: feats.length } });
      row.push({ text: f, options: { color: TXT, fill: { color: PANEL }, fontSize: 8.5 } });
      body.push(row);
    }));
    s.addTable([hdr].concat(body), { x, y: 1.28, w: tw, colW: [catw, tw - catw], rowH: [0.28].concat(body.map(() => RH)), fontFace: BF, valign: "middle", border: { type: "solid", pt: 1, color: BG }, margin: [0.02, 0.06, 0.02, 0.06] });
    return body.length;
  };
  const nl = table(left, M), nr = table(right, M + tw + gap);
  s.addText(`${nl + nr} alpha features, hidden until Preferences → Show alpha features is on. Early versions: expect changes.`, { x: M, y: 7.12, w: W - 2 * M, h: 0.28, fontFace: BF, fontSize: 9, italic: true, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  notes(s, "Everything built from the future list ships behind the alpha switch in Preferences, off by default.");
}

{
  const s = base();
  header(s, { kicker: "Introduction", title: "Six steps from plate to hit list" });
  const mods = [
    ["mask", "Mask", "Find every cell, nucleus, parasite"],
    ["measure", "Measure", "Shape, intensity, texture per object"],
    ["annotate", "Annotate", "Label example cells"],
    ["classify", "Classify", "Score every cell's phenotype"],
    ["map_barcodes", "Map Barcodes", "Which gRNAs are in which well"],
    ["regression", "Regression", "Which genes change the phenotype"],
  ];
  const n = 6, gap = 0.25, tw = (W - 2 * M - gap * 5) / n, ts = 1.35, ty = 1.85;
  mods.forEach(([ic, name, d], i) => {
    const x = M + i * (tw + gap);
    iconTile(s, ic, x + (tw - ts) / 2, ty, ts, i < 4 ? TEAL : (i === 4 ? MAG : BLUE));
    s.addText(String(i + 1), { x: x + (tw - ts) / 2 - 0.14, y: ty - 0.14, w: 0.4, h: 0.4, shape: pres.shapes.OVAL, fill: { color: WHITE }, fontFace: HF, fontSize: 13, bold: true, color: BG, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(name, { x, y: ty + ts + 0.2, w: tw, h: 0.4, fontFace: HF, fontSize: 17, bold: true, color: TXT, align: "center", margin: 0, isTextBox: true });
    s.addText(d, { x, y: ty + ts + 0.62, w: tw, h: 0.7, fontFace: BF, fontSize: 13, color: MUTED, align: "center", valign: "top", margin: 0, isTextBox: true });
    if (i < 5) s.addText("›", { x: x + tw - 0.05, y: ty + ts / 2 - 0.3, w: gap + 0.1, h: 0.6, fontFace: HF, fontSize: 26, color: DIM, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("images", { x: M, y: 4.85, w: 4 * (tw + gap) - gap, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: NEUT, align: "center", charSpacing: 2, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: M + 0.2, y: 4.82, w: 4 * (tw + gap) - gap - 0.4, h: 0, line: { color: NEUT, width: 1.5 } });
  s.addText("sequencing + statistics", { x: M + 4 * (tw + gap), y: 4.85, w: 2 * tw + gap, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: MAG, align: "center", charSpacing: 2, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: M + 4 * (tw + gap) + 0.2, y: 4.82, w: 2 * tw + gap - 0.4, h: 0, line: { color: MAG, width: 1.5 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.55, w: W - 2 * M, h: 1.3, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
  s.addText([
    { text: "One project, one database.  ", options: { bold: true, color: TXT } },
    { text: "Images, masks, crops, measurements, annotations, predictions, barcodes and well IDs all live in one SQLite file (measurements.db). Every module reads from it and writes back to it, and every run saves its settings so it can be replayed.", options: { color: MUTED } },
  ], { x: M + 0.3, y: 5.6, w: W - 2 * M - 0.6, h: 1.2, fontFace: BF, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  notes(s, "The six core modules in order. Steps 1-4 are imaging; 5-6 add the screen. Everything is stored in one project database.");
}

{
  const s = base();
  header(s, { kicker: "Introduction", title: "19 modules in one desktop app" });
  fit(s, "app_1513/home_tiles.png", M, 1.45, 7.9, 5.4, { caption: "The home screen: every module is a tile, grouped by what it is for" });
  const cats = [
    ["Core", "6", "Mask · Measure · Annotate · Classify · Map Barcodes · Regression", TEAL],
    ["Data", "7", "Import · Embeddings · Run Compare · Experiment Design · Power · Dose–Response · QC", BLUE],
    ["Tools", "5", "Make Masks · Align & Stitch · Image UMAP · Gate Editor · Graph Builder", AMBER],
    ["Organism", "1", "Toxoplasma: one module per organism, its assays inside. More organisms are alpha species", MAG],
  ];
  const x = 8.75, w = W - M - x;
  cats.forEach(([n, k, d, c], i) => {
    const y = 1.5 + i * 1.12;
    s.addText(k, { x, y, w: 0.7, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: c, margin: 0, valign: "top", isTextBox: true });
    s.addText([{ text: n, options: { bold: true, color: TXT, fontSize: 16, breakLine: true } }, { text: d, options: { color: MUTED, fontSize: 12 } }],
      { x: x + 0.8, y, w: w - 0.8, h: 1.0, fontFace: BF, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText([{ text: "Plus ~40 sub-tools ", options: { bold: true, color: TXT } }, { text: "inside the modules. Ctrl+K opens a command palette that reaches every one of them.", options: { color: MUTED } }],
    { x, y: 6.0, w, h: 0.8, fontFace: BF, fontSize: 13, margin: 0, valign: "top", isTextBox: true });
  notes(s, "Each module carries a maturity badge: stable, beta or alpha. Preferences can hide alpha/beta modules.");
}

{
  const s = base();
  header(s, { kicker: "Introduction", title: "Runs on a laptop, a workstation or a cluster" });
  const cards = [
    ["Desktop app", "One borderless window that opens with a logo animation (replay it from Help). Live previews, a console per module, settings search, recipes, and a Run button.", TEAL],
    ["Headless", "The same modules and settings files run without the window, on a server or a cluster. Export settings… on every form saves it as a settings CSV for spacr-run.", MAG],
    ["Reproducible", "Every run records its settings. spacr-repro RUN_DIR replays it. Run History and Run Compare show what changed.", BLUE],
  ];
  const gap = 0.3, cw = (W - 2 * M - 2 * gap) / 3;
  cards.forEach(([h, d, c], i) => {
    const x = M + i * (cw + gap), y = 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: 2.1, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
    s.addText(h, { x: x + 0.25, y: y + 0.2, w: cw - 0.5, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: y + 0.72, w: cw - 0.5, h: 1.3, fontFace: BF, fontSize: 13, color: TXT, valign: "top", margin: 0, isTextBox: true });
  });
  const stats = [["4", "kinds of graphics card", "NVIDIA, AMD, Apple and Intel: or just the CPU"], ["10", "interface languages", "switch in Preferences without restarting"], ["84", "interactive tutorials", "narrated videos from install to hit investigation"], ["3.9–3.14", "Python versions", "3.12 recommended; Linux, macOS, Windows"]];
  const sw = (W - 2 * M - 3 * gap) / 4;
  stats.forEach(([v, l, d], i) => {
    const x = M + i * (sw + gap), y = 4.1;
    s.addText(v, { x, y, w: sw, h: 0.95, fontFace: HF, fontSize: i === 3 ? 38 : 50, bold: true, color: [TEAL, MAG, BLUE, AMBER][i], margin: 0, valign: "bottom", isTextBox: true });
    s.addText(l, { x, y: y + 1.0, w: sw, h: 0.35, fontFace: HF, fontSize: 14, bold: true, color: TXT, margin: 0, isTextBox: true });
    s.addText(d, { x, y: y + 1.38, w: sw, h: 0.8, fontFace: BF, fontSize: 12, color: MUTED, valign: "top", margin: 0, isTextBox: true });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.3, w: W - 2 * M, h: 0.85, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
  s.addText([
    { text: "Accessibility  ", options: { bold: true, color: TEAL } },
    { text: "Preferences → Appearance → Accessibility: High contrast, Text size, Reduce motion, stronger focus outlines and a keyboard-shortcut editor. Screen readers announce every control, in the interface language.", options: { color: TXT } },
  ], { x: M + 0.3, y: 6.33, w: W - 2 * M - 0.6, h: 0.79, fontFace: BF, fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
  notes(s, "UMAP/clustering is GPU-accelerated on NVIDIA only; AMD, Apple and Intel GPUs are beta for Cellpose and Torch.");
}

{
  const s = base();
  header(s, { kicker: "Introduction", title: "Install spaCR" });
  const cards = [
    ["Desktop installer", "Easiest. Bundles its own Python: no conda needed.", ["GitHub → Releases → latest", "Windows  .exe", "macOS    .pkg  (Intel + Apple silicon)", "Linux    chmod +x SpaCR-*.run && ./SpaCR-*.run"], TEAL],
    ["pip (conda env)", "Most flexible; add optional extras.", ["conda create -n spacr python=3.12 -y", "conda activate spacr", "pip install \"spacr[qt]\"", "spacr"], MAG],
    ["conda-forge", "One command into an existing env.", ["conda create -n spacr python=3.12 -y", "conda activate spacr", "conda install conda-forge::spacr", "spacr"], BLUE],
    ["From source", "For development (nightly branch).", ["git clone --branch nightly \\", "  https://github.com/EinarOlafsson/spacr", "cd spacr && pip install -e .", "spacr"], AMBER],
  ];
  const gap = 0.25, cw = (W - 2 * M - gap) / 2, chh = 2.3;
  cards.forEach(([h, d, code, c], i) => {
    const x = M + (i % 2) * (cw + gap), y = 1.45 + Math.floor(i / 2) * (chh + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: chh, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
    s.addText(h, { x: x + 0.25, y: y + 0.15, w: 3, h: 0.4, fontFace: HF, fontSize: 18, bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 3.0, y: y + 0.17, w: cw - 3.2, h: 0.4, fontFace: BF, fontSize: 12, italic: true, color: MUTED, align: "right", margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: chh - 0.85, fill: { color: "0A0E12" }, line: { color: "0A0E12" }, rectRadius: 0.06 });
    s.addText(code.join("\n"), { x: x + 0.4, y: y + 0.7, w: cw - 0.8, h: chh - 0.95, fontFace: MONO, fontSize: 12.5, color: "C8F2EE", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText([{ text: "Update and remove  ", options: { bold: true, color: TEAL } }, { text: "Help → Check for updates installs the new version in place of the old one and keeps your data. Preferences → Performance → Uninstall… lists the installed versions and the categories of data to remove.", options: { color: MUTED } }],
    { x: M, y: 6.38, w: W - 2 * M, h: 0.45, fontFace: BF, fontSize: 12.5, margin: 0, valign: "middle", isTextBox: true });
  s.addText([{ text: "Something off?  ", options: { bold: true, color: AMBER } }, { text: "spacr-doctor", options: { fontFace: MONO, color: "C8F2EE" } }, { text: " checks the install and says how to fix it.   Headless server: ", options: { color: MUTED } }, { text: "pip install spacr", options: { fontFace: MONO, color: "C8F2EE" } }, { text: " (no Qt).", options: { color: MUTED } }],
    { x: M, y: 6.85, w: W - 2 * M, h: 0.4, fontFace: BF, fontSize: 13, margin: 0, valign: "middle", isTextBox: true });
  notes(s, "Current release 1.5.1.5. macOS pkg is not notarized yet: System Settings > Privacy & Security > Open Anyway. Optional extras: [zarr], [omero], [napari], [czi,nd2,lif].");
}

// ---------------------------------------------------------------- ask the group
if (!PUBLIC) {
  const s = base();
  header(s, { kicker: "Your turn", kickerColor: MAG, title: "What would make spaCR better for you?" });
  s.addText("Keep these in mind during the demos: I would love to hear your answers, today or any time after.", { x: M, y: 1.3, w: W - 2 * M, h: 0.45, fontFace: BF, fontSize: 16, italic: true, color: NEUT, margin: 0, isTextBox: true });
  const qs = [
    ["New features", "What analysis do you still do by hand, in Excel, Fiji or a script, that spaCR could do for you?", "measure", TEAL],
    ["Your assays", "Which readouts or assays from your own projects would you want as a ready-made module?", "analyze_plaques", MAG],
    ["Easier to use", "Which step is confusing, slow or has too many settings? Where did you get stuck?", "qc_dashboard", BLUE],
    ["Getting started", "What got in the way of installing or trying spaCR: installers, example data, tutorials, documentation?", "foreign", AMBER],
  ];
  const gap = 0.3, cw = (W - 2 * M - gap) / 2, ch = 2.05;
  qs.forEach(([t, q, ic, c], i) => {
    const x = M + (i % 2) * (cw + gap), y = 1.95 + Math.floor(i / 2) * (ch + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: PANEL }, line: { color: c, width: 1.25 }, rectRadius: 0.1 });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.75, h: 0.75, shape: pres.shapes.OVAL, fill: { color: c }, fontFace: HF, fontSize: 24, bold: true, color: BG, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 1.3, y: y + 0.28, w: cw - 1.55, h: 0.45, fontFace: HF, fontSize: 19, bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(q, { x: x + 1.3, y: y + 0.78, w: cw - 1.55, h: ch - 0.95, fontFace: BF, fontSize: 15, color: TXT, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText([{ text: "Later ideas:  ", options: { bold: true, color: AMBER } }, { text: "tell me, or open an issue at github.com/EinarOlafsson/spacr/issues", options: { color: MUTED } }],
    { x: M, y: 6.85, w: W - 2 * M, h: 0.4, fontFace: BF, fontSize: 14, margin: 0, valign: "middle", isTextBox: true });
  notes(s, "Ask the group now, then keep a list during the demos. Come back to it after the last module.");
}

// ---------------------------------------------------------------- sections
function section(num, title, sub, mods, color) {
  const s = base();
  s.addText(num, { x: M + 0.2, y: 1.3, w: 3, h: 1.6, fontFace: HF, fontSize: 110, bold: true, color, margin: 0, isTextBox: true });
  s.addText(title, { x: M + 0.2, y: 3.0, w: 8, h: 1.0, fontFace: HF, fontSize: 48, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText(sub, { x: M + 0.2, y: 4.05, w: 7.5, h: 0.9, fontFace: BF, fontSize: 18, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  const n = mods.length, ts = 0.95, gap = 0.28;
  const cols = n > 4 ? Math.ceil(n / 2) : n;
  const gx = W - M - cols * ts - (cols - 1) * gap;
  mods.forEach(([ic, name], i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = gx + c * (ts + gap), y = 2.0 + r * 1.75;
    iconTile(s, ic, x, y, ts, color);
    s.addText(name, { x: x - 0.2, y: y + ts + 0.08, w: ts + 0.4, h: 0.5, fontFace: BF, fontSize: 11, color: TXT, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
  return s;
}

// ---------------------------------------------------------------- module slide builder
function moduleSlide(m) {
  const s = base();
  header(s, { icon: m.icon, iconColor: m.color, kicker: m.kicker, kickerColor: m.color === TEAL ? NEUT : m.color, title: m.title, stage: m.stage, demo: m.demo });
  const top = 1.45;
  const bottom = m.sub && m.sub.length ? (H - 0.3 - Math.ceil(m.sub.length / Math.min(m.sub.length, 4)) * 0.78 - 0.45) : H - 0.4;
  if (m.sub) subtools(s, m.sub, bottom + 0.15);
  const lw = m.lw === undefined ? 4.55 : m.lw;
  if (lw > 0) leftColumn(s, m, M, top, lw, bottom - top);
  const rx = lw > 0 ? M + lw + 0.35 : M, rw = W - M - rx, rh = bottom - top;
  if (m.draw) m.draw(s, rx, top, rw, rh);
  else if (m.images) layoutImages(s, m.images, rx, top, rw, rh, m.layout || "row");
  if (m.notes) notes(s, m.notes);
  return s;
}

function layoutImages(s, imgs, x, y, w, h, mode) {
  const n = imgs.length, gap = 0.2;
  if (n === 1) return fit(s, imgs[0].src, x, y, w, h, imgs[0]);
  if (mode === "col") {
    const hs = imgs.map((im) => im.frac || 1 / n);
    let cy = y;
    imgs.forEach((im, i) => { const hh = (h - gap * (n - 1)) * hs[i]; fit(s, im.src, x, cy, w, hh, im); cy += hh + gap; });
  } else {
    const ws = imgs.map((im) => im.frac || 1 / n);
    let cx = x;
    imgs.forEach((im, i) => { const ww = (w - gap * (n - 1)) * ws[i]; fit(s, im.src, cx, y, ww, h, im); cx += ww + gap; });
  }
}

// ---------------------------------------------------------------- CORE
section("01", "Core pipeline", "From raw plate images and FASTQ reads to a ranked list of genes.", [["mask", "Mask"], ["measure", "Measure"], ["annotate", "Annotate"], ["classify", "Classify"], ["map_barcodes", "Map Barcodes"], ["regression", "Regression"]], TEAL);

const CORE = (i) => `Core · ${i} of 6`;

moduleSlide({
  icon: "mask", color: TEAL, kicker: CORE(1), title: "Mask: segmentation", stage: "stable",
  purpose: "Finds and outlines every cell, nucleus, parasite and organelle in your images.",
  can: ["Reads files from most microscopes (TIFF, Leica, Zeiss, Nikon)", "Cellpose-SAM by default; other segmentation models optional", "Cell, nucleus, pathogen and up to 26 organelle classes", "Plug in fine-tuned Model Zoo models (e.g. Toxoplasma vacuoles)", "2-D, 3-D z-stacks and time-lapse; live preview before a full run"],
  out: ["One mask per object type, aligned to the images", "Overlay images and a quality score per field", "The settings used, saved with the run"],
  images: [{ src: "keyframes/mask_cell_objects.png", caption: "Live preview: host cells (left) and the label mask Cellpose-SAM returns (right)" }],
  notes: "Channels are mapped to biology (nuclei, ER, lipid droplets, parasites in the tutorial set). Preview on one field, then run the plate.",
});

moduleSlide({
  icon: "mask", color: TEAL, kicker: CORE(1), title: "Mask: every object class", stage: "stable", demo: true, lw: 0,
  // item 590: four square panels with their captions directly beneath, the
  // group centred in the space above the sub-tools.
  panels: [
    { src: "mask_new/mask_nuclei.png", caption: "Nuclei: Cellpose-SAM (Hoechst)" },
    { src: "mask_new/mask_cells.png", caption: "Cells: Cellpose-SAM (cell stain)" },
    { src: "mask_new/mask_pathogen.png", caption: "Pathogen: Toxoplasma, zoomed" },
    { src: "mask_new/mask_droplets.png", caption: "Organelles: lipid droplets, zoomed" },
  ],
  draw: function (s, x, y, w, h) {
    const n = this.panels.length, gap = 0.2, pw = (w - gap * (n - 1)) / n, capH = 0.42;
    const top = y + Math.max(0, (h - pw - capH) / 2);
    this.panels.forEach((im, i) => fit(s, im.src, x + i * (pw + gap), top, pw, pw + capH, im));
  },
  sub: [["Timelapse", "Follow each cell and parasite from frame to frame in movies"], ["Cellpose flows", "See why two cells were merged, or one was split"]],
  notes: "Each class is its own Cellpose pass with its own filters (size, intensity). Pathogens can be filtered by size to drop debris.",
});

moduleSlide({
  icon: "measure", color: TEAL, kicker: CORE(2), title: "Measure: features and crops", stage: "stable", demo: true,
  purpose: "Turns every segmented object into a row of numbers: shape, brightness, texture and position.",
  can: ["Shape, size, brightness and texture of every object", "Position: neighbours, distances, touching the edge", "Channel overlap; measure inside a chosen region", "Saves object crops for Annotate and Classify", "3-D objects measured in µm", "Checks free RAM first and sizes its workers to fit"],
  out: ["One table per object type (cells, nuclei, parasites, …)", "Per-object PNG crops"],
  layout: "col",
  images: [{ src: "keyframes/measure_crops.png", caption: "Crop preview, grouped by nucleated / infected state", frac: 0.55 }, { src: "keyframes/db_table.png", caption: "The cell table in the Database Browser", frac: 0.45 }],
  sub: [["Illumination", "Flat-field correction estimated from the plate itself"], ["AnnData export", "Export for single-cell analysis tools (scanpy)"], ["Motility assay", "Track velocity and straightness, split by infection state"], ["Database Browser", "Browse, filter and export any project table"]],
});

moduleSlide({
  icon: "annotate", color: TEAL, kicker: CORE(3), title: "Annotate: label cells by hand", stage: "stable", demo: true,
  purpose: "Label single-cell crops so a classifier can learn what each phenotype looks like.",
  can: ["Pages through a grid of crops from the project", "Click, or press 1–9 with auto-advance; 0 clears, u undoes", "Several annotation columns side by side (e.g. two annotators)", "Active learning: shows the cells the model is least sure about first", "Change channels and colours for display only; files stay untouched"],
  out: ["Labels in an annotation column of the project", "Class counts and coverage per well"],
  images: [{ src: "keyframes/annotate_active.png", caption: "Annotation grid. Selected crops are outlined; the queue can be ranked by model uncertainty" }],
  sub: [["Annotator Agreement", "How often two people label a cell the same way: and a list of the cells they disagreed on"]],
});

moduleSlide({
  icon: "classify", color: TEAL, kicker: CORE(4), title: "Classify: score every cell", stage: "alpha",
  purpose: "Teach the computer what a phenotype looks like, then let it score every cell in the screen.",
  can: [["Image model:", "learns straight from the cell pictures (ResNet, EfficientNet, ViT, MaxViT, …)"], ["Measurement model:", "learns from the measured features (XGBoost, random forest, …)"], "Examples come from control wells or from your Annotate labels", "Tested on wells it never trained on, so the accuracy is honest", "Scores every cell and writes the result back to the project"],
  out: ["A class and a probability for every cell", "The mean score per well → input to Regression"],
  images: [{ src: "manuscript/fig2_classify.png", caption: "Each cell is scored by the classifier; a well's score is the mean of its cells (preprint Fig. 2)", card: true }],
  notes: "Two families: image models on crops, and measurement models on the Measure table. The badge says alpha because the app's maturity table has Classify at alpha.",
});

moduleSlide({
  icon: "classify", color: TEAL, kicker: CORE(4), title: "Classify: image model", stage: "alpha", lw: 0,
  images: [
    { src: "manuscript/fig4E_crops.png", caption: "Input: single-cell crops from negative (NC) and positive (PC) control wells", card: true },
    { src: "manuscript/fig4F_saliency.png", caption: "Saliency: where MaxViT looked: at the vacuole in PC cells", card: true },
  ],
  sub: [["Activation Maps", "Show which pixels drove each decision"], ["Classifier Evaluation", "Accuracy and mistakes per class and per plate, on unseen wells"], ["Training Runs", "Compare training curves between runs"], ["Explain CV Model", "Explain an image model in terms of measurements"], ["Feature Explorer", "Which measurements separate the classes best"]],
  notes: "Preprint Fig. 4E-F. Red DsRed parasites, green GFP-TSG101, blue Hoechst. TSG101 gathers at the vacuole in NC but not in PC (GRA14 knockout).",
});

moduleSlide({
  icon: "classify", color: TEAL, kicker: CORE(4), title: "Classify: measurement model", stage: "alpha", demo: true, lw: 0,
  layout: "col",
  images: [
    { src: "manuscript/fig4BCD_features.png", caption: "XGBoost on the measurements: which measurements decide the class, three ways of ranking them", card: true, frac: 0.44 },
    { src: "manuscript/fig3CD_accuracy.png", caption: "Accuracy with different channel combinations (grey, XGBoost) vs the image model (teal, MaxViT)", card: true, frac: 0.56 },
  ],
  notes: "Preprint Fig. 4B-D and 3C-D. The top feature is TSG101-DsRed colocalization at the cell level; the image model reaches the highest accuracy.",
});

moduleSlide({
  icon: "map_barcodes", color: MAG, kicker: CORE(5), title: "Map Barcodes: which guides, which well", stage: "stable", demo: true,
  purpose: "Reads the sequencing data and works out which gRNAs are in each well, and how much of each.",
  can: ["Reads the row, column and gRNA barcodes from the sequencing files", "You give it the barcode lists and where they sit in the read", "Paired-end or single reads; set how many mismatches to allow", "Quick test on a small part of the file first"],
  out: ["Guide counts per well", "QC table and run manifest", "gRNA fraction per well → input to Regression"],
  images: [{ src: "outputs/01_plate_heatmap_unique_counts.png", caption: "Unique gRNAs per well, four plates", card: true, frac: 0.58 }, { src: "outputs/18_fraction_histogram.png", caption: "gRNA fraction per well", card: true, frac: 0.42 }],
  sub: [["Barcode QC", "Reads per well, reads that did not match, library coverage: and the cut-off that gives the expected number of gRNAs per well"]],
});

moduleSlide({
  icon: "regression", color: BLUE, kicker: CORE(6), title: "Regression: from wells to genes", stage: "stable",
  purpose: "Works out which genes change the phenotype, and ranks them.",
  can: ["Pairs each well's score with the fraction of each gRNA in that well", "Default: a mixed model where the gRNAs of a gene are grouped", "Other models for counts, fractions or skewed data", "Permutation test: shuffles gRNAs to check a hit is not chance"],
  out: ["A ranked table of genes with effect size and p-value", "Volcano, top effects and quality-check plots"],
  layout: "col",
  images: [{ src: "manuscript/fig2_mlr.png", caption: "The model: a well's score = its gRNA fractions × each gene's effect (preprint Fig. 2)", card: true }, { src: "manuscript/figS7_volcano.png", caption: "The result: EAF1 and GRA14 stand out (preprint Fig. S7)", card: true }],
});

moduleSlide({
  icon: "regression", color: BLUE, kicker: CORE(6), title: "Regression: choose the model", stage: "stable", lw: 3.7,
  purpose: "21 model families, each explained in plain words, with what it assumes about your data.",
  can: [["Mixed:", "the gRNAs of a gene grouped, genes allowed to vary"], ["OLS / WLS:", "classic least squares, optionally weighted by cells per well"], ["GLM, Poisson, beta, logit:", "for counts, fractions and yes/no readouts"], ["Lasso, ridge, elastic net, group lasso, horseshoe:", "when there are many more genes than wells"], ["Robust, quantile, RRA:", "when a few wells are extreme"]],
  out: ["Plate, row and column covariates; permutation test on gRNAs", "The formula and what it assumes, shown for the chosen model"],
  images: [{ src: "regression_app/crop_models.png", caption: "Regression type: every model with the assumption it makes (OLS selected)" }],
});

moduleSlide({
  icon: "regression", color: BLUE, kicker: CORE(6), title: "Regression: pick a gene, see its cells", stage: "stable", lw: 0,
  images: [
    { src: "regression_app/crop_volcano.png", caption: "Interactive volcano: click a point. EAF1 (TGGT1_225160) is selected, its gene card below", frac: 0.45 },
    { src: "regression_app/crop_cells.png", caption: "Cells tab: the cells in an EAF1 well whose scores fit the effect, with why they were picked", frac: 0.55 },
  ],
  sub: [["Volcano Explorer", "Publication version of the volcano; vector export"], ["Investigate Hit", "Test the hit at the cell level against control wells"], ["Hit List", "Ranked genes with intervals and gRNA agreement"], ["Prediction Profiler", "How the prediction changes as one gene's fraction changes"]],
  notes: "Real run of the Regression module on TSG101 plate 1 (MaxViT scores, OLS at gene level). EAF1 is the 3rd hit (effect +0.26). The cells shown are candidates: the screen reports which FRACTION of a well carried the guide, not which cell did.",
});

moduleSlide({
  icon: "regression", color: BLUE, kicker: CORE(6), title: "Regression: assumption checks", stage: "stable", lw: 3.7,
  purpose: "Every run checks its own assumptions and says, in red, which ones fail and what to do.",
  can: [["Equal variance:", "Breusch-Pagan test on the residuals"], ["Normal residuals:", "skew, kurtosis, D'Agostino test"], ["Independence:", "Durbin-Watson; wells on one plate share handling"], ["Influence:", "leverage and Cook's distance per well"], ["Collinearity:", "VIF and condition number"], ["Calibration:", "p-value histogram and genomic inflation"]],
  out: ["A run summary: the answer, what was fitted, each assumption with its verdict", "Residual, Q-Q, scale-location and influence plots"],
  layout: "col", images: [{ src: "regression_app/crop_diagnostics.png", caption: "Summary tab: each assumption with its test and verdict; equal variance and normality rejected (red)", frac: 0.5 }, { src: "regression_app/crop_diag_plots.png", caption: "The diagnostic plots behind the verdicts", frac: 0.5 }],
});

moduleSlide({
  icon: "regression", color: BLUE, kicker: CORE(6), title: "Regression: the hit list", stage: "stable", demo: true, lw: 0,
  images: [
    { src: "manuscript/fig5BC_volcano.png", caption: "Volcanos from the screen: XGBoost (left) and MaxViT (right) scores both find EAF1 and GRA14 (preprint Fig. 5)", card: true, frac: 0.46 },
    { src: "outputs/17_phenotype_plot.png", caption: "Genes ranked by phenotype", card: true, frac: 0.24 },
    { src: "outputs/03_coefficient_forest.png", caption: "Strongest effects with 95% intervals", card: true, frac: 0.3 },
  ],
  sub: [["Volcano Explorer", "Click a point to see the gene; export for publication"], ["Hit List", "Ranked genes with confidence intervals and gRNA agreement; export as a table"], ["Investigate Hit", "Jump from a gene to its wells and cells"], ["Diagnostics", "Checks that the model fits the data (plate, row and column effects)"], ["Methods & Results", "Drafts the methods text from the run itself"], ["Prediction Profiler", "How a prediction changes across one input"], ["Parameter Sweep", "Re-run across settings to see how stable the hits are"]],
  notes: "Volcano and rank plot from the MaxViT screen analysis (figure_5b); forest plot from the example dataset.",
});

// ---------------------------------------------------------------- DATA
section("02", "Data", "Bring data in, plan experiments, and check that a run can be trusted.", [["foreign", "Import"], ["embeddings", "Embeddings"], ["run_compare", "Run Compare"], ["experiment_design", "Experiment Design"], ["power", "Power / Design"], ["dose_response", "Dose–Response"], ["qc_dashboard", "QC"]], BLUE);
const DATA = (i) => `Data · ${i} of 7`;

moduleSlide({
  icon: "foreign", color: BLUE, kicker: DATA(1), title: "Import: bring outside data in", stage: "stable", demo: true,
  purpose: "Turns images, masks and tables from other tools into a spaCR project.",
  can: ["Pair images, instance masks and a measurement CSV", "Match their column names to spaCR's, converting units", "Choose what happens when names collide", "Preview before anything is written; save and reuse the mapping"],
  out: ["A spaCR project that records where the data came from"],
  layout: "col",
  images: [{ src: "keyframes/import_map2.png", caption: "Column mapping: their name → spaCR column, with unit conversion" }, { src: "keyframes/converter2.png", caption: "Format Converter: plan for turning vendor files into spaCR's TIFF layout" }],
  sub: [["Import Images", "Reads any microscope's folder and works out the naming itself"], ["Format Converter", "Turns Nikon, Zeiss and Leica files into spaCR's layout"], ["External Masks", "Measure masks made elsewhere, without re-segmenting"]],
});

moduleSlide({
  icon: "embeddings", color: BLUE, kicker: DATA(2), title: "Embeddings: one feature vector per cell", stage: "alpha", demo: true,
  purpose: "Gives every cell a label-free feature vector from a pretrained image model, next to the measured features.",
  can: ["Crops from the project (filter by object class or plate) or from a folder", "Choose the network: DINOv2 (default; Cell-DINO when installed), ResNet-18, ResNet-50, ConvNeXt, ViT", "Each stain is described separately, or all together", "Runs on the graphics card when there is one"],
  out: ["One row of numbers per cell, next to its measurements", "Which network was used is recorded"],
  draw: (s, x, y, w, h) => {
    const cy = y + 0.3;
    fit(s, "keyframes/measure_crops.png", x, cy, 2.6, 2.2, { caption: "cell crops" });
    s.addText("→", { x: x + 2.6, y: cy + 0.7, w: 0.5, h: 0.6, fontFace: HF, fontSize: 28, color: DIM, align: "center", margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 3.1, y: cy + 0.3, w: 1.9, h: 1.5, fill: { color: PANEL2 }, line: { color: BLUE, width: 1.5 }, rectRadius: 0.12 });
    s.addText([{ text: "pretrained\nbackbone", options: { bold: true, color: TXT, breakLine: true } }, { text: "e.g. DINOv2", options: { color: MUTED, fontSize: 11 } }], { x: x + 3.1, y: cy + 0.3, w: 1.9, h: 1.5, fontFace: BF, fontSize: 14, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText("→", { x: x + 5.0, y: cy + 0.7, w: 0.5, h: 0.6, fontFace: HF, fontSize: 28, color: DIM, align: "center", margin: 0, isTextBox: true });
    const vx = x + 5.55, vw = w - 5.55;
    for (let r = 0; r < 6; r++) for (let c = 0; c < 8; c++) {
      const v = Math.abs(Math.sin(r * 3.1 + c * 1.7));
      const col = v > 0.66 ? BLUE : v > 0.33 ? "2D5A8F" : "1B2A3B";
      s.addShape(pres.shapes.RECTANGLE, { x: vx + c * (vw / 8), y: cy + 0.15 + r * 0.3, w: vw / 8 - 0.03, h: 0.27, fill: { color: col }, line: { color: col } });
    }
    s.addText("one row of numbers per cell", { x: vx, y: cy + 2.0, w: vw, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 3.1, w, h: 1.5, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
    s.addText([{ text: "Use it for  ", options: { bold: true, color: BLUE } }, { text: "clustering and UMAPs without choosing features, finding cells that look like a hit, and checking whether a classifier is picking up something the measurements miss.", options: { color: TXT } }],
      { x: x + 0.3, y: y + 3.15, w: w - 0.6, h: 1.4, fontFace: BF, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  },
  notes: "Diagram only; no saved embedding figure exists yet. Full matrix via spacr.embeddings.embed_array().",
});

moduleSlide({
  icon: "run_compare", color: BLUE, kicker: DATA(3), title: "Run Compare: differences between runs", stage: "beta", demo: true,
  purpose: "Puts two runs of the same project side by side, so you know why the results moved.",
  can: ["Pick a project and any two of its runs", "Settings diff, grouped by settings panel", "Object, well and field counts per plate", "Hits that appeared, vanished or changed rank", "Incompatible runs are blocked unless you choose 'compare anyway'"],
  out: ["Side-by-side comparison tables"],
  draw: (s, x, y, w, h) => {
    const cards = [
      ["Settings", [["pathogen_model", "cpsam  →  toxoplasma_pv_v3"], ["cell_diameter", "90  →  110"], ["regression_type", "ols  →  mixed"]], BLUE],
      ["Counts per plate", [["cells", "182,410  →  176,902"], ["pathogens", "40,112  →  52,774"], ["wells", "384  →  384"]], TEAL],
      ["Hits", [["appeared", "3 genes"], ["vanished", "1 gene"], ["rank changed", "12 genes"]], MAG],
    ];
    const gap = 0.2, ch = (h - 2 * gap - 0.35) / 3;
    cards.forEach(([t, rows, c], i) => {
      const yy = y + i * (ch + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w, h: ch, fill: { color: PANEL }, line: { color: c, width: 1 }, rectRadius: 0.08 });
      s.addText(t, { x: x + 0.25, y: yy + 0.12, w: 3, h: 0.35, fontFace: HF, fontSize: 15, bold: true, color: c, margin: 0, isTextBox: true });
      rows.forEach(([k, v], j) => {
        s.addText([{ text: k, options: { color: MUTED } }], { x: x + 0.3, y: yy + 0.52 + j * 0.3, w: 2.4, h: 0.3, fontFace: MONO, fontSize: 11.5, margin: 0, isTextBox: true });
        s.addText(v, { x: x + 2.8, y: yy + 0.52 + j * 0.3, w: w - 3.1, h: 0.3, fontFace: MONO, fontSize: 11.5, color: TXT, margin: 0, isTextBox: true });
      });
    });
    s.addText("Illustration of the three comparison panels (values made up)", { x, y: y + h - 0.3, w, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
  },
});

moduleSlide({
  icon: "experiment_design", color: BLUE, kicker: DATA(4), title: "Experiment Design: plan the plate", stage: "alpha", demo: true,
  purpose: "Lay out conditions, controls and replicates before you image anything.",
  can: ["Assign conditions and control roles to wells", "Warnings for capacity, too few replicates, and row / column / edge confounding", "Randomise with a seed, so the layout can be rebuilt exactly"],
  out: ["A plate map that later spaCR steps use directly", "The settings used, so the layout can be rebuilt"],
  draw: (s, x, y, w, h) => {
    const rows = 16, cols = 24, gw = w - 0.5, cell = Math.min(gw / cols, (h - 1.1) / rows);
    const ox = x + 0.45, oy = y + 0.35;
    const colors = { pc: MAG, nc: TEAL, a: BLUE, b: AMBER, e: "2A333D" };
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const letters = "ABCDEFGHIJKLMNOP";
    for (let r = 0; r < rows; r++) {
      s.addText(letters[r], { x: x + 0.1, y: oy + r * cell, w: 0.3, h: cell, fontFace: BF, fontSize: 8, color: DIM, align: "center", valign: "middle", margin: 0, isTextBox: true });
      for (let c = 0; c < cols; c++) {
        const edge = r === 0 || r === rows - 1 || c === 0 || c === cols - 1;
        let k = "e";
        if (!edge) { const v = rnd(); k = v < 0.1 ? "pc" : v < 0.2 ? "nc" : v < 0.6 ? "a" : "b"; }
        s.addShape(pres.shapes.OVAL, { x: ox + c * cell + cell * 0.1, y: oy + r * cell + cell * 0.1, w: cell * 0.8, h: cell * 0.8, fill: { color: colors[k] }, line: { color: colors[k] } });
      }
    }
    const ly = oy + rows * cell + 0.2;
    [["positive control", MAG], ["negative control", TEAL], ["condition A", BLUE], ["condition B", AMBER], ["edge (left empty)", "2A333D"]].forEach(([t, c], i) => {
      const lx = x + 0.45 + i * (gw / 5);
      s.addShape(pres.shapes.OVAL, { x: lx, y: ly + 0.05, w: 0.18, h: 0.18, fill: { color: c }, line: { color: c === "2A333D" ? DIM : c } });
      s.addText(t, { x: lx + 0.25, y: ly, w: gw / 5 - 0.3, h: 0.28, fontFace: BF, fontSize: 11, color: MUTED, margin: 0, valign: "middle", isTextBox: true });
    });
    s.addText("Illustration: a randomised 384-well layout with edge wells left empty", { x, y: ly + 0.38, w, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
  },
});

moduleSlide({
  icon: "power", color: BLUE, kicker: DATA(5), title: "Power / Design: how big a screen?", stage: "beta", demo: true,
  purpose: "Estimates how many cells per well and how many plates you need to find a real hit.",
  can: ["Enter the library: genes, gRNAs per gene, constructs per well", "Enter the plate format and number of plates", "Enter the classifier's background rate and the effect you expect", "Simulates the screen many times to see how often the real hit is found"],
  out: ["Detection probability vs cells per well", "A one-sentence answer, with caveats"],
  images: [{ src: "manuscript/fig1DE_power.png", caption: "spaCRPower simulations: hit detection vs plates, by library size and genes per well (preprint Fig. 1D–E)", card: true }],
});

moduleSlide({
  icon: "dose_response", color: BLUE, kicker: DATA(6), title: "Dose–Response: EC50 estimates", stage: "alpha", demo: true,
  purpose: "Fits dose–response curves per gene or compound, and says so when an EC50 can't be pinned down.",
  can: ["Choose concentration, response and grouping columns", "Fits the standard S-shaped curve", "A confidence interval on every EC50"],
  out: ["Curves on a log-dose axis", "EC50 table with CI. Out-of-range → a bound ('EC50 > 30 µM'); failed fits listed as 'refused'", "Warns when a curve does not fit well"],
  draw: (s, x, y, w, h) => {
    const doses = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100];
    const f = (d, ec, hill) => 5 + 90 / (1 + Math.pow(ec / d, hill));
    s.addChart(pres.charts.LINE, [
      { name: "compound A (EC50 ≈ 0.8 µM)", labels: doses.map(String), values: doses.map((d) => +f(d, 0.8, 1.2).toFixed(1)) },
      { name: "compound B (EC50 ≈ 6 µM)", labels: doses.map(String), values: doses.map((d) => +f(d, 6, 1.6).toFixed(1)) },
      { name: "compound C (EC50 > 100 µM)", labels: doses.map(String), values: doses.map((d) => +f(d, 400, 1).toFixed(1)) },
    ], {
      titleFontFace: HF, catAxisLabelFontFace: BF, valAxisLabelFontFace: BF, legendFontFace: BF, dataLabelFontFace: BF, catAxisTitleFontFace: BF, valAxisTitleFontFace: BF, x, y, w, h: h - 0.35, chartColors: [TEAL, MAG, AMBER], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      catAxisTitle: "dose (µM, log steps)", showCatAxisTitle: true, catAxisTitleColor: MUTED, catAxisLabelColor: MUTED, catAxisLabelFontSize: 11, catAxisTitleFontSize: 11,
      valAxisTitle: "response (%)", showValAxisTitle: true, valAxisTitleColor: MUTED, valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisTitleFontSize: 11,
      valAxisMinVal: 0, valAxisMaxVal: 100, valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendColor: TXT, legendFontSize: 11,
    });
    s.addText("Schematic of the output (simulated curves)", { x, y: y + h - 0.3, w, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
  },
});

moduleSlide({
  icon: "qc_dashboard", color: BLUE, kicker: DATA(7), title: "QC: run quality checks", stage: "alpha", demo: true,
  purpose: "Five stored checks on one page. Reads what the run already wrote; recomputes nothing.",
  can: ["Segmentation QC scorecards", "Pixel size recorded", "Classifier tested on truly unseen wells", "Plate effects (row / column / edge)", "Annotator agreement"],
  out: ["Verdict cards: ok · warn · fail · missing · stale", "Nothing is written to disk"],
  draw: (s, x, y, w, h) => {
    const cards = [["Segmentation", "ok", "98% of fields pass", TEAL], ["Units", "ok", "0.325 µm / pixel", TEAL], ["Classifier test", "warn", "same well used to train and test", AMBER], ["Plate effects", "fail", "row gradient, ρ = 0.41", MAG], ["Annotator agreement", "missing", "no second annotator", DIM]];
    const gap = 0.2, cw = (w - gap) / 2, ch = (h - 0.4 - 2 * gap) / 3;
    cards.forEach(([t, v, d, c], i) => {
      const cx = x + (i % 2) * (cw + gap), cy = y + Math.floor(i / 2) * (ch + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: ch, fill: { color: PANEL }, line: { color: c, width: 1.25 }, rectRadius: 0.08 });
      s.addText(t, { x: cx + 0.25, y: cy + 0.15, w: cw - 1.5, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: TXT, margin: 0, isTextBox: true });
      pill(s, v.toUpperCase(), cx + cw - 1.2, cy + 0.17, c, 1.0);
      s.addText(d, { x: cx + 0.25, y: cy + 0.6, w: cw - 0.5, h: 0.4, fontFace: BF, fontSize: 12.5, color: MUTED, margin: 0, isTextBox: true });
    });
    s.addText("Illustration of the dashboard's verdict cards (values made up)", { x, y: y + h - 0.3, w, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
  },
  sub: [["Layer Viewer", "Images and masks as layers you can switch on and off"], ["Control Charts", "Follow a control over time and flag drift"], ["Outliers", "Find unusual cells or wells"]],
});

// ---------------------------------------------------------------- TOOLS
section("03", "Tools", "Point these at a project: fix masks, stitch tiles, explore cells, gate populations, plot.", [["make_masks", "Make Masks"], ["align", "Align & Stitch"], ["umap", "Image UMAP"], ["gate_editor", "Gate Editor"], ["graph_builder", "Graph Builder"]], AMBER);
const TOOLS = (i) => `Tools · ${i} of 5`;

moduleSlide({
  icon: "make_masks", color: AMBER, kicker: TOOLS(1), title: "Make Masks: fix masks by hand", stage: "beta",
  purpose: "Correct segmentation by hand, and build training data for your own Cellpose model.",
  can: ["Tools: brush, erase, erase object, magic wand + / −, draw, box, divide, zoom, recrop, ruler", "Live magnifier: a box that zooms in and outlines cells as you move", "Edit Label and Time mode: object numbers kept across frames (32-bit labels); + Cell and + Daughter record divisions", "3D in the same view: slice slider, Z wheel, View list; a 3D switch for object detection", "TEM images (DM3, DM4, MRC) with an Electron microscopy contrast preset; the pixel size is kept", "Open a folder or Load test data; Keep, Skip or Discard per field"],
  out: ["Curated masks, ready for training or measuring", "A log of every edit, so curated masks are told apart from model output"],
  images: [{ src: "app_1513/make_masks.png", caption: "Make Masks: tools across the top, settings left, shortcuts right, field actions below. Inset: the Orthogonal 3D view on FIB-SEM test data" }],
});

moduleSlide({
  icon: "make_masks", color: AMBER, kicker: TOOLS(1), title: "Make Masks: live magnifier", stage: "beta", lw: 0,
  draw: (s, x, y, w, h) => {
    // two synchronised recordings from the app, side by side (same path, frames and timing)
    const pair = SIZES["images/magnifier_gif/magnifier_cells.gif"] && SIZES["images/magnifier_gif/magnifier_toxo.gif"];
    const gh = h - 1.35, gap = 0.4, gw = Math.min(gh, (w - gap) / 2), gx = x + (w - 2 * gw - gap) / 2;
    if (pair) {
      fit(s, "magnifier_gif/magnifier_cells.gif", gx, y, gw, gh, { caption: "Cell from Hoechst: whole cells found from the nuclear (Hoechst) image" });
      fit(s, "magnifier_gif/magnifier_toxo.gif", gx + gw + gap, y, gw, gh, { caption: "Toxoplasma from Cell Mask: vacuoles found from the cell-mask image" });
    } else {
      fit(s, "magnifier_gif/magnifier_app.gif", x, y, w, gh, { caption: "Recorded in the app: Cellpose-SAM finds the nuclei once; each click adds the one under the box to the mask" });
    }
    const items = [
      ["Box", "follows the mouse, zoomed and contrast-adjusted"],
      ["Model", "Cellpose-SAM, your own model, or a threshold, outlining live"],
      ["Click", "adds what the box found; every click can be undone"],
      ["Drag", "along a cell bigger than the box joins its pieces into one object"],
    ];
    const cw = (w - 0.45) / 4, by = y + h - 1.05;
    items.forEach(([k, v], i) => {
      const bx = x + i * (cw + 0.15);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: cw, h: 1.0, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.08 });
      s.addText([{ text: k, options: { bold: true, color: AMBER, breakLine: true } }, { text: v, options: { color: TXT } }], { x: bx + 0.15, y: by + 0.06, w: cw - 0.3, h: 0.88, fontFace: BF, fontSize: 12, valign: "middle", margin: 0, isTextBox: true });
    });
  },
  notes: "Both recordings come from the real Make Masks screen with the magnifier in whole-image mode and play in step. Left: Cell from Hoechst on the Hoechst channel. Right: Toxoplasma from Cell Mask on the cell-mask channel of the same field. Wheel changes zoom, Shift+wheel box size.",
});

moduleSlide({
  icon: "make_masks", color: AMBER, kicker: TOOLS(1), title: "Make Masks: Features", stage: "beta",
  purpose: "Measure the masks you drew, without leaving Make Masks.",
  can: ["One row per field; columns are the image channels and the mask types", "Fill it by dropping files, clicking a cell to pick a file, or one filename pattern", "Press Measure: the run goes through the Measure module itself"],
  out: ["The same measurements database and crops a Measure run makes, from your hand-corrected masks"],
  images: [{ src: "makemasks_current/features.png", caption: "Features: four fields with their cell masks, ready to measure" }],
});

moduleSlide({
  icon: "make_masks", color: AMBER, kicker: TOOLS(1), title: "Make Masks: train and compare models", stage: "beta", demo: true, lw: 0,
  layout: "row",
  images: [
    { src: "keyframes/train_cellpose_fig.png", caption: "Cellpose Workbench, training: image / label pairs it will learn from" },
    { src: "keyframes/cellpose_masks_fig.png", caption: "Cellpose Workbench, applying a model: image, mask overlay and flows" },
  ],
  sub: [["Cellpose Workbench", "Fine-tune Cellpose on your masks, then mask the whole folder"], ["Model Compare", "Two models' masks side by side, and how much they agree"], ["Model Zoo", "Browse, download and test trained models"], ["Curate", "Edit masks and tracks, with an edit log"], ["Napari Bridge", "Edit a mask in napari and bring it back"]],
});

moduleSlide({
  icon: "align", color: AMBER, kicker: TOOLS(2), title: "Align & Stitch: tiles to mosaic", stage: "stable", demo: true,
  purpose: "Stitches overlapping tiles into one large image without running out of memory.",
  can: ["Set grid, overlap, tile order and reference channel", "Plans first, reading only the overlaps: fast on huge plates", "Finds how far each tile is off by matching nuclei in the overlaps", "Shows how sure it is about every tile", "Tells you the final size and memory needed before it starts"],
  out: ["The stitched mosaic, written piece by piece so memory stays low", "Where every tile ended up, saved with the project"],
  lw: 3.6, images: [{ src: "stitch_new/stitch_figure_notitle.png", caption: "A real Hoechst image cut into 9 overlapping tiles, each up to 20 px off (simulated); spaCR put every tile back where it came from" }],
});

moduleSlide({
  icon: "umap", color: AMBER, kicker: TOOLS(3), title: "Image UMAP: map all cells", stage: "beta", demo: true,
  purpose: "Places every cell on a 2-D map where similar cells sit together: with the real images.",
  can: ["UMAP, t-SNE or PCA maps", "Finds groups of similar cells automatically", "Embed by controls (positive vs negative groups)", "Can choose the map settings for you", "Click a point to see that cell's crop"],
  out: ["Scatter coloured by cluster or condition, with thumbnails", "Table with a cluster column per cell"],
  lw: 3.6, images: [{ src: "umap_new/umap_clusters_notitle.png", caption: "5,000 real cells from TSG101 screen plate 1: they fall into groups by number of nuclei and vacuoles" }],
  sub: [["Image Scatter", "Any two measured features, with the crop shown on hover"], ["PCA", "Which measurements drive the main differences"]],
});

moduleSlide({
  icon: "gate_editor", color: AMBER, kicker: TOOLS(4), title: "Gate Editor: gate cells like FACS", stage: "alpha", demo: true,
  purpose: "Select cell populations the way a flow cytometrist gates: in 1, 2 or 3 dimensions.",
  can: ["Load tables from one or several projects, or a CSV", "Threshold, rectangle, oval, polygon, wand gates in 1-D and 2-D", "Box, cylinder and prism gates in 3-D", "Nest and combine gates; count and % of parent for each", "Publish a gate as a filter every linked view follows", "Keys: Shift+E Edit, Shift+S Spin, Shift+D Draw, Shift+X / Y / Z for the plane (rebindable)", "Overlap picker: a click on stacked gates lists them all; choose which one to move"],
  out: ["A gating strategy you can reuse on other plates", "Gate-membership columns in the project; PNG / PDF plots"],
  layout: "col",
  images: [{ src: "keyframes/gate_2d.png", caption: "2-D rectangle gate on cell area vs nuclear intensity" }, { src: "keyframes/gate_walk.png", caption: "Cluster walk: populations found and gated automatically" }],
});

moduleSlide({
  icon: "graph_builder", color: AMBER, kicker: TOOLS(5), title: "Graph Builder: plots without code", stage: "beta", demo: true,
  purpose: "Build a plot by dragging columns onto axes. The chart type follows the data.",
  can: ["Six drop zones: x, y, colour, size and two facets", "Histogram, bar, scatter, box or heatmap chosen for you; violin and line on request", "Brush points to select cells: linked to the UMAP and plate map", "Annotate conditions from metadata; reusable schemas", "Merge tables by name; save a chart and load it back"],
  out: ["Interactive charts and annotated tables, exportable"],
  images: [{ src: "keyframes/plate_heat.png", caption: "Plate Viewer: mean cell intensity per well", frac: 0.55 }, { src: "manuscript/fig3B_recruitment.png", caption: "Violin plot of per-well values", card: true, frac: 0.45 }],
  sub: [["Plate Viewer", "Plate heatmaps and edge-effect detection"], ["Small Multiples", "One plot per group, shared or independent axes"]],
});

// ---------------------------------------------------------------- ASSAYS
section("04", "Assays", "One module per organism, each with its own assays. Toxoplasma ships today; more organisms are alpha species.", [["replication", "Toxoplasma"]], MAG);
const AS = (i) => `Assays · Toxoplasma · ${i} of 4`;

{
  const s = base();
  header(s, { kicker: "Assays", title: "One module per organism" });
  fit(s, "app_1513/toxoplasma.png", M, 1.45, W - 2 * M, 3.6, { caption: "The Toxoplasma module: the parasite's compartments on the left, its assays as tiles" });
  const orgs = [
    ["Toxoplasma", "Plaque · Recruitment · Host–Pathogen · Invasion · Replication · Gliding motility, with Starplast's hyperLOPIT compartments"],
    ["More organisms (alpha)", "Plasmodium, Candida, Trypanosoma, Leishmania, Giardia, virus and mammalian modules: Preferences > Show alpha species"],
  ];
  const cw = (W - 2 * M - 0.2) / 2;
  orgs.forEach(([n, d], i) => {
    s.addText([{ text: n, options: { bold: true, color: TXT, fontSize: 15, breakLine: true } }, { text: d, options: { color: MUTED, fontSize: 11 } }],
      { x: M + i * (cw + 0.2), y: 5.35, w: cw, h: 1.3, fontFace: BF, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText("Greyed tiles are assays still being built.", { x: M, y: 6.8, w: W - 2 * M, h: 0.3, fontFace: BF, fontSize: 10, italic: true, color: MUTED, margin: 0, isTextBox: true });
  notes(s, "Home's Assays group now holds one module per organism; each organism module opens its own assay tiles.");
}

moduleSlide({
  icon: "analyze_plaques", color: MAG, kicker: AS(1), title: "Plaque Assay: count and size plaques", stage: "beta", demo: true,
  purpose: "Finds every plaque in whole-well images and measures it in mm².",
  can: ["Segments plaques with the Model Zoo plaque model (Cellpose-SAM)", "A YOLO detector finds and splits wells first", "Areas converted to mm² from the well diameter", "'Figure' mode for plaque images taken from papers", "Can reuse existing masks"],
  out: ["A table of every plaque, and a summary per well", "Area, perimeter, diameter, eccentricity, solidity per plaque"],
  images: [{ src: "models/plaque_examples.png", caption: "Curated plaque outlines: our microscope (left) and a published figure (right)", card: true, cardColor: WHITE, frac: 0.68 }, { src: "models/welldet_example.png", caption: "Wells found in a figure", card: true, frac: 0.32 }],
});

moduleSlide({
  icon: "recruitment", color: MAG, kicker: AS(2), title: "Recruitment: marker at the vacuole", stage: "stable", demo: true,
  purpose: "How much does a fluorescent marker gather at the parasite vacuole compared with the host cytoplasm?",
  can: ["Map wells to cell type, pathogen and treatment", "Size and intensity filters per object class", "Choose the channel of interest", "Minimum number of cells per well"],
  out: ["Tables per cell and per well", "Bar plots of recruitment by condition, with per-well points"],
  layout: "col",
  images: [{ src: "outputs/06_PLATE1_E01_1_1.png", caption: "One field: channels and the cell / nucleus / parasite masks", frac: 0.3 }, { src: "outputs/recruitment_viridis.png", caption: "GFP-TSG101 recruitment to the vacuole, NC vs PC, 4 plates (preprint data)", frac: 0.7 }],
});

moduleSlide({
  icon: "invasion", color: MAG, kicker: AS(3), title: "Invasion: attached or invaded?", stage: "stable", demo: true,
  purpose: "Scores each parasite as outside (attached) or inside (invaded) from two-colour differential staining.",
  can: ["Set the outside-stain and total-stain channels", "Thresholds per field, or from control wells", "Flags fields where inside and outside cannot be told apart", "Checks the answer holds if the threshold moves"],
  out: ["Per-parasite calls, per-field thresholds, per-well invasion", "Statistics comparing conditions, per well", "Plots per well, by condition, and of the thresholds"],
  images: [{ src: "outputs/invasion_viridis.png", caption: "spaCR invasion analysis of the test dataset (4 wells per condition, mean ± SD)" }],
});

moduleSlide({
  icon: "replication", color: MAG, kicker: AS(4), title: "Replication: parasites per vacuole", stage: "beta", demo: true,
  purpose: "Counts parasites per vacuole (1, 2, 4, 8, 16 …) and compares the distributions between conditions.",
  can: ["Groups parasites into vacuoles: auto, spatial, by cell, or by object", "Area filters for parasites and vacuoles", "Warns when too many vacuoles have a non-power-of-two count"],
  out: ["Vacuole counts, per-well distributions, condition summary", "Statistics comparing conditions", "Stacked-bar plots per well and by condition"],
  draw: (s, x, y, w, h) => {
    fit(s, "outputs/replication_synthetic.png", x, y, w, h - 1.25, { caption: "Example output on synthetic data: 6 wells per condition, mean ± SD" });
    const by = y + h - 1.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w, h: 1.05, fill: { color: PANEL }, line: { color: MAG, width: 1.25 }, rectRadius: 0.1 });
    s.addText([{ text: "In progress:  ", options: { bold: true, color: MAG } }, { text: "a classification model that finds and counts Toxoplasma from the Toxoplasma stain, from the nuclear stain, or from the cell mask, so parasites per vacuole can be scored without a dedicated parasite channel.", options: { color: TXT } }],
      { x: x + 0.25, y: by + 0.05, w: w - 0.5, h: 0.95, fontFace: BF, fontSize: 13.5, valign: "middle", margin: 0, isTextBox: true });
  },
});

// ---------------------------------------------------------------- infrastructure
{
  const s = base();
  header(s, { kicker: "Behind the tiles", title: "Project tools in the Help menu" });
  const items = [
    ["report", "Report", "HTML / PDF report of a run: figures, statistics, settings"],
    ["pipeline_graph", "Pipeline Graph", "Which run produced which file, as a graph"],
    ["batch", "Batch Runner", "Queue modules with dependencies; validate, then run unattended"],
    ["mask", "Plate Queue", "Line up plates for Mask / Measure and let them run overnight"],
    ["run_compare", "Run History", "Every run, its settings and its outputs"],
    ["foreign", "Distributed Jobs", "Send work to SSH hosts, Slurm or the cloud"],
    ["measure", "Database Browser", "Browse, search, filter and export project tables"],
    ["embeddings", "Data Manager", "What is on disk, how big, and what can go"],
    ["annotate", "Project Browser", "All projects on this machine"],
    ["regression", "Feature Dictionary", "What every measured column means, and its unit"],
  ];
  const gap = 0.2, cw = (W - 2 * M - gap) / 2, ch = 0.92;
  items.forEach(([ic, n, d], i) => {
    const x = M + (i % 2) * (cw + gap), y = 1.45 + Math.floor(i / 2) * (ch + 0.13);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.08 });
    iconTile(s, ic, x + 0.15, y + 0.13, 0.66, PANEL2);
    s.addText([{ text: n, options: { bold: true, color: TXT, fontSize: 15, breakLine: true } }, { text: d, options: { color: MUTED, fontSize: 12 } }],
      { x: x + 1.0, y: y + 0.08, w: cw - 1.2, h: ch - 0.16, fontFace: BF, valign: "middle", margin: 0, isTextBox: true });
  });
  notes(s, "Icons here are generic; these tools have no home-screen tile. Ctrl+K reaches all of them.");
}

// ---------------------------------------------------------------- MODELS
section("05", "Trained models", L("Models I trained for our images, shipped through the spaCR Model Zoo. Newest version of each.", "Models trained for Toxoplasma and host-cell images, shipped through the spaCR Model Zoo. Newest version of each."), [["mask", "PV"], ["analyze_plaques", "Plaque"], ["model_zoo", "Wells"], ["measure", "Cross-channel"]], TEAL);

const MODELS = [
  ["Toxoplasma PV v4", "toxoplasma_pv_v4", "Parasite vacuoles", "Cellpose-SAM", "F1 0.761 (stock 0.377)"],
  ["Toxoplasma Plaque v5", "toxoplasma_plaque_v5", "Plaques in wells", "Cellpose-SAM", "F1 0.816 (stock 0.227)"],
  ["Plaque Well Detector v3", "toxoplasma_well_detector_v3", "Wells in plates / figures", "YOLO11n", "mAP50 0.994"],
  ["Toxoplasma from Cell Mask", "toxoplasma_from_cellmask_v1", "PVs from the cell channel", "Cellpose-SAM", "F1 0.606 (stock 0.021)"],
  ["Nuclei from Cell Mask", "nuclei_from_cellmask_v1", "Nuclei from the cell channel", "Cellpose-SAM", "F1 0.888 (stock 0.201)"],
  ["Cell from Hoechst", "cell_from_hoechst_v1", "Cells from the nuclear channel", "Cellpose-SAM", "F1 0.870 (stock 0.301)"],
  ["Toxoplasma from Hoechst", "toxoplasma_from_hoechst_v1", "PVs from the nuclear channel", "Cellpose-SAM", "F1 0.569 (stock 0.002)"],
  ["Live cell v1", "live_cell_v1", "Unstained cells (phase, DIC)", "Cellpose-SAM", "F1 0.960 (stock 0.885) *"],
];

{
  const s = base();
  header(s, { icon: "model_zoo", iconColor: TEAL, kicker: "Model Zoo", title: "Eight models, one line in a settings file" });
  const rows = [[
    { text: "Model", options: { bold: true, color: NEUT } }, { text: "Settings key", options: { bold: true, color: NEUT } }, { text: "Finds", options: { bold: true, color: NEUT } }, { text: "Architecture", options: { bold: true, color: NEUT } }, { text: "Score on unseen images", options: { bold: true, color: NEUT } },
  ]].concat(MODELS.map((r) => r.map((c, j) => ({ text: c, options: { color: j === 0 ? TXT : j === 1 ? "C8F2EE" : MUTED, fontFace: j === 1 ? MONO : BF, bold: j === 0 } }))));
  s.addTable(rows, { x: M, y: 1.45, w: W - 2 * M, colW: [2.7, 3.2, 2.6, 1.55, 2.28], fontFace: BF, fontSize: 12.5, fill: { color: PANEL }, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.4, valign: "middle" });
  s.addText("* Live cell v1: F1 0.960 vs 0.885 on data stock never saw; 0.694 vs 0.738 over all 2,199 test fields (stock trained on LIVECell).", { x: M, y: 5.1, w: W - 2 * M, h: 0.34, fontFace: BF, fontSize: 11.5, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.55, w: W - 2 * M, h: 1.4, fill: { color: PANEL2 }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
  s.addText([
    { text: "How it works  ", options: { bold: true, color: NEUT } },
    { text: "Name a key in a settings file: ", options: { color: TXT } }, { text: "pathogen_model: toxoplasma_pv_v4", options: { fontFace: MONO, color: "C8F2EE" } },
    { text: ": and spaCR downloads it the first time it is needed. Anyone can contribute a trained model: it is uploaded to the spaCR Hugging Face and then appears in every user's Model Zoo.", options: { color: TXT } },
  ], { x: M + 0.3, y: 5.6, w: W - 2 * M - 0.6, h: 1.3, fontFace: BF, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  notes(s, "Only the newest default version of each model line is shown. Older versions and alternatives (PV v1-v3 and v5, plaque v1-v4, well detector v1 and v2) remain in the zoo. Live cell v1: F1 0.960 against 0.885 on the datasets stock Cellpose-SAM never trained on; over all 2,199 test fields 0.694 against 0.738, because stock trained on LIVECell.");
}

const SEG_DEFS = [
  ["F1", "share of objects found correctly; lower if objects are missed or invented (1 = perfect). Found = outlines overlap by at least half."],
  ["AJI", "how well each outline matches; lower if a cell is split in two or two cells merged."],
  ["Dice", "pixel overlap between the model's mask and the true mask."],
  ["stock", "vanilla Cellpose-SAM (cpsam_v2), never trained on our images."],
];
const DET_DEFS = [
  ["precision", "of the boxes it draws, the share that really are wells."],
  ["recall", "of the real wells, the share it finds."],
  ["mAP50", "overall detection score; a box counts when it overlaps the true well by at least half."],
  ["mAP50-95", "the same, but also rewards boxes drawn tightly around the well."],
];

function modelSlide(o) {
  const s = base();
  header(s, { icon: o.icon, iconColor: o.color || TEAL, kicker: "Model · " + o.key, title: o.title });
  const lx = M, lw = 7.1, top = 1.45, bh = 5.55, ih = 4.05;
  if (o.draw) o.draw(s, lx, top, lw, ih); else layoutImages(s, o.images, lx, top, lw, ih, o.layout || "row");
  const defs = o.defs || SEG_DEFS, dy = top + ih + 0.12;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: dy, w: lw, h: top + bh - dy, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.08 });
  s.addText("WHAT THE SCORES MEAN", { x: lx + 0.18, y: dy + 0.08, w: 4, h: 0.24, fontFace: HF, fontSize: 10, bold: true, color: NEUT, charSpacing: 1.5, margin: 0, isTextBox: true });
  s.addText(defs.map(([k, v], i) => [{ text: k + "  ", options: { bold: true, color: TXT } }, { text: v, options: { color: MUTED, breakLine: i < defs.length - 1 } }]).flat(),
    { x: lx + 0.18, y: dy + 0.34, w: lw - 0.36, h: top + bh - dy - 0.4, fontFace: BF, fontSize: 10.5, valign: "top", paraSpaceAfter: 1, margin: 0, isTextBox: true });
  const rx = lx + lw + 0.35, rw = W - M - rx;
  let y = top;
  s.addText([{ text: o.arch, options: { bold: true, color: TXT, breakLine: true } }, { text: o.task, options: { color: MUTED } }], { x: rx, y, w: rw, h: 0.7, fontFace: BF, fontSize: 13, valign: "top", margin: 0, isTextBox: true });
  y += 0.75;
  s.addChart(pres.charts.BAR, o.chart.series, {
    titleFontFace: HF, catAxisLabelFontFace: BF, valAxisLabelFontFace: BF, legendFontFace: BF, dataLabelFontFace: BF, catAxisTitleFontFace: BF, valAxisTitleFontFace: BF, x: rx - 0.1, y, w: rw + 0.1, h: 2.35, barDir: "col", barGapWidthPct: 50,
    chartColors: o.chart.colors || [TEAL, "5F6B78"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: TXT,
    valAxisMinVal: 0, valAxisMaxVal: o.chart.max || 1, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: MUTED, catAxisLabelFontSize: 10.5, catAxisLineShow: false,
    showLegend: o.chart.series.length > 1, legendPos: "t", legendColor: TXT, legendFontSize: 10.5,
    showTitle: true, title: o.chart.title, titleColor: TXT, titleFontSize: 12,
  });
  y += 2.45;
  heading(s, "Trained on", rx, y, rw);
  s.addText(o.data, { x: rx, y: y + 0.28, w: rw, h: 0.9, fontFace: BF, fontSize: 12, color: TXT, valign: "top", margin: 0, isTextBox: true });
  y += 1.2;
  heading(s, "Good to know", rx, y, rw, AMBER);
  bullets(s, o.notes, rx, y + 0.28, rw, top + bh - y - 0.28, 11.5);
  if (o.speaker) s.addNotes(o.speaker);
}

const PRED = (k) => `predictions/${k}.png`;
const havePred = (k) => !!SIZES["images/" + PRED(k)];
const predImages = (k, fallback, cap) => havePred(k) ? [{ src: PRED(k), caption: cap || "Unseen test field: orange = vanilla Cellpose-SAM (untrained), magenta = our fine-tuned model, cyan = curated truth" }] : fallback;

modelSlide({
  title: "Toxoplasma PV v4", key: "toxoplasma_pv_v4", icon: "mask",
  arch: "Cellpose-SAM (cpsam_v2), round 7", task: "Outlines Toxoplasma parasitophorous vacuoles in anti-Toxoplasma or DsRed images (RH and ME49).",
  images: predImages("toxoplasma_pv_v4", [{ src: "models/pv_examples.png", caption: "Curated vacuole outlines (cyan)", card: true }], "Unseen test field. Orange = vanilla Cellpose-SAM (cpsam_v2, never trained on our images), magenta = PV v4, cyan = curated truth"),
  chart: { title: "209 held-out fields, never seen in training", series: [{ name: "PV v4", labels: ["F1", "AJI", "Dice"], values: [0.761, 0.582, 0.835] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.377, 0.040, 0.091] }] },
  data: "Round 6's 556 curated fields plus 80 from a new plate: 23,340 vacuoles in 502 training fields. 11 wells have never been used for training.",
  notes: ["Re-scored on curated ground truth: F1 0.761, precision 0.77, recall 0.75", "On the new plate it was trained on, PV v3 finds more vacuoles (F1 0.90 vs 0.75): use v3 there", "Weak on published-figure crops, as is every PV model"],
  speaker: "Re-scored 2026-10-11 on curated ground truth, strict hold-out: only fields this model never saw (209 of 655). Numbers differ between models because the unseen sets differ; the card gives a common subset too.",
});

modelSlide({
  title: "Toxoplasma Plaque v5", key: "toxoplasma_plaque_v5", icon: "analyze_plaques", color: MAG,
  arch: "Cellpose-SAM (cpsam_v2), round 7", task: "Outlines plaques in crystal-violet wells: from gel imagers, our plates and published figures.",
  images: [{ src: "models/plaque_examples.png", caption: "Curated plaque outlines (cyan): our microscope (left) and a published figure (right)", card: true }],
  chart: { title: "153 held-out fields, never seen in training", series: [{ name: "Plaque v5", labels: ["F1", "AJI", "Dice"], values: [0.816, 0.737, 0.896] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.227, 0.178, 0.329] }] },
  data: "1,507 curated fields (1,201 train, 153 validation, 153 test): Gel Doc wells, lab plates and published figures; 49,360 training plaques.",
  notes: ["Default since 2026-10-17: built on cpsam_v2", "Precision 0.87, recall 0.77; on the 153 test fields v4 scores F1 0.725 and v3 0.741", "On Gel Doc wells v3 scores higher (0.954 vs 0.868): use v3 there", "v4 stays selectable"],
  speaker: "Example images show curated ground truth, not predictions.",
});

modelSlide({
  title: "Plaque Well Detector v3", key: "toxoplasma_well_detector_v3", icon: "model_zoo", color: AMBER,
  arch: "YOLO11n object detector (the default)", task: "Draws a box around every plaque well in a plate photo or a published figure: the first step before plaque segmentation.",
  images: [{ src: "models/welldet_example.png", caption: "Detected wells (cyan) in a published figure; fluorescence and EM panels ignored. Shown with an earlier detector", card: true }],
  defs: DET_DEFS,
  chart: { title: "73 held-out images", max: 1.15, colors: [AMBER, AMBER, AMBER, AMBER], series: [{ name: "Detector v3", labels: ["mAP50", "mAP50-95", "precision", "recall"], values: [0.994, 0.884, 0.970, 0.998] }] },
  data: "452 reviewed training images (1,679 wells): our plates, Gel Doc plates and figures from published papers, grouped by plate and figure.",
  notes: ["The default well detector since 2026-10-12", "Its tight-box score (mAP50-95) is below the previous default's on the 73 clean images, and above it on 3 Gel Doc plates", "Finds wells, not plaques; the well size also gives the mm² scale"],
  speaker: "The example image is from an earlier detector. v3 scores are on the clean incumbent-validation images; the Gel Doc plates selected the checkpoint, so they are validation rather than independent test.",
});

const CC = (title, key, task, data, chart, notesList, icon) => modelSlide({
  title, key, icon: icon || "measure", color: BLUE,
  arch: "Cellpose-SAM (cpsam_v2), cross-channel", task,
  images: predImages(key, [{ src: "models/pv_examples.png", caption: "example" }]),
  chart, data, notes: notesList,
  speaker: "Cross-channel: the model sees only one channel and predicts objects that are stained in another. Caveat: the test wells were also used to pick the best checkpoint.",
});

CC("Toxoplasma from Cell Mask", "toxoplasma_from_cellmask_v1", "Finds Toxoplasma vacuoles from the host cell-mask channel alone: no parasite stain needed.",
  "2,567 training fields; tested on 463 fields from other wells. HFF, HeLa and THP-1 hosts.",
  { title: "463 unseen fields", colors: [BLUE, "5F6B78"], series: [{ name: "fine-tuned", labels: ["F1", "AJI", "Dice"], values: [0.606, 0.494, 0.610] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.021, 0.008, 0.020] }] },
  ["Stock Cellpose cannot do this at all", "F1 by host: HeLa 0.71 · HFF 0.56 · THP-1 0.47", "For counting, occupancy and area: not fine morphology"]);

CC("Nuclei from Cell Mask", "nuclei_from_cellmask_v1", "Finds nuclei from the cell-mask channel alone: frees the DAPI / Hoechst channel for another marker.",
  "Cell-mask images paired with curated nuclear masks; tested on 453 fields from wells never used for training. HFF, HeLa and THP-1.",
  { title: "453 unseen fields", colors: [BLUE, "5F6B78"], series: [{ name: "fine-tuned", labels: ["F1", "AJI", "Dice"], values: [0.888, 0.792, 0.877] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.201, 0.286, 0.449] }] },
  ["F1 by host: HFF 0.93 · HeLa 0.86 · THP-1 0.86", "Learns nuclei from cell shape: expect worse on unusual or very confluent cells"]);

CC("Cell from Hoechst", "cell_from_hoechst_v1", "Finds whole host cells from the Hoechst (nuclear) channel alone: the counterpart of nuclei-from-cell-mask.",
  "2,578 training fields (237,957 cells); tested on 451 fields (45,098 cells) from other wells.",
  { title: "451 unseen fields", colors: [BLUE, "5F6B78"], series: [{ name: "fine-tuned", labels: ["F1", "AJI", "Dice"], values: [0.870, 0.799, 0.895] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.301, 0.351, 0.524] }] },
  ["Misses some cells but rarely invents one (precision 0.94, recall 0.81)", "F1 0.57 higher than stock"]);

CC("Toxoplasma from Hoechst", "toxoplasma_from_hoechst_v1", "Finds Toxoplasma vacuoles from the Hoechst channel alone: no parasite stain needed.",
  "2,567 training fields; tested on 463 fields from other wells. HFF, HeLa and THP-1 hosts.",
  { title: "463 unseen fields", colors: [BLUE, "5F6B78"], series: [{ name: "fine-tuned", labels: ["F1", "AJI", "Dice"], values: [0.569, 0.421, 0.546] }, { name: "vanilla Cellpose-SAM", labels: ["F1", "AJI", "Dice"], values: [0.002, 0.006, 0.016] }] },
  ["Harder than the cell-mask route (F1 0.57 vs 0.61)", "Stock Cellpose finds essentially nothing", "For counting, occupancy and area"]);

{
  const s = base();
  header(s, { icon: "model_zoo", iconColor: PANEL2, kicker: "Model Zoo", title: "Other models available in spaCR" });
  const groups = [
    ["Segmentation", TEAL, ["Cellpose 4: Cellpose-SAM (default) and Cellpose-DINO", "Cellpose 3 models (cyto3, nuclei, …)", "Cellpose models from bioimage.io", "Models other users have contributed"]],
    ["Image classifiers (Classify)", MAG, ["ResNet (default ResNet-50), ResNeXt, Wide ResNet", "EfficientNet, MobileNet, RegNet", "ConvNeXt, VGG, DenseNet", "Vision transformers: ViT, Swin, MaxViT", "All start from networks pre-trained on everyday photos"]],
    ["Feature classifiers (Classify)", BLUE, ["XGBoost (default), LightGBM, CatBoost", "Random forest and other tree models", "Logistic regression, support vector machine, small neural network"]],
    ["Embeddings & detection", AMBER, ["Embeddings: DINOv2 (default; Cell-DINO when installed), ResNet-18, ResNet-50, ConvNeXt, ViT", "YOLO object detection (behind the well detector)"]],
  ];
  const gap = 0.25, cw = (W - 2 * M - gap) / 2, ch = 2.62;
  groups.forEach(([t, c, items], i) => {
    const x = M + (i % 2) * (cw + gap), y = 1.45 + Math.floor(i / 2) * (ch + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: PANEL }, line: { color: LINE, width: 0.75 }, rectRadius: 0.1 });
    s.addText(t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.4, fontFace: HF, fontSize: 17, bold: true, color: c, margin: 0, isTextBox: true });
    bullets(s, items, x + 0.3, y + 0.65, cw - 0.55, ch - 0.8, 12.5);
  });
  s.addNotes("Legacy Cellpose names (cyto, cyto2, cyto3, nuclei) in Cellpose 4 are redirected to cpsam. Not in spaCR: Mesmer.");
}

// ---------------------------------------------------------------- close
{
  const s = base();
  s.addText(L("Try it", "Get started"), { x: M + 0.2, y: 0.8, w: 7, h: 0.9, fontFace: HF, fontSize: 44, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 0.2, y: 1.95, w: 7.4, h: 0.85, fill: { color: "0A0E12" }, line: { color: LINE }, rectRadius: 0.08 });
  s.addText('pip install "spacr[qt]"   &&   spacr', { x: M + 0.5, y: 1.95, w: 7.0, h: 0.85, fontFace: MONO, fontSize: 20, color: "C8F2EE", valign: "middle", margin: 0, isTextBox: true });
  const links = [["Docs & tutorials", "einarolafsson.github.io/spacr"], ["Source & installers", "github.com/EinarOlafsson/spacr"], ["Preprint", "bioRxiv 10.64898/2026.07.08.737057"], ["Models", "huggingface.co/einarolafsson"], ["Example data", "spacr-download --list"]];
  links.forEach(([k, v], i) => {
    const y = 3.25 + i * 0.62;
    s.addText(k, { x: M + 0.2, y, w: 2.6, h: 0.45, fontFace: BF, fontSize: 16, bold: true, color: NEUT, margin: 0, valign: "middle", isTextBox: true });
    s.addText(v, { x: M + 2.9, y, w: 5.5, h: 0.45, fontFace: BF, fontSize: 16, color: TXT, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addImage({ path: IMG("icons/logo.png"), x: 8.6, y: PUBLIC ? 2.9 : 1.9, w: 4.2, h: 4.2 / 2.42 });
  if (!PUBLIC) s.addText("Questions?", { x: 8.6, y: 4.0, w: 4.2, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: MAG, align: "center", margin: 0, isTextBox: true });
  s.addNotes("Bugs and feature requests: GitHub issues.");
}

pres.writeFile({ fileName: path.join(ROOT, PUBLIC ? "spacr_overview.pptx" : "spacr_lab_meeting.pptx") }).then((f) => console.log("wrote", f));
