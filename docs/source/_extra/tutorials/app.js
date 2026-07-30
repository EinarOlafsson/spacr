"use strict";

const VOICE_CATALOG = window.SPACR_VOICE_CATALOG || [];
const DEFAULT_LANGUAGE = "en";
const DEFAULT_VOICE = "af_heart";
const PRODUCTION_ROOT =
  document.documentElement.dataset.productionRoot || "../production";

const READY = {
  pypi: {
    id: "01_pypi_github",
    title: "PyPI & GitHub",
    duration: "1 min 46 sec",
    durationShort: "1:46",
    description: "Understand where releases, installers, source code, issues, and the nightly branch live.",
    objectives: [
      "Use PyPI as the reference for the current Python package.",
      "Find desktop installers and release notes on GitHub.",
      "Distinguish the packaged release from the nightly development branch."
    ],
    prerequisite: "No installation is required for this orientation.",
    poster: `${PRODUCTION_ROOT}/01_pypi_github/keyframes/01_pypi_top.png`,
    video: `${PRODUCTION_ROOT}/01_pypi_github/video/01_pypi_github_af_heart.mp4`,
    silent: `${PRODUCTION_ROOT}/01_pypi_github/video/01_pypi_github_silent.mp4`,
    captions: "captions/01_pypi_github_en.vtt",
    scenes: `${PRODUCTION_ROOT}/01_pypi_github/scenes.json`,
    timings: `${PRODUCTION_ROOT}/01_pypi_github/audio/timings_af_heart.json`
  },
  conda: {
    id: "02_conda_install",
    title: "Installation with Conda",
    duration: "1 min 5 sec",
    durationShort: "1:05",
    description: "Create an isolated Python 3.12 environment and install spaCR.",
    objectives: [
      "Create and activate a dedicated Conda environment.",
      "Install spaCR with pip inside the active environment.",
      "Understand the current Conda Forge status."
    ],
    prerequisite: "Install Miniconda, Anaconda, or another Conda-compatible environment manager.",
    poster: `${PRODUCTION_ROOT}/02_conda_install/keyframes/01.png`,
    video: `${PRODUCTION_ROOT}/02_conda_install/video/02_conda_install_af_heart.mp4`,
    silent: `${PRODUCTION_ROOT}/02_conda_install/video/02_conda_install_silent.mp4`,
    captions: "captions/02_conda_install_en.vtt",
    scenes: `${PRODUCTION_ROOT}/02_conda_install/scenes.json`,
    timings: `${PRODUCTION_ROOT}/02_conda_install/audio/timings_af_heart.json`
  },
  pip: {
    id: "03_pip_install",
    title: "Installation with pip",
    duration: "59 sec",
    durationShort: "0:59",
    description: "Install spaCR in an existing environment for desktop or command-line use.",
    objectives: [
      "Confirm a supported Python version.",
      "Install spaCR with a simple pip command.",
      "Use the same installation for command-line workflows on headless systems."
    ],
    prerequisite: "Use an isolated Python environment when possible; Python 3.12 is recommended.",
    poster: `${PRODUCTION_ROOT}/03_pip_install/keyframes/01.png`,
    video: `${PRODUCTION_ROOT}/03_pip_install/video/03_pip_install_af_heart.mp4`,
    silent: `${PRODUCTION_ROOT}/03_pip_install/video/03_pip_install_silent.mp4`,
    captions: "captions/03_pip_install_en.vtt",
    scenes: `${PRODUCTION_ROOT}/03_pip_install/scenes.json`,
    timings: `${PRODUCTION_ROOT}/03_pip_install/audio/timings_af_heart.json`
  },
  installers: {
    id: "04_platform_installers",
    title: "Platform installers",
    duration: "1 min 15 sec",
    durationShort: "1:15",
    description: "Choose the Windows, macOS, or Linux online installer when you do not want to manage Python.",
    objectives: [
      "Select the installer that matches your operating system.",
      "Handle Gatekeeper and Linux executable permissions.",
      "Understand private runtimes, acceleration options, and install logs."
    ],
    prerequisite: "No existing Python or Conda installation is required.",
    poster: `${PRODUCTION_ROOT}/04_platform_installers/keyframes/01.png`,
    video: `${PRODUCTION_ROOT}/04_platform_installers/video/04_platform_installers_af_heart.mp4`,
    silent: `${PRODUCTION_ROOT}/04_platform_installers/video/04_platform_installers_silent.mp4`,
    captions: "captions/04_platform_installers_en.vtt",
    scenes: `${PRODUCTION_ROOT}/04_platform_installers/scenes.json`,
    timings: `${PRODUCTION_ROOT}/04_platform_installers/audio/timings_af_heart.json`
  },
  home: {
    id: "05_home",
    title: "Home screen & navigation",
    duration: "1 min 34 sec",
    durationShort: "1:34",
    description: "Tour the nightly application, understand module groups, and open the Mask workflow.",
    objectives: [
      "Navigate with the sidebar, Home tiles, and category tabs.",
      "Read system, run-history, and module-maturity panels.",
      "Open the Mask module from the Core section."
    ],
    prerequisite: "Install and launch the spaCR desktop application.",
    poster: `${PRODUCTION_ROOT}/05_home/keyframes/01_home.png`,
    video: `${PRODUCTION_ROOT}/05_home/video/05_home_af_heart.mp4`,
    silent: `${PRODUCTION_ROOT}/05_home/video/05_home_silent.mp4`,
    captions: "captions/05_home_en.vtt",
    scenes: `${PRODUCTION_ROOT}/05_home/scenes.json`,
    timings: `${PRODUCTION_ROOT}/05_home/audio/timings_af_heart.json`
  }
};

const SERIES = [
  {
    title: "Getting started & core analysis",
    lessons: [
      READY.pypi,
      READY.conda,
      READY.pip,
      READY.installers,
      READY.home,
      planned("06_api", "Python API & headless workflows",
        "Run spaCR without the desktop interface and translate reproducible settings into Python workflows.",
        ["Choose an API entry point.", "Load reproducible settings.", "Run and monitor a headless workflow."],
        "Complete one installation tutorial first."),
      planned("07_mask", "Mask",
        "Segment the supplied four-channel experiment into cell, nucleus, pathogen, and lipid-droplet organelle masks.",
        ["Map the four acquisition channels.", "Preview four segmentation passes.", "Run Mask and inspect its outputs."],
        "Use the supplied 13-field tutorial dataset."),
      planned("08_measure", "Measure",
        "Extract per-object intensity, morphology, texture, colocalization, and radial-distribution features."),
      planned("09_annotate", "Annotate",
        "Review object crops and assign reliable labels for downstream model training."),
      planned("10_classify_cv", "Classify · computer vision",
        "Train and apply an image-based classifier to annotated object crops."),
      planned("11_classify_ml", "Classify · machine learning",
        "Train a feature-based model from measured single-object data."),
      planned("12_map_barcodes", "Map Barcodes",
        "Connect sequencing reads and guide identities to microscopy phenotypes."),
      planned("13_regression", "Regression",
        "Estimate guide effects and identify phenotype-associated perturbations."),
      planned("14_make_masks", "Make Masks",
        "Interactively construct or refine masks for model-training data."),
      planned("15_image_umap", "Image UMAP",
        "Explore object-image embeddings and phenotype neighborhoods."),
      planned("16_activation", "Activation maps",
        "Inspect which image regions influence a trained classifier.")
    ]
  },
  {
    title: "Time-resolved analysis & segmentation models",
    lessons: [
      planned("17_timelapse", "Timelapse", "Segment, link, and measure objects across time."),
      planned("18_motility", "Motility assay", "Quantify motion, displacement, and trajectory-level phenotypes."),
      planned("19_train_cellpose", "Train Cellpose", "Train a custom CPSAM-compatible segmentation checkpoint."),
      planned("20_cellpose_masks", "Cellpose Masks", "Review, compare, and manage Cellpose mask outputs."),
      planned("21_model_compare", "Model Compare", "Compare segmentation models on the same images and metrics."),
      planned("22_model_zoo", "Model Zoo", "Browse, import, and apply reusable segmentation models.")
    ]
  },
  {
    title: "Annotation QC & biological assays",
    lessons: [
      planned("23_agreement", "Annotator Agreement", "Quantify agreement and resolve inconsistent labels."),
      planned("24_plaque", "Plaque assay", "Measure plaque formation and treatment-dependent effects."),
      planned("25_recruitment", "Recruitment", "Quantify recruitment of an ER-channel signal to pathogens."),
      planned("26_invasion", "Invasion assay", "Classify and quantify intracellular versus extracellular parasites."),
      planned("27_replication", "Replication assay", "Measure parasite replication distributions per host cell.")
    ]
  },
  {
    title: "Operations, reporting & data utilities",
    lessons: [
      planned("28_training_runs", "Training Runs", "Inspect model runs, parameters, metrics, and artifacts."),
      planned("29_report", "Report", "Assemble reproducible QC and analysis summaries."),
      planned("30_plate_queue", "Plate Queue", "Schedule and monitor multi-plate processing."),
      planned("31_external_masks", "External Masks", "Import third-party masks into a valid spaCR project."),
      planned("32_align_stitch", "Align & Stitch", "Register channels and assemble tiled acquisitions."),
      planned("33_plate_viewer", "Plate Viewer", "Explore fields, wells, masks, and measurements spatially."),
      planned("34_database", "Database Browser", "Inspect and query spaCR measurement databases."),
      planned("35_converter", "Format Converter", "Convert microscopy formats into mapped Yokogawa-style TIFFs.")
    ]
  },
  {
    title: "Additional registered modules",
    lessons: [
      planned("36_import", "Import Project", "Bring an existing dataset into spaCR with traceable mappings."),
      planned("37_batch", "Batch Runner", "Run reproducible workflows across multiple projects.")
    ]
  }
];

function planned(id, title, description, objectives, prerequisite) {
  return {
    id,
    title,
    description,
    objectives: objectives || [
      `Understand the purpose and inputs of ${title}.`,
      "Configure the important settings.",
      "Run the workflow and locate its outputs."
    ],
    prerequisite: prerequisite || "Earlier workflow tutorials will provide the required project outputs.",
    duration: "In production",
    durationShort: "Soon",
    state: "planned"
  };
}

SERIES.forEach((series, seriesIndex) => {
  series.number = seriesIndex + 1;
  series.lessons.forEach(lesson => {
    lesson.series = series.number;
    lesson.seriesTitle = series.title;
    lesson.state = lesson.state || "ready";
  });
});

const LESSONS = SERIES.flatMap(series => series.lessons);
const READY_LESSONS = LESSONS.filter(lesson => lesson.state === "ready");
const STORAGE_KEY = "spacr-tutorial-progress-v1";
const WATCH_KEY = "spacr-tutorial-watch-v1";
const LANGUAGE_KEY = "spacr-tutorial-language-v1";
const VOICE_KEY = "spacr-tutorial-voice-v1";

const $ = selector => document.querySelector(selector);
const elements = {
  curriculum: $("#curriculum"),
  search: $("#lesson-search"),
  seriesLabel: $("#series-label"),
  position: $("#lesson-position"),
  status: $("#status-pill"),
  title: $("#lesson-title"),
  description: $("#lesson-description"),
  duration: $("#duration-badge"),
  objectives: $("#objective-list"),
  prerequisite: $("#prerequisite-copy"),
  player: $("#ready-player"),
  planned: $("#planned-card"),
  plannedTitle: $("#planned-title"),
  plannedCopy: $("#planned-copy"),
  plannedQueue: $("#planned-queue"),
  video: $("#tutorial-video"),
  loading: $("#video-loading"),
  captionTrack: $("#caption-track"),
  voice: $("#voice-select"),
  language: $("#language-select"),
  watchTime: $("#watch-time"),
  watchBar: $("#watch-bar"),
  chapters: $("#chapter-list"),
  transcript: $("#transcript-list"),
  chapterCard: $("#chapter-card"),
  previous: $("#previous-button"),
  next: $("#next-button"),
  previousTitle: $("#previous-title"),
  nextTitle: $("#next-title"),
  complete: $("#complete-button"),
  completeLabel: $("#complete-label"),
  copyLink: $("#copy-link-button"),
  continue: $("#continue-button"),
  progressLabel: $("#progress-label"),
  progressBar: $("#progress-bar"),
  availableCount: $("#available-count"),
  totalCount: $("#total-count"),
  sidebar: $("#sidebar"),
  scrim: $("#sidebar-scrim"),
  menuButton: $("#menu-button"),
  closeMenu: $("#close-menu-button"),
  toast: $("#toast")
};

let activeLesson = null;
let chapterData = [];
let toastTimer = null;
let completed = readStoredSet(STORAGE_KEY);
let watchProgress = readStoredObject(WATCH_KEY);

function languageById(id) {
  return VOICE_CATALOG.find(language => language.id === id) ||
    VOICE_CATALOG.find(language => language.id === DEFAULT_LANGUAGE) ||
    VOICE_CATALOG[0];
}

function voiceById(language, id) {
  return language?.voices.find(voice => voice.id === id);
}

function renderedVoiceSource(lesson, languageId, voiceId) {
  if (voiceId === "silent") return lesson.silent;
  const configured = lesson.narration?.[languageId]?.[voiceId];
  if (configured) return configured;
  // Heart is the initial mastered narration already muxed into each video.
  if (languageId === "en" && voiceId === "af_heart") return lesson.video;
  return "";
}

function populateVoiceSelector(preferredVoice = "") {
  const language = languageById(elements.language.value);
  if (!language) return;

  elements.voice.innerHTML = "";
  language.voices.forEach(voice => {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = `${voice.name} · ${voice.variant}`;
    option.dataset.engineCode = voice.engineCode;
    elements.voice.appendChild(option);
  });

  const silent = document.createElement("option");
  silent.value = "silent";
  silent.textContent = "Silent master";
  elements.voice.appendChild(silent);

  const preferredExists = preferredVoice === "silent" ||
    Boolean(voiceById(language, preferredVoice));
  elements.voice.value = preferredExists ? preferredVoice : language.voices[0]?.id || "silent";
}

function setupNarrationSelectors() {
  const storedLanguage = localStorage.getItem(LANGUAGE_KEY) || DEFAULT_LANGUAGE;
  const language = languageById(storedLanguage);

  elements.language.innerHTML = "";
  VOICE_CATALOG.forEach(item => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.label} · ${item.voices.length} ${item.voices.length === 1 ? "voice" : "voices"}`;
    elements.language.appendChild(option);
  });
  elements.language.value = language?.id || DEFAULT_LANGUAGE;

  const storedVoice = localStorage.getItem(VOICE_KEY) ||
    (elements.language.value === DEFAULT_LANGUAGE ? DEFAULT_VOICE : "");
  populateVoiceSelector(storedVoice);
}

function selectedVideoSource(lesson, notify = false) {
  const language = languageById(elements.language.value);
  const voice = elements.voice.value;
  const rendered = renderedVoiceSource(lesson, language?.id, voice);
  if (rendered) return rendered;

  if (notify) {
    const selected = voiceById(language, voice);
    const label = selected ? `${selected.name} in ${language.label}` : language?.label;
    showToast(`${label} is configured; narration rendering is pending. Playing the silent master.`);
  }
  return lesson.silent;
}

function renderCurriculum(query = "") {
  const normalized = query.trim().toLowerCase();
  elements.curriculum.innerHTML = "";
  let matches = 0;

  SERIES.forEach(series => {
    const filtered = series.lessons.filter(lesson => {
      if (!normalized) return true;
      return `${lesson.title} ${lesson.description} ${series.title}`
        .toLowerCase().includes(normalized);
    });
    if (!filtered.length) return;
    matches += filtered.length;

    const block = document.createElement("section");
    block.className = "series-block";
    block.dataset.series = series.number;

    const toggle = document.createElement("button");
    toggle.className = "series-toggle";
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "true");
    toggle.innerHTML = `
      <span>
        <small>Series ${series.number}</small>
        <strong>${escapeHTML(series.title)}</strong>
      </span>
      <span class="series-count">${filtered.length}</span>
      <svg class="chevron" viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6 9 6 6 6-6"/>
      </svg>`;
    toggle.addEventListener("click", () => {
      const collapsed = block.classList.toggle("collapsed");
      toggle.setAttribute("aria-expanded", String(!collapsed));
    });

    const lessonList = document.createElement("div");
    lessonList.className = "series-lessons";
    filtered.forEach(lesson => lessonList.appendChild(makeLessonLink(lesson)));

    block.append(toggle, lessonList);
    elements.curriculum.appendChild(block);
  });

  if (!matches) {
    elements.curriculum.innerHTML =
      `<div class="empty-search">No tutorials match “${escapeHTML(query)}”.</div>`;
  }
}

function makeLessonLink(lesson) {
  const index = LESSONS.indexOf(lesson) + 1;
  const button = document.createElement("button");
  button.type = "button";
  button.className = `lesson-link ${lesson.state}`;
  button.dataset.lesson = lesson.id;
  button.setAttribute("aria-current", lesson.id === activeLesson?.id ? "page" : "false");
  if (lesson.id === activeLesson?.id) button.classList.add("active");
  if (completed.has(lesson.id)) button.classList.add("complete");
  button.innerHTML = `
    <span class="lesson-number">${String(index).padStart(2, "0")}</span>
    <span class="lesson-link-copy">
      <strong>${escapeHTML(lesson.title)}</strong>
      <small>${lesson.state === "ready" ? lesson.durationShort : "In production"}</small>
    </span>
    <span class="lesson-state-dot" aria-hidden="true"></span>`;
  button.addEventListener("click", () => selectLesson(lesson.id));
  return button;
}

async function selectLesson(id, options = {}) {
  const lesson = LESSONS.find(item => item.id === id) || LESSONS[0];
  const prior = activeLesson;
  if (prior?.id === lesson.id && !options.force) {
    closeSidebar();
    return;
  }

  if (prior?.state === "ready") saveWatchPosition(prior);
  activeLesson = lesson;
  if (!options.skipHash) history.replaceState(null, "", `#lesson=${lesson.id}`);

  renderCurriculum(elements.search.value);
  updateLessonHeader(lesson);
  updateGuide(lesson);
  updatePagination(lesson);
  updateCompleteButton();
  closeSidebar();

  if (lesson.state === "ready") {
    elements.player.hidden = false;
    elements.planned.hidden = true;
    elements.chapterCard.hidden = false;
    await loadReadyLesson(lesson);
  } else {
    elements.player.hidden = true;
    elements.planned.hidden = false;
    elements.chapterCard.hidden = true;
    elements.video.pause();
    showPlannedLesson(lesson);
  }

  if (options.focus) $("#lesson-content").focus({ preventScroll: true });
}

function updateLessonHeader(lesson) {
  const index = LESSONS.indexOf(lesson) + 1;
  elements.seriesLabel.textContent = `Series ${lesson.series}`;
  elements.position.textContent = `Lesson ${index} of ${LESSONS.length}`;
  elements.status.textContent = lesson.state === "ready" ? "Ready" : "In production";
  elements.status.className = `status-pill ${lesson.state === "ready" ? "ready" : ""}`;
  elements.title.textContent = lesson.title;
  elements.description.textContent = lesson.description;
  document.title = `${lesson.title} · spaCR Learning Path`;
}

function updateGuide(lesson) {
  elements.duration.textContent = lesson.duration;
  elements.objectives.innerHTML = lesson.objectives
    .map(item => `<li>${escapeHTML(item)}</li>`).join("");
  elements.prerequisite.textContent = lesson.prerequisite;
}

function showPlannedLesson(lesson) {
  const plannedLessons = LESSONS.filter(item => item.state !== "ready");
  const queuePosition = plannedLessons.indexOf(lesson) + 1;
  elements.plannedTitle.textContent = `${lesson.title} is in production`;
  elements.plannedCopy.textContent = lesson.description;
  elements.plannedQueue.textContent = `Production queue ${queuePosition} of ${plannedLessons.length}`;
}

async function loadReadyLesson(lesson) {
  elements.loading.classList.remove("hidden");
  const source = selectedVideoSource(lesson);
  const priorTime = watchProgress[lesson.id] || 0;

  elements.video.poster = lesson.poster;
  elements.captionTrack.src = lesson.captions;
  elements.video.src = source;
  elements.video.load();

  const metadataReady = () => {
    elements.loading.classList.add("hidden");
    if (Number.isFinite(priorTime) && priorTime > 0 &&
        priorTime < elements.video.duration - 2) {
      elements.video.currentTime = priorTime;
    }
    updateWatchUI();
  };
  elements.video.addEventListener("loadedmetadata", metadataReady, { once: true });

  await loadLessonDetail(lesson);
}

async function loadLessonDetail(lesson) {
  elements.chapters.innerHTML = `<div class="detail-empty">Loading chapters…</div>`;
  elements.transcript.innerHTML = "";
  chapterData = [];

  try {
    const [sceneResponse, timingResponse] = await Promise.all([
      fetch(lesson.scenes),
      fetch(lesson.timings)
    ]);
    if (!sceneResponse.ok || !timingResponse.ok) throw new Error("Lesson metadata unavailable");
    const sceneSpec = await sceneResponse.json();
    const timingSpec = await timingResponse.json();
    chapterData = sceneSpec.scenes.map((scene, index) => ({
      index: index + 1,
      start: timingSpec.scenes[index]?.speech_start || 0,
      end: timingSpec.scenes[index]?.scene_end || 0,
      text: scene.narration,
      label: chapterLabel(scene.narration, index)
    }));
    renderChapters();
    renderTranscript();
  } catch (error) {
    elements.chapters.innerHTML = `
      <div class="detail-empty">
        Start this portal through its local web server to load chapters and transcripts.
      </div>`;
    elements.transcript.innerHTML = elements.chapters.innerHTML;
  }
}

function renderChapters() {
  elements.chapters.innerHTML = "";
  chapterData.forEach(chapter => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chapter-button";
    button.dataset.start = chapter.start;
    button.innerHTML = `
      <span class="chapter-time">${formatTime(chapter.start)}</span>
      <span class="chapter-copy">
        <strong>${escapeHTML(chapter.label)}</strong>
        <small>${escapeHTML(truncate(chapter.text, 82))}</small>
      </span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>`;
    button.addEventListener("click", () => seekTo(chapter.start));
    elements.chapters.appendChild(button);
  });
}

function renderTranscript() {
  elements.transcript.innerHTML = chapterData.map(chapter => `
    <div class="transcript-entry">
      <button type="button" data-seek="${chapter.start}">${formatTime(chapter.start)}</button>
      <p>${escapeHTML(chapter.text)}</p>
    </div>`).join("");
  elements.transcript.querySelectorAll("[data-seek]").forEach(button => {
    button.addEventListener("click", () => seekTo(Number(button.dataset.seek)));
  });
}

function seekTo(seconds) {
  if (!activeLesson || activeLesson.state !== "ready") return;
  elements.video.currentTime = seconds;
  elements.video.play().catch(() => {});
}

function chapterLabel(text, index) {
  const sentence = text.split(/[.!?]/)[0]
    .replace(/^This is\s+/i, "")
    .replace(/^The\s+/i, "");
  const words = sentence.split(/\s+/).slice(0, 6).join(" ");
  return words || `Chapter ${index + 1}`;
}

function updatePagination(lesson) {
  const index = LESSONS.indexOf(lesson);
  const previous = LESSONS[index - 1];
  const next = LESSONS[index + 1];

  elements.previous.disabled = !previous;
  elements.previousTitle.textContent = previous?.title || "Beginning of path";
  elements.next.disabled = !next;
  elements.nextTitle.textContent = next?.title || "End of path";
  elements.previous.onclick = previous ? () => selectLesson(previous.id, { focus: true }) : null;
  elements.next.onclick = next ? () => selectLesson(next.id, { focus: true }) : null;
}

function updateWatchUI() {
  const current = Number.isFinite(elements.video.currentTime) ? elements.video.currentTime : 0;
  const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
  const percent = duration ? Math.min(100, current / duration * 100) : 0;
  elements.watchTime.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
  elements.watchBar.style.width = `${percent}%`;
  updateCurrentChapter(current);
}

function updateCurrentChapter(currentTime) {
  const currentIndex = chapterData.findIndex((chapter, index) => {
    const next = chapterData[index + 1];
    return currentTime >= chapter.start && (!next || currentTime < next.start);
  });
  elements.chapters.querySelectorAll(".chapter-button").forEach((button, index) => {
    button.classList.toggle("current", index === currentIndex);
  });
}

function saveWatchPosition(lesson = activeLesson) {
  if (!lesson || lesson.state !== "ready") return;
  watchProgress[lesson.id] = elements.video.currentTime || 0;
  localStorage.setItem(WATCH_KEY, JSON.stringify(watchProgress));
}

function toggleComplete() {
  if (!activeLesson || activeLesson.state !== "ready") return;
  if (completed.has(activeLesson.id)) {
    completed.delete(activeLesson.id);
    showToast("Lesson marked incomplete");
  } else {
    completed.add(activeLesson.id);
    showToast("Lesson complete");
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
  updateCompleteButton();
  updateCourseProgress();
  renderCurriculum(elements.search.value);
}

function updateCompleteButton() {
  const available = activeLesson?.state === "ready";
  elements.complete.hidden = !available;
  if (!available) return;
  const isComplete = completed.has(activeLesson.id);
  elements.complete.classList.toggle("completed", isComplete);
  elements.completeLabel.textContent = isComplete ? "Completed" : "Mark complete";
  elements.complete.setAttribute("aria-pressed", String(isComplete));
}

function updateCourseProgress() {
  const readyComplete = READY_LESSONS.filter(lesson => completed.has(lesson.id)).length;
  const percentage = READY_LESSONS.length ? readyComplete / READY_LESSONS.length * 100 : 0;
  elements.progressLabel.textContent = `${readyComplete} of ${READY_LESSONS.length} complete`;
  elements.progressBar.style.width = `${percentage}%`;
}

function continueCourse() {
  const next = READY_LESSONS.find(lesson => !completed.has(lesson.id)) || READY_LESSONS.at(-1);
  selectLesson(next.id, { focus: true });
}

function switchVoice() {
  if (!activeLesson || activeLesson.state !== "ready") return;
  localStorage.setItem(LANGUAGE_KEY, elements.language.value);
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  const time = elements.video.currentTime || 0;
  const wasPlaying = !elements.video.paused;
  const source = selectedVideoSource(activeLesson, true);
  elements.loading.classList.remove("hidden");
  elements.video.src = source;
  elements.video.load();
  elements.video.addEventListener("loadedmetadata", () => {
    elements.video.currentTime = Math.min(time, Math.max(0, elements.video.duration - .25));
    elements.loading.classList.add("hidden");
    if (wasPlaying) elements.video.play().catch(() => {});
  }, { once: true });
}

function copyLessonLink() {
  const url = new URL(window.location.href);
  url.hash = `lesson=${activeLesson.id}`;
  navigator.clipboard?.writeText(url.href)
    .then(() => showToast("Tutorial link copied"))
    .catch(() => {
      window.prompt("Copy this tutorial link:", url.href);
    });
}

function setupDetailTabs() {
  const chapterTab = $("#chapters-tab");
  const transcriptTab = $("#transcript-tab");
  const chapterPanel = $("#chapters-panel");
  const transcriptPanel = $("#transcript-panel");

  function select(tab) {
    const showChapters = tab === "chapters";
    chapterTab.classList.toggle("active", showChapters);
    transcriptTab.classList.toggle("active", !showChapters);
    chapterTab.setAttribute("aria-selected", String(showChapters));
    transcriptTab.setAttribute("aria-selected", String(!showChapters));
    chapterPanel.hidden = !showChapters;
    transcriptPanel.hidden = showChapters;
  }

  chapterTab.addEventListener("click", () => select("chapters"));
  transcriptTab.addEventListener("click", () => select("transcript"));
}

function openSidebar() {
  elements.sidebar.classList.add("open");
  elements.scrim.hidden = false;
  elements.menuButton.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  elements.sidebar.classList.remove("open");
  elements.scrim.hidden = true;
  elements.menuButton.setAttribute("aria-expanded", "false");
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2200);
}

function lessonFromHash() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  return params.get("lesson");
}

function readStoredSet(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function readStoredObject(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

function truncate(value, length) {
  return value.length > length ? `${value.slice(0, length - 1).trim()}…` : value;
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.search.addEventListener("input", event => renderCurriculum(event.target.value));
elements.video.addEventListener("timeupdate", () => {
  updateWatchUI();
  if (activeLesson?.state === "ready") {
    watchProgress[activeLesson.id] = elements.video.currentTime || 0;
  }
});
elements.video.addEventListener("pause", () => saveWatchPosition());
elements.video.addEventListener("ended", () => {
  if (activeLesson?.state === "ready" && !completed.has(activeLesson.id)) {
    completed.add(activeLesson.id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
    updateCompleteButton();
    updateCourseProgress();
    renderCurriculum(elements.search.value);
    showToast("Lesson completed");
  }
});
elements.video.addEventListener("canplay", () => elements.loading.classList.add("hidden"));
elements.video.addEventListener("error", () => {
  elements.loading.innerHTML = "Video unavailable. Start the portal from the tutorial workspace.";
  elements.loading.classList.remove("hidden");
});
elements.voice.addEventListener("change", switchVoice);
elements.language.addEventListener("change", () => {
  localStorage.setItem(LANGUAGE_KEY, elements.language.value);
  populateVoiceSelector();
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  switchVoice();
});
elements.complete.addEventListener("click", toggleComplete);
elements.copyLink.addEventListener("click", copyLessonLink);
elements.continue.addEventListener("click", continueCourse);
elements.menuButton.addEventListener("click", openSidebar);
elements.closeMenu.addEventListener("click", closeSidebar);
elements.scrim.addEventListener("click", closeSidebar);

window.addEventListener("beforeunload", () => saveWatchPosition());
window.addEventListener("hashchange", () => {
  const id = lessonFromHash();
  if (id && id !== activeLesson?.id) selectLesson(id, { skipHash: true });
});
window.addEventListener("keydown", event => {
  const tag = document.activeElement?.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
  if (event.key === "/") {
    event.preventDefault();
    elements.search.focus();
  } else if (event.key === "[") {
    event.preventDefault();
    elements.previous.click();
  } else if (event.key === "]") {
    event.preventDefault();
    elements.next.click();
  } else if (event.key === "Escape") {
    closeSidebar();
  }
});

elements.availableCount.textContent = READY_LESSONS.length;
elements.totalCount.textContent = LESSONS.length;
setupNarrationSelectors();
setupDetailTabs();
updateCourseProgress();
renderCurriculum();
selectLesson(lessonFromHash() || LESSONS[0].id, { skipHash: !location.hash, force: true });
