"use strict";

const VOICE_CATALOG = window.SPACR_VOICE_CATALOG || [];
const BASE_CATALOG = window.SPACR_LESSON_CATALOG;
const DEFAULT_LANGUAGE = "en";
const DEFAULT_VOICE = "af_heart";
const PRODUCTION_ROOT = document.documentElement.dataset.productionRoot || "../production";
const STORAGE_KEY = "spacr-tutorial-progress-v2";
const WATCH_KEY = "spacr-tutorial-watch-v2";
const LANGUAGE_KEY = "spacr-tutorial-language-v2";
const VOICE_KEY = "spacr-tutorial-voice-v2";

const $ = selector => document.querySelector(selector);
const elements = {
  curriculum: $("#curriculum"), search: $("#lesson-search"),
  seriesLabel: $("#series-label"), position: $("#lesson-position"),
  status: $("#status-pill"), title: $("#lesson-title"),
  description: $("#lesson-description"), duration: $("#duration-badge"),
  objectives: $("#objective-list"), prerequisite: $("#prerequisite-copy"),
  player: $("#ready-player"), planned: $("#planned-card"),
  video: $("#tutorial-video"), audio: $("#narration-audio"),
  loading: $("#video-loading"), captionTrack: $("#caption-track"),
  voice: $("#voice-select"), language: $("#language-select"),
  watchTime: $("#watch-time"), watchBar: $("#watch-bar"),
  chapters: $("#chapter-list"), transcript: $("#transcript-list"),
  chapterCard: $("#chapter-card"), previous: $("#previous-button"),
  next: $("#next-button"), previousTitle: $("#previous-title"),
  nextTitle: $("#next-title"), complete: $("#complete-button"),
  completeLabel: $("#complete-label"), copyLink: $("#copy-link-button"),
  continue: $("#continue-button"), progressLabel: $("#progress-label"),
  progressBar: $("#progress-bar"), availableCount: $("#available-count"),
  totalCount: $("#total-count"), sidebar: $("#sidebar"),
  scrim: $("#sidebar-scrim"), menuButton: $("#menu-button"),
  closeMenu: $("#close-menu-button"), toast: $("#toast")
};

const LESSONS = BASE_CATALOG.lessons;
const localizedCatalogs = new Map([["en", BASE_CATALOG]]);
let localizedCatalog = BASE_CATALOG;
let activeLesson = null;
let chapterData = [];
let audioTimings = null;
let syncRate = 1;
let captionUrl = "";
let toastTimer = null;
let completed = readStoredSet(STORAGE_KEY);
let watchProgress = readStoredObject(WATCH_KEY);

function languageById(id) {
  return VOICE_CATALOG.find(item => item.id === id) ||
    VOICE_CATALOG.find(item => item.id === DEFAULT_LANGUAGE) || VOICE_CATALOG[0];
}

function localizedLesson(id) {
  return localizedCatalog.lessons.find(item => item.id === id) ||
    LESSONS.find(item => item.id === id);
}

function baseLesson(id) {
  return LESSONS.find(item => item.id === id);
}

function seriesBlocks() {
  return localizedCatalog.series.map(series => ({
    ...series,
    lessons: LESSONS.filter(lesson => lesson.series === series.number)
  }));
}

function voiceById(language, id) {
  return language?.voices.find(voice => voice.id === id);
}

function audioSource(lesson = activeLesson) {
  if (!lesson || elements.voice.value === "silent") return "";
  return `${PRODUCTION_ROOT}/${lesson.id}/audio/${elements.language.value}/${elements.voice.value}.m4a`;
}

function timingSource(lesson = activeLesson) {
  return `${PRODUCTION_ROOT}/${lesson.id}/audio/${elements.language.value}/${elements.voice.value}.json`;
}

function populateVoiceSelector(preferredVoice = "") {
  const language = languageById(elements.language.value);
  elements.voice.innerHTML = "";
  language.voices.forEach(voice => {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = `${voice.name} · ${voice.variant}`;
    elements.voice.appendChild(option);
  });
  const silent = document.createElement("option");
  silent.value = "silent";
  silent.textContent = "Silent master";
  elements.voice.appendChild(silent);
  const available = preferredVoice === "silent" || voiceById(language, preferredVoice);
  elements.voice.value = available ? preferredVoice : language.voices[0]?.id || "silent";
}

function setupNarrationSelectors() {
  const preferredLanguage = localStorage.getItem(LANGUAGE_KEY) || DEFAULT_LANGUAGE;
  elements.language.innerHTML = "";
  VOICE_CATALOG.forEach(language => {
    const option = document.createElement("option");
    option.value = language.id;
    option.textContent = `${language.label} · ${language.voices.length} ${language.voices.length === 1 ? "voice" : "voices"}`;
    elements.language.appendChild(option);
  });
  elements.language.value = languageById(preferredLanguage).id;
  const preferredVoice = localStorage.getItem(VOICE_KEY) ||
    (elements.language.value === DEFAULT_LANGUAGE ? DEFAULT_VOICE : "");
  populateVoiceSelector(preferredVoice);
}

async function loadLocalizedCatalog(language) {
  if (localizedCatalogs.has(language)) return localizedCatalogs.get(language);
  const response = await fetch(`catalog/lessons_${language}.json`);
  if (!response.ok) throw new Error(`The ${language} tutorial catalog is unavailable`);
  const catalog = await response.json();
  localizedCatalogs.set(language, catalog);
  return catalog;
}

function renderCurriculum(query = "") {
  const normalized = query.trim().toLowerCase();
  elements.curriculum.innerHTML = "";
  let matches = 0;
  seriesBlocks().forEach(series => {
    const filtered = series.lessons.filter(base => {
      const lesson = localizedLesson(base.id);
      return !normalized || `${lesson.title} ${lesson.description} ${series.title}`
        .toLowerCase().includes(normalized);
    });
    if (!filtered.length) return;
    matches += filtered.length;
    const block = document.createElement("section");
    block.className = "series-block";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "series-toggle";
    toggle.setAttribute("aria-expanded", "true");
    toggle.innerHTML = `<span><small>Series ${series.number}</small><strong>${escapeHTML(series.title)}</strong></span><span class="series-count">${filtered.length}</span><svg class="chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>`;
    toggle.addEventListener("click", () => {
      const collapsed = block.classList.toggle("collapsed");
      toggle.setAttribute("aria-expanded", String(!collapsed));
    });
    const list = document.createElement("div");
    list.className = "series-lessons";
    filtered.forEach(lesson => list.appendChild(makeLessonLink(lesson)));
    block.append(toggle, list);
    elements.curriculum.appendChild(block);
  });
  if (!matches) elements.curriculum.innerHTML = `<div class="empty-search">No tutorials match “${escapeHTML(query)}”.</div>`;
}

function makeLessonLink(base) {
  const lesson = localizedLesson(base.id);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "lesson-link ready";
  button.dataset.lesson = base.id;
  if (base.id === activeLesson?.id) button.classList.add("active");
  if (completed.has(base.id)) button.classList.add("complete");
  button.setAttribute("aria-current", base.id === activeLesson?.id ? "page" : "false");
  button.innerHTML = `<span class="lesson-number">${String(base.number).padStart(2, "0")}</span><span class="lesson-link-copy"><strong>${escapeHTML(lesson.title)}</strong><small>4K video</small></span><span class="lesson-state-dot" aria-hidden="true"></span>`;
  button.addEventListener("click", () => selectLesson(base.id));
  return button;
}

async function selectLesson(id, options = {}) {
  const lesson = baseLesson(id) || LESSONS[0];
  if (activeLesson?.id === lesson.id && !options.force) {
    closeSidebar();
    return;
  }
  if (activeLesson) saveWatchPosition();
  elements.video.pause();
  activeLesson = lesson;
  if (!options.skipHash) history.replaceState(null, "", `#lesson=${lesson.id}`);
  renderCurriculum(elements.search.value);
  updateLessonHeader();
  updateGuide();
  updatePagination();
  updateCompleteButton();
  closeSidebar();
  elements.player.hidden = false;
  elements.planned.hidden = true;
  elements.chapterCard.hidden = false;
  await loadReadyLesson();
  if (options.focus) $("#lesson-content").focus({ preventScroll: true });
}

function updateLessonHeader() {
  const lesson = localizedLesson(activeLesson.id);
  const series = localizedCatalog.series.find(item => item.number === activeLesson.series);
  elements.seriesLabel.textContent = `Series ${activeLesson.series}`;
  elements.position.textContent = `Lesson ${activeLesson.number} of ${LESSONS.length}`;
  elements.status.textContent = "Ready";
  elements.status.className = "status-pill ready";
  elements.title.textContent = lesson.title;
  elements.description.textContent = lesson.description;
  document.title = `${lesson.title} · spaCR Learning Path`;
}

function updateGuide() {
  const lesson = localizedLesson(activeLesson.id);
  elements.duration.textContent = "Selectable narration";
  elements.objectives.innerHTML = lesson.objectives.map(item => `<li>${escapeHTML(item)}</li>`).join("");
  elements.prerequisite.textContent = lesson.prerequisite;
}

function mediaReady(media) {
  if (media.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    media.addEventListener("loadedmetadata", resolve, { once: true });
    media.addEventListener("error", reject, { once: true });
  });
}

async function loadReadyLesson() {
  elements.loading.classList.remove("hidden");
  elements.video.poster = `${PRODUCTION_ROOT}/${activeLesson.poster}`;
  elements.video.src = `${PRODUCTION_ROOT}/${activeLesson.silent}`;
  elements.video.load();
  const saved = watchProgress[activeLesson.id] || 0;
  await loadNarration(false);
  await mediaReady(elements.video);
  if (saved > 0 && saved < elements.video.duration - 0.25) elements.video.currentTime = saved;
  configureMediaSync();
  await loadLessonDetail();
  elements.loading.classList.add("hidden");
  updateWatchUI();
}

async function loadNarration(resume = true) {
  const wasPlaying = resume && !elements.video.paused;
  const normalized = elements.video.duration ? elements.video.currentTime / elements.video.duration : 0;
  elements.audio.pause();
  if (elements.voice.value === "silent") {
    elements.audio.removeAttribute("src");
    elements.audio.load();
    audioTimings = null;
    chapterData = [];
    renderChapters();
    renderTranscript();
    if (wasPlaying) elements.video.play().catch(() => {});
    return;
  }
  elements.loading.classList.remove("hidden");
  elements.audio.src = audioSource();
  elements.audio.load();
  try {
    await Promise.all([mediaReady(elements.video), mediaReady(elements.audio)]);
    elements.video.currentTime = normalized * elements.video.duration;
    configureMediaSync();
    await loadLessonDetail();
    if (wasPlaying) await elements.video.play();
  } catch (error) {
    showToast("This narration track could not be loaded.");
  } finally {
    elements.loading.classList.add("hidden");
  }
}

function configureMediaSync() {
  if (!elements.audio.duration || !elements.video.duration || elements.voice.value === "silent") {
    syncRate = 1;
    elements.video.playbackRate = 1;
    return;
  }
  syncRate = elements.video.duration / elements.audio.duration;
  elements.video.defaultPlaybackRate = syncRate;
  elements.video.playbackRate = syncRate;
  elements.audio.playbackRate = 1;
  elements.audio.volume = elements.video.volume;
  elements.audio.muted = elements.video.muted;
  syncAudio(true);
}

function syncAudio(force = false) {
  if (elements.voice.value === "silent" || !elements.audio.duration || !elements.video.duration) return;
  const target = elements.video.currentTime / elements.video.duration * elements.audio.duration;
  if (force || Math.abs(elements.audio.currentTime - target) > 0.16) {
    elements.audio.currentTime = Math.min(target, Math.max(0, elements.audio.duration - 0.02));
  }
}

async function loadLessonDetail() {
  const lesson = localizedLesson(activeLesson.id);
  if (elements.voice.value === "silent") {
    chapterData = lesson.scenes.map((scene, index) => ({ index: index + 1, start: 0, text: scene.narration, label: chapterLabel(scene.narration, index) }));
    renderChapters();
    renderTranscript();
    return;
  }
  elements.chapters.innerHTML = `<div class="detail-empty">Loading chapters…</div>`;
  try {
    const response = await fetch(timingSource());
    if (!response.ok) throw new Error("timings unavailable");
    audioTimings = await response.json();
    chapterData = lesson.scenes.map((scene, index) => ({
      index: index + 1,
      start: audioTimings.scenes[index]?.speech_start || 0,
      end: audioTimings.scenes[index]?.speech_end || 0,
      text: scene.narration,
      label: chapterLabel(scene.narration, index)
    }));
    renderCaptions();
    renderChapters();
    renderTranscript();
  } catch (error) {
    elements.chapters.innerHTML = `<div class="detail-empty">Chapter metadata is unavailable for this track.</div>`;
    elements.transcript.innerHTML = elements.chapters.innerHTML;
  }
}

function renderCaptions() {
  if (captionUrl) URL.revokeObjectURL(captionUrl);
  const scale = elements.audio.duration ? elements.video.duration / elements.audio.duration : 1;
  const lines = ["WEBVTT", ""];
  chapterData.forEach(chapter => {
    lines.push(String(chapter.index), `${vttTime(chapter.start * scale)} --> ${vttTime(chapter.end * scale)}`, chapter.text, "");
  });
  captionUrl = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/vtt" }));
  elements.captionTrack.src = captionUrl;
  elements.captionTrack.srclang = elements.language.value;
  elements.captionTrack.label = languageById(elements.language.value).label;
  elements.captionTrack.track.mode = "showing";
}

function renderChapters() {
  elements.chapters.innerHTML = "";
  chapterData.forEach(chapter => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chapter-button";
    button.innerHTML = `<span class="chapter-time">${formatTime(chapter.start)}</span><span class="chapter-copy"><strong>${escapeHTML(chapter.label)}</strong><small>${escapeHTML(truncate(chapter.text, 82))}</small></span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>`;
    button.addEventListener("click", () => seekTo(chapter.start));
    elements.chapters.appendChild(button);
  });
}

function renderTranscript() {
  elements.transcript.innerHTML = chapterData.map(chapter => `<div class="transcript-entry"><button type="button" data-seek="${chapter.start}">${formatTime(chapter.start)}</button><p>${escapeHTML(chapter.text)}</p></div>`).join("");
  elements.transcript.querySelectorAll("[data-seek]").forEach(button => button.addEventListener("click", () => seekTo(Number(button.dataset.seek))));
}

function seekTo(audioSeconds) {
  const audioDuration = elements.audio.duration || audioTimings?.total_duration || elements.video.duration;
  elements.video.currentTime = audioDuration ? audioSeconds / audioDuration * elements.video.duration : audioSeconds;
  syncAudio(true);
  elements.video.play().catch(() => {});
}

function chapterLabel(text, index) {
  const sentence = text.split(/[.!?。！？]/)[0].replace(/^This is\s+/i, "").replace(/^The\s+/i, "");
  return truncate(sentence.split(/\s+/).slice(0, 7).join(" ") || `Chapter ${index + 1}`, 48);
}

function updatePagination() {
  const index = LESSONS.indexOf(activeLesson);
  const previous = LESSONS[index - 1];
  const next = LESSONS[index + 1];
  elements.previous.disabled = !previous;
  elements.previousTitle.textContent = previous ? localizedLesson(previous.id).title : "Beginning of path";
  elements.next.disabled = !next;
  elements.nextTitle.textContent = next ? localizedLesson(next.id).title : "End of path";
  elements.previous.onclick = previous ? () => selectLesson(previous.id, { focus: true }) : null;
  elements.next.onclick = next ? () => selectLesson(next.id, { focus: true }) : null;
}

function mediaClock() {
  const ratio = elements.video.duration ? elements.video.currentTime / elements.video.duration : 0;
  const duration = elements.voice.value === "silent" ? elements.video.duration : elements.audio.duration;
  return { current: ratio * (duration || 0), duration: duration || 0, ratio };
}

function updateWatchUI() {
  const clock = mediaClock();
  elements.watchTime.textContent = `${formatTime(clock.current)} / ${formatTime(clock.duration)}`;
  elements.watchBar.style.width = `${Math.min(100, clock.ratio * 100)}%`;
  const currentIndex = chapterData.findIndex((chapter, index) => clock.current >= chapter.start && (!chapterData[index + 1] || clock.current < chapterData[index + 1].start));
  elements.chapters.querySelectorAll(".chapter-button").forEach((button, index) => button.classList.toggle("current", index === currentIndex));
}

function saveWatchPosition() {
  if (!activeLesson) return;
  watchProgress[activeLesson.id] = elements.video.currentTime || 0;
  localStorage.setItem(WATCH_KEY, JSON.stringify(watchProgress));
}

function toggleComplete() {
  if (completed.has(activeLesson.id)) { completed.delete(activeLesson.id); showToast("Lesson marked incomplete"); }
  else { completed.add(activeLesson.id); showToast("Lesson complete"); }
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
  updateCompleteButton(); updateCourseProgress(); renderCurriculum(elements.search.value);
}

function markCompleteAtEnd() {
  if (!activeLesson || completed.has(activeLesson.id)) return;
  completed.add(activeLesson.id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
  updateCompleteButton(); updateCourseProgress(); renderCurriculum(elements.search.value);
  showToast("Lesson completed");
}

function updateCompleteButton() {
  const done = completed.has(activeLesson?.id);
  elements.complete.hidden = !activeLesson;
  elements.complete.classList.toggle("completed", done);
  elements.completeLabel.textContent = done ? "Completed" : "Mark complete";
  elements.complete.setAttribute("aria-pressed", String(done));
}

function updateCourseProgress() {
  const count = LESSONS.filter(lesson => completed.has(lesson.id)).length;
  elements.progressLabel.textContent = `${count} of ${LESSONS.length} complete`;
  elements.progressBar.style.width = `${count / LESSONS.length * 100}%`;
}

function continueCourse() {
  selectLesson((LESSONS.find(lesson => !completed.has(lesson.id)) || LESSONS.at(-1)).id, { focus: true });
}

async function switchVoice() {
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  await loadNarration(true);
}

async function switchLanguage() {
  localStorage.setItem(LANGUAGE_KEY, elements.language.value);
  populateVoiceSelector();
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  elements.loading.classList.remove("hidden");
  try {
    localizedCatalog = await loadLocalizedCatalog(elements.language.value);
    renderCurriculum(elements.search.value); updateLessonHeader(); updateGuide(); updatePagination();
    await loadNarration(true);
  } catch (error) { showToast(error.message); }
  finally { elements.loading.classList.add("hidden"); }
}

function copyLessonLink() {
  const url = new URL(window.location.href); url.hash = `lesson=${activeLesson.id}`;
  navigator.clipboard?.writeText(url.href).then(() => showToast("Tutorial link copied")).catch(() => window.prompt("Copy this tutorial link:", url.href));
}

function setupDetailTabs() {
  const chapterTab = $("#chapters-tab"), transcriptTab = $("#transcript-tab");
  const chapterPanel = $("#chapters-panel"), transcriptPanel = $("#transcript-panel");
  const select = chapters => {
    chapterTab.classList.toggle("active", chapters); transcriptTab.classList.toggle("active", !chapters);
    chapterTab.setAttribute("aria-selected", String(chapters)); transcriptTab.setAttribute("aria-selected", String(!chapters));
    chapterPanel.hidden = !chapters; transcriptPanel.hidden = chapters;
  };
  chapterTab.addEventListener("click", () => select(true)); transcriptTab.addEventListener("click", () => select(false));
}

function openSidebar() { elements.sidebar.classList.add("open"); elements.scrim.hidden = false; elements.menuButton.setAttribute("aria-expanded", "true"); }
function closeSidebar() { elements.sidebar.classList.remove("open"); elements.scrim.hidden = true; elements.menuButton.setAttribute("aria-expanded", "false"); }
function showToast(message) { clearTimeout(toastTimer); elements.toast.textContent = message; elements.toast.classList.add("visible"); toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2600); }
function lessonFromHash() { return new URLSearchParams(location.hash.replace(/^#/, "")).get("lesson"); }
function readStoredSet(key) { try { const value = JSON.parse(localStorage.getItem(key) || "[]"); return new Set(Array.isArray(value) ? value : []); } catch { return new Set(); } }
function readStoredObject(key) { try { const value = JSON.parse(localStorage.getItem(key) || "{}"); return value && typeof value === "object" ? value : {}; } catch { return {}; } }
function formatTime(seconds) { if (!Number.isFinite(seconds) || seconds < 0) return "0:00"; const total = Math.floor(seconds); return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`; }
function vttTime(seconds) { const total = Math.max(0, Math.round(seconds * 1000)); const hours = Math.floor(total / 3600000); const minutes = Math.floor(total % 3600000 / 60000); const secs = Math.floor(total % 60000 / 1000); return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(total % 1000).padStart(3, "0")}`; }
function truncate(value, length) { return value.length > length ? `${value.slice(0, length - 1).trim()}…` : value; }
function escapeHTML(value) { return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }

elements.search.addEventListener("input", event => renderCurriculum(event.target.value));
elements.video.addEventListener("play", () => { syncAudio(true); if (elements.voice.value !== "silent") elements.audio.play().catch(() => showToast("Select play again to start narration.")); });
elements.video.addEventListener("pause", () => { elements.audio.pause(); saveWatchPosition(); });
elements.video.addEventListener("seeking", () => syncAudio(true));
elements.video.addEventListener("timeupdate", () => { syncAudio(false); updateWatchUI(); });
elements.video.addEventListener("volumechange", () => { elements.audio.volume = elements.video.volume; elements.audio.muted = elements.video.muted; });
elements.video.addEventListener("ratechange", () => { if (syncRate && elements.voice.value !== "silent") elements.audio.playbackRate = elements.video.playbackRate / syncRate; });
elements.video.addEventListener("ended", markCompleteAtEnd);
elements.audio.addEventListener("ended", markCompleteAtEnd);
elements.video.addEventListener("canplay", () => elements.loading.classList.add("hidden"));
elements.voice.addEventListener("change", switchVoice);
elements.language.addEventListener("change", switchLanguage);
elements.complete.addEventListener("click", toggleComplete); elements.copyLink.addEventListener("click", copyLessonLink);
elements.continue.addEventListener("click", continueCourse); elements.menuButton.addEventListener("click", openSidebar);
elements.closeMenu.addEventListener("click", closeSidebar); elements.scrim.addEventListener("click", closeSidebar);
window.addEventListener("beforeunload", () => { saveWatchPosition(); if (captionUrl) URL.revokeObjectURL(captionUrl); });
window.addEventListener("hashchange", () => { const id = lessonFromHash(); if (id && id !== activeLesson?.id) selectLesson(id, { skipHash: true }); });
window.addEventListener("keydown", event => { const tag = document.activeElement?.tagName; if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return; if (event.key === "/") { event.preventDefault(); elements.search.focus(); } else if (event.key === "[") { event.preventDefault(); elements.previous.click(); } else if (event.key === "]") { event.preventDefault(); elements.next.click(); } else if (event.key === "Escape") closeSidebar(); });

elements.availableCount.textContent = LESSONS.length;
elements.totalCount.textContent = LESSONS.length;
setupNarrationSelectors(); setupDetailTabs(); updateCourseProgress(); renderCurriculum();
loadLocalizedCatalog(elements.language.value).then(catalog => { localizedCatalog = catalog; renderCurriculum(); return selectLesson(lessonFromHash() || LESSONS[0].id, { skipHash: !location.hash, force: true }); }).catch(error => { showToast(error.message); selectLesson(lessonFromHash() || LESSONS[0].id, { skipHash: !location.hash, force: true }); });
