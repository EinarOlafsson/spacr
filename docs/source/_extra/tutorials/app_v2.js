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
const THEME_KEY = "spacr-tutorial-theme-v1";

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
  closeMenu: $("#close-menu-button"), toast: $("#toast"),
  themeToggle: $("#theme-toggle"), themeColor: $('meta[name="theme-color"]')
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
let openCustomSelect = null;
let languageRequest = 0;
let narrationRequest = 0;
let completed = readStoredSet(STORAGE_KEY);
let watchProgress = readStoredObject(WATCH_KEY);
const mobileSidebarQuery = window.matchMedia("(max-width: 780px)");

function applyTheme(theme, persist = false) {
  const normalized = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = normalized;
  const light = normalized === "light";
  elements.themeToggle?.setAttribute("aria-pressed", String(light));
  elements.themeToggle?.setAttribute("aria-label", "Light mode");
  if (elements.themeToggle) elements.themeToggle.title = `Switch to ${light ? "dark" : "light"} mode`;
  if (elements.themeColor) elements.themeColor.content = light ? "#f4f7fb" : "#070a0f";
  if (persist) {
    try { localStorage.setItem(THEME_KEY, normalized); } catch (error) { /* Storage may be disabled. */ }
  }
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light", true);
}

function enhanceSelect(select) {
  if (!select || select.customSelect) return;
  const control = select.closest(".select-control");
  const controlLabel = select.getAttribute("aria-label") || "Choose an option";
  const custom = document.createElement("div");
  custom.className = "custom-select";
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "custom-select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  trigger.setAttribute("aria-label", controlLabel);
  const value = document.createElement("strong");
  const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  chevron.setAttribute("viewBox", "0 0 24 24");
  chevron.setAttribute("aria-hidden", "true");
  chevron.innerHTML = '<path d="m6 9 6 6 6-6"/>';
  trigger.append(value, chevron);
  const menu = document.createElement("div");
  menu.className = "custom-select-menu";
  menu.id = `${select.id}-listbox`;
  menu.setAttribute("role", "listbox");
  menu.setAttribute("aria-label", controlLabel);
  trigger.setAttribute("aria-controls", menu.id);
  custom.append(trigger, menu);
  select.insertAdjacentElement("afterend", custom);
  select.tabIndex = -1;
  select.setAttribute("aria-hidden", "true");
  control?.classList.add("enhanced-select");

  let typeahead = "";
  let typeaheadTimer = null;

  const buttons = () => [...menu.querySelectorAll(".custom-select-option")];
  const selectedIndex = () => Math.max(0, [...select.options].findIndex(option => option.value === select.value));
  const positionMenu = () => {
    custom.classList.remove("opens-down");
    const triggerRect = trigger.getBoundingClientRect();
    const menuHeight = Math.min(menu.scrollHeight, 340, window.innerHeight * .52);
    const spaceAbove = triggerRect.top - 12;
    const spaceBelow = window.innerHeight - triggerRect.bottom - 12;
    const opensDown = spaceAbove < menuHeight && spaceBelow > spaceAbove;
    if (opensDown) custom.classList.add("opens-down");
    const availableHeight = Math.max(0, opensDown ? spaceBelow : spaceAbove);
    menu.style.maxHeight = `${Math.floor(Math.min(menuHeight, availableHeight))}px`;
  };
  const focusAt = index => {
    const items = buttons();
    if (!items.length) return;
    const wrapped = (index + items.length) % items.length;
    items.forEach((item, itemIndex) => item.classList.toggle("active", itemIndex === wrapped));
    items[wrapped].focus({ preventScroll: true });
    items[wrapped].scrollIntoView({ block: "nearest" });
  };
  const close = (restoreFocus = false) => {
    custom.classList.remove("open", "opens-down");
    menu.style.maxHeight = "";
    trigger.setAttribute("aria-expanded", "false");
    buttons().forEach(item => item.classList.remove("active"));
    clearTimeout(typeaheadTimer);
    typeahead = "";
    if (openCustomSelect === close) openCustomSelect = null;
    if (restoreFocus) trigger.focus();
  };
  const open = (focusSelected = false) => {
    if (openCustomSelect && openCustomSelect !== close) openCustomSelect();
    openCustomSelect = close;
    custom.classList.add("open");
    trigger.setAttribute("aria-expanded", "true");
    positionMenu();
    if (focusSelected) requestAnimationFrame(() => focusAt(selectedIndex()));
  };
  const choose = optionValue => {
    if (select.value !== optionValue) {
      select.value = optionValue;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
    sync();
    close(true);
  };
  const search = character => {
    clearTimeout(typeaheadTimer);
    typeahead += character.toLocaleLowerCase();
    typeaheadTimer = setTimeout(() => { typeahead = ""; }, 700);
    const items = buttons();
    const match = items.findIndex(item => item.textContent.trim().toLocaleLowerCase().startsWith(typeahead));
    if (match >= 0) focusAt(match);
  };
  const onKeydown = event => {
    const items = buttons();
    const current = Math.max(0, items.indexOf(document.activeElement));
    const handled = () => { event.preventDefault(); event.stopPropagation(); };
    if (event.key === "Tab" && custom.classList.contains("open")) {
      close(true);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      handled();
      if (!custom.classList.contains("open")) open(true);
      else focusAt(current + (event.key === "ArrowDown" ? 1 : -1));
    } else if (event.key === "Home" && custom.classList.contains("open")) {
      handled(); focusAt(0);
    } else if (event.key === "End" && custom.classList.contains("open")) {
      handled(); focusAt(items.length - 1);
    } else if (event.key === "Escape" && custom.classList.contains("open")) {
      handled(); close(true);
    } else if ((event.key === "Enter" || event.key === " ") && document.activeElement?.classList.contains("custom-select-option")) {
      handled(); document.activeElement.click();
    } else if ((event.key === "Enter" || event.key === " ") && document.activeElement === trigger) {
      handled();
      if (custom.classList.contains("open")) close(true); else open(true);
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      handled();
      if (!custom.classList.contains("open")) open();
      search(event.key);
    }
  };
  const sync = () => {
    const selected = select.selectedOptions[0] || select.options[0];
    value.textContent = selected?.textContent || "";
    trigger.setAttribute("aria-label", `${controlLabel}: ${value.textContent}`);
    buttons().forEach(item => {
      const active = item.dataset.value === select.value;
      item.setAttribute("aria-selected", String(active));
    });
  };
  const rebuild = () => {
    menu.innerHTML = "";
    [...select.options].forEach(option => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "custom-select-option";
      button.setAttribute("role", "option");
      button.tabIndex = -1;
      button.dataset.value = option.value;
      button.textContent = option.textContent;
      button.addEventListener("click", () => choose(option.value));
      menu.appendChild(button);
    });
    sync();
  };

  trigger.addEventListener("click", () => custom.classList.contains("open") ? close() : open(true));
  trigger.addEventListener("keydown", onKeydown);
  menu.addEventListener("keydown", onKeydown);
  select.addEventListener("change", sync);
  custom.addEventListener("focusout", () => requestAnimationFrame(() => {
    if (custom.classList.contains("open") && !custom.contains(document.activeElement)) close();
  }));
  document.addEventListener("pointerdown", event => {
    if (custom.classList.contains("open") && !custom.contains(event.target)) close();
  });
  window.addEventListener("resize", () => {
    if (custom.classList.contains("open")) close();
  });
  new MutationObserver(rebuild).observe(select, { childList: true });
  select.customSelect = { rebuild, sync, close };
  rebuild();
}

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
  const lessonId = activeLesson.id;
  elements.loading.classList.remove("hidden");
  elements.video.poster = `${PRODUCTION_ROOT}/${activeLesson.poster}`;
  elements.video.src = `${PRODUCTION_ROOT}/${activeLesson.silent}`;
  elements.video.load();
  const saved = watchProgress[activeLesson.id] || 0;
  const expectedNarrationRequest = narrationRequest + 1;
  const isCurrent = () => activeLesson?.id === lessonId && narrationRequest === expectedNarrationRequest;
  await loadNarration(false, () => activeLesson?.id === lessonId);
  if (!isCurrent()) return;
  await mediaReady(elements.video);
  if (!isCurrent()) return;
  if (saved > 0 && saved < elements.video.duration - 0.25) elements.video.currentTime = saved;
  configureMediaSync();
  elements.loading.classList.add("hidden");
  updateWatchUI();
}

async function loadNarration(resume = true, outerCurrent = () => true) {
  const request = ++narrationRequest;
  const isCurrent = () => request === narrationRequest && outerCurrent();
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
    if (!isCurrent()) return;
    elements.video.currentTime = normalized * elements.video.duration;
    configureMediaSync();
    await loadLessonDetail(isCurrent);
    if (!isCurrent()) return;
    if (wasPlaying) await elements.video.play();
  } catch (error) {
    if (isCurrent()) showToast("This narration track could not be loaded.");
  } finally {
    if (isCurrent()) elements.loading.classList.add("hidden");
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

async function loadLessonDetail(isCurrent = () => true) {
  if (!isCurrent()) return;
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
    const timings = await response.json();
    if (!isCurrent()) return;
    audioTimings = timings;
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
    if (!isCurrent()) return;
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
  const requestedLanguage = elements.language.value;
  const request = ++languageRequest;
  const isCurrent = () => request === languageRequest && elements.language.value === requestedLanguage;
  localStorage.setItem(LANGUAGE_KEY, requestedLanguage);
  populateVoiceSelector();
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  elements.loading.classList.remove("hidden");
  try {
    const catalog = await loadLocalizedCatalog(requestedLanguage);
    if (!isCurrent()) return;
    localizedCatalog = catalog;
    renderCurriculum(elements.search.value); updateLessonHeader(); updateGuide(); updatePagination();
    await loadNarration(true, isCurrent);
  } catch (error) { if (isCurrent()) showToast(error.message); }
  finally { if (isCurrent()) elements.loading.classList.add("hidden"); }
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

function syncSidebarAccessibility() {
  const closedOnMobile = mobileSidebarQuery.matches && !elements.sidebar.classList.contains("open");
  elements.sidebar.inert = closedOnMobile;
  if (closedOnMobile) elements.sidebar.setAttribute("aria-hidden", "true");
  else elements.sidebar.removeAttribute("aria-hidden");
  const drawerOpen = mobileSidebarQuery.matches && !closedOnMobile;
  $(".topbar").inert = drawerOpen;
  elements.player.closest(".lesson-content").inert = drawerOpen;
}
function openSidebar() {
  elements.sidebar.classList.add("open");
  elements.scrim.hidden = false;
  elements.menuButton.setAttribute("aria-expanded", "true");
  syncSidebarAccessibility();
  requestAnimationFrame(() => elements.closeMenu.focus());
}
function closeSidebar(restoreFocus = false) {
  const focusWasInside = elements.sidebar.contains(document.activeElement);
  elements.sidebar.classList.remove("open");
  elements.scrim.hidden = true;
  elements.menuButton.setAttribute("aria-expanded", "false");
  syncSidebarAccessibility();
  if (mobileSidebarQuery.matches && (restoreFocus || focusWasInside)) elements.menuButton.focus();
}
function handleSidebarBreakpoint() {
  if (!mobileSidebarQuery.matches) {
    elements.sidebar.classList.remove("open");
    elements.scrim.hidden = true;
    elements.menuButton.setAttribute("aria-expanded", "false");
  }
  syncSidebarAccessibility();
}
function sidebarFocusableElements() {
  return [...elements.sidebar.querySelectorAll("button:not([disabled]), input:not([disabled]), a[href]")]
    .filter(element => !element.hidden && element.getClientRects().length);
}
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
elements.closeMenu.addEventListener("click", () => closeSidebar(true)); elements.scrim.addEventListener("click", () => closeSidebar(true));
elements.sidebar.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    event.preventDefault(); event.stopPropagation(); closeSidebar(true); return;
  }
  if (event.key !== "Tab" || !mobileSidebarQuery.matches) return;
  const focusable = sidebarFocusableElements();
  if (!focusable.length) return;
  const first = focusable[0], last = focusable.at(-1);
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault(); last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault(); first.focus();
  }
});
elements.themeToggle?.addEventListener("click", toggleTheme);
window.addEventListener("beforeunload", () => { saveWatchPosition(); if (captionUrl) URL.revokeObjectURL(captionUrl); });
window.addEventListener("hashchange", () => { const id = lessonFromHash(); if (id && id !== activeLesson?.id) selectLesson(id, { skipHash: true }); });
window.addEventListener("keydown", event => { const tag = document.activeElement?.tagName; if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || event.target.closest?.(".custom-select")) return; if (event.key === "/") { event.preventDefault(); elements.search.focus(); } else if (event.key === "[") { event.preventDefault(); elements.previous.click(); } else if (event.key === "]") { event.preventDefault(); elements.next.click(); } else if (event.key === "Escape") closeSidebar(true); });
if (mobileSidebarQuery.addEventListener) mobileSidebarQuery.addEventListener("change", handleSidebarBreakpoint);
else mobileSidebarQuery.addListener(handleSidebarBreakpoint);

elements.availableCount.textContent = LESSONS.length;
elements.totalCount.textContent = LESSONS.length;
applyTheme(document.documentElement.dataset.theme);
syncSidebarAccessibility();
setupNarrationSelectors();
enhanceSelect(elements.language); enhanceSelect(elements.voice);
setupDetailTabs(); updateCourseProgress(); renderCurriculum();
loadLocalizedCatalog(elements.language.value).then(catalog => { localizedCatalog = catalog; renderCurriculum(); return selectLesson(lessonFromHash() || LESSONS[0].id, { skipHash: !location.hash, force: true }); }).catch(error => { showToast(error.message); selectLesson(lessonFromHash() || LESSONS[0].id, { skipHash: !location.hash, force: true }); });
