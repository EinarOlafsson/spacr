"use strict";

const VOICE_CATALOG = window.SPACR_VOICE_CATALOG || [];
const BASE_CATALOG = window.SPACR_LESSON_CATALOG;
const DEFAULT_LANGUAGE = "en";
const DEFAULT_VOICE = "af_heart";
const PRODUCTION_ROOT = document.documentElement.dataset.productionRoot || "../production";
// Narration is served separately from the video. All 54 voices are 2,662 MiB,
// which no GitHub Pages site can carry -- publishing one voice per language
// was the only way to fit, and it left 27 of the 28 English voices unusable.
// Pointing narration at external storage removes that constraint entirely:
// the audio, its timing sidecars, and nothing else resolve against this root.
// Video, posters and captions stay on Pages, so a lesson still plays if the
// audio host is unreachable -- it degrades to silent, not to broken.
// Defaults to PRODUCTION_ROOT, which is the all-local behaviour.
const AUDIO_ROOT = document.documentElement.dataset.audioRoot || PRODUCTION_ROOT;
// The 4K silent masters, when they are hosted somewhere with room for them.
// Same silent cut as the 1440p copy at the same timings, so narration syncs
// against either without recomputation -- a plain <video src>, which is why
// this can carry the voice picker while the player drives the element at an
// arbitrary playback rate to match the selected narration.
// Empty means no 4K is available and the quality control stays hidden.
const VIDEO_4K_ROOT = document.documentElement.dataset.video4kRoot || "";
const STORAGE_KEY = "spacr-tutorial-progress-v2";
const WATCH_KEY = "spacr-tutorial-watch-v2";
const LANGUAGE_KEY = "spacr-tutorial-language-v2";
const VOICE_KEY = "spacr-tutorial-voice-v2";
const THEME_KEY = "spacr-tutorial-theme-v1";
const CAPTION_SETTINGS_KEY = "spacr-tutorial-captions-v1";
const QUALITY_KEY = "spacr-tutorial-quality-v1";

const CAPTION_LANGUAGES = [
  { id: "auto", label: "Same as narration", shortLabel: "Same as narration" },
  { id: "en", label: "English", shortLabel: "English" },
  { id: "de", label: "Deutsch · German", shortLabel: "Deutsch" },
  { id: "sv", label: "Svenska · Swedish", shortLabel: "Svenska" },
  { id: "is", label: "Íslenska · Icelandic", shortLabel: "Íslenska" },
  { id: "ja", label: "日本語 · Japanese", shortLabel: "日本語" },
  { id: "nb", label: "Norsk bokmål · Norwegian", shortLabel: "Norsk bokmål" },
  { id: "ko", label: "한국어 · Korean", shortLabel: "한국어" },
  { id: "da", label: "Dansk · Danish", shortLabel: "Dansk" },
  { id: "es", label: "Español · Spanish", shortLabel: "Español" },
  { id: "fr", label: "Français · French", shortLabel: "Français" },
  { id: "it", label: "Italiano · Italian", shortLabel: "Italiano" },
  { id: "pt-BR", label: "Português do Brasil", shortLabel: "Português" },
  { id: "zh-CN", label: "简体中文 · Mandarin", shortLabel: "简体中文" },
  { id: "hi", label: "हिन्दी · Hindi", shortLabel: "हिन्दी" }
];
const CAPTION_ONLY_LANGUAGES = new Set(["de", "sv", "is", "nb", "ko", "da"]);

const DEFAULT_CAPTION_SETTINGS = Object.freeze({
  enabled: true,
  language: "auto",
  size: 100,
  color: "#FFFFFF",
  background: "#020509",
  backgroundOpacity: 82
});

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
  captionSettings: $("#caption-settings"),
  captionSettingsButton: $("#caption-settings-button"),
  captionSettingsPanel: $("#caption-settings-panel"),
  captionSettingsClose: $("#caption-settings-close"),
  captionSettingsSummary: $("#caption-settings-summary"),
  captionEnabled: $("#caption-enabled"),
  captionLanguage: $("#caption-language-select"),
  captionSize: $("#caption-size"), captionSizeValue: $("#caption-size-value"),
  captionTextColor: $("#caption-text-color"),
  captionTextColorValue: $("#caption-text-color-value"),
  captionBackgroundColor: $("#caption-background-color"),
  captionBackgroundColorValue: $("#caption-background-color-value"),
  captionBackgroundOpacity: $("#caption-background-opacity"),
  captionBackgroundOpacityValue: $("#caption-background-opacity-value"),
  captionPreviewText: $("#caption-preview-text"),
  captionReset: $("#caption-reset-button"),
  watchTime: $("#watch-time"), watchBar: $("#watch-bar"),
  chapters: $("#chapter-list"), transcript: $("#transcript-list"),
  chapterCard: $("#chapter-card"), previous: $("#previous-button"),
  next: $("#next-button"), previousTitle: $("#previous-title"),
  nextTitle: $("#next-title"), complete: $("#complete-button"),
  completeLabel: $("#complete-label"), copyLink: $("#copy-link-button"),
  quality: $("#quality-select"), qualityControl: $("#quality-control"),
  continue: $("#continue-button"), progressLabel: $("#progress-label"),
  progressBar: $("#progress-bar"), availableCount: $("#available-count"),
  totalCount: $("#total-count"), sidebar: $("#sidebar"),
  scrim: $("#sidebar-scrim"), menuButton: $("#menu-button"),
  closeMenu: $("#close-menu-button"), toast: $("#toast"),
  themeToggle: $("#theme-toggle"), themeColor: $('meta[name="theme-color"]')
};

const LESSONS = BASE_CATALOG.lessons;
const localizedCatalogs = new Map([["en", BASE_CATALOG]]);
const captionCatalogs = new Map([["en", BASE_CATALOG]]);
let localizedCatalog = BASE_CATALOG;
let captionCatalog = BASE_CATALOG;
let activeLesson = null;
let chapterData = [];
let audioTimings = null;
let visualTimings = null;
let syncRate = 1;
let userPlaybackRate = 1;
let programmedVideoRate = 1;
let captionUrl = "";
let captionTrackLoading = false;
let toastTimer = null;
let openCustomSelect = null;
let languageRequest = 0;
let narrationRequest = 0;
let captionLanguageRequest = 0;
let narrationAudioAvailable = false;
let narrationFetchController = null;
let narrationObjectUrl = "";
let captionSettings = readCaptionSettings();
let completed = readStoredSet(STORAGE_KEY);
let watchProgress = readStoredObject(WATCH_KEY);
const mobileSidebarQuery = window.matchMedia("(max-width: 780px)");

function applyTheme(theme, persist = false) {
  const normalized = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = normalized;
  const light = normalized === "light";
  elements.themeToggle?.setAttribute("aria-pressed", String(light));
  elements.themeToggle?.setAttribute("aria-label", `Switch to ${light ? "dark" : "light"} mode`);
  if (elements.themeToggle) elements.themeToggle.title = `Switch to ${light ? "dark" : "light"} mode`;
  if (elements.themeColor) elements.themeColor.content = light ? "#f4f7fb" : "#070a0f";
  if (persist) {
    try { localStorage.setItem(THEME_KEY, normalized); } catch (error) { /* Storage may be disabled. */ }
  }
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light", true);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Number(value)));
}

function validHexColor(value, fallback) {
  return /^#[0-9a-f]{6}$/i.test(String(value || ""))
    ? String(value).toUpperCase() : fallback;
}

function normalizeCaptionSettings(value = {}) {
  const language = CAPTION_LANGUAGES.some(item => item.id === value.language)
    ? value.language : DEFAULT_CAPTION_SETTINGS.language;
  return {
    enabled: value.enabled !== false,
    language,
    size: clamp(Number.isFinite(Number(value.size)) ? value.size : DEFAULT_CAPTION_SETTINGS.size, 75, 200),
    color: validHexColor(value.color, DEFAULT_CAPTION_SETTINGS.color),
    background: validHexColor(value.background, DEFAULT_CAPTION_SETTINGS.background),
    backgroundOpacity: clamp(
      Number.isFinite(Number(value.backgroundOpacity))
        ? value.backgroundOpacity : DEFAULT_CAPTION_SETTINGS.backgroundOpacity,
      0, 100)
  };
}

function readCaptionSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(CAPTION_SETTINGS_KEY) || "{}");
    return normalizeCaptionSettings(stored && typeof stored === "object" ? stored : {});
  } catch (error) {
    return { ...DEFAULT_CAPTION_SETTINGS };
  }
}

function persistCaptionSettings() {
  try { localStorage.setItem(CAPTION_SETTINGS_KEY, JSON.stringify(captionSettings)); }
  catch (error) { /* Storage may be disabled. */ }
}

function captionLanguageById(id) {
  return CAPTION_LANGUAGES.find(item => item.id === id) || CAPTION_LANGUAGES[0];
}

function effectiveCaptionLanguage() {
  if (captionSettings.language !== "auto") return captionSettings.language;
  const narrationLanguage = elements.language?.value || DEFAULT_LANGUAGE;
  return CAPTION_LANGUAGES.some(item => item.id === narrationLanguage)
    ? narrationLanguage : DEFAULT_LANGUAGE;
}

function hexToRgb(hex) {
  const value = validHexColor(hex, "#000000").slice(1);
  return [0, 2, 4].map(index => Number.parseInt(value.slice(index, index + 2), 16));
}

function updateCaptionCueStyle() {
  let style = document.querySelector("#caption-cue-style");
  if (!style) {
    style = document.createElement("style");
    style.id = "caption-cue-style";
    document.head.appendChild(style);
  }
  const [red, green, blue] = hexToRgb(captionSettings.background);
  const alpha = (captionSettings.backgroundOpacity / 100).toFixed(2);
  style.textContent = `#tutorial-video::cue { color: ${captionSettings.color}; background-color: rgba(${red}, ${green}, ${blue}, ${alpha}); font-size: ${captionSettings.size}%; }`;
}

function updateCaptionTrackMode() {
  if (!elements.captionTrack?.track) return;
  const showing = captionSettings.enabled && Boolean(captionUrl);
  elements.captionTrack.track.mode = showing ? "showing" : "disabled";
}

function syncCaptionPreferenceFromNativeControls() {
  if (!captionUrl || !elements.captionTrack?.track) return;
  const enabled = elements.captionTrack.track.mode === "showing";
  if (captionSettings.enabled === enabled) return;
  captionSettings.enabled = enabled;
  applyCaptionSettings({ persist: true });
}

function syncCaptionControls() {
  if (!elements.captionEnabled) return;
  elements.captionEnabled.checked = captionSettings.enabled;
  elements.captionLanguage.value = captionSettings.language;
  elements.captionLanguage.customSelect?.sync();
  elements.captionSize.value = captionSettings.size;
  elements.captionSizeValue.value = `${captionSettings.size}%`;
  elements.captionSizeValue.textContent = `${captionSettings.size}%`;
  elements.captionTextColor.value = captionSettings.color.toLowerCase();
  elements.captionTextColorValue.value = captionSettings.color;
  elements.captionTextColorValue.textContent = captionSettings.color;
  elements.captionBackgroundColor.value = captionSettings.background.toLowerCase();
  elements.captionBackgroundColorValue.value = captionSettings.background;
  elements.captionBackgroundColorValue.textContent = captionSettings.background;
  elements.captionBackgroundOpacity.value = captionSettings.backgroundOpacity;
  const opacityLabel = captionSettings.backgroundOpacity === 0
    ? "Transparent" : `${captionSettings.backgroundOpacity}%`;
  elements.captionBackgroundOpacityValue.value = opacityLabel;
  elements.captionBackgroundOpacityValue.textContent = opacityLabel;
  const language = captionLanguageById(captionSettings.language);
  elements.captionSettingsSummary.textContent = captionSettings.enabled
    ? language.shortLabel : "Off";
  const [red, green, blue] = hexToRgb(captionSettings.background);
  elements.captionPreviewText.style.color = captionSettings.color;
  elements.captionPreviewText.style.backgroundColor =
    `rgba(${red}, ${green}, ${blue}, ${captionSettings.backgroundOpacity / 100})`;
  elements.captionPreviewText.style.fontSize = `${Math.round(14 * captionSettings.size / 100)}px`;
}

function applyCaptionSettings({ persist = false } = {}) {
  captionSettings = normalizeCaptionSettings(captionSettings);
  updateCaptionCueStyle();
  syncCaptionControls();
  updateCaptionTrackMode();
  if (persist) persistCaptionSettings();
}

function setupCaptionControls() {
  elements.captionLanguage.innerHTML = "";
  CAPTION_LANGUAGES.forEach(language => {
    const option = document.createElement("option");
    option.value = language.id;
    option.textContent = language.label;
    elements.captionLanguage.appendChild(option);
  });
  applyCaptionSettings();
}

function openCaptionSettings() {
  if (!elements.captionSettingsPanel.hidden) return;
  if (openCustomSelect) openCustomSelect();
  elements.captionSettingsPanel.hidden = false;
  elements.captionSettings.classList.add("open");
  elements.captionSettingsButton.setAttribute("aria-expanded", "true");
  positionCaptionSettingsPanel();
  requestAnimationFrame(() => elements.captionEnabled.focus({ preventScroll: true }));
}

function positionCaptionSettingsPanel() {
  if (elements.captionSettingsPanel.hidden) return;
  const panel = elements.captionSettingsPanel;
  panel.classList.remove("opens-down");
  panel.style.maxHeight = "";
  const trigger = elements.captionSettingsButton.getBoundingClientRect();
  const desired = Math.min(panel.scrollHeight, window.innerHeight - 24);
  const above = Math.max(0, trigger.top - 12);
  const below = Math.max(0, window.innerHeight - trigger.bottom - 12);
  const opensDown = above < desired && below > above;
  if (opensDown) panel.classList.add("opens-down");
  const available = opensDown ? below : above;
  panel.style.maxHeight = `${Math.max(1, Math.floor(Math.min(desired, available)))}px`;
}

function closeCaptionSettings(restoreFocus = false) {
  if (elements.captionSettingsPanel.hidden) return;
  elements.captionLanguage.customSelect?.close();
  elements.captionSettingsPanel.hidden = true;
  elements.captionSettingsPanel.classList.remove("opens-down");
  elements.captionSettingsPanel.style.maxHeight = "";
  elements.captionSettings.classList.remove("open");
  elements.captionSettingsButton.setAttribute("aria-expanded", "false");
  if (restoreFocus) elements.captionSettingsButton.focus();
}

function enhanceSelect(select) {
  if (!select || select.customSelect) return;
  const control = select.closest(".select-control");
  const portalMenu = Boolean(select.closest(".caption-settings-panel"));
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
    if (portalMenu) {
      const viewportPadding = 12;
      const maxWidth = Math.max(1, Math.min(360, window.innerWidth - viewportPadding * 2));
      const width = Math.max(
        Math.min(triggerRect.width, maxWidth),
        Math.min(menu.scrollWidth, maxWidth)
      );
      const opensDown = spaceBelow >= menuHeight || spaceBelow >= spaceAbove;
      const availableHeight = Math.max(1, opensDown ? spaceBelow : spaceAbove);
      const height = Math.max(1, Math.floor(Math.min(menuHeight, availableHeight)));
      const left = Math.min(
        Math.max(viewportPadding, triggerRect.left),
        Math.max(viewportPadding, window.innerWidth - width - viewportPadding)
      );
      const top = opensDown
        ? Math.min(window.innerHeight - viewportPadding - height, triggerRect.bottom + 8)
        : Math.max(viewportPadding, triggerRect.top - 8 - height);
      menu.style.width = `${Math.floor(width)}px`;
      menu.style.maxHeight = `${height}px`;
      menu.style.top = `${Math.floor(top)}px`;
      menu.style.right = "auto";
      menu.style.bottom = "auto";
      menu.style.left = `${Math.floor(left)}px`;
      return;
    }
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
    menu.classList.remove("portaled");
    menu.removeAttribute("style");
    if (menu.parentElement !== custom) custom.appendChild(menu);
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
    if (portalMenu) {
      document.body.appendChild(menu);
      menu.classList.add("portaled");
    }
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
  const setDisabled = disabled => {
    select.disabled = Boolean(disabled);
    trigger.disabled = Boolean(disabled);
    custom.classList.toggle("disabled", Boolean(disabled));
    if (disabled) close();
  };

  trigger.addEventListener("click", () => custom.classList.contains("open") ? close() : open(true));
  trigger.addEventListener("keydown", onKeydown);
  menu.addEventListener("keydown", onKeydown);
  select.addEventListener("change", sync);
  custom.addEventListener("focusout", () => requestAnimationFrame(() => {
    if (custom.classList.contains("open") &&
        !custom.contains(document.activeElement) && !menu.contains(document.activeElement)) close();
  }));
  document.addEventListener("pointerdown", event => {
    if (custom.classList.contains("open") &&
        !custom.contains(event.target) && !menu.contains(event.target)) close();
  });
  document.addEventListener("scroll", event => {
    if (custom.classList.contains("open") &&
        event.target !== menu && !menu.contains(event.target)) close();
  }, true);
  window.addEventListener("resize", () => {
    if (custom.classList.contains("open")) close();
  });
  new MutationObserver(rebuild).observe(select, { childList: true });
  select.customSelect = { rebuild, sync, close, setDisabled };
  rebuild();
  setDisabled(select.disabled);
}

function setMediaSelectorsDisabled(disabled) {
  [elements.language, elements.voice, elements.captionLanguage].forEach(select => {
    if (!select) return;
    select.disabled = Boolean(disabled);
    select.customSelect?.setDisabled(disabled);
  });
}

function languageById(id) {
  return VOICE_CATALOG.find(item => item.id === id) ||
    VOICE_CATALOG.find(item => item.id === DEFAULT_LANGUAGE) || VOICE_CATALOG[0];
}

function localizedLesson(id) {
  return localizedCatalog.lessons.find(item => item.id === id) ||
    LESSONS.find(item => item.id === id);
}

function captionLesson(id) {
  return captionCatalog.lessons.find(item => item.id === id) ||
    localizedLesson(id) || baseLesson(id);
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

function narrationRoot() {
  return AUDIO_ROOT;
}

function fourKAvailable() {
  return Boolean(VIDEO_4K_ROOT);
}

// The 4K cut is the same silent master at a higher resolution, published
// under the same per-lesson path, so only the root changes.
function videoSource(lesson = activeLesson) {
  if (!lesson) return "";
  if (fourKAvailable() && elements.quality?.value === "4k") {
    return `${VIDEO_4K_ROOT}/${lesson.silent}`;
  }
  return `${PRODUCTION_ROOT}/${lesson.silent}`;
}

// Swapping src resets the element, so the position, the rate mapping and the
// play state all have to be put back by hand. Narration is untouched: both
// cuts share one timeline, so the audio element keeps playing underneath and
// only needs re-syncing to the new video clock.
async function applyQualityChange() {
  if (!activeLesson) return;
  try { localStorage.setItem(QUALITY_KEY, elements.quality.value); }
  catch (error) { /* Storage may be disabled. */ }

  const resumeAt = elements.video.currentTime || 0;
  const wasPlaying = !elements.video.paused;
  elements.video.pause();
  elements.loading.classList.remove("hidden");

  const lessonId = activeLesson.id;
  elements.video.src = videoSource();
  elements.video.load();
  try {
    await mediaReady(elements.video);
  } catch (error) {
    // The 4K host is unreachable or the file is missing. Fall back rather
    // than leaving the lesson dead, and say so -- a silent downgrade would
    // look like the quality control simply does nothing.
    if (elements.quality.value === "4k") {
      elements.quality.value = "1440p";
      elements.video.src = videoSource();
      elements.video.load();
      try { await mediaReady(elements.video); } catch (again) { return; }
      showToast("4K is unavailable right now — playing 1440p");
    } else {
      return;
    }
  }
  if (activeLesson?.id !== lessonId) return;

  if (resumeAt > 0 && resumeAt < elements.video.duration - 0.25) {
    elements.video.currentTime = resumeAt;
  }
  configureMediaSync();
  elements.loading.classList.add("hidden");
  if (wasPlaying) elements.video.play().catch(() => { /* Autoplay may be refused. */ });
}

function setupQualityControl() {
  if (!elements.quality || !elements.qualityControl) return;
  elements.qualityControl.hidden = !fourKAvailable();
  if (!fourKAvailable()) return;
  let saved = "";
  try { saved = localStorage.getItem(QUALITY_KEY) || ""; }
  catch (error) { /* Storage may be disabled. */ }
  if (saved === "4k" || saved === "1440p") elements.quality.value = saved;
  elements.quality.addEventListener("change", applyQualityChange);
}

function audioSource(lesson = activeLesson) {
  if (!lesson || elements.voice.value === "silent") return "";
  return `${narrationRoot()}/${lesson.id}/audio/${elements.language.value}/${elements.voice.value}.m4a`;
}

function discardNarrationAudio() {
  elements.audio.pause();
  elements.audio.removeAttribute("src");
  elements.audio.load();
  if (narrationObjectUrl) URL.revokeObjectURL(narrationObjectUrl);
  narrationObjectUrl = "";
}

async function fetchNarrationAudio(source, signal) {
  // Safari's media stack can ask an MP4/M4A host for multiple byte ranges in
  // one request. The Hugging Face Xet endpoint currently rejects that request
  // shape even though a normal CORS GET succeeds. Narration files are small
  // (the largest release track is under 1.5 MB), so download one complete file
  // and let the media element read a local Blob URL instead of issuing ranges.
  const response = await fetch(source, {
    mode: "cors",
    credentials: "omit",
    signal,
  });
  if (!response.ok) throw new Error(`Narration download failed (${response.status})`);
  const blob = await response.blob();
  if (!blob.size) throw new Error("Narration download was empty");
  return blob;
}

function timingSource(lesson = activeLesson) {
  if (elements.voice.value === "silent") {
    return defaultTimingSource(lesson);
  }
  return `${narrationRoot()}/${lesson.id}/audio/${elements.language.value}/${elements.voice.value}.json`;
}

function defaultTimingSource(lesson = activeLesson) {
  return `${narrationRoot()}/${lesson.id}/audio/${DEFAULT_LANGUAGE}/${DEFAULT_VOICE}.json`;
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

function validateCaptionCatalog(catalog, language) {
  if (!catalog || !Array.isArray(catalog.lessons)) {
    throw new Error(`The ${language} caption catalog is invalid`);
  }
  const byId = new Map(catalog.lessons.map(lesson => [lesson.id, lesson]));
  const aligned = LESSONS.every(source => {
    const lesson = byId.get(source.id);
    return lesson && Array.isArray(lesson.scenes) &&
      lesson.scenes.length === source.scenes.length &&
      lesson.scenes.every(scene => typeof scene.narration === "string" && scene.narration.trim());
  });
  if (!aligned) throw new Error(`The ${language} captions do not align with this tutorial version`);
  return catalog;
}

async function loadCaptionCatalog(language) {
  if (captionCatalogs.has(language)) return captionCatalogs.get(language);
  if (localizedCatalogs.has(language)) {
    const catalog = validateCaptionCatalog(localizedCatalogs.get(language), language);
    captionCatalogs.set(language, catalog);
    return catalog;
  }
  const filename = CAPTION_ONLY_LANGUAGES.has(language)
    ? `captions_${language}.json` : `lessons_${language}.json`;
  const response = await fetch(`catalog/${filename}`);
  if (!response.ok) throw new Error(`The ${captionLanguageById(language).shortLabel} captions are unavailable`);
  const catalog = validateCaptionCatalog(await response.json(), language);
  captionCatalogs.set(language, catalog);
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
  if (media.error) return Promise.reject(media.error);
  if (media.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      media.removeEventListener("loadedmetadata", loaded);
      media.removeEventListener("error", failed);
    };
    const loaded = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(media.error || new Error("Media could not be loaded")); };
    media.addEventListener("loadedmetadata", loaded, { once: true });
    media.addEventListener("error", failed, { once: true });
  });
}

function mediaReadyOrDeferred(media, timeoutMs = 1500) {
  if (media.error) return Promise.reject(media.error);
  if (media.readyState >= 1) return Promise.resolve(true);
  return new Promise((resolve, reject) => {
    let timer = 0;
    const cleanup = () => {
      clearTimeout(timer);
      media.removeEventListener("loadedmetadata", loaded);
      media.removeEventListener("error", failed);
    };
    const loaded = () => { cleanup(); resolve(true); };
    const failed = () => {
      cleanup();
      reject(media.error || new Error("Media could not be loaded"));
    };
    const deferred = () => { cleanup(); resolve(false); };
    media.addEventListener("loadedmetadata", loaded, { once: true });
    media.addEventListener("error", failed, { once: true });
    timer = setTimeout(deferred, timeoutMs);
  });
}

async function loadReadyLesson() {
  const lessonId = activeLesson.id;
  elements.loading.classList.remove("hidden");
  elements.video.poster = `${PRODUCTION_ROOT}/${activeLesson.poster}`;
  elements.video.src = videoSource();
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
  if (narrationFetchController) narrationFetchController.abort();
  narrationFetchController = null;
  discardNarrationAudio();
  narrationAudioAvailable = false;
  audioTimings = null;
  visualTimings = null;
  clearCaptions();
  if (elements.voice.value === "silent") {
    configureMediaSync();
    try {
      await mediaReady(elements.video);
    } catch (error) {
      if (isCurrent()) showToast("This tutorial video could not be loaded.");
      return;
    }
    if (!isCurrent()) return;
    await loadLessonDetail(isCurrent);
    if (!isCurrent()) return;
    if (wasPlaying) elements.video.play().catch(() => {});
    return;
  }
  elements.loading.classList.remove("hidden");
  const controller = new AbortController();
  narrationFetchController = controller;
  // Resolve failures into data immediately so an early video error cannot
  // leave a rejected fetch promise unobserved.
  const download = fetchNarrationAudio(audioSource(), controller.signal)
    .then(blob => ({ blob, error: null }))
    .catch(error => ({ blob: null, error }));
  try {
    await mediaReady(elements.video);
    if (!isCurrent()) return;
    try {
      const result = await download;
      if (!isCurrent()) return;
      if (result.error) throw result.error;
      narrationObjectUrl = URL.createObjectURL(result.blob);
      elements.audio.src = narrationObjectUrl;
      elements.audio.load();
      // iPhone browsers may defer audio metadata until a user presses Play.
      // The complete Blob is already present, so a metadata timeout means
      // "finish on the play gesture", not "narration is unavailable".
      await mediaReadyOrDeferred(elements.audio);
      narrationAudioAvailable = true;
    } catch (audioError) {
      if (!isCurrent()) return;
      discardNarrationAudio();
      showToast("Narration is unavailable; the GitHub-hosted video remains available.");
    }
    elements.video.currentTime = normalized * elements.video.duration;
    configureMediaSync();
    await loadLessonDetail(isCurrent);
    if (!isCurrent()) return;
    if (wasPlaying) await elements.video.play();
  } catch (error) {
    if (isCurrent()) showToast("This narration track could not be loaded.");
  } finally {
    if (narrationFetchController === controller) narrationFetchController = null;
    if (isCurrent()) elements.loading.classList.add("hidden");
  }
}

function configureMediaSync() {
  if (!narrationAudioAvailable || !elements.audio.duration ||
      !elements.video.duration || elements.voice.value === "silent") {
    syncRate = 1;
    userPlaybackRate = 1;
    programmedVideoRate = 1;
    elements.video.defaultPlaybackRate = 1;
    elements.video.playbackRate = 1;
    return;
  }
  userPlaybackRate = 1;
  elements.audio.playbackRate = 1;
  elements.audio.volume = elements.video.volume;
  elements.audio.muted = elements.video.muted;
  syncAudio(true);
}

function timingDuration(timings, fallback = 0) {
  return Number(timings?.total_duration) || fallback || 0;
}

function timingScene(timings, seconds) {
  const scenes = timings?.scenes;
  if (!Array.isArray(scenes) || !scenes.length) return null;
  const bounded = Math.max(0, Math.min(seconds, timingDuration(timings, seconds)));
  let index = scenes.findIndex(scene => bounded < Number(scene.scene_end));
  if (index < 0) index = scenes.length - 1;
  return { index, scene: scenes[index] };
}

function mapTiming(seconds, fromTimings, toTimings) {
  const fromDuration = timingDuration(fromTimings);
  const toDuration = timingDuration(toTimings);
  if (!fromDuration || !toDuration) return fromDuration ? seconds / fromDuration * toDuration : seconds;
  const located = timingScene(fromTimings, seconds);
  const target = located && toTimings?.scenes?.[located.index];
  if (!located || !target) return seconds / fromDuration * toDuration;
  const sourceStart = Number(located.scene.speech_start) || 0;
  const sourceEnd = Number(located.scene.scene_end) || sourceStart;
  const targetStart = Number(target.speech_start) || 0;
  const targetEnd = Number(target.scene_end) || targetStart;
  const progress = sourceEnd > sourceStart
    ? clamp((seconds - sourceStart) / (sourceEnd - sourceStart), 0, 1) : 0;
  return targetStart + progress * (targetEnd - targetStart);
}

function videoTimeFromAudio(audioSeconds) {
  if (!elements.video.duration) return audioSeconds;
  const referenceDuration = timingDuration(visualTimings, elements.video.duration);
  const referenceTime = mapTiming(audioSeconds, audioTimings, visualTimings || audioTimings);
  return referenceDuration ? referenceTime / referenceDuration * elements.video.duration : audioSeconds;
}

function audioTimeFromVideo(videoSeconds) {
  if (!elements.video.duration) return videoSeconds;
  const referenceDuration = timingDuration(visualTimings, elements.video.duration);
  const referenceTime = videoSeconds / elements.video.duration * referenceDuration;
  return mapTiming(referenceTime, visualTimings || audioTimings, audioTimings);
}

function updateSceneSyncRate(audioSeconds) {
  if (!narrationAudioAvailable || !elements.video.duration ||
      !audioTimings || !visualTimings) return;
  const selected = timingScene(audioTimings, audioSeconds);
  const reference = selected && visualTimings.scenes?.[selected.index];
  if (!selected || !reference) return;
  const selectedDuration = Number(selected.scene.scene_end) - Number(selected.scene.speech_start);
  const referenceDuration = Number(reference.scene_end) - Number(reference.speech_start);
  const referenceTotal = timingDuration(visualTimings, elements.video.duration);
  if (selectedDuration <= 0 || referenceDuration <= 0 || referenceTotal <= 0) return;
  syncRate = elements.video.duration / referenceTotal * referenceDuration / selectedDuration;
  programmedVideoRate = clamp(syncRate * userPlaybackRate, 0.0625, 16);
  elements.video.defaultPlaybackRate = syncRate;
  if (Math.abs(elements.video.playbackRate - programmedVideoRate) > 0.001) {
    elements.video.playbackRate = programmedVideoRate;
  }
  elements.audio.playbackRate = userPlaybackRate;
}

function syncAudio(force = false) {
  if (!narrationAudioAvailable || elements.voice.value === "silent" ||
      !elements.audio.duration || !elements.video.duration) return;
  const target = audioTimeFromVideo(elements.video.currentTime);
  if (force || Math.abs(elements.audio.currentTime - target) > 0.16) {
    elements.audio.currentTime = Math.min(target, Math.max(0, elements.audio.duration - 0.02));
  }
  updateSceneSyncRate(target);
}

async function loadLessonDetail(isCurrent = () => true) {
  if (!isCurrent()) return;
  elements.chapters.innerHTML = `<div class="detail-empty">Loading chapters…</div>`;
  try {
    const requestedSource = timingSource();
    let response = await fetch(requestedSource);
    if (!response.ok && requestedSource !== defaultTimingSource()) {
      response = await fetch(defaultTimingSource());
    }
    if (!response.ok) throw new Error("timings unavailable");
    const timings = await response.json();
    let referenceTimings = timings;
    if (requestedSource !== defaultTimingSource()) {
      const referenceResponse = await fetch(defaultTimingSource());
      if (referenceResponse.ok) referenceTimings = await referenceResponse.json();
    }
    if (!isCurrent()) return;
    audioTimings = timings;
    visualTimings = referenceTimings;
    configureMediaSync();
    rebuildChapterData();
    renderCaptions();
    renderChapters();
    renderTranscript();
  } catch (error) {
    if (!isCurrent()) return;
    audioTimings = null;
    clearCaptions();
    elements.chapters.innerHTML = `<div class="detail-empty">Chapter metadata is unavailable for this track.</div>`;
    elements.transcript.innerHTML = elements.chapters.innerHTML;
  }
}

function rebuildChapterData() {
  if (!activeLesson) {
    chapterData = [];
    return;
  }
  const lesson = captionLesson(activeLesson.id);
  chapterData = lesson.scenes.map((scene, index) => {
    const timing = audioTimings?.scenes?.[index];
    const start = timing?.speech_start || 0;
    const end = timing?.speech_end || start;
    return {
      index: index + 1,
      start,
      end,
      text: scene.narration,
      label: chapterLabel(scene.narration, index)
    };
  });
}

function clearCaptions() {
  if (captionUrl) URL.revokeObjectURL(captionUrl);
  captionUrl = "";
  captionTrackLoading = false;
  if (!elements.captionTrack) return;
  if (elements.captionTrack.track) elements.captionTrack.track.mode = "disabled";
  elements.captionTrack.removeAttribute("src");
}

function splitCaptionText(text) {
  const sentences = String(text).match(/[^.!?。！？।]+(?:[.!?。！？।]+|$)/gu) || [String(text)];
  return sentences.map(sentence => sentence.trim()).filter(Boolean);
}

function renderCaptions() {
  if (!audioTimings) {
    clearCaptions();
    return;
  }
  if (captionUrl) URL.revokeObjectURL(captionUrl);
  const lines = ["WEBVTT", ""];
  let cueIndex = 0;
  chapterData.forEach(chapter => {
    const sentences = splitCaptionText(chapter.text);
    const weights = sentences.map(sentence => Math.max(1, [...sentence].length));
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    const duration = Math.max(0, chapter.end - chapter.start);
    let cursor = chapter.start;
    sentences.forEach((sentence, index) => {
      const end = index === sentences.length - 1
        ? chapter.end : cursor + duration * weights[index] / totalWeight;
      lines.push(
        String(++cueIndex),
        `${vttTime(videoTimeFromAudio(cursor))} --> ${vttTime(videoTimeFromAudio(end))}`,
        sentence.replaceAll("-->", "→"),
        ""
      );
      cursor = end;
    });
  });
  captionUrl = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/vtt" }));
  captionTrackLoading = true;
  elements.captionTrack.src = captionUrl;
  const language = effectiveCaptionLanguage();
  elements.captionTrack.srclang = language;
  elements.captionTrack.label = captionLanguageById(language).shortLabel;
  updateCaptionTrackMode();
  elements.captionTrack.addEventListener("load", () => {
    captionTrackLoading = false;
    updateCaptionTrackMode();
  }, { once: true });
  elements.captionTrack.addEventListener("error", () => {
    captionTrackLoading = false;
  }, { once: true });
}

function renderChapters() {
  elements.chapters.innerHTML = "";
  elements.chapters.lang = effectiveCaptionLanguage();
  elements.chapters.dir = "auto";
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
  elements.transcript.lang = effectiveCaptionLanguage();
  elements.transcript.dir = "auto";
  elements.transcript.innerHTML = chapterData.map(chapter => `<div class="transcript-entry"><button type="button" data-seek="${chapter.start}">${formatTime(chapter.start)}</button><p>${escapeHTML(chapter.text)}</p></div>`).join("");
  elements.transcript.querySelectorAll("[data-seek]").forEach(button => button.addEventListener("click", () => seekTo(Number(button.dataset.seek))));
}

function seekTo(audioSeconds) {
  elements.video.currentTime = videoTimeFromAudio(audioSeconds);
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
  const duration = audioTimings?.total_duration ||
    (narrationAudioAvailable ? elements.audio.duration : elements.video.duration);
  const current = audioTimings ? audioTimeFromVideo(elements.video.currentTime) : ratio * (duration || 0);
  return { current, duration: duration || 0, ratio };
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
  const captionsFollowNarration = captionSettings.language === "auto";
  localStorage.setItem(LANGUAGE_KEY, requestedLanguage);
  populateVoiceSelector();
  localStorage.setItem(VOICE_KEY, elements.voice.value);
  elements.loading.classList.remove("hidden");
  const captionResult = captionsFollowNarration
    ? loadCaptionCatalog(requestedLanguage)
      .then(catalog => ({ catalog, error: null }))
      .catch(error => ({ catalog: null, error }))
    : Promise.resolve(null);
  try {
    let catalog;
    try {
      catalog = await loadLocalizedCatalog(requestedLanguage);
    } catch (error) {
      catalog = BASE_CATALOG;
      if (isCurrent()) showToast(error.message);
    }
    if (!isCurrent()) return;
    localizedCatalog = catalog;
    const captions = await captionResult;
    if (!isCurrent()) return;
    if (captionsFollowNarration && captionSettings.language === "auto") {
      if (captions.error) {
        captionSettings.language = DEFAULT_LANGUAGE;
        captionCatalog = BASE_CATALOG;
        applyCaptionSettings({ persist: true });
        showToast(captions.error.message);
      } else {
        captionCatalog = captions.catalog;
      }
    }
    renderCurriculum(elements.search.value); updateLessonHeader(); updateGuide(); updatePagination();
    await loadNarration(true, isCurrent);
  } catch (error) { if (isCurrent()) showToast(error.message); }
  finally { if (isCurrent()) elements.loading.classList.add("hidden"); }
}

async function switchCaptionLanguage() {
  captionSettings.language = elements.captionLanguage.value;
  applyCaptionSettings({ persist: true });
  const requestedLanguage = effectiveCaptionLanguage();
  const request = ++captionLanguageRequest;
  const isCurrent = () => request === captionLanguageRequest &&
    requestedLanguage === effectiveCaptionLanguage();
  try {
    const catalog = await loadCaptionCatalog(requestedLanguage);
    if (!isCurrent()) return;
    captionCatalog = catalog;
    rebuildChapterData();
    renderCaptions();
    renderChapters();
    renderTranscript();
  } catch (error) {
    if (!isCurrent()) return;
    captionSettings.language = DEFAULT_LANGUAGE;
    captionCatalog = BASE_CATALOG;
    applyCaptionSettings({ persist: true });
    rebuildChapterData();
    renderCaptions();
    renderChapters();
    renderTranscript();
    showToast(error.message);
  }
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

function updateCaptionAppearanceFromControls() {
  captionSettings = {
    ...captionSettings,
    size: Number(elements.captionSize.value),
    color: elements.captionTextColor.value,
    background: elements.captionBackgroundColor.value,
    backgroundOpacity: Number(elements.captionBackgroundOpacity.value)
  };
  applyCaptionSettings({ persist: true });
}

function resetCaptionAppearance() {
  captionSettings = {
    ...DEFAULT_CAPTION_SETTINGS,
    enabled: captionSettings.enabled,
    language: captionSettings.language
  };
  applyCaptionSettings({ persist: true });
}

elements.search.addEventListener("input", event => renderCurriculum(event.target.value));
elements.video.addEventListener("play", () => {
  syncAudio(true);
  if (narrationAudioAvailable) {
    elements.audio.play().catch(() => showToast("Select play again to start narration."));
  }
});
elements.video.addEventListener("pause", () => { elements.audio.pause(); saveWatchPosition(); });
elements.video.addEventListener("seeking", () => syncAudio(true));
elements.video.addEventListener("timeupdate", () => { syncAudio(false); updateWatchUI(); });
elements.video.addEventListener("volumechange", () => { elements.audio.volume = elements.video.volume; elements.audio.muted = elements.video.muted; });
elements.video.addEventListener("ratechange", () => {
  if (narrationAudioAvailable && syncRate && elements.voice.value !== "silent") {
    if (Math.abs(elements.video.playbackRate - programmedVideoRate) > 0.001) {
      userPlaybackRate = clamp(elements.video.playbackRate / syncRate, 0.25, 4);
      programmedVideoRate = elements.video.playbackRate;
    }
    elements.audio.playbackRate = userPlaybackRate;
  }
});
elements.video.addEventListener("ended", markCompleteAtEnd);
elements.audio.addEventListener("ended", markCompleteAtEnd);
elements.audio.addEventListener("loadedmetadata", () => {
  if (!narrationAudioAvailable) return;
  configureMediaSync();
  if (!elements.video.paused) {
    syncAudio(true);
    elements.audio.play().catch(() => {
      showToast("Select play again to start narration.");
    });
  }
});
elements.audio.addEventListener("error", () => {
  // Loading errors are handled by loadNarration while availability is false.
  // This catches a deferred phone decoder failure after the player is ready.
  if (!narrationAudioAvailable || !elements.audio.getAttribute("src")) return;
  narrationAudioAvailable = false;
  discardNarrationAudio();
  configureMediaSync();
  showToast("Narration is unavailable; the GitHub-hosted video remains available.");
});
elements.video.addEventListener("canplay", () => elements.loading.classList.add("hidden"));
elements.video.textTracks?.addEventListener?.("change", syncCaptionPreferenceFromNativeControls);
elements.voice.addEventListener("change", switchVoice);
elements.language.addEventListener("change", switchLanguage);
elements.captionLanguage.addEventListener("change", switchCaptionLanguage);
elements.captionEnabled.addEventListener("change", () => {
  captionSettings.enabled = elements.captionEnabled.checked;
  applyCaptionSettings({ persist: true });
});
elements.captionSize.addEventListener("input", updateCaptionAppearanceFromControls);
elements.captionTextColor.addEventListener("input", updateCaptionAppearanceFromControls);
elements.captionBackgroundColor.addEventListener("input", updateCaptionAppearanceFromControls);
elements.captionBackgroundOpacity.addEventListener("input", updateCaptionAppearanceFromControls);
elements.captionReset.addEventListener("click", resetCaptionAppearance);
elements.captionSettingsButton.addEventListener("click", () => {
  if (elements.captionSettingsPanel.hidden) openCaptionSettings();
  else closeCaptionSettings(true);
});
elements.captionSettingsClose.addEventListener("click", () => closeCaptionSettings(true));
document.addEventListener("pointerdown", event => {
  const insidePortaledMenu = event.target.closest?.(".custom-select-menu.portaled");
  if (!elements.captionSettingsPanel.hidden &&
      !elements.captionSettings.contains(event.target) && !insidePortaledMenu) {
    closeCaptionSettings();
  }
});
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
window.addEventListener("beforeunload", () => {
  saveWatchPosition();
  if (narrationFetchController) narrationFetchController.abort();
  if (narrationObjectUrl) URL.revokeObjectURL(narrationObjectUrl);
  if (captionUrl) URL.revokeObjectURL(captionUrl);
});
window.addEventListener("resize", () => {
  if (!elements.captionSettingsPanel.hidden) positionCaptionSettingsPanel();
});
window.addEventListener("hashchange", () => {
  const id = lessonFromHash();
  if (activeLesson && id && id !== activeLesson.id) selectLesson(id, { skipHash: true });
});
window.addEventListener("keydown", event => {
  if (event.key === "Escape" && !elements.captionSettingsPanel.hidden) {
    event.preventDefault(); event.stopPropagation(); closeCaptionSettings(true); return;
  }
  const tag = document.activeElement?.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || event.target.closest?.(".custom-select")) return;
  if (event.key === "/") { event.preventDefault(); elements.search.focus(); }
  else if (event.key === "[") { event.preventDefault(); elements.previous.click(); }
  else if (event.key === "]") { event.preventDefault(); elements.next.click(); }
  else if (event.key === "Escape") closeSidebar(true);
});
if (mobileSidebarQuery.addEventListener) mobileSidebarQuery.addEventListener("change", handleSidebarBreakpoint);
else mobileSidebarQuery.addListener(handleSidebarBreakpoint);

elements.availableCount.textContent = LESSONS.length;
elements.totalCount.textContent = LESSONS.length;
applyTheme(document.documentElement.dataset.theme);
syncSidebarAccessibility();
setupNarrationSelectors();
setupCaptionControls();
setMediaSelectorsDisabled(true);
enhanceSelect(elements.language); enhanceSelect(elements.voice); enhanceSelect(elements.captionLanguage);
setupDetailTabs(); updateCourseProgress(); renderCurriculum();

async function initializeApp() {
  setupQualityControl();
  const requestedLanguage = elements.language.value;
  const requestedCaptionLanguage = effectiveCaptionLanguage();
  const captionTask = (async () => {
    try {
      const catalog = await loadCaptionCatalog(requestedCaptionLanguage);
      if (effectiveCaptionLanguage() !== requestedCaptionLanguage) return;
      captionCatalog = catalog;
      if (activeLesson) {
        rebuildChapterData();
        renderCaptions();
        renderChapters();
        renderTranscript();
      }
    } catch (error) {
      if (effectiveCaptionLanguage() !== requestedCaptionLanguage) return;
      captionSettings.language = DEFAULT_LANGUAGE;
      captionCatalog = BASE_CATALOG;
      applyCaptionSettings({ persist: true });
      if (activeLesson) {
        rebuildChapterData();
        renderCaptions();
        renderChapters();
        renderTranscript();
      }
      showToast(error.message);
    }
  })();

  try {
    localizedCatalog = await loadLocalizedCatalog(requestedLanguage);
  } catch (error) {
    localizedCatalog = BASE_CATALOG;
    showToast(error.message);
  }
  renderCurriculum();
  try {
    await selectLesson(lessonFromHash() || LESSONS[0].id, {
      skipHash: !location.hash,
      force: true
    });
  } finally {
    setMediaSelectorsDisabled(false);
  }
  await captionTask;
}

initializeApp().catch(error => {
  setMediaSelectorsDisabled(false);
  showToast(error?.message || "The tutorial player could not be initialized.");
});
