"use strict";

const VOICE_CATALOG = window.SPACR_VOICE_CATALOG || [];
const BASE_CATALOG = window.SPACR_LESSON_CATALOG;
const MODULE_NAVIGATION = window.SPACR_TUTORIAL_NAVIGATION;
const DEFAULT_LANGUAGE = "en";
const DEFAULT_VOICE = "af_heart";
const PRODUCTION_ROOT = document.documentElement.dataset.productionRoot || "../production";
// Narration is served separately from the video. The 50 supported voices are
// collectively too large for a GitHub Pages site. Publishing one voice per
// language was the only way to fit, and it left 23 of 24 English voices
// unusable.
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
// Web copies that live on the same immutable media revision instead of the
// Pages tree. Only a lesson whose catalog entry names its hosted copy
// (lesson.web) uses it; every other lesson keeps its local copy.
const WEB_VIDEO_ROOT = document.documentElement.dataset.webRoot || "";
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
  content: $("#lesson-content"),
  seriesLabel: $("#series-label"), position: $("#lesson-position"),
  status: $("#status-pill"), title: $("#lesson-title"),
  route: $("#lesson-route"), description: $("#lesson-description"),
  duration: $("#duration-badge"),
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

const navigationOrder = MODULE_NAVIGATION
  ? [...(MODULE_NAVIGATION.intro?.lessons || []),
     ...MODULE_NAVIGATION.sections.flatMap(section => section.groups.flatMap(group => group.lessons))]
  : [];
const lessonByIdentity = new Map(BASE_CATALOG.lessons.map(lesson => [lesson.id, lesson]));
const orderedIdentities = [...new Set([...navigationOrder, ...lessonByIdentity.keys()])];
const LESSONS = orderedIdentities.map(id => lessonByIdentity.get(id)).filter(Boolean);
const isPlayable = lesson => Boolean(lesson) && lesson.status !== "coming_soon";
const AVAILABLE_LESSONS = LESSONS.filter(isPlayable);
const playerToolbar = elements.player.querySelector('.player-toolbar');
const playerToolbarNext = playerToolbar.nextSibling;
const localizedCatalogs = new Map([["en", BASE_CATALOG]]);
const captionCatalogs = new Map([["en", BASE_CATALOG]]);
let localizedCatalog = BASE_CATALOG;
let captionCatalog = BASE_CATALOG;
let activeLesson = null;
let chapterData = [];
let audioTimings = null;
let visualTimings = null;
let syncRate = 1;
let programmedVideoRate = 1;
// The narration is the audible clock. A phone must never have to repair drift
// by seeking or rate-shifting speech while it is playing; either operation is
// heard as a skip, repeated consonant, or stretched syllable. The silent
// visual master can absorb a sub-frame correction without an audible cost.
const VIDEO_CLOCK_DRIFT_SECONDS = 0.4;
const VIDEO_END_PARK_SECONDS = 0.10;
const VIDEO_END_PARK_RATE = 0.0625;
let videoClockCorrectionPending = false;
// True from a narration-led video skip until the picture plays again: the
// narration is the master clock and must never wait for the picture.
let videoCatchingUpToNarration = false;
let nativeFrameSyncHandle = null;
let completeNativeSceneIndex = null;
let completeNativeNarrationWaiting = false;
// A localized narration can outlast the silent visual master. The video stays
// genuinely playing at a browser-safe minimum rate near its final frame so
// native phone controls still offer a truthful Pause button while audio ends.
let videoParkedForNarration = false;
let narratedPlaybackCompleted = false;
let programmaticVideoPlayPending = false;
let programmaticVideoPausePending = false;
// Narration waits for the picture. While the video is seeking or has too
// little data to advance, the narration clock is held (paused) without
// touching the viewer's play state, and both resume together once the
// picture can play. Otherwise speech runs on over a frozen frame and the
// clock correction chases a moving target on a slow media host.
let narrationHeldForVideo = false;
let narrationLoadPending = false;
let videoHeldForAudio = false;
let audioHoldPauseEventsPending = 0;
let resumeAfterAudioHold = false;
let captionUrl = "";
let captionTrackLoading = false;
let toastTimer = null;
let openCustomSelect = null;
let languageRequest = 0;
let narrationRequest = 0;
let nativeIdleTransition = null;
let nativeIdleTransitionTimer = null;
let captionLanguageRequest = 0;
let narrationAudioAvailable = false;
let narrationFetchController = null;
let narrationObjectUrl = "";
let captionSettings = readCaptionSettings();
let completed = readStoredSet(STORAGE_KEY);
let watchProgress = readStoredObject(WATCH_KEY);
const mobileSidebarQuery = window.matchMedia("(max-width: 900px)");

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
  const showing = captionSettings.enabled && Boolean(captionUrl) && !nativeIdleTransition;
  elements.captionTrack.track.mode = showing ? "showing" : "disabled";
}

function syncCaptionPreferenceFromNativeControls() {
  if (nativeIdleTransition) return;
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

function routedLesson(lesson) {
  const location = MODULE_NAVIGATION?.routes?.[lesson?.id];
  const hostKey = location ? location.host_app_key : lesson?.host_app_key;
  const appKey = hostKey || lesson?.app_key || null;
  const host = hostKey
    ? LESSONS.find(item => item.app_key === hostKey && !item.host_app_key)
    : null;
  return { appKey, host };
}

function seriesBlocks() {
  if (MODULE_NAVIGATION) {
    const language = elements.language?.value || DEFAULT_LANGUAGE;
    const labels = MODULE_NAVIGATION.labels[language] || MODULE_NAVIGATION.labels.en;
    return MODULE_NAVIGATION.sections.map(section => ({
      id: section.id,
      title: labels[section.label_index],
      groups: section.groups.map(group => ({
        ...group,
        title: group.label_index != null ? labels[group.label_index]
          : group.host_lesson && language !== "en"
            ? localizedLesson(group.host_lesson).title : group.title,
        lessons: group.lessons.map(baseLesson).filter(Boolean)
      }))
    }));
  }
  return localizedCatalog.series.map(series => ({
    ...series,
    id: `series-${series.number}`,
    groups: [{title: "", lessons: LESSONS.filter(lesson => lesson.series === series.number)}]
  }));
}

function voiceById(language, id) {
  return language?.voices.find(voice => voice.id === id);
}

function lessonVoices(language, lesson = activeLesson) {
  // An absent declaration preserves the published fifty-voice lessons.
  // An explicit declaration offers only the tracks verified for this lesson.
  if (!lesson || !Object.prototype.hasOwnProperty.call(lesson, "narration_voices")) {
    return language.voices;
  }
  const declared = lesson.narration_voices?.[language.id];
  if (!Array.isArray(declared)) return [];
  return language.voices.filter(voice => declared.includes(voice.id));
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
  if (!isPlayable(lesson)) return "";
  if (fourKAvailable() && elements.quality?.value === "4k") {
    return `${VIDEO_4K_ROOT}/${lesson.silent}`;
  }
  if (WEB_VIDEO_ROOT && lesson.web) {
    return `${WEB_VIDEO_ROOT}/${lesson.web}`;
  }
  return `${PRODUCTION_ROOT}/${lesson.silent}`;
}

// Swapping src resets the element, so the position, the rate mapping and the
// play state all have to be put back by hand. Narration is untouched: both
// cuts share one timeline, so the audio element keeps playing underneath and
// only needs re-syncing to the new video clock.
async function applyQualityChange() {
  if (!isPlayable(activeLesson)) return;
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
    setVideoTimeWithoutNarrationSeek(resumeAt);
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
  if (!isPlayable(lesson) || elements.voice.value === "silent") return "";
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

function visualTimingSource(lesson = activeLesson) {
  return lesson.visual_timings
    ? `${narrationRoot()}/${lesson.visual_timings}` : defaultTimingSource(lesson);
}

function populateVoiceSelector(preferredVoice = "") {
  const language = languageById(elements.language.value);
  const voices = lessonVoices(language);
  elements.voice.innerHTML = "";
  voices.forEach(voice => {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = `${voice.name} · ${voice.variant}`;
    elements.voice.appendChild(option);
  });
  const silent = document.createElement("option");
  silent.value = "silent";
  silent.textContent = "Silent master";
  elements.voice.appendChild(silent);
  const available = preferredVoice === "silent" || voices.some(voice => voice.id === preferredVoice);
  elements.voice.value = available ? preferredVoice : voices[0]?.id || "silent";
}

function populateNarrationLanguages(preferredLanguage) {
  elements.language.innerHTML = "";
  VOICE_CATALOG.forEach(language => {
    const count = lessonVoices(language).length;
    const option = document.createElement("option");
    option.value = language.id;
    option.textContent = count
      ? `${language.label} · ${count} ${count === 1 ? "voice" : "voices"}`
      : `${language.label} · silent`;
    elements.language.appendChild(option);
  });
  elements.language.value = languageById(preferredLanguage).id;
}

function setupNarrationSelectors() {
  const preferredLanguage = localStorage.getItem(LANGUAGE_KEY) || DEFAULT_LANGUAGE;
  populateNarrationLanguages(preferredLanguage);
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
  if (MODULE_NAVIGATION?.intro) {
    const language = elements.language?.value || DEFAULT_LANGUAGE;
    const labels = MODULE_NAVIGATION.labels[language] || MODULE_NAVIGATION.labels.en;
    const title = labels[MODULE_NAVIGATION.intro.label_index];
    const lessons = MODULE_NAVIGATION.intro.lessons.map(baseLesson).filter(base => {
      const lesson = localizedLesson(base.id);
      return !normalized || `${lesson.title} ${lesson.description} ${title}`.toLowerCase().includes(normalized);
    });
    if (lessons.length) {
      elements.curriculum.appendChild(makeModuleGroup({id: "getting-started", title, lessons}));
      matches += lessons.length;
    }
  }
  seriesBlocks().forEach(series => {
    const groups = series.groups.map(group => ({...group,
      lessons: group.lessons.filter(base => {
        const lesson = localizedLesson(base.id);
        const host = routedLesson(base).host;
        const hostTitle = host ? localizedLesson(host.id).title : "";
        return !normalized || `${lesson.title} ${lesson.description} ${hostTitle} ${group.title} ${series.title}`
          .toLowerCase().includes(normalized);
      })
    })).filter(group => group.lessons.length);
    const count = groups.reduce((sum, group) => sum + group.lessons.length, 0);
    if (!count) return;
    matches += count;
    const block = document.createElement("section");
    block.className = "series-block";
    block.dataset.section = series.id;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "series-toggle";
    toggle.setAttribute("aria-expanded", "true");
    toggle.innerHTML = `<span><strong>${escapeHTML(series.title)}</strong></span><span class="series-count">${count}</span><svg class="chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>`;
    toggle.addEventListener("click", () => {
      const collapsed = block.classList.toggle("collapsed");
      toggle.setAttribute("aria-expanded", String(!collapsed));
    });
    const list = document.createElement("div");
    list.className = "series-lessons";
    groups.forEach(group => list.appendChild(makeModuleGroup(group)));
    block.append(toggle, list);
    elements.curriculum.appendChild(block);
  });
  if (!matches) elements.curriculum.innerHTML = `<div class="empty-search">No tutorials match “${escapeHTML(query)}”.</div>`;
}

function makeModuleGroup(group) {
  const container = document.createElement("div");
  container.className = "module-group";
  container.dataset.host = group.id || "";
  if (group.title) {
    const heading = document.createElement("h3");
    heading.className = "module-group-heading";
    heading.textContent = `${group.help_host ? "Help → " : ""}${group.title}`;
    container.appendChild(heading);
  }
  group.lessons.forEach(lesson => container.appendChild(makeLessonLink(lesson)));
  return container;
}

function makeLessonLink(base) {
  const lesson = localizedLesson(base.id);
  const button = document.createElement("button");
  button.type = "button";
  button.className = `lesson-link ${isPlayable(base) ? "ready" : "planned"}`;
  button.dataset.lesson = base.id;
  if (base.id === activeLesson?.id) button.classList.add("active");
  if (isPlayable(base) && completed.has(base.id)) button.classList.add("complete");
  button.setAttribute("aria-current", base.id === activeLesson?.id ? "page" : "false");
  button.innerHTML = `<span class="lesson-number">${String(LESSONS.indexOf(base) + 1).padStart(2, "0")}</span><span class="lesson-link-copy"><strong>${escapeHTML(lesson.title)}</strong><small>${escapeHTML(isPlayable(base) ? "4K video" : lesson.availability_title || "Coming soon")}</small></span><span class="lesson-state-dot" aria-hidden="true"></span>`;
  button.addEventListener("click", () => selectLesson(base.id));
  return button;
}

async function selectLesson(id, options = {}) {
  id = BASE_CATALOG.lesson_aliases?.[id] || id;
  const lesson = baseLesson(id) || LESSONS[0];
  if (activeLesson?.id === lesson.id && !options.force) {
    closeSidebar();
    return;
  }
  if (activeLesson) saveWatchPosition();
  elements.video.pause();
  activeLesson = lesson;
  const preferredVoice = elements.voice.value;
  populateNarrationLanguages(elements.language.value);
  populateVoiceSelector(preferredVoice);
  if (!options.skipHash) history.replaceState(null, "", `#lesson=${lesson.id}`);
  renderCurriculum(elements.search.value);
  updateLessonHeader();
  updateGuide();
  updatePagination();
  updateCompleteButton();
  closeSidebar();
  if (!isPlayable(lesson)) {
    ++narrationRequest;
    narrationFetchController?.abort();
    narrationAudioAvailable = false;
    discardNarrationAudio();
    elements.video.removeAttribute("src");
    elements.video.removeAttribute("poster");
    elements.video.load();
    if (captionUrl) URL.revokeObjectURL(captionUrl);
    captionUrl = "";
    elements.captionTrack.removeAttribute("src");
    chapterData = [];
    audioTimings = visualTimings = null;
    elements.chapters.replaceChildren();
    elements.transcript.replaceChildren();
    elements.player.hidden = true;
    elements.planned.hidden = false;
    elements.chapterCard.hidden = true;
    document.querySelector('.lesson-guide').hidden = true;
    elements.planned.appendChild(playerToolbar);
    updateComingSoon();
    if (options.focus) elements.content.focus({ preventScroll: true });
    return;
  }
  elements.player.insertBefore(playerToolbar, playerToolbarNext);
  document.querySelector('.lesson-guide').hidden = false;
  elements.voice.disabled = false;
  elements.voice.customSelect?.setDisabled(false);
  if (elements.quality) elements.quality.disabled = false;
  elements.player.hidden = false;
  elements.planned.hidden = true;
  elements.chapterCard.hidden = false;
  await loadReadyLesson();
  if (options.focus) $("#lesson-content").focus({ preventScroll: true });
}

function updateLessonHeader() {
  const lesson = localizedLesson(activeLesson.id);
  const series = localizedCatalog.series.find(item => item.number === activeLesson.series);
  const route = routedLesson(activeLesson);
  const navigationSection = seriesBlocks().find(section => section.groups.some(
    group => group.lessons.some(item => item.id === activeLesson.id)));
  const navLabels = MODULE_NAVIGATION?.labels[elements.language?.value || DEFAULT_LANGUAGE]
    || MODULE_NAVIGATION?.labels.en;
  elements.seriesLabel.textContent = navigationSection?.title
    || (navLabels ? navLabels[2] : `Series ${activeLesson.series}`);
  elements.position.textContent = `Lesson ${LESSONS.indexOf(activeLesson) + 1} of ${LESSONS.length}`;
  elements.status.textContent = isPlayable(activeLesson) ? "Ready" : lesson.availability_title || "Coming soon";
  elements.status.className = `status-pill ${isPlayable(activeLesson) ? "ready" : "planned"}`;
  elements.title.textContent = lesson.title;
  elements.content.dataset.appKey = route.appKey || "";
  elements.route.replaceChildren();
  if (route.host) {
    const host = localizedLesson(route.host.id);
    const hostName = document.createElement("span");
    const arrow = document.createElement("span");
    const workflowName = document.createElement("span");
    hostName.className = "lesson-route-host";
    hostName.textContent = host.title;
    arrow.className = "lesson-route-arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "→";
    workflowName.className = "lesson-route-workflow";
    workflowName.textContent = lesson.title;
    elements.route.append(hostName, arrow, workflowName);
    elements.route.setAttribute(
      "aria-label", `${host.title}: ${lesson.title}`);
    elements.route.hidden = false;
  } else {
    elements.route.removeAttribute("aria-label");
    elements.route.hidden = true;
  }
  elements.description.textContent = lesson.description;
  document.title = `${lesson.title} · spaCR Learning Path`;
}

function updateComingSoon() {
  if (!activeLesson || isPlayable(activeLesson)) return;
  const lesson = captionSettings.language === "auto"
    ? localizedLesson(activeLesson.id) : captionLesson(activeLesson.id);
  $("#planned-title").textContent = lesson.availability_title || "Coming soon";
  $("#planned-copy").textContent = lesson.description;
  $("#planned-copy").parentElement.lang = effectiveCaptionLanguage();
  elements.status.textContent = lesson.availability_title || "Coming soon";
  elements.voice.disabled = true;
  elements.voice.customSelect?.setDisabled(true);
  if (elements.quality) elements.quality.disabled = true;
  elements.loading.classList.add("hidden");
}

function tutorialExampleFiles(lesson) {
  if (!Array.isArray(lesson.example_files)) return [];
  return [...new Set(lesson.example_files.filter(name =>
    typeof name === "string" && name.length <= 128 &&
    /^[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:csv|zip)$/.test(name)))];
}

function updateGuide() {
  const lesson = localizedLesson(activeLesson.id);
  elements.duration.textContent = "Selectable narration";
  elements.objectives.innerHTML = lesson.objectives.map(item => `<li>${escapeHTML(item)}</li>`).join("");
  elements.prerequisite.textContent = lesson.prerequisite;
  for (const name of tutorialExampleFiles(lesson)) {
    const link = document.createElement("a");
    link.href = "examples/" + encodeURIComponent(name);
    link.download = name;
    link.textContent = name;
    elements.prerequisite.append(document.createElement("br"), link);
  }
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
  videoParkedForNarration = false;
  narratedPlaybackCompleted = false;
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
  if (saved > 0 && saved < elements.video.duration - 0.25) {
    setVideoTimeWithoutNarrationSeek(saved);
  }
  configureMediaSync();
  elements.loading.classList.add("hidden");
  updateWatchUI();
}

async function loadNarration(resume = true, outerCurrent = () => true) {
  cancelNativeIdleTransition();
  if (!isPlayable(activeLesson)) { updateComingSoon(); return; }
  nativeIdlePauseClock = null;
  const request = ++narrationRequest;
  const isCurrent = () => request === narrationRequest && outerCurrent();
  const wasPlaying = resume && (!elements.video.paused ||
    (videoHeldForAudio && resumeAfterAudioHold));
  if (!resume) {
    videoHeldForAudio = false;
    resumeAfterAudioHold = false;
  }
  narrationLoadPending = elements.voice.value !== "silent";
  if (wasPlaying && narrationLoadPending) holdVideoForAudio();
  videoParkedForNarration = false;
  narratedPlaybackCompleted = false;
  if (narrationFetchController) narrationFetchController.abort();
  narrationFetchController = null;
  discardNarrationAudio();
  narrationAudioAvailable = false;
  // Keep the previous mapping for a chapter/seek made during a voice load.
  // Narration cannot use it until the selected track's metadata is ready.
  if (!resume) {
    audioTimings = null;
    visualTimings = null;
  }
  completeNativeSceneIndex = null;
  completeNativeNarrationWaiting = false;
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
    releaseVideoForAudio();
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
      if (!isCurrent()) return;
      narrationAudioAvailable = true;
    } catch (audioError) {
      if (!isCurrent()) return;
      discardNarrationAudio();
      showToast("Narration is unavailable; the GitHub-hosted video remains available.");
    }
    await loadLessonDetail(isCurrent);
    if (!isCurrent()) return;
    if (narrationAudioAvailable && (!audioTimings || !visualTimings)) {
      narrationAudioAvailable = false;
      discardNarrationAudio();
    }
    narrationLoadPending = false;
    configureMediaSync();
    if (videoHeldForAudio && resumeAfterAudioHold) {
      if (narrationAudioAvailable) await playNarration();
      else releaseVideoForAudio();
    }
  } catch (error) {
    if (isCurrent()) showToast("This narration track could not be loaded.");
  } finally {
    if (narrationFetchController === controller) narrationFetchController = null;
    if (isCurrent()) {
      narrationLoadPending = false;
      if (!videoHeldForAudio) elements.loading.classList.add("hidden");
    }
  }
}

function configureMediaSync() {
  if (narrationLoadPending) return;
  if (!narrationAudioAvailable || !elements.audio.duration ||
      !elements.video.duration || elements.voice.value === "silent") {
    videoParkedForNarration = false;
    syncRate = 1;
    programmedVideoRate = 1;
    elements.video.defaultPlaybackRate = 1;
    elements.video.playbackRate = 1;
    return;
  }
  elements.audio.defaultPlaybackRate = 1;
  elements.audio.playbackRate = 1;
  elements.audio.volume = elements.video.volume;
  elements.audio.muted = elements.video.muted;
  seekNarrationToVideo();
}

function completeNativePlayback() {
  return (visualTimings?.complete_native_clips === true ||
    visualTimings?.preserve_visual_intervals === true) &&
    visualTimings?.native_live_video === true;
}

function narrationOwnsCompletion() {
  return !completeNativePlayback() && narrationAudioAvailable && elements.voice.value !== "silent" &&
    Number.isFinite(elements.audio.duration) && elements.audio.duration > 0 &&
    !elements.audio.ended;
}

function shouldPauseNarrationForVideoPause(
  narrationIsCompletionClock,
  videoIsEnded
) {
  return !(narrationIsCompletionClock && videoIsEnded);
}

function narratedVideoPlayAction(
  programmaticPlay,
  completedNarration,
  parkedForNarration
) {
  if (programmaticPlay) return "programmatic";
  if (completedNarration) return "replay";
  if (parkedForNarration) return "resume";
  return "sync";
}

function setVideoTimeWithoutNarrationSeek(target) {
  if (!Number.isFinite(elements.video.duration)) return;
  const bounded = clamp(target, 0, elements.video.duration);
  if (Math.abs(elements.video.currentTime - bounded) <= 0.005) return;
  videoClockCorrectionPending = true;
  elements.video.currentTime = bounded;
}

function playVideoWithoutNarrationSync() {
  if (!elements.video.paused) return;
  programmaticVideoPlayPending = true;
  elements.video.play().catch(() => {
    programmaticVideoPlayPending = false;
  });
}

function parkVideoForNarration(force = false) {
  if (!narrationOwnsCompletion() || !Number.isFinite(elements.video.duration) ||
      elements.video.duration <= 0) return false;
  if (!force && videoParkedForNarration) return true;
  if (!force) {
    // timeupdate can be sparse on phones and the visual rate can be high.
    // Enter the park early enough to prevent the next tick crossing EOF.
    const guard = Math.max(0.12, elements.video.playbackRate * 0.35);
    if (elements.video.currentTime < elements.video.duration - guard) return false;
  }
  videoParkedForNarration = true;
  narratedPlaybackCompleted = false;
  programmedVideoRate = VIDEO_END_PARK_RATE;
  elements.video.defaultPlaybackRate = VIDEO_END_PARK_RATE;
  if (Math.abs(elements.video.playbackRate - VIDEO_END_PARK_RATE) > 0.001) {
    elements.video.playbackRate = VIDEO_END_PARK_RATE;
  }
  setVideoTimeWithoutNarrationSeek(
    Math.max(0, elements.video.duration - VIDEO_END_PARK_SECONDS)
  );
  // If a sparse phone timeupdate allowed natural EOF first, restart only the
  // parked video clock. The still-audible narration clock is never sought.
  playVideoWithoutNarrationSync();
  return true;
}

function finishNarratedPlayback() {
  if (completeNativePlayback()) {
    completeNativeNarrationWaiting = true;
    updateWatchUI();
    return;
  }
  if (narrationAudioAvailable && elements.voice.value !== "silent") {
    videoParkedForNarration = false;
    narratedPlaybackCompleted = true;
    const videoWasPlaying = !elements.video.paused;
    if (videoWasPlaying) {
      programmaticVideoPausePending = true;
      elements.video.pause();
    }
    if (Number.isFinite(elements.video.duration) && elements.video.duration > 0) {
      setVideoTimeWithoutNarrationSeek(elements.video.duration);
    }
  }
  markCompleteAtEnd();
}

function restartNarratedPlayback() {
  narratedPlaybackCompleted = false;
  videoParkedForNarration = false;
  if (Number.isFinite(elements.audio.duration) && elements.audio.duration > 0) {
    elements.audio.currentTime = 0;
  }
  setVideoTimeWithoutNarrationSeek(0);
  programmedVideoRate = 1;
  elements.video.defaultPlaybackRate = 1;
  elements.video.playbackRate = 1;
  updateSceneSyncRate(0);
  playNarration()
    .then(() => syncVideoToNarration(true))
    .catch(() => showToast("Select play again to start narration."));
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
  if ((visualTimings?.preserve_visual_intervals === true &&
      visualTimings?.native_live_video === true) ||
      located.scene.native_live_video || target.native_live_video) {
    // Preserved intervals and native footage map speech by elapsed time.
    // Ordinary native playback skips unused tails; preserved playback holds speech.
    const elapsed = Math.max(0, seconds - sourceStart);
    return targetStart + Math.min(elapsed, Math.max(0, targetEnd - targetStart - 0.001));
  }
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
  // Complete native masters can also contain held-image cards. Their clocks
  // still use the same duration ratio as mapTiming; only preserved intervals
  // and actual native footage must remain at their recorded speed.
  const elapsedMapping = (visualTimings.preserve_visual_intervals === true &&
    visualTimings.native_live_video === true) || reference.native_live_video;
  syncRate = elapsedMapping ? 1
    : elements.video.duration / referenceTotal * referenceDuration / selectedDuration;
  programmedVideoRate = clamp(syncRate, 0.0625, 16);
  elements.video.defaultPlaybackRate = programmedVideoRate;
  if (Math.abs(elements.video.playbackRate - programmedVideoRate) > 0.001) {
    elements.video.playbackRate = programmedVideoRate;
  }
  // Narration remains natural-speed. Scene-duration differences are handled
  // exclusively by the silent visual master above.
  elements.audio.defaultPlaybackRate = 1;
  if (Math.abs(elements.audio.playbackRate - 1) > 0.001) {
    elements.audio.playbackRate = 1;
  }
}

// This is the only routine that seeks narration, and callers use it only for
// an explicit video seek or a load/resume boundary. Normal playback flows in
// the other direction: stable narration time drives the silent video.
function seekNarrationToVideo() {
  if (narrationLoadPending) return;
  if (completeNativePlayback()) {
    syncCompleteNativeNarration(true);
    return;
  }
  if (videoParkedForNarration || !narrationAudioAvailable ||
      elements.voice.value === "silent" ||
      !elements.audio.duration || !elements.video.duration) return;
  const target = audioTimeFromVideo(elements.video.currentTime);
  if (Math.abs(elements.audio.currentTime - target) > 0.015) {
    elements.audio.currentTime = Math.min(target, Math.max(0, elements.audio.duration - 0.02));
  }
  updateSceneSyncRate(target);
}

function videoIsLoading() {
  if (videoCatchingUpToNarration) return false;
  return elements.video.seeking ||
    elements.video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA;
}

function narrationHoldApplies() {
  return narrationAudioAvailable && elements.voice.value !== "silent" &&
    !elements.video.paused && !elements.video.ended &&
    !videoParkedForNarration && !narratedPlaybackCompleted;
}

// Native pictures keep their recorded speed. Padding after actual completed
// speech can be skipped; the selected voice owns the pause deadline.
// Native scene padding ends for the selected voice after speech, rather
// than making every shorter voice wait for the longest captured hold.
function nativeIdleTailCut(visuals, audio, index, maxGap = 1) {
  const visual = visuals?.scenes?.[index];
  const spoken = audio?.scenes?.[index];
  const next = audio?.scenes?.[index + 1];
  const nextVisual = visuals?.scenes?.[index + 1];
  const sentences = spoken?.sentences;
  const nextSentences = next?.sentences;
  const final = index === audio?.scenes?.length - 1 && index === visuals?.scenes?.length - 1;
  if (!visual || !Array.isArray(sentences) || !sentences.length ||
      (!final && (!nextVisual || !Array.isArray(nextSentences) || !nextSentences.length))) return null;
  // Only an explicit source-bound safe idle fence permits a picture cut.
  const protectedEnd = visual.idle_tail_start;
  if (typeof protectedEnd !== "number" || !Number.isFinite(protectedEnd) || protectedEnd < 0) return null;
  const start = Number(spoken.speech_start);
  const end = Number(spoken.speech_end);
  const last = Number(sentences[sentences.length - 1].audible_end);
  const nextStart = final ? end : Number(next.speech_start);
  const nextSound = final ? end : Number(nextSentences[0].audible_start);
  const visualStart = Number(visual.speech_start);
  const visualEnd = Number(visual.scene_end);
  const nextVideo = final ? visualEnd : Number(nextVisual.speech_start);
  if (![start, end, last, nextStart, nextSound, visualStart, visualEnd, nextVideo, maxGap]
      .every(Number.isFinite) || maxGap <= 0 || last < start || end < last ||
      nextSound < nextStart || nextStart < end || visualEnd <= visualStart) return null;
  const gap = nativeIdleGapBudget(last, nextSound, final, maxGap);
  if (gap === null) return null;
  const cutElapsed = last - start + gap - (nextSound - nextStart);
  // Preserve the entire speech clip including its conservative quiet edge,
  // every protected action, and the next sentence's original leading audio.
  if (protectedEnd > end - start || cutElapsed < end - start || cutElapsed < protectedEnd ||
      cutElapsed >= visualEnd - visualStart) return null;
  return { at: visualStart + cutElapsed, next: nextVideo, final };
}

function nativeIdleTailReady(state) {
  return state.sameTrack && state.sameScene && !state.loadPending &&
    !state.audioSeeking && !state.videoSeeking && !state.videoPaused &&
    !state.pictureLoading && !state.explicitSeek && state.readyState >= 3 &&
    state.bufferedSpeechEnd && Number.isFinite(state.audioTime) &&
    Number.isFinite(state.speechEnd) && state.audioTime >= state.speechEnd;
}

function nativeIdleGapBudget(lastSound, nextSound, final, maxGap = 1) {
  if (![lastSound, nextSound, maxGap].every(Number.isFinite) ||
      typeof final !== "boolean" || maxGap <= 0 || nextSound < lastSound) return null;
  return final ? maxGap : Math.min(maxGap, nextSound - lastSound);
}

let nativeIdlePauseClock = null;

function nativeIdleTailWait(previous, state, now, maxGap = 1) {
  if (!nativeIdleTailReady(state) || !Number.isFinite(now) ||
      !Number.isFinite(state.lastSound) || !Number.isFinite(state.nextLead) ||
      state.lastSound > state.speechEnd || state.nextLead < 0) {
    return { clock: null, ready: false };
  }
  // Anchor the first OBSERVED completion of this actual loaded speech clip.
  // A late/buffered voice must not inherit a deadline from the video clock.
  const clock = previous?.key === state.key ? previous
    : { key: state.key, completedAt: now };
  const quietAlreadyHeard = state.speechEnd - state.lastSound;
  const gap = nativeIdleGapBudget(state.lastSound, state.nextSound, state.final, maxGap);
  if (gap === null) return { clock: null, ready: false };
  const remaining = Math.max(0, gap - quietAlreadyHeard - state.nextLead);
  return { clock, ready: now >= clock.completedAt + remaining * 1000 };
}

function currentNativeIdleTailState(spoken, index, seek) {
  let bufferedSpeechEnd = false;
  const end = Number(spoken.speech_end);
  for (let i = 0; i < elements.audio.buffered.length; i += 1) {
    if (elements.audio.buffered.start(i) <= end && elements.audio.buffered.end(i) >= end) {
      bufferedSpeechEnd = true;
      break;
    }
  }
  const next = audioTimings.scenes[index + 1];
  return {
    key: `${activeLesson.id}/${narrationRequest}/${audioTimings.language}/${audioTimings.voice}/${index}`,
    lastSound: Number(spoken.sentences?.at(-1)?.audible_end),
    nextLead: next ? Number(next.sentences?.[0]?.audible_start) - Number(next.speech_start) : 0,
    nextSound: next ? Number(next.sentences?.[0]?.audible_start) : end,
    final: !next,
    sameTrack: audioTimings?.voice === elements.voice.value &&
      audioTimings?.language === elements.language.value &&
      !!narrationObjectUrl && elements.audio.currentSrc === narrationObjectUrl,
    sameScene: completeNativeSceneIndex === index,
    loadPending: narrationLoadPending,
    audioSeeking: elements.audio.seeking,
    videoSeeking: elements.video.seeking || videoClockCorrectionPending,
    videoPaused: elements.video.paused,
    pictureLoading: videoIsLoading(),
    explicitSeek: seek,
    readyState: elements.audio.readyState,
    bufferedSpeechEnd,
    audioTime: elements.audio.currentTime,
    speechEnd: end
  };
}

function cancelNativeIdleTransition() {
  if (nativeIdleTransitionTimer !== null) clearTimeout(nativeIdleTransitionTimer);
  nativeIdleTransitionTimer = null;
  nativeIdleTransition = null;
  nativeIdlePauseClock = null;
  elements.audio.playbackRate = 1;
  updateCaptionTrackMode();
}

function scheduleNativeIdleTransition() {
  if (!nativeIdleTransition) return;
  if (nativeIdleTransitionTimer !== null) clearTimeout(nativeIdleTransitionTimer);
  const remaining = nativeIdleTransition.resumeAt - performance.now();
  nativeIdleTransitionTimer = setTimeout(() => {
    nativeIdleTransitionTimer = null;
    serviceNativeIdleTransition();
  }, nativeIdleTransition.final && remaining > 0 ? remaining : 16);
}

function nativeIdleQuietRate(audioTime, quietEnd, now, endAt) {
  // Only post-clip silence is elastic. Return to ordinary speed before the
  // next complete speech clip, preserving its entire conservative lead.
  const reserve = 0.1;
  const source = quietEnd - audioTime - reserve;
  const wall = (endAt - now) / 1000 - reserve;
  return source <= 0 || wall <= 0 ? 1 : clamp(source / wall, 0.0625, 1);
}

function serviceNativeIdleTransition() {
  const transition = nativeIdleTransition;
  if (!transition) return;
  if (transition.mode === "late") { serviceNativeLateIdleTransition(transition); return; }
  if (transition.request !== narrationRequest || transition.lesson !== activeLesson?.id ||
      transition.voice !== elements.voice.value || transition.language !== elements.language.value ||
      narrationLoadPending || !narrationObjectUrl || elements.audio.currentSrc !== narrationObjectUrl) {
    cancelNativeIdleTransition();
    return;
  }
  if (!transition.final && elements.audio.currentTime < transition.quietEnd) {
    elements.audio.playbackRate = nativeIdleQuietRate(elements.audio.currentTime,
      transition.quietEnd, performance.now(), transition.resumeAt);
    scheduleNativeIdleTransition();
    return;
  }
  elements.audio.playbackRate = 1;
  if (transition.final) elements.audio.pause();
  if ((transition.final && performance.now() < transition.resumeAt) || videoClockCorrectionPending ||
      videoIsLoading() || (!transition.final &&
        (elements.audio.seeking || elements.audio.readyState < HTMLMediaElement.HAVE_FUTURE_DATA))) {
    scheduleNativeIdleTransition();
    return;
  }
  nativeIdleTransition = null;
  if (nativeIdleTransitionTimer !== null) clearTimeout(nativeIdleTransitionTimer);
  nativeIdleTransitionTimer = null;
  completeNativeSceneIndex = transition.scene;
  completeNativeNarrationWaiting = transition.final;
  updateCaptionTrackMode();
  if (transition.final) {
    videoHeldForAudio = false;
    resumeAfterAudioHold = false;
    elements.loading.classList.add("hidden");
    playVideoWithoutNarrationSync();
  } else {
    // The same audio stream is already playing: no device restart can add
    // an unmeasured pause or swallow the next word.
    releaseVideoForAudio();
  }
}

function beginNativeIdleTransition(cut, state, clock, index) {
  if (nativeIdleTransition || !nativeIdleTailReady(state) || (!cut.final && elements.audio.paused)) return;
  const gap = nativeIdleGapBudget(state.lastSound, state.nextSound, state.final);
  if (gap === null) return;
  const quietHeard = state.speechEnd - state.lastSound;
  const finalFrame = cut.final ? 2 / Number(visualTimings.fps || 30) : 0;
  nativeIdleTransition = {
    request: narrationRequest, lesson: activeLesson.id,
    voice: elements.voice.value, language: elements.language.value,
    scene: cut.final ? index : index + 1, final: cut.final,
    quietEnd: cut.final ? state.speechEnd : Number(audioTimings.scenes[index + 1].speech_start),
    resumeAt: clock.completedAt + Math.max(0, gap - quietHeard - state.nextLead - finalFrame) * 1000
  };
  // Decode the next picture during the quiet window, after the entire actual
  // speech clip. Continuous audio avoids variable pause/play device latency.
  completeNativeNarrationWaiting = true;
  if (cut.final) elements.audio.pause();
  else elements.audio.playbackRate = nativeIdleQuietRate(elements.audio.currentTime,
    nativeIdleTransition.quietEnd, performance.now(), nativeIdleTransition.resumeAt);
  holdVideoForAudio();
  updateCaptionTrackMode();
  // Stay inside the first next-scene frame when its fractional media clock
  // rounds down, otherwise scene synchronization can replay the previous clip.
  const destination = cut.final ? Math.max(0, elements.video.duration - finalFrame)
    : (cut.next + 1 / 90000) / timingDuration(visualTimings, elements.video.duration) * elements.video.duration;
  setVideoTimeWithoutNarrationSeek(destination);
  scheduleNativeIdleTransition();
}

function nativeLateIdleTailCut(visuals, audio, index) {
  const visual = visuals?.scenes?.[index];
  const spoken = audio?.scenes?.[index];
  const next = audio?.scenes?.[index + 1];
  const final = index === visuals?.scenes?.length - 1 && index === audio?.scenes?.length - 1;
  if (!visual || !spoken || (!final && !next)) return null;
  const start = Number(visual.speech_start), end = Number(visual.scene_end);
  const speechStart = Number(spoken.speech_start), speechEnd = Number(spoken.speech_end);
  const fence = visual.idle_tail_start;
  const nextStart = final ? speechEnd : Number(next.speech_start);
  const nextVideo = final ? end : Number(visuals.scenes[index + 1]?.speech_start);
  if (typeof fence !== "number" || ![start, end, speechStart, speechEnd, fence, nextStart, nextVideo].every(Number.isFinite) ||
      speechEnd < speechStart || fence <= speechEnd - speechStart ||
      fence >= end - start - 1 / Number(visuals.fps || 30) || nextStart < speechEnd) return null;
  return { final, fence: start + fence, next: nextVideo, speechEnd, nextStart };
}

function beginNativeLateIdleTransition(cut, state, index, visualTime) {
  if (nativeIdleTransition || !nativeIdleTailReady(state) || visualTime < cut.fence ||
      state.speechEnd !== cut.speechEnd ||
      (!cut.final && elements.audio.currentTime >= cut.nextStart)) return false;
  const frame = 2 / Number(visualTimings.fps || 30);
  if (cut.final && elements.video.currentTime >= elements.video.duration - frame - 0.000001) return false;
  nativeIdleTransition = {
    mode: "late", phase: "decode", request: narrationRequest, lesson: activeLesson.id,
    voice: elements.voice.value, language: elements.language.value,
    scene: cut.final ? index : index + 1, final: cut.final,
    speechEnd: cut.speechEnd, quietEnd: cut.nextStart,
    fenceReachedAt: performance.now()
  };
  // All demonstrated actions through the bound fence have already played at1x.
  // Preserve the completed audio position: only the picture may skip idle time.
  completeNativeNarrationWaiting = true;
  elements.audio.playbackRate = 1;
  elements.audio.pause();
  holdVideoForAudio();
  updateCaptionTrackMode();
  const destination = cut.final ? Math.max(0, elements.video.duration - frame)
    : (cut.next + 1 / 90000) / timingDuration(visualTimings, elements.video.duration) * elements.video.duration;
  setVideoTimeWithoutNarrationSeek(destination);
  scheduleNativeIdleTransition();
  return true;
}

function serviceNativeLateIdleTransition(transition) {
  if (nativeIdleTransition !== transition) return;
  if (transition.request !== narrationRequest || transition.lesson !== activeLesson?.id ||
      transition.voice !== elements.voice.value || transition.language !== elements.language.value ||
      narrationLoadPending || !narrationObjectUrl || elements.audio.currentSrc !== narrationObjectUrl ||
      elements.audio.currentTime < transition.speechEnd || elements.audio.seeking) {
    cancelNativeIdleTransition();
    return;
  }
  if (transition.phase === "decode") {
    if (videoClockCorrectionPending || videoIsLoading() ||
        (!transition.final && elements.audio.readyState < HTMLMediaElement.HAVE_FUTURE_DATA)) {
      scheduleNativeIdleTransition();
      return;
    }
    if (transition.final) {
      // Play the genuine final two frames to the browser's actual ended event.
      nativeIdleTransition = null;
      if (nativeIdleTransitionTimer !== null) clearTimeout(nativeIdleTransitionTimer);
      nativeIdleTransitionTimer = null;
      completeNativeNarrationWaiting = true;
      videoHeldForAudio = false;
      resumeAfterAudioHold = false;
      updateCaptionTrackMode();
      elements.loading.classList.add("hidden");
      playVideoWithoutNarrationSync();
      return;
    }
    if (elements.audio.currentTime >= transition.quietEnd) {
      cancelNativeIdleTransition();
      return;
    }
    transition.phase = "starting";
    startNarrationAudio().then(() => {
      if (nativeIdleTransition !== transition) return;
      transition.phase = "quiet";
      scheduleNativeIdleTransition();
    }).catch(() => {
      if (nativeIdleTransition !== transition) return;
      cancelNativeIdleTransition();
      videoHeldForAudio = false;
      resumeAfterAudioHold = false;
      elements.loading.classList.add("hidden");
      showToast("Select play again to start narration.");
    });
    scheduleNativeIdleTransition();
    return;
  }
  if (transition.phase === "starting") {
    scheduleNativeIdleTransition();
    return;
  }
  if (elements.audio.paused) {
    // Never turn a viewer pause or cancelled start into an automatic replay.
    cancelNativeIdleTransition();
    return;
  }
  if (elements.audio.currentTime < transition.quietEnd) {
    scheduleNativeIdleTransition();
    return;
  }
  // The intervening PCM silence has played without any seek or speed change.
  nativeIdleTransition = null;
  if (nativeIdleTransitionTimer !== null) clearTimeout(nativeIdleTransitionTimer);
  nativeIdleTransitionTimer = null;
  completeNativeSceneIndex = transition.scene;
  completeNativeNarrationWaiting = false;
  updateCaptionTrackMode();
  releaseVideoForAudio();
}

function syncCompleteNativeNarration(seek = false) {
  if (nativeIdleTransition) { serviceNativeIdleTransition(); return; }
  if (narrationLoadPending || !completeNativePlayback() || !narrationAudioAvailable ||
      elements.voice.value === "silent" || !elements.video.duration ||
      !elements.audio.duration || !audioTimings) return;
  if (videoClockCorrectionPending || elements.video.seeking) {
    if (!elements.audio.paused) elements.audio.pause();
    return;
  }
  const visualTime = elements.video.currentTime / elements.video.duration *
    timingDuration(visualTimings, elements.video.duration);
  const located = timingScene(visualTimings, visualTime);
  const spoken = located && audioTimings.scenes?.[located.index];
  if (!spoken) return;
  const idleCut = nativeIdleTailCut(visualTimings, audioTimings, located.index);
  const idleState = currentNativeIdleTailState(spoken, located.index, seek);
  const idleWait = nativeIdleTailWait(nativeIdlePauseClock, idleState, performance.now());
  nativeIdlePauseClock = idleWait.clock;
  const lateCut = nativeLateIdleTailCut(visualTimings, audioTimings, located.index);
  if (lateCut && beginNativeLateIdleTransition(lateCut, idleState, located.index, visualTime)) return;
  const protectedEnd = Number(located.scene.idle_tail_start);
  // Decode two genuine final frames: seeking to the final decoded PTS can
  // already mark the video ended, making play() rewind instead of finish.
  // Do not schedule that same transition again during its remaining frames.
  const finalFramePending = idleCut?.final &&
    visualTime >= timingDuration(visualTimings, elements.video.duration) - 2 / Number(visualTimings.fps || 30) - 0.000001;
  if (idleCut && idleWait.clock && !finalFramePending && (idleCut.final || !elements.audio.paused) && visualTime >= Number(located.scene.speech_start) + protectedEnd) {
    beginNativeIdleTransition(idleCut, idleState, idleWait.clock, located.index);
    return;
  }
  const start = Number(spoken.speech_start) || 0;
  const end = Number(spoken.speech_end);
  if (!Number.isFinite(end) || end < start) return;
  const changed = completeNativeSceneIndex !== located.index;
  if (changed || seek) {
    completeNativeSceneIndex = located.index;
    // Hybrid masters contain both native clips and ordinary held cards.
    // A viewer seek must invert the same per-scene mapping used to seek the
    // picture; only native scenes map by unscaled elapsed time.
    const target = seek ? Math.min(audioTimeFromVideo(elements.video.currentTime), end) : start;
    if (Math.abs(elements.audio.currentTime - target) > 0.015) {
      elements.audio.currentTime = Math.min(target, elements.audio.duration);
    }
    completeNativeNarrationWaiting = target >= end;
  } else if (elements.audio.currentTime >= end) {
    completeNativeNarrationWaiting = true;
  }
  updateSceneSyncRate(start);
  // Retain an explicit mid-scene seek target even before decoding is ready.
  if (videoHeldForAudio && audioNeedsData()) return;
  if (completeNativeNarrationWaiting || (elements.video.paused && !videoHeldForAudio) ||
      elements.video.ended || videoIsLoading()) {
    if (!elements.audio.paused) elements.audio.pause();
  } else if (elements.audio.paused && !elements.audio.ended) {
    startNarrationAudio().catch(() => showToast("Select play again to start narration."));
  }
  if (completeNativeNarrationWaiting) releaseVideoForAudio();
}

function holdVideoForAudio() {
  if (!elements.video.paused) {
    videoHeldForAudio = true;
    resumeAfterAudioHold = true;
    audioHoldPauseEventsPending += 1;
    elements.video.pause();
  }
  if (videoHeldForAudio) elements.loading.classList.remove("hidden");
}

function audioNeedsData() {
  return narrationAudioAvailable && elements.voice.value !== "silent" &&
    !elements.audio.ended && !completeNativeNarrationWaiting &&
    !elements.audio.seeking &&
    elements.audio.readyState < HTMLMediaElement.HAVE_FUTURE_DATA;
}

function releaseVideoForAudio() {
  if (nativeIdleTransition) return;
  if (!videoHeldForAudio || !resumeAfterAudioHold || narrationLoadPending || audioNeedsData()) return;
  if (narrationAudioAvailable && elements.voice.value !== "silent" &&
      elements.audio.paused && !completeNativeNarrationWaiting) return;
  videoHeldForAudio = false;
  resumeAfterAudioHold = false;
  elements.loading.classList.add("hidden");
  playVideoWithoutNarrationSync();
}

function startNarrationAudio() {
  if (narrationLoadPending) return Promise.resolve();
  const request = narrationRequest;
  if (audioNeedsData() && !videoIsLoading()) holdVideoForAudio();
  return elements.audio.play().then(() => {
    if (request === narrationRequest) releaseVideoForAudio();
  });
}

function updateNarrationHold() {
  // A requested picture seek already holds speech. Do not turn its ordinary
  // playing/buffering state into a second, reciprocal picture pause.
  if (!completeNativePlayback() && narrationHoldApplies() && videoIsLoading()) {
    narrationHeldForVideo = true;
    if (!elements.audio.paused) elements.audio.pause();
    return;
  }
  if (narrationLoadPending || audioNeedsData()) {
    holdVideoForAudio();
    // A deferred mobile decoder needs the play gesture to obtain metadata.
    // It still cannot advance the picture before real decoded data arrives.
    if (!elements.audio.paused && elements.audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
      elements.audio.pause();
    }
    return;
  }
  if (completeNativePlayback()) {
    syncCompleteNativeNarration(false);
    return;
  }
  if (!narrationHeldForVideo && !videoHeldForAudio) return;
  narrationHeldForVideo = false;
  // Resume only what was held: a paused or finished lesson stays as it is.
  if ((narrationHoldApplies() || videoHeldForAudio) && elements.audio.paused && !elements.audio.ended) {
    startNarrationAudio()
      .catch(() => showToast("Select play again to start narration."));
  }
}

// Every narration start goes through here: while the picture is loading the
// narration is held instead (updateNarrationHold resumes it with the picture),
// so no play() races a pause() and no false "select play again" appears.
function playNarration() {
  if (narrationLoadPending) {
    holdVideoForAudio();
    return Promise.resolve();
  }
  if (completeNativePlayback()) {
    if (narrationAudioAvailable && !elements.audio.duration) return startNarrationAudio();
    syncCompleteNativeNarration(false);
    return Promise.resolve();
  }
  if (narrationHoldApplies() && videoIsLoading()) {
    narrationHeldForVideo = true;
    return Promise.resolve();
  }
  return startNarrationAudio();
}

function setVideoTimeFromNarration(target) {
  if (videoParkedForNarration || !elements.video.duration ||
      videoClockCorrectionPending) return;
  const bounded = Math.min(
    Math.max(0, target), Math.max(0, elements.video.duration - 0.02));
  if (Math.abs(elements.video.currentTime - bounded) <= 0.015) return;
  videoClockCorrectionPending = true;
  videoCatchingUpToNarration = true;
  elements.video.currentTime = bounded;
}

function syncVideoToNarration(force = false) {
  if (narrationLoadPending) return;
  if (completeNativePlayback()) {
    syncCompleteNativeNarration(false);
    return;
  }
  if (videoParkedForNarration || !narrationAudioAvailable ||
      elements.voice.value === "silent" ||
      !elements.audio.duration || !elements.video.duration) return;
  const audioSeconds = elements.audio.currentTime;
  const target = videoTimeFromAudio(audioSeconds);
  updateSceneSyncRate(audioSeconds);
  const drift = Math.abs(elements.video.currentTime - target);
  if (drift > (force ? 0.015 : VIDEO_CLOCK_DRIFT_SECONDS)) {
    setVideoTimeFromNarration(target);
  }
}

function startNativeFrameSync() {
  if (nativeFrameSyncHandle !== null || !visualTimings?.native_live_video ||
      elements.video.paused || elements.video.ended) return;
  const tick = () => {
    nativeFrameSyncHandle = null;
    if (!visualTimings?.native_live_video || elements.video.paused ||
        elements.video.ended) return;
    syncVideoToNarration(false);
    updateNarrationHold();
    nativeFrameSyncHandle = requestAnimationFrame(tick);
  };
  nativeFrameSyncHandle = requestAnimationFrame(tick);
}

function boundNativeIdleTimings(reference, lesson, timingSha) {
  const result = { ...reference, scenes: reference.scenes.map(scene => {
    const copy = { ...scene };
    delete copy.idle_tail_start;
    return copy;
  }) };
  const policy = lesson.native_idle_tails;
  if (!policy) return result;
  // The normal candidate validates this master hash against actual media bytes;
  // the browser additionally binds the small timing document it just loaded.
  if (policy.schema !== 1 || policy.lesson !== lesson.id ||
      !/^[0-9a-f]{64}$/.test(policy.master_sha256 || "") ||
      policy.visual_timing_sha256 !== timingSha || !Array.isArray(policy.scenes)) return result;
  const fences = new Map();
  for (const fence of policy.scenes) {
    if (![fence.source_start, fence.source_end, fence.idle_tail_start].every(value =>
        typeof value === "number" && Number.isFinite(value)) ||
        fence.idle_tail_start < 0 || fence.idle_tail_start >= fence.source_end - fence.source_start) return result;
    const matches = result.scenes.map((scene, index) => ({ scene, index })).filter(({ scene }) =>
      scene.visual === fence.visual &&
      Math.abs(fence.source_start - Number(scene.speech_start)) <= 1e-7 &&
      Math.abs(fence.source_end - Number(scene.scene_end)) <= 1e-7);
    if (matches.length !== 1 || fences.has(matches[0].index)) return result;
    fences.set(matches[0].index, fence.idle_tail_start);
  }
  for (const [index, scene] of result.scenes.entries()) {
    if (fences.has(index)) scene.idle_tail_start = fences.get(index);
  }
  return result;
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
    const timingText = await response.text();
    const timings = JSON.parse(timingText);
    let referenceText = timingText;
    let referenceTimings = timings;
    if (requestedSource !== visualTimingSource()) {
      const referenceResponse = await fetch(visualTimingSource());
      if (referenceResponse.ok) {
        referenceText = await referenceResponse.text();
        referenceTimings = JSON.parse(referenceText);
      }
      else if (activeLesson.visual_timings) throw new Error("native visual timings unavailable");
    }
    if (activeLesson.visual_timings) {
      const native = referenceTimings.kind === "native_visual_timing" ||
        referenceTimings.native_live_video;
      const accepted = native
        ? referenceTimings.kind === "native_visual_timing" &&
          referenceTimings.native_live_video === true && referenceTimings.coverage?.accepted
        : referenceTimings.schema === 1 && !referenceTimings.kind &&
          !referenceTimings.scenes?.some(scene => scene.native_live_video);
      if (!accepted || referenceTimings.scenes?.length !== timings.scenes?.length) {
        throw new Error("visual timings are incomplete");
      }
    }
    if (!isCurrent()) return;
    const referenceHash = Array.from(new Uint8Array(await crypto.subtle.digest(
      "SHA-256", new TextEncoder().encode(referenceText))), byte => byte.toString(16).padStart(2, "0")).join("");
    if (!isCurrent()) return;
    audioTimings = timings;
    visualTimings = boundNativeIdleTimings(referenceTimings, activeLesson, referenceHash);
    completeNativeSceneIndex = null;
    completeNativeNarrationWaiting = false;
    configureMediaSync();
    startNativeFrameSync();
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
    // A hosted voice may have a newer script than the page catalog. Its
    // paired transcript owns same-language captions, not the stale page.
    // Never replace a requested translation with another language, or use
    // fallback timing metadata from a different voice as spoken evidence.
    const followsAudio = audioTimings?.language === effectiveCaptionLanguage()
      && audioTimings?.voice === elements.voice.value;
    const text = followsAudio && typeof timing?.text === "string" && timing.text.trim()
      ? timing.text : scene.narration;
    return {
      index: index + 1,
      start,
      end,
      text,
      related: relatedLessonsForScene(activeLesson, index),
      label: chapterLabel(text, index)
    };
  });
}

function relatedLessonsForScene(lesson, index) {
  return [...new Set(lesson?.scenes?.[index]?.related_lessons || [])]
    .filter(id => id !== lesson.id && LESSONS.some(item => item.id === id));
}

function makeRelatedLessons(ids) {
  const links = document.createElement("div");
  links.className = "related-lessons";
  ids.forEach(id => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "related-lesson-link";
    button.dataset.relatedLesson = id;
    button.textContent = localizedLesson(id).title;
    button.addEventListener("click", () => selectLesson(id, { focus: true }));
    links.appendChild(button);
  });
  return links;
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
  const source = String(text).trim();
  if (!source) return [];
  const abbreviations = new Set([
    "dr.", "fig.", "i.e.", "jr.", "mr.", "mrs.", "ms.", "no.",
    "prof.", "sr.", "st.", "vs.", "e.g."
  ]);
  const closings = /["'”’)\]}」』】）]+$/gu;
  const openings = /^["'“‘({\[「『【（¿¡]+/gu;
  const sentences = [];
  let start = 0;
  for (const separator of source.matchAll(/[。！？।]+["'”’)\]}」』】）]*\s*|\s+/gu)) {
    const explicit = /^[。！？।]/u.test(separator[0]);
    const boundary = Number(separator.index) + (explicit ? separator[0].trimEnd().length : 0);
    const next = Number(separator.index) + separator[0].length;
    const prefix = source.slice(start, boundary).trimEnd();
    if (!prefix) continue;
    const bare = prefix.replace(closings, "");
    const terminal = bare.slice(-1);
    if (!".!?。！？।".includes(terminal)) continue;
    if (terminal === ".") {
      const lower = bare.toLowerCase();
      const token = lower.match(/(?:^|\s)([^\s]+)$/u)?.[1] || lower;
      const modelLabel = /\bmodel\s+[a-z]\.$/u.test(lower);
      if (!modelLabel && (abbreviations.has(token) || /^(?:[a-z]\.)+$/u.test(token))) {
        continue;
      }
    }
    const following = source.slice(next)
      .replace(openings, "");
    if (!following || (!explicit && !(/^[\p{Lu}\p{Lt}\p{Lo}\p{Nd}]/u.test(following) || /^spaCR\b/u.test(following)))) {
      continue;
    }
    sentences.push(source.slice(start, boundary).trim());
    start = next;
  }
  sentences.push(source.slice(start).trim());
  return sentences.filter(Boolean);
}

function captionCueIntervals(chapter, timing, sentences) {
  const precise = Array.isArray(timing?.sentences) ? timing.sentences : [];
  let previousEnd = chapter.start;
  const preciseIsUsable = precise.length === sentences.length && precise.every(item => {
    const start = Number(item?.speech_start);
    const end = Number(item?.speech_end);
    const valid = Number.isFinite(start) && Number.isFinite(end) && end > start &&
      start >= previousEnd - 0.001 && start >= chapter.start - 0.001 &&
      end <= chapter.end + 0.001;
    previousEnd = end;
    return valid;
  });
  if (preciseIsUsable) {
    return sentences.map((text, index) => ({
      start: Number(precise[index].speech_start),
      end: Number(precise[index].speech_end),
      text
    }));
  }

  const weights = sentences.map(sentence => Math.max(1, [...sentence].length));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const duration = Math.max(0, chapter.end - chapter.start);
  let cursor = chapter.start;
  return sentences.map((text, index) => {
    const end = index === sentences.length - 1
      ? chapter.end : cursor + duration * weights[index] / totalWeight;
    const cue = { start: cursor, end, text };
    cursor = end;
    return cue;
  });
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
    const timing = audioTimings?.scenes?.[chapter.index - 1];
    captionCueIntervals(chapter, timing, sentences).forEach(cue => {
      lines.push(
        String(++cueIndex),
        `${vttTime(videoTimeFromAudio(cue.start))} --> ${vttTime(videoTimeFromAudio(cue.end))}`,
        cue.text.replaceAll("-->", "→"),
        ""
      );
    });
  });
  captionUrl = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/vtt" }));
  captionTrackLoading = true;
  // Start a fresh native load when replacing a previous/revoked blob.
  const track = elements.captionTrack.cloneNode(false);
  track.src = captionUrl;
  const language = effectiveCaptionLanguage();
  track.srclang = language;
  track.label = captionLanguageById(language).shortLabel;
  track.addEventListener("load", () => {
    if (elements.captionTrack !== track) return;
    captionTrackLoading = false;
    updateCaptionTrackMode();
  }, { once: true });
  track.addEventListener("error", () => {
    if (elements.captionTrack !== track) return;
    captionTrackLoading = false;
  }, { once: true });
  elements.captionTrack.replaceWith(track);
  elements.captionTrack = track;
  updateCaptionTrackMode();
}

function chapterDisplayTime(chapter) {
  return completeNativePlayback() ? videoTimeFromAudio(chapter.start) : chapter.start;
}

function renderChapters() {
  elements.chapters.innerHTML = "";
  elements.chapters.lang = effectiveCaptionLanguage();
  elements.chapters.dir = "auto";
  chapterData.forEach(chapter => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chapter-button";
    button.innerHTML = `<span class="chapter-time">${formatTime(chapterDisplayTime(chapter))}</span><span class="chapter-copy"><strong>${escapeHTML(chapter.label)}</strong><small>${escapeHTML(truncate(chapter.text, 82))}</small></span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>`;
    button.addEventListener("click", () => seekTo(chapter.start));
    elements.chapters.appendChild(button);
    if (chapter.related.length) elements.chapters.appendChild(makeRelatedLessons(chapter.related));
  });
}

function renderTranscript() {
  elements.transcript.lang = effectiveCaptionLanguage();
  elements.transcript.dir = "auto";
  elements.transcript.innerHTML = chapterData.map(chapter => `<div class="transcript-entry"><button type="button" data-seek="${chapter.start}">${formatTime(chapterDisplayTime(chapter))}</button><p>${escapeHTML(chapter.text)}</p></div>`).join("");
  elements.transcript.querySelectorAll("[data-seek]").forEach(button => button.addEventListener("click", () => seekTo(Number(button.dataset.seek))));
  elements.transcript.querySelectorAll(".transcript-entry").forEach((entry, index) => {
    const related = chapterData[index].related;
    if (related.length) entry.appendChild(makeRelatedLessons(related));
  });
}

function seekTo(audioSeconds) {
  cancelNativeIdleTransition();
  if (completeNativePlayback()) {
    completeNativeSceneIndex = null;
    completeNativeNarrationWaiting = false;
  }
  if (narrationAudioAvailable && elements.audio.duration &&
      elements.voice.value !== "silent") {
    const bounded = Math.min(
      Math.max(0, audioSeconds), Math.max(0, elements.audio.duration - 0.02));
    elements.audio.currentTime = bounded;
    updateSceneSyncRate(bounded);
  }
  setVideoTimeFromNarration(videoTimeFromAudio(audioSeconds));
  // This is a viewer's requested chapter, not an automatic drift correction.
  // Its loading picture must hold narration at the requested speech offset.
  videoCatchingUpToNarration = false;
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
  if (narrationAudioAvailable && elements.voice.value !== "silent" &&
      elements.audio.duration) {
    const duration = audioTimings?.total_duration || elements.audio.duration;
    const current = Math.min(elements.audio.currentTime || 0, duration);
    return {
      current,
      duration,
      ratio: duration ? current / duration : 0,
    };
  }
  const ratio = elements.video.duration ? elements.video.currentTime / elements.video.duration : 0;
  const duration = elements.video.duration;
  const current = audioTimings ? audioTimeFromVideo(elements.video.currentTime) : ratio * (duration || 0);
  return { current, duration: duration || 0, ratio };
}

function updateWatchUI() {
  const clock = mediaClock();
  const shownCurrent = completeNativePlayback() ? elements.video.currentTime : clock.current;
  const shownDuration = completeNativePlayback() ? elements.video.duration : clock.duration;
  elements.watchTime.textContent = `${formatTime(shownCurrent)} / ${formatTime(shownDuration)}`;
  const shownRatio = completeNativePlayback() && elements.video.duration
    ? elements.video.currentTime / elements.video.duration : clock.ratio;
  elements.watchBar.style.width = `${Math.min(100, shownRatio * 100)}%`;
  const currentIndex = chapterData.findIndex((chapter, index) => clock.current >= chapter.start && (!chapterData[index + 1] || clock.current < chapterData[index + 1].start));
  elements.chapters.querySelectorAll(".chapter-button").forEach((button, index) => button.classList.toggle("current", index === currentIndex));
}

function saveWatchPosition() {
  if (!isPlayable(activeLesson)) return;
  watchProgress[activeLesson.id] = elements.video.currentTime || 0;
  localStorage.setItem(WATCH_KEY, JSON.stringify(watchProgress));
}

function toggleComplete() {
  if (!isPlayable(activeLesson)) return;
  if (completed.has(activeLesson.id)) { completed.delete(activeLesson.id); showToast("Lesson marked incomplete"); }
  else { completed.add(activeLesson.id); showToast("Lesson complete"); }
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
  updateCompleteButton(); updateCourseProgress(); renderCurriculum(elements.search.value);
}

function markCompleteAtEnd() {
  if (!isPlayable(activeLesson) || completed.has(activeLesson.id)) return;
  completed.add(activeLesson.id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
  updateCompleteButton(); updateCourseProgress(); renderCurriculum(elements.search.value);
  showToast("Lesson completed");
}

function updateCompleteButton() {
  const done = completed.has(activeLesson?.id);
  elements.complete.hidden = !isPlayable(activeLesson);
  elements.complete.classList.toggle("completed", done);
  elements.completeLabel.textContent = done ? "Completed" : "Mark complete";
  elements.complete.setAttribute("aria-pressed", String(done));
}

function updateCourseProgress() {
  const count = AVAILABLE_LESSONS.filter(lesson => completed.has(lesson.id)).length;
  elements.progressLabel.textContent = `${count} of ${AVAILABLE_LESSONS.length} complete`;
  elements.progressBar.style.width = `${count / Math.max(1, AVAILABLE_LESSONS.length) * 100}%`;
}

function continueCourse() {
  const next = AVAILABLE_LESSONS.find(lesson => !completed.has(lesson.id)) || AVAILABLE_LESSONS.at(-1);
  if (next) selectLesson(next.id, { focus: true });
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
    updateComingSoon();
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
  startNativeFrameSync();
  if (narrationLoadPending || (audioNeedsData() && !videoIsLoading())) {
    programmaticVideoPlayPending = false;
    holdVideoForAudio();
    if (!narrationLoadPending && narrationAudioAvailable) {
      playNarration().catch(() => showToast("Select play again to start narration."));
    }
    return;
  }
  const action = narratedVideoPlayAction(
    programmaticVideoPlayPending,
    narratedPlaybackCompleted,
    videoParkedForNarration
  );
  if (action === "programmatic") {
    programmaticVideoPlayPending = false;
    return;
  }
  if (action === "replay") {
    restartNarratedPlayback();
    return;
  }
  if (action === "resume") {
    playNarration()
      .then(updateNarrationHold)
      .catch(() => showToast("Select play again to start narration."));
    return;
  }
  videoParkedForNarration = false;
  seekNarrationToVideo();
  if (narrationAudioAvailable) {
    playNarration()
      .then(() => {
        syncVideoToNarration(true);
        updateNarrationHold();
      })
      .catch(() => showToast("Select play again to start narration."));
  }
});
elements.video.addEventListener("pause", () => {
  if (audioHoldPauseEventsPending > 0) {
    audioHoldPauseEventsPending -= 1;
    return;
  }
  if (programmaticVideoPausePending) {
    programmaticVideoPausePending = false;
    return;
  }
  cancelNativeIdleTransition();
  videoHeldForAudio = false;
  resumeAfterAudioHold = false;
  // A sparse phone tick can still reach natural EOF before parking; that
  // browser-generated pause must not clip narration. A parked video remains
  // genuinely playing, so every native user pause has video.ended === false
  // and pauses both clocks here.
  if (shouldPauseNarrationForVideoPause(
    narrationOwnsCompletion(), elements.video.ended
  )) {
    elements.audio.pause();
    saveWatchPosition();
  }
});
elements.video.addEventListener("pause", () => { nativeIdlePauseClock = null; });
elements.video.addEventListener("seeking", () => {
  nativeIdlePauseClock = null;
  if (!videoClockCorrectionPending) {
    cancelNativeIdleTransition();
    videoCatchingUpToNarration = false;
    videoParkedForNarration = false;
    narratedPlaybackCompleted = false;
    seekNarrationToVideo();
  }
});
elements.video.addEventListener("seeked", () => {
  if (videoClockCorrectionPending) {
    videoClockCorrectionPending = false;
    if (completeNativePlayback()) {
      syncCompleteNativeNarration(true);
      return;
    }
    // One correction per settled seek. A chapter jump made while it was in
    // flight is applied now; held narration cannot have moved meanwhile.
    syncVideoToNarration(false);
    return;
  }
  seekNarrationToVideo();
});
// Registered after the handlers above so a finished correction or a viewer
// seek has updated both clocks before narration is held or released.
elements.audio.addEventListener("play", updateNarrationHold);
for (const type of ["waiting", "stalled", "seeking", "seeked", "canplay"]) {
  elements.audio.addEventListener(type, updateNarrationHold);
}
elements.audio.addEventListener("playing", releaseVideoForAudio);
for (const type of ["waiting", "stalled", "seeking", "seeked", "loadeddata",
                    "canplay", "canplaythrough", "playing", "pause", "emptied"]) {
  elements.video.addEventListener(type, updateNarrationHold);
}
elements.video.addEventListener("playing", () => {
  if (!videoClockCorrectionPending) videoCatchingUpToNarration = false;
});
elements.video.addEventListener("emptied", () => {
  videoCatchingUpToNarration = false;
  // Replacing/removing src cancels an in-flight seek without a seeked event.
  // Its guard belongs to the discarded resource, not the next lesson.
  videoClockCorrectionPending = false;
});
elements.video.addEventListener("timeupdate", () => {
  if (!parkVideoForNarration(false)) syncVideoToNarration(false);
  updateWatchUI();
});
elements.video.addEventListener("volumechange", () => { elements.audio.volume = elements.video.volume; elements.audio.muted = elements.video.muted; });
elements.video.addEventListener("ratechange", () => {
  if (narrationAudioAvailable && elements.voice.value !== "silent") {
    if (Math.abs(elements.video.playbackRate - programmedVideoRate) > 0.001) {
      // Native controls may offer a visual playback-rate menu. While
      // narration is selected the visual rate is dictated by scene timing;
      // accepting that menu's rate would require stretching the voice.
      elements.video.playbackRate = programmedVideoRate;
    }
    elements.audio.defaultPlaybackRate = 1;
    if (Math.abs(elements.audio.playbackRate - 1) > 0.001) {
      elements.audio.playbackRate = 1;
    }
  }
});
elements.video.addEventListener("ended", () => {
  if (!parkVideoForNarration(true)) markCompleteAtEnd();
});
elements.audio.addEventListener("ended", finishNarratedPlayback);
elements.audio.addEventListener("timeupdate", () => {
  syncVideoToNarration(false);
  updateWatchUI();
});
elements.audio.addEventListener("loadedmetadata", () => {
  if (narrationLoadPending || !narrationAudioAvailable) return;
  configureMediaSync();
  if (!elements.video.paused || (videoHeldForAudio && resumeAfterAudioHold)) {
    playNarration().catch(() => {
      showToast("Select play again to start narration.");
    });
  }
});
elements.audio.addEventListener("error", () => {
  cancelNativeIdleTransition();
  // Loading errors are handled by loadNarration while availability is false.
  // This catches a deferred phone decoder failure after the player is ready.
  if (!narrationAudioAvailable || !elements.audio.getAttribute("src")) return;
  narrationAudioAvailable = false;
  videoParkedForNarration = false;
  narratedPlaybackCompleted = false;
  programmaticVideoPlayPending = false;
  discardNarrationAudio();
  configureMediaSync();
  releaseVideoForAudio();
  showToast("Narration is unavailable; the GitHub-hosted video remains available.");
});
elements.video.addEventListener("canplay", () => {
  if (!narrationLoadPending && !videoHeldForAudio) elements.loading.classList.add("hidden");
});
elements.video.textTracks?.addEventListener?.("change", syncCaptionPreferenceFromNativeControls);
document.addEventListener("visibilitychange", () => {
  // A throttled background timer must never carry a quiet-only rate into speech.
  if (document.hidden) cancelNativeIdleTransition();
});
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

elements.availableCount.textContent = AVAILABLE_LESSONS.length;
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
    updateComingSoon();
  }
  await captionTask;
  updateComingSoon();
}

initializeApp().catch(error => {
  setMediaSelectorsDisabled(false);
  showToast(error?.message || "The tutorial player could not be initialized.");
});
