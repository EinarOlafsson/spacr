/* On-demand API docstring translations generated outside Python sources. */
(() => {
  "use strict";

  const LANGUAGES = {
    en: "English", sv: "Svenska", de: "Deutsch", es: "Español",
    zh_CN: "简体中文", pt: "Português", hi: "हिन्दी", ko: "한국어",
    is: "Íslenska", fr: "Français",
  };
  const LABELS = {
    sv: "Översatt API-dokumentation", de: "Übersetzte API-Dokumentation",
    es: "Documentación de API traducida", zh_CN: "翻译的 API 文档",
    pt: "Documentação da API traduzida", hi: "अनुवादित API दस्तावेज़",
    ko: "번역된 API 문서", is: "Þýdd API-skjölun",
    fr: "Documentation API traduite",
  };
  const SELECT_LABELS = {
    en: "API language", sv: "API-språk", de: "API-Sprache",
    es: "Idioma de la API", zh_CN: "API 语言", pt: "Idioma da API",
    hi: "API भाषा", ko: "API 언어", is: "Tungumál API",
    fr: "Langue de l’API",
  };

  const script = [...document.scripts].find((item) =>
    /(?:^|\/)api_i18n\.js(?:\?|$)/.test(item.src));
  if (!script) return;
  const catalogRoot = new URL("./i18n/api/", script.src);
  let active = "en";

  function escapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = value;
    return node.innerHTML;
  }

  function inlineMarkup(value) {
    return escapeHtml(value).replace(/``([^`]+)``|`([^`]+)`/g,
      (_match, literal, role) => `<code>${literal || role}</code>`);
  }

  function renderRst(value) {
    const blocks = value.split(/\n\s*\n/).filter((item) => item.trim());
    return blocks.map((block) => {
      const lines = block.split("\n");
      if (lines.length > 1 && /^[=~^`'\-:#*+]{3,}$/.test(lines[1].trim())) {
        return `<h4>${inlineMarkup(lines[0])}</h4>`;
      }
      if (lines.every((line) => /^\s*[*-]\s+/.test(line))) {
        return `<ul>${lines.map((line) => `<li>${inlineMarkup(line.replace(/^\s*[*-]\s+/, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((line) => /^\s*:[^:]+:/.test(line))) {
        return `<dl>${lines.map((line) => {
          const match = line.match(/^\s*:([^:]+):\s*(.*)$/);
          return `<dt>${inlineMarkup(match[1])}</dt><dd>${inlineMarkup(match[2])}</dd>`;
        }).join("")}</dl>`;
      }
      if (lines.some((line) => /^\s*(?:>>>|\.\.\.)/.test(line))) {
        return `<pre><code>${escapeHtml(lines.join("\n"))}</code></pre>`;
      }
      return `<p>${inlineMarkup(lines.join(" "))}</p>`;
    }).join("");
  }

  function clearTranslations() {
    document.querySelectorAll(".spacr-api-translation").forEach((node) => node.remove());
  }

  async function selectLanguage(language) {
    active = language;
    localStorage.setItem("spacr-doc-language", language);
    const label = document.querySelector(".spacr-api-language__label");
    if (label) label.textContent = SELECT_LABELS[language] || SELECT_LABELS.en;
    clearTranslations();
    if (language === "en") return;
    const response = await fetch(new URL(`${language}.json`, catalogRoot));
    if (!response.ok) throw new Error(`API translation catalog returned ${response.status}`);
    const catalog = await response.json();
    if (language !== active) return;
    const heading = document.querySelector("main h1");
    if (heading) {
      const moduleKey = heading.textContent.replace(/[¶#]\s*$/, "")
        .replace(/\s+module$/i, "").trim();
      const record = catalog.symbols && catalog.symbols[moduleKey];
      if (record && record.text && /^spacr(?:\.|$)/.test(moduleKey)) {
        const panel = document.createElement("section");
        panel.className = "spacr-api-translation";
        panel.lang = language.replace("_", "-");
        panel.innerHTML = `<div class="spacr-api-translation__label">${LABELS[language] || "Translated API documentation"}</div>${renderRst(record.text)}`;
        heading.insertAdjacentElement("afterend", panel);
      }
    }
    document.querySelectorAll("dl.py dt[id]").forEach((signature) => {
      const record = catalog.symbols && catalog.symbols[signature.id];
      const body = signature.parentElement && signature.parentElement.querySelector(":scope > dd");
      if (!record || !record.text || !body) return;
      const panel = document.createElement("section");
      panel.className = "spacr-api-translation";
      panel.lang = language.replace("_", "-");
      panel.innerHTML = `<div class="spacr-api-translation__label">${LABELS[language] || "Translated API documentation"}</div>${renderRst(record.text)}`;
      body.prepend(panel);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!document.querySelector("dl.py dt[id]") && !/\/api\//.test(location.pathname)) return;
    const wrapper = document.createElement("label");
    wrapper.className = "spacr-api-language";
    const label = document.createElement("span");
    label.className = "spacr-api-language__label";
    label.textContent = SELECT_LABELS.en;
    wrapper.append(label);
    const select = document.createElement("select");
    Object.entries(LANGUAGES).forEach(([code, label]) => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = label;
      select.append(option);
    });
    wrapper.append(select);
    const content = document.querySelector("main") || document.body;
    content.prepend(wrapper);
    const requested = new URLSearchParams(location.search).get("lang");
    const preferred = (requested && LANGUAGES[requested] && requested) ||
      localStorage.getItem("spacr-doc-language") ||
      Object.keys(LANGUAGES).find((code) => navigator.language.toLowerCase().startsWith(code.split("_")[0])) || "en";
    select.value = preferred;
    select.addEventListener("change", () => selectLanguage(select.value));
    selectLanguage(preferred).catch(() => {
      active = "en";
      select.value = "en";
      label.textContent = SELECT_LABELS.en;
      clearTranslations();
    });
  });
})();
