/* On-demand API docstring translations generated outside Python sources. */
(() => {
  "use strict";

  const LANGUAGES = Object.freeze({
    en: "English", sv: "Svenska", de: "Deutsch", es: "Español",
    zh_CN: "简体中文", pt: "Português", hi: "हिन्दी", ko: "한국어",
    is: "Íslenska", fr: "Français",
  });
  const LABELS = Object.freeze({
    sv: "Översatt API-dokumentation", de: "Übersetzte API-Dokumentation",
    es: "Documentación de API traducida", zh_CN: "翻译的 API 文档",
    pt: "Documentação da API traduzida", hi: "अनुवादित API दस्तावेज़",
    ko: "번역된 API 문서", is: "Þýdd API-skjölun",
    fr: "Documentation API traduite",
  });
  const SELECT_LABELS = Object.freeze({
    en: "API language", sv: "API-språk", de: "API-Sprache",
    es: "Idioma de la API", zh_CN: "API 语言", pt: "Idioma da API",
    hi: "API भाषा", ko: "API 언어", is: "Tungumál API",
    fr: "Langue de l’API",
  });
  // These labels are UI supplied by this renderer, not docstring content.
  // Keeping them here prevents protected RST field/directive names from
  // leaking English into an otherwise localized translation panel.
  const RST_TERMS = Object.freeze({
    en: {
      note: "Note", warning: "Warning", tip: "Tip", important: "Important",
      caution: "Caution", attention: "Attention", danger: "Danger",
      error: "Error", hint: "Hint", parameter: "Parameter",
      argument: "Argument", keyword: "Keyword", type: "Type",
      variable: "Variable", raises: "Raises", returns: "Returns",
      return_type: "Return type", yields: "Yields",
    },
    sv: {
      note: "Anteckning", warning: "Varning", tip: "Tips", important: "Viktigt",
      caution: "Försiktighet", attention: "Observera", danger: "Fara",
      error: "Fel", hint: "Ledtråd", parameter: "Parameter",
      argument: "Argument", keyword: "Nyckelord", type: "Typ",
      variable: "Variabel", raises: "Undantag", returns: "Returvärde",
      return_type: "Returtyp", yields: "Genererar",
    },
    de: {
      note: "Bemerkung", warning: "Warnung", tip: "Tipp", important: "Wichtig",
      caution: "Vorsicht", attention: "Achtung", danger: "Gefahr",
      error: "Fehler", hint: "Hinweis", parameter: "Parameter",
      argument: "Argument", keyword: "Schlüsselwort", type: "Typ",
      variable: "Variable", raises: "Ausnahmen", returns: "Rückgabe",
      return_type: "Rückgabetyp", yields: "Erzeugt",
    },
    es: {
      note: "Nota", warning: "Advertencia", tip: "Consejo",
      important: "Importante", caution: "Precaución", attention: "Atención",
      danger: "Peligro", error: "Error", hint: "Sugerencia",
      parameter: "Parámetro", argument: "Argumento", keyword: "Palabra clave",
      type: "Tipo", variable: "Variable", raises: "Excepciones",
      returns: "Devuelve", return_type: "Tipo devuelto", yields: "Genera",
    },
    zh_CN: {
      note: "备注", warning: "警告", tip: "提示", important: "重要",
      caution: "小心", attention: "注意", danger: "危险", error: "错误",
      hint: "提示", parameter: "参数", argument: "参数", keyword: "关键字",
      type: "类型", variable: "变量", raises: "抛出", returns: "返回",
      return_type: "返回类型", yields: "生成",
    },
    pt: {
      note: "Nota", warning: "Aviso", tip: "Dica", important: "Importante",
      caution: "Cuidado", attention: "Atenção", danger: "Perigo",
      error: "Erro", hint: "Sugestão", parameter: "Parâmetro",
      argument: "Argumento", keyword: "Palavra-chave", type: "Tipo",
      variable: "Variável", raises: "Exceções", returns: "Retorna",
      return_type: "Tipo de retorno", yields: "Produz",
    },
    hi: {
      note: "टिप्पणी", warning: "चेतावनी", tip: "सुझाव",
      important: "महत्वपूर्ण", caution: "सावधानी", attention: "ध्यान दें",
      danger: "खतरा", error: "त्रुटि", hint: "संकेत", parameter: "पैरामीटर",
      argument: "आर्ग्युमेंट", keyword: "कीवर्ड", type: "प्रकार",
      variable: "चर", raises: "अपवाद", returns: "लौटाता है",
      return_type: "वापसी प्रकार", yields: "उत्पन्न करता है",
    },
    ko: {
      note: "참고", warning: "경고", tip: "팁", important: "중요",
      caution: "조심", attention: "주의", danger: "위험", error: "오류",
      hint: "힌트", parameter: "매개변수", argument: "인수", keyword: "키워드",
      type: "형식", variable: "변수", raises: "예외 발생", returns: "반환",
      return_type: "반환 형식", yields: "생성",
    },
    is: {
      note: "Athugasemd", warning: "Aðvörun", tip: "Ábending",
      important: "Mikilvægt", caution: "Aðgát", attention: "Athugið",
      danger: "Hætta", error: "Villa", hint: "Ábending", parameter: "Færibreyta",
      argument: "Viðfang", keyword: "Lykilorð", type: "Gerð",
      variable: "Breyta", raises: "Undantekningar", returns: "Skilar",
      return_type: "Skilagerð", yields: "Gefur",
    },
    fr: {
      note: "Remarque", warning: "Avertissement", tip: "Astuce",
      important: "Important", caution: "Prudence", attention: "Attention",
      danger: "Danger", error: "Erreur", hint: "Indication", parameter: "Paramètre",
      argument: "Argument", keyword: "Mot-clé", type: "Type",
      variable: "Variable", raises: "Exceptions", returns: "Renvoie",
      return_type: "Type renvoyé", yields: "Produit",
    },
  });
  const FIELD_TERMS = Object.freeze({
    param: "parameter", parameter: "parameter",
    arg: "argument", argument: "argument",
    keyword: "keyword", key: "keyword",
    type: "type", vartype: "type", ivar: "variable",
    raise: "raises", raises: "raises",
    return: "returns", returns: "returns", rtype: "return_type",
    yield: "yields", yields: "yields",
  });
  const STORAGE_KEY = "spacr-doc-language";
  const SHA256 = /^[0-9a-f]{64}$/;
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

  const script = [...document.scripts].find((item) =>
    /(?:^|\/)api_i18n\.js(?:\?|$)/.test(item.src));
  if (!script) return;
  const scriptUrl = new URL(script.src, document.baseURI);
  const catalogRoot = new URL("./i18n/api/", scriptUrl);
  // Sphinx supplies a digest of every catalog. Retain a schema fallback for
  // direct/source-tree use, where the data attribute is intentionally absent.
  const catalogVersion = script.dataset.apiCatalogVersion ||
    scriptUrl.searchParams.get("v") || "schema-2";

  let requestedLanguage = "en";
  let requestSerial = 0;
  let requestController = null;
  let languageSelect = null;
  let languageLabel = null;
  let apiArticle = null;

  function plainObject(value) {
    return value !== null && typeof value === "object" &&
      !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
  }

  function validHashes(value) {
    return Array.isArray(value) && value.length > 0 &&
      value.every((item) => typeof item === "string" && SHA256.test(item));
  }

  function sameArray(left, right) {
    return left.length === right.length &&
      left.every((item, index) => item === right[index]);
  }

  function safeStorageGet() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (_error) {
      return null;
    }
  }

  function safeStorageSet(language) {
    try {
      window.localStorage.setItem(STORAGE_KEY, language);
    } catch (_error) {
      // Storage can be unavailable in private, sandboxed, or hardened contexts.
    }
  }

  function normalizedLanguage(value, allowRegional = true) {
    if (typeof value !== "string" || !value.trim()) return null;
    const tag = value.trim().replace(/_/g, "-");
    const exact = Object.keys(LANGUAGES).find(
      (code) => code.replace(/_/g, "-").toLowerCase() === tag.toLowerCase());
    if (exact) return exact;
    if (!allowRegional) return null;
    const base = tag.split("-")[0].toLowerCase();
    // A Chinese region/script is semantically significant. In particular,
    // Traditional Chinese (zh-TW/zh-Hant) must not select the zh_CN catalog.
    if (base === "zh") return null;
    return Object.keys(LANGUAGES).find((code) => code === base) || null;
  }

  function browserLanguage() {
    const candidates = [];
    try {
      if (Array.isArray(navigator.languages)) candidates.push(...navigator.languages);
      if (navigator.language) candidates.push(navigator.language);
    } catch (_error) {
      return "en";
    }
    for (const candidate of candidates) {
      const language = normalizedLanguage(candidate, true);
      if (language) return language;
    }
    return "en";
  }

  function queryLanguage() {
    try {
      return normalizedLanguage(
        new URLSearchParams(location.search).get("lang"), true);
    } catch (_error) {
      return null;
    }
  }

  function initialLanguage() {
    return queryLanguage() || normalizedLanguage(safeStorageGet(), true) ||
      browserLanguage();
  }

  function updateHistory(language, mode) {
    if (!mode || !window.history || typeof window.history[mode] !== "function") return;
    try {
      const url = new URL(location.href);
      url.searchParams.set("lang", language);
      const previous = plainObject(history.state) ? history.state : {};
      history[mode]({...previous, spacrApiLanguage: language}, "", url);
    } catch (_error) {
      // file:// previews and embedded webviews may deny History API writes.
    }
  }

  function escapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = String(value);
    return node.innerHTML;
  }

  function inlineMarkup(value) {
    return escapeHtml(value)
      .replace(/:([A-Za-z][\w:-]*):`([^`]+)`/g, (_match, role, text) => {
        const roleClass = role.replace(/[^A-Za-z0-9_-]/g, "-");
        return `<code class="rst-role rst-role-${roleClass}">${text}</code>`;
      })
      .replace(/``([^`]+)``|`([^`]+)`/g,
        (_match, literal, interpreted) => `<code>${literal || interpreted}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  function preformattedRst(value) {
    return `<pre class="spacr-api-translation__rst">${escapeHtml(value)}</pre>`;
  }

  function dedent(lines) {
    const nonblank = lines.filter((line) => line.trim());
    const indent = nonblank.length ? Math.min(...nonblank.map(
      (line) => (line.match(/^\s*/) || [""])[0].length)) : 0;
    return lines.map((line) => line.slice(Math.min(indent, line.length)));
  }

  function indentedBody(lines, start) {
    let index = start;
    while (index < lines.length && !lines[index].trim()) index += 1;
    const body = [];
    while (index < lines.length) {
      const line = lines[index];
      if (line.trim() && !/^\s+/.test(line)) break;
      body.push(line);
      index += 1;
    }
    while (body.length && !body[body.length - 1].trim()) body.pop();
    return {body: dedent(body), index};
  }

  function fieldMatch(line) {
    const match = line.match(/^\s*:([^:]+):\s*(.*)$/);
    if (!match) return null;
    return /^(?:param|parameter|arg|argument|keyword|key|type|vartype|ivar|raises?|returns?|rtype|yields?)(?:\s|$)/i
      .test(match[1]) ? match : null;
  }

  function localizedTerm(language, term) {
    const terms = RST_TERMS[language] || RST_TERMS.en;
    return terms[term] || RST_TERMS.en[term] || "";
  }

  function localizedFieldName(value, language) {
    const match = String(value).trim().match(/^(\S+)(?:\s+(.*))?$/);
    if (!match) return value;
    const term = FIELD_TERMS[match[1].toLowerCase()];
    if (!term) return value;
    const suffix = match[2] ? ` ${match[2]}` : "";
    return `${localizedTerm(language, term)}${suffix}`;
  }

  function structuredRst(value, language, depth = 0) {
    if (depth > 3) return null;
    const lines = String(value).replace(/\r\n?/g, "\n").split("\n");
    // Raw HTML, substitutions, targets, tables, and unknown directives are
    // deliberately shown as escaped RST rather than partially/lossily parsed.
    const supportedDirective = /^\s*\.\.\s+(?:(?:note|warning|tip|important|caution|attention|danger|error|hint)|(?:code|code-block))::/i;
    if (lines.some((line) => /^\s*\.\.\s+/.test(line) &&
      !supportedDirective.test(line))) return null;
    if (lines.some((line) => /^\s*(?:\|[^|]+\||\+[-+=]+\+|\|.*\|\s*$)/.test(line))) return null;

    const output = [];
    let index = 0;
    while (index < lines.length) {
      if (!lines[index].trim()) {
        index += 1;
        continue;
      }

      if (index + 1 < lines.length &&
          /^[=~^`'\-:#*+]{3,}$/.test(lines[index + 1].trim()) &&
          lines[index + 1].trim().length >= lines[index].trim().length) {
        output.push(`<h4>${inlineMarkup(lines[index].trim())}</h4>`);
        index += 2;
        continue;
      }

      const directive = lines[index].match(/^\s*\.\.\s+([\w-]+)::\s*(.*)$/);
      if (directive) {
        const name = directive[1].toLowerCase();
        const captured = indentedBody(lines, index + 1);
        if (name === "code" || name === "code-block") {
          const codeLines = captured.body.filter(
            (line) => !/^\s*:[\w-]+:\s*/.test(line));
          output.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        } else {
          const bodyLines = directive[2] ?
            [directive[2], "", ...captured.body] : captured.body;
          const bodyText = bodyLines.join("\n");
          const body = structuredRst(bodyText, language, depth + 1);
          if (body === null) return null;
          const title = localizedTerm(language, name);
          output.push(`<aside class="admonition ${name}" role="note">` +
            `<p class="admonition-title">${escapeHtml(title)}</p>${body}</aside>`);
        }
        index = captured.index;
        continue;
      }

      if (/^\s*(?:>>>|\.\.\.)\s?/.test(lines[index])) {
        const code = [];
        while (index < lines.length && lines[index].trim()) {
          code.push(lines[index]);
          index += 1;
        }
        output.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        continue;
      }

      if (/^\s*[*+-]\s+/.test(lines[index])) {
        const items = [];
        while (index < lines.length) {
          const item = lines[index].match(/^\s*[*+-]\s+(.*)$/);
          if (!item) break;
          const text = [item[1]];
          index += 1;
          while (index < lines.length && /^\s{2,}\S/.test(lines[index]) &&
              !/^\s*[*+-]\s+/.test(lines[index])) {
            text.push(lines[index].trim());
            index += 1;
          }
          items.push(`<li>${inlineMarkup(text.join(" "))}</li>`);
          while (index < lines.length && !lines[index].trim()) index += 1;
        }
        output.push(`<ul>${items.join("")}</ul>`);
        continue;
      }

      if (/^\s*(?:\d+[.)]|#\.)\s+/.test(lines[index])) {
        const items = [];
        while (index < lines.length) {
          const item = lines[index].match(/^\s*(?:\d+[.)]|#\.)\s+(.*)$/);
          if (!item) break;
          items.push(`<li>${inlineMarkup(item[1])}</li>`);
          index += 1;
          while (index < lines.length && !lines[index].trim()) index += 1;
        }
        output.push(`<ol>${items.join("")}</ol>`);
        continue;
      }

      if (fieldMatch(lines[index])) {
        const fields = [];
        while (index < lines.length) {
          const field = fieldMatch(lines[index]);
          if (!field) break;
          const description = [field[2]];
          index += 1;
          while (index < lines.length && /^\s+\S/.test(lines[index]) &&
              !fieldMatch(lines[index])) {
            description.push(lines[index].trim());
            index += 1;
          }
          fields.push(`<dt>${inlineMarkup(localizedFieldName(field[1], language))}</dt>` +
            `<dd>${inlineMarkup(description.join(" "))}</dd>`);
          while (index < lines.length && !lines[index].trim()) index += 1;
        }
        output.push(`<dl class="field-list">${fields.join("")}</dl>`);
        continue;
      }

      const paragraph = [];
      while (index < lines.length && lines[index].trim()) {
        if (paragraph.length && (supportedDirective.test(lines[index]) ||
            /^\s*(?:[*+-]|(?:\d+[.)]|#\.))\s+/.test(lines[index]) ||
            fieldMatch(lines[index]))) break;
        paragraph.push(lines[index].trim());
        index += 1;
      }
      if (!paragraph.length) return null;
      const joined = paragraph.join(" ");
      if (/::$/.test(joined)) {
        const captured = indentedBody(lines, index);
        if (!captured.body.length) {
          output.push(`<p>${inlineMarkup(joined)}</p>`);
        } else {
          output.push(`<p>${inlineMarkup(joined.slice(0, -1))}</p>`);
          output.push(`<pre><code>${escapeHtml(captured.body.join("\n"))}</code></pre>`);
          index = captured.index;
        }
      } else {
        output.push(`<p>${inlineMarkup(joined)}</p>`);
      }
    }
    return output.join("");
  }

  function renderRst(value, language) {
    const rendered = structuredRst(value, language);
    return rendered === null ? preformattedRst(value) : rendered;
  }

  function clearTranslations() {
    if (!apiArticle) return;
    apiArticle.querySelectorAll(".spacr-api-translation")
      .forEach((node) => node.remove());
  }

  function catalogUrl(language) {
    const url = new URL(`${language}.json`, catalogRoot);
    url.searchParams.set("v", catalogVersion);
    return url;
  }

  async function fetchJson(language, signal) {
    const response = await fetch(catalogUrl(language), {
      signal,
      headers: {Accept: "application/json"},
    });
    if (!response.ok) {
      throw new Error(`API translation catalog returned ${response.status}`);
    }
    return response.json();
  }

  function validateCatalog(payload, language) {
    if (!plainObject(payload) || payload.schema !== 2 ||
        payload.language !== language || !plainObject(payload.symbols)) {
      throw new Error("Malformed API translation catalog metadata");
    }
    const keys = Object.keys(payload.symbols);
    if (!keys.length || keys.some((key) => !key.startsWith("spacr"))) {
      throw new Error("Malformed API translation symbol map");
    }
    for (const key of keys) {
      const record = payload.symbols[key];
      if (!plainObject(record) || typeof record.text !== "string" ||
          !record.text.trim() || typeof record.source_sha256 !== "string" ||
          !SHA256.test(record.source_sha256) ||
          !validHashes(record.source_blocks_sha256)) {
        throw new Error(`Malformed API translation record: ${key}`);
      }
      if (language !== "en" &&
          (!validHashes(record.translation_source_blocks_sha256) ||
           record.translation_source_blocks_sha256.length !==
             record.source_blocks_sha256.length)) {
        throw new Error(`Malformed API translation freshness record: ${key}`);
      }
    }
    return payload.symbols;
  }

  function validateAgainstEnglish(localized, english) {
    const localizedKeys = Object.keys(localized);
    const englishKeys = Object.keys(english);
    if (localizedKeys.length !== englishKeys.length) {
      throw new Error("Stale API translation symbol set");
    }
    for (const key of englishKeys) {
      if (!own(localized, key) ||
          localized[key].source_sha256 !== english[key].source_sha256 ||
          !sameArray(localized[key].source_blocks_sha256,
            english[key].source_blocks_sha256)) {
        throw new Error(`Stale API translation source hash: ${key}`);
      }
    }
  }

  function translationPanel(language, text) {
    const panel = document.createElement("section");
    panel.className = "spacr-api-translation";
    panel.lang = language.replace("_", "-");
    panel.innerHTML = `<div class="spacr-api-translation__label">` +
      `${LABELS[language] || "Translated API documentation"}</div>` +
      renderRst(text, language);
    return panel;
  }

  function renderCatalog(language, symbols) {
    const insertions = [];
    // Furo wraps an AutoAPI module page in a top-level ``section#module-*``.
    // Retain the direct-child fallback for small standalone/source fixtures.
    const heading = apiArticle.querySelector(
      ':scope > section[id^="module-spacr"] > h1') ||
      apiArticle.querySelector(":scope > h1");
    if (heading) {
      const moduleKey = heading.textContent.replace(/[¶#]\s*$/, "")
        .replace(/\s+module$/i, "").trim();
      const record = symbols[moduleKey];
      if (record && /^spacr(?:\.|$)/.test(moduleKey)) {
        insertions.push({
          target: heading,
          position: "afterend",
          panel: translationPanel(language, record.text),
        });
      }
    }
    apiArticle.querySelectorAll("dl.py dt[id]").forEach((signature) => {
      const record = own(symbols, signature.id) ? symbols[signature.id] : null;
      const body = signature.parentElement &&
        signature.parentElement.querySelector(":scope > dd");
      if (!record || !body) return;
      insertions.push({
        target: body,
        position: "prepend",
        panel: translationPanel(language, record.text),
      });
    });
    clearTranslations();
    insertions.forEach(({target, position, panel}) => {
      if (position === "prepend") target.prepend(panel);
      else target.insertAdjacentElement(position, panel);
    });
  }

  function commitEnglish({historyMode = null, persist = true} = {}) {
    requestedLanguage = "en";
    clearTranslations();
    if (languageSelect) languageSelect.value = "en";
    if (languageLabel) languageLabel.textContent = SELECT_LABELS.en;
    if (persist) safeStorageSet("en");
    updateHistory("en", historyMode);
  }

  async function selectLanguage(language, options = {}) {
    const selected = normalizedLanguage(language, false) || "en";
    const serial = ++requestSerial;
    requestedLanguage = selected;
    if (requestController) requestController.abort();
    requestController = typeof AbortController === "function" ?
      new AbortController() : null;
    const signal = requestController ? requestController.signal : undefined;

    if (languageSelect) languageSelect.value = selected;
    if (languageLabel) {
      languageLabel.textContent = SELECT_LABELS[selected] || SELECT_LABELS.en;
    }

    if (selected === "en") {
      commitEnglish(options);
      return true;
    }

    try {
      const [englishPayload, localizedPayload] = await Promise.all([
        fetchJson("en", signal), fetchJson(selected, signal),
      ]);
      if (serial !== requestSerial || selected !== requestedLanguage) return false;
      const english = validateCatalog(englishPayload, "en");
      const localized = validateCatalog(localizedPayload, selected);
      validateAgainstEnglish(localized, english);
      if (serial !== requestSerial || selected !== requestedLanguage) return false;
      renderCatalog(selected, localized);
      if (languageSelect) languageSelect.value = selected;
      if (languageLabel) languageLabel.textContent = SELECT_LABELS[selected];
      if (options.persist !== false) safeStorageSet(selected);
      updateHistory(selected, options.historyMode || null);
      return true;
    } catch (error) {
      if (serial !== requestSerial || selected !== requestedLanguage) return false;
      console.warn("Unable to load localized API documentation; using English.", error);
      // Invalidate this request before rollback so a late continuation cannot
      // reinsert a failed/stale locale after the English state is committed.
      requestSerial += 1;
      commitEnglish({historyMode: options.failureHistoryMode || "replaceState"});
      return false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    apiArticle = document.querySelector('article[role="main"]');
    if (!apiArticle || (!apiArticle.querySelector("dl.py dt[id]") &&
        !/\/api(?:\/|$)/.test(location.pathname))) return;

    const wrapper = document.createElement("label");
    wrapper.className = "spacr-api-language";
    languageLabel = document.createElement("span");
    languageLabel.className = "spacr-api-language__label";
    languageLabel.textContent = SELECT_LABELS.en;
    wrapper.append(languageLabel);
    languageSelect = document.createElement("select");
    Object.entries(LANGUAGES).forEach(([code, displayName]) => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = displayName;
      languageSelect.append(option);
    });
    wrapper.append(languageSelect);
    apiArticle.prepend(wrapper);

    const preferred = initialLanguage();
    try {
      const state = plainObject(history.state) ? history.state : {};
      history.replaceState({...state, spacrApiLanguage: preferred}, "", location.href);
    } catch (_error) {
      // History state is an enhancement, never a prerequisite for translation.
    }
    languageSelect.value = preferred;
    languageSelect.addEventListener("change", () => {
      void selectLanguage(languageSelect.value, {historyMode: "pushState"});
    });
    window.addEventListener("popstate", (event) => {
      const stateLanguage = plainObject(event.state) ?
        normalizedLanguage(event.state.spacrApiLanguage, false) : null;
      const preferredAtHistoryEntry = stateLanguage || queryLanguage() || "en";
      void selectLanguage(preferredAtHistoryEntry, {persist: true});
    });
    void selectLanguage(preferred);
  });
})();
