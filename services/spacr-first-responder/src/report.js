import { SQL } from "./sql.js";
import { createComment, createIssue, issueState } from "./github.js";
import { causeLabel, ensureCauseLabel, hubFor, linkToHub, openIssuesWithLabel, splitCause } from "./causes.js";
import { causeOnFile, dropError } from "./cache.js";
import { sha256Hex } from "./util.js";

export const ISSUE_LABEL = "user-report";
export const USER_ERROR_LABEL = "user-error";

export function issueLabels(r) {
  return r.kind === "user_error" ? [ISSUE_LABEL, USER_ERROR_LABEL] : [ISSUE_LABEL];
}
export const SIGNATURE_FRAMES = 5;

export function topFrames(frames, n = SIGNATURE_FRAMES) {
  return frames.slice(-n);
}

export function normalizedProblems(message) {
  return problemLines(message)
    .map((l) =>
      l
        .toLowerCase()
        .replace(/<[^<>]*>/g, " ")
        .replace(/\S*[\\/]\S*/g, " ")
        .replace(/[0-9]+/g, "")
        .replace(/\s+/g, " ")
        .trim(),
    )
    .filter((l) => l)
    .sort();
}

export async function signatureOf(report) {
  const parts = [report.exception_type, report.module];
  for (const f of topFrames(report.frames)) parts.push(`${f.file}:${f.function}`);
  if (report.kind === "user_error") parts.push(...normalizedProblems(report.message));
  return sha256Hex(parts.join("\n"));
}

function fence(text) {
  const runs = String(text).match(/`+/g) || [];
  const longest = runs.reduce((m, r) => Math.max(m, r.length), 0);
  const f = "`".repeat(Math.max(3, longest + 1));
  return `${f}text\n${text}\n${f}`;
}

function quote(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/@/g, "@​")
    .split(/\r?\n/)
    .map((l) => `> ${l}`)
    .join("\n");
}

function oneLine(text, max) {
  const s = String(text).replace(/\s+/g, " ").trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

export function environmentLine(r) {
  return [
    `spaCR ${r.spacr_version}`,
    r.os,
    `Python ${r.python}`,
    `Qt ${r.qt ?? "n/a"}`,
    `torch ${r.torch ?? "n/a"}`,
    `GPU ${r.gpu ?? "n/a"}`,
  ].join(" · ");
}

export function issueTitle(r) {
  const msg = oneLine(headline(r.message), 80);
  const prefix = r.kind === "user_error" ? "[user-report][settings]" : "[user-report]";
  const base = `${prefix} ${r.exception_type} in ${r.module}`;
  return msg ? `${base}: ${msg}` : base;
}

export function tracebackText(frames) {
  if (frames.length === 0) return "(no frames)";
  return frames
    .map((f) => (f.code ? `  ${f.file}:${f.line} in ${f.function}\n      ${f.code.trim()}` : `  ${f.file}:${f.line} in ${f.function}`))
    .join("\n");
}

export function problemLines(message) {
  return String(message || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => /^[-*•]\s+\S/.test(l))
    .map((l) => l.replace(/^[-*•]\s+/, ""));
}

export function headline(message) {
  return String(message || "").split(/\r?\n/)[0].trim();
}

function plainText(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/@/g, "@​");
}

export function errorSections(r) {
  const problems = problemLines(r.message);
  const lines = [
    "### Error",
    "",
    `**${r.exception_type}** in \`${r.module}\`${r.screen ? ` on the **${plainText(r.screen)}** screen` : ""}`,
    ...(r.kind === "user_error" ? ["", "Kind: **user error** (settings or usage), reported from spaCR."] : []),
    "",
    fence(problems.length ? headline(r.message) : r.message || ""),
  ];
  if (problems.length) {
    lines.push("", `### Problems (${problems.length})`, "", ...problems.map((p, i) => `${i + 1}. ${plainText(p)}`));
  }
  if (r.error_text) {
    lines.push("", "### Console error (scrubbed)", "", fence(r.error_text));
  }
  if (r.chain && r.chain.length) {
    lines.push(
      "",
      "### Exception chain (oldest first)",
      "",
      ...r.chain.map((c) => `- \`${c.type}\`${c.message ? `: ${plainText(oneLine(c.message, 300))}` : ""}`),
    );
  }
  const refused = r.kind === "user_error" && r.frames.length > 0;
  lines.push(
    "",
    refused
      ? "### Where the run was refused (call site, most recent call last, file names only)"
      : "### Traceback (most recent call last, file names only)",
    "",
    ...details(`${r.frames.length} frame${r.frames.length === 1 ? "" : "s"}`, fence(tracebackText(r.frames)), r.frames.length <= 12),
  );
  return lines;
}

function details(summary, inner, open = false) {
  return [open ? "<details open>" : "<details>", `<summary>${plainText(summary)}</summary>`, "", inner, "", "</details>"];
}

function yesNo(v) {
  return v === true ? "yes" : v === false ? "no" : "n/a";
}

export function runSection(r) {
  const run = r.run || {};
  const rows = [
    ["Screen", r.screen ? plainText(r.screen) : "n/a"],
    ["Module", `\`${r.module}\``],
    ["Stage", run.stage ? `\`${run.stage}\`` : "n/a"],
    ["Seconds before the failure", run.elapsed_s !== undefined ? String(run.elapsed_s) : "n/a"],
    ["Dry or preview run", yesNo(run.dry_run)],
  ];
  return ["### Run", "", "| Item | Value |", "| --- | --- |", ...rows.map(([k, v]) => `| ${k} | ${v} |`)];
}

export function inputSection(inputs) {
  if (!inputs || !Object.keys(inputs).length) return [];
  const lines = ["### Input (no names)", ""];
  if (inputs.images !== undefined) lines.push(`- Images found: ${inputs.images}`);
  if (inputs.extensions && inputs.extensions.length) lines.push(`- Extensions: ${inputs.extensions.join(", ")}`);
  if (inputs.shape && inputs.shape.length) lines.push(`- First image shape: ${inputs.shape.join(" × ")}`);
  if (inputs.dtype) lines.push(`- First image dtype: ${plainText(inputs.dtype)}`);
  if (inputs.channels !== undefined) lines.push(`- Likely channels: ${inputs.channels}`);
  return lines;
}

export function environmentRows(r) {
  const e = r.environment || {};
  const v = e.versions || {};
  const rows = [
    ["spaCR", r.spacr_version],
    ["OS family", r.os],
    ["OS version", e.os_version],
    ["CPU", e.arch !== undefined || e.cores !== undefined ? `${e.arch ?? "n/a"}, ${e.cores ?? "?"} cores` : undefined],
    ["RAM", e.ram_gb !== undefined ? `${e.ram_gb} GB` : undefined],
    ["Python", r.python],
    ["Qt", v.qt ?? r.qt ?? "n/a"],
    ["PySide6", v.pyside6],
    ["torch", v.torch ?? r.torch ?? "n/a"],
    ["cellpose", v.cellpose],
    ["numpy", v.numpy],
    ["scikit-image", v.scikit_image],
    ["GPU class", r.gpu ?? "n/a"],
    ["GPU model", e.gpu_model],
    ["VRAM", e.vram_gb !== undefined ? `${e.vram_gb} GB` : undefined],
    ["CUDA available", e.cuda !== undefined ? yesNo(e.cuda) : undefined],
    ["MPS available", e.mps !== undefined ? yesNo(e.mps) : undefined],
    ["Install", e.install],
  ];
  return rows.filter(([, val]) => val !== undefined && val !== null).map(([k, val]) => `| ${k} | ${plainText(val)} |`);
}

function settingText(value) {
  return JSON.stringify(value);
}

export function settingsSection(settings) {
  const keys = Object.keys(settings || {});
  if (!keys.length) return [];
  const width = Math.min(40, Math.max(...keys.map((k) => k.length)));
  const text = keys.map((k) => `${k.padEnd(width)} = ${settingText(settings[k])}`).join("\n");
  return ["### Settings in effect (paths and free text replaced by markers)", "", ...details(`${keys.length} settings`, fence(text))];
}

export function logSection(logTail) {
  if (!logTail) return [];
  const n = logTail.split(/\r?\n/).length;
  return ["### Console before the failure (scrubbed)", "", ...details(`last ${n} lines`, fence(logTail))];
}

export function relatedSection(cause, related, self = null) {
  if (!cause) return [];
  const others = [...new Set((related || []).filter((n) => n && n !== self))];
  const lines = ["### Same likely cause", "", `Likely cause key: \`${cause}\` (label \`${causeLabel(cause)}\`).`];
  if (others.length) lines.push("", ...others.map((n) => `- #${n}`));
  else lines.push("", "No other open report has this cause yet.");
  return lines;
}

export function summarySection(summary) {
  if (!summary) return [];
  const source =
    summary.source === "client"
      ? "summary written by the reporter's own AI"
      : `summary written by spaCR AI (${summary.model})`;
  return ["### AI summary (generated, may be wrong)", "", quote(summary.text), "", `_${source}_`];
}

export function issueBody(r, signature, summary = null, grouping = null) {
  const lines = [
    "An anonymised error report was sent from spaCR by the spacr-first-responder relay.",
    "The reporter has no GitHub account link; paths, names and other personal data were removed before sending.",
    "",
    ...errorSections(r),
    "",
    ...runSection(r),
  ];
  const input = inputSection(r.inputs);
  if (input.length) lines.push("", ...input);
  lines.push("", "### Environment", "", "| Item | Value |", "| --- | --- |", ...environmentRows(r));
  const settings = settingsSection(r.settings);
  if (settings.length) lines.push("", ...settings);
  const log = logSection(r.log_tail);
  if (log.length) lines.push("", ...log);
  if (grouping && grouping.cause) lines.push("", ...relatedSection(grouping.cause, grouping.related));
  if (r.user_note) {
    lines.push("", "### Note from the user", "", quote(r.user_note));
  }
  if (summary) lines.push("", "---", "", ...summarySection(summary));
  lines.push(
    "",
    "---",
    `Signature: \`${signature.slice(0, 16)}\`. Further identical reports add one comment per day.`,
    `<!-- spacr-signature: ${signature} -->`,
  );
  return lines.join("\n");
}

export function plusOneBody(r, count, summary = null) {
  const lines = [`+1 from ${environmentLine(r)}. Reports so far: ${count}.`];
  lines.push(
    "",
    "<details>",
    `<summary>This occurrence: ${plainText(oneLine(headline(r.message) || r.exception_type, 100))}</summary>`,
    "",
    ...errorSections(r),
    "",
    "</details>",
  );
  if (r.user_note) lines.push("", quote(r.user_note));
  if (summary) lines.push("", ...summarySection(summary));
  return lines.join("\n");
}

export function clientSummary(r) {
  if (!r.ai_summary) return null;
  const parsed = splitCause(r.ai_summary);
  return parsed.text ? { text: parsed.text, source: "client", model: null, cause: parsed.cause } : null;
}

export async function causeFor(env, r, signature, summary) {
  return r.cause || (summary && summary.cause) || (await causeOnFile(env.DB, signature)) || null;
}

export async function groupingFor(env, r, signature, summary) {
  const cause = await causeFor(env, r, signature, summary);
  if (!cause) return { cause: null, label: null, related: [], titleSuffix: "" };
  const label = await ensureCauseLabel(env, cause);
  const related = label ? await openIssuesWithLabel(env, label) : [];
  const hub = await hubFor(env.DB, cause);
  if (hub && !related.includes(hub)) related.unshift(hub);
  return { cause, label, related, titleSuffix: label ? "" : ` · Cause: ${cause}` };
}

export async function obtainSummary(r, getSummary) {
  const own = clientSummary(r);
  if (own) return own;
  if (!r.summarize || typeof getSummary !== "function") return null;
  try {
    return await getSummary();
  } catch (_) {
    return null;
  }
}

export function sourceOf(summary) {
  return summary ? summary.source : "none";
}

export function summaryFields(summary) {
  const out = { ai_summary_source: sourceOf(summary) };
  if (summary && summary.source === "relay" && summary.text) out.ai_summary = summary.text;
  if (summary && summary.source === "relay") out.ai_summary_cached = summary.cached === true;
  return out;
}

function issueUrl(env, number) {
  return number ? `https://github.com/${env.REPO}/issues/${number}` : null;
}

async function openNewIssue(env, r, signature, now, getSummary) {
  const claimed = await env.DB.prepare(SQL.claimSignature).bind(signature, now).first();
  if (!claimed) return null;
  let issue;
  let summary = null;
  let grouping;
  try {
    summary = await obtainSummary(r, getSummary);
    grouping = await groupingFor(env, r, signature, summary);
    issue = await createIssue(
      env,
      issueTitle(r) + grouping.titleSuffix,
      issueBody(r, signature, summary, grouping),
      grouping.label ? [...issueLabels(r), grouping.label] : issueLabels(r),
    );
  } catch (err) {
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
    throw err;
  }
  await env.DB.prepare(SQL.setIssueNumber).bind(signature, issue.number, summary ? 1 : 0).run();
  if (grouping.cause) await linkToHub(env, grouping.cause, issue.number, now);
  return {
    issue_number: issue.number,
    issue_url: issue.url,
    deduplicated: false,
    ...summaryFields(summary),
  };
}

async function reopenAsRecurred(env, r, signature, closedNumber, getSummary) {
  await dropError(env.DB, signature);
  const summary = await obtainSummary(r, getSummary);
  const grouping = await groupingFor(env, r, signature, summary);
  const title = `[recurred] ${issueTitle(r)}${grouping.titleSuffix}`;
  const body = `${issueBody(r, signature, summary, grouping)}\n\nSeen again after #${closedNumber} was closed.`;
  const issue = await createIssue(env, title, body, grouping.label ? [...issueLabels(r), grouping.label] : issueLabels(r));
  await env.DB.prepare(SQL.resetSignature).bind(signature, issue.number, summary ? 1 : 0).run();
  if (grouping.cause) await linkToHub(env, grouping.cause, issue.number, new Date().toISOString());
  return {
    issue_number: issue.number,
    issue_url: issue.url,
    deduplicated: false,
    recurred_from: closedNumber,
    ...summaryFields(summary),
  };
}

async function commentSummary(env, r, signature, claim, getSummary) {
  if (claim.has_summary) return null;
  if (!clientSummary(r) && !r.summarize) return null;
  const won = await env.DB.prepare(SQL.claimSummary).bind(signature).first();
  if (!won) return null;
  const summary = await obtainSummary(r, getSummary);
  if (!summary) await env.DB.prepare(SQL.unclaimSummary).bind(signature).run();
  return summary;
}

export async function fileReport(env, r, { day, now, getSummary }) {
  const signature = await signatureOf(r);
  for (let attempt = 0; attempt < 2; attempt++) {
    const bumped = await env.DB.prepare(SQL.bumpSignature).bind(signature, now).first();
    if (!bumped) {
      const created = await openNewIssue(env, r, signature, now, getSummary);
      if (created) return created;
      continue;
    }
    if (!bumped.issue_number) {
      return { issue_number: null, issue_url: null, deduplicated: true, pending: true, ai_summary_source: "none" };
    }
    const claim = await env.DB.prepare(SQL.claimDailyComment).bind(signature, day).first();
    if (!claim) {
      return {
        issue_number: bumped.issue_number,
        issue_url: issueUrl(env, bumped.issue_number),
        deduplicated: true,
        commented: false,
        ai_summary_source: "none",
      };
    }
    const state = await issueState(env, claim.issue_number);
    if (state === "closed") return reopenAsRecurred(env, r, signature, claim.issue_number, getSummary);
    if (state === "gone") {
      await env.DB.prepare(SQL.dropSignature).bind(signature).run();
      continue;
    }
    const summary = await commentSummary(env, r, signature, claim, getSummary);
    let result;
    try {
      result = await createComment(env, claim.issue_number, plusOneBody(r, claim.count, summary));
    } catch (err) {
      if (summary) await env.DB.prepare(SQL.unclaimSummary).bind(signature).run();
      throw err;
    }
    if (!result.gone) {
      return {
        issue_number: claim.issue_number,
        issue_url: issueUrl(env, claim.issue_number),
        deduplicated: true,
        commented: true,
        ...summaryFields(summary),
      };
    }
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
  }
  return { issue_number: null, issue_url: null, deduplicated: true, pending: true, ai_summary_source: "none" };
}
