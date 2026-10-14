import { SQL } from "./sql.js";
import { createComment, createIssue, issueState } from "./github.js";
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

export async function signatureOf(report) {
  const parts = [report.exception_type, report.module];
  for (const f of topFrames(report.frames)) parts.push(`${f.file}:${f.function}`);
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
  const msg = oneLine(r.message || "", 80);
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

export function summarySection(summary) {
  if (!summary) return [];
  const source =
    summary.source === "client"
      ? "summary written by the reporter's own AI"
      : `summary written by spaCR AI (${summary.model})`;
  return ["### AI summary (generated, may be wrong)", "", quote(summary.text), "", `_${source}_`];
}

export function issueBody(r, signature, summary = null) {
  const lines = [
    "An anonymised error report was sent from spaCR by the spacr-first-responder relay.",
    "The reporter has no GitHub account link; paths, names and other personal data were removed before sending.",
    "",
    "### Error",
    "",
    `**${r.exception_type}** in \`${r.module}\``,
    ...(r.kind === "user_error" ? ["", "Kind: **user error** (settings or usage), reported from spaCR."] : []),
    "",
    fence(r.message || ""),
    "",
    "### Traceback (most recent call last, file names only)",
    "",
    fence(tracebackText(r.frames)),
    "",
    "### Environment",
    "",
    "| Item | Value |",
    "| --- | --- |",
    `| spaCR | ${r.spacr_version} |`,
    `| OS family | ${r.os} |`,
    `| Python | ${r.python} |`,
    `| Qt | ${r.qt ?? "n/a"} |`,
    `| torch | ${r.torch ?? "n/a"} |`,
    `| GPU class | ${r.gpu ?? "n/a"} |`,
  ];
  if (r.user_note) {
    lines.push("", "### Note from the user", "", quote(r.user_note));
  }
  if (summary) lines.push("", ...summarySection(summary));
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
  if (r.user_note) lines.push("", quote(r.user_note));
  if (summary) lines.push("", ...summarySection(summary));
  return lines.join("\n");
}

export function clientSummary(r) {
  return r.ai_summary ? { text: r.ai_summary, source: "client", model: null } : null;
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

function issueUrl(env, number) {
  return number ? `https://github.com/${env.REPO}/issues/${number}` : null;
}

async function openNewIssue(env, r, signature, now, getSummary) {
  const claimed = await env.DB.prepare(SQL.claimSignature).bind(signature, now).first();
  if (!claimed) return null;
  let issue;
  let summary = null;
  try {
    summary = await obtainSummary(r, getSummary);
    issue = await createIssue(env, issueTitle(r), issueBody(r, signature, summary), issueLabels(r));
  } catch (err) {
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
    throw err;
  }
  await env.DB.prepare(SQL.setIssueNumber).bind(signature, issue.number, summary ? 1 : 0).run();
  return {
    issue_number: issue.number,
    issue_url: issue.url,
    deduplicated: false,
    ai_summary_source: sourceOf(summary),
  };
}

async function reopenAsRecurred(env, r, signature, closedNumber, getSummary) {
  const summary = await obtainSummary(r, getSummary);
  const title = `[recurred] ${issueTitle(r)}`;
  const body = `${issueBody(r, signature, summary)}\n\nSeen again after #${closedNumber} was closed.`;
  const issue = await createIssue(env, title, body, issueLabels(r));
  await env.DB.prepare(SQL.resetSignature).bind(signature, issue.number, summary ? 1 : 0).run();
  return {
    issue_number: issue.number,
    issue_url: issue.url,
    deduplicated: false,
    recurred_from: closedNumber,
    ai_summary_source: sourceOf(summary),
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
        ai_summary_source: sourceOf(summary),
      };
    }
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
  }
  return { issue_number: null, issue_url: null, deduplicated: true, pending: true, ai_summary_source: "none" };
}
