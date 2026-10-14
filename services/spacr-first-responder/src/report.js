import { SQL } from "./sql.js";
import { createComment, createIssue } from "./github.js";
import { sha256Hex } from "./util.js";

export const ISSUE_LABEL = "user-report";
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
  const base = `[user-report] ${r.exception_type} in ${r.module}`;
  return msg ? `${base}: ${msg}` : base;
}

export function tracebackText(frames) {
  if (frames.length === 0) return "(no frames)";
  return frames.map((f) => `  ${f.file}:${f.line} in ${f.function}`).join("\n");
}

export function issueBody(r, signature) {
  const lines = [
    "An anonymised error report was sent from spaCR by the spacr-first-responder relay.",
    "The reporter has no GitHub account link; paths, names and other personal data were removed before sending.",
    "",
    "### Error",
    "",
    `**${r.exception_type}** in \`${r.module}\``,
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
  lines.push(
    "",
    "---",
    `Signature: \`${signature.slice(0, 16)}\`. Further identical reports add one comment per day.`,
    `<!-- spacr-signature: ${signature} -->`,
  );
  return lines.join("\n");
}

export function plusOneBody(r, count) {
  const lines = [`+1 from ${environmentLine(r)}. Reports so far: ${count}.`];
  if (r.user_note) lines.push("", quote(r.user_note));
  return lines.join("\n");
}

function issueUrl(env, number) {
  return number ? `https://github.com/${env.REPO}/issues/${number}` : null;
}

async function openNewIssue(env, r, signature, now) {
  const claimed = await env.DB.prepare(SQL.claimSignature).bind(signature, now).first();
  if (!claimed) return null;
  let issue;
  try {
    issue = await createIssue(env, issueTitle(r), issueBody(r, signature), [ISSUE_LABEL]);
  } catch (err) {
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
    throw err;
  }
  await env.DB.prepare(SQL.setIssueNumber).bind(signature, issue.number).run();
  return { issue_number: issue.number, issue_url: issue.url, deduplicated: false };
}

export async function fileReport(env, r, { day, now }) {
  const signature = await signatureOf(r);
  for (let attempt = 0; attempt < 2; attempt++) {
    const bumped = await env.DB.prepare(SQL.bumpSignature).bind(signature, now).first();
    if (!bumped) {
      const created = await openNewIssue(env, r, signature, now);
      if (created) return created;
      continue;
    }
    if (!bumped.issue_number) {
      return { issue_number: null, issue_url: null, deduplicated: true, pending: true };
    }
    const claim = await env.DB.prepare(SQL.claimDailyComment).bind(signature, day).first();
    if (!claim) {
      return {
        issue_number: bumped.issue_number,
        issue_url: issueUrl(env, bumped.issue_number),
        deduplicated: true,
        commented: false,
      };
    }
    const result = await createComment(env, claim.issue_number, plusOneBody(r, claim.count));
    if (!result.gone) {
      return {
        issue_number: claim.issue_number,
        issue_url: issueUrl(env, claim.issue_number),
        deduplicated: true,
        commented: true,
      };
    }
    await env.DB.prepare(SQL.dropSignature).bind(signature).run();
  }
  return { issue_number: null, issue_url: null, deduplicated: true, pending: true };
}
