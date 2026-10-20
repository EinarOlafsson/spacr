import { createComment, githubRequest, issueState } from "./github.js";
import { SQL } from "./sql.js";

export const CAUSE_MAX = 40;
export const CAUSE_WORDS = 5;
export const CAUSE_LABEL_PREFIX = "cause:";
export const CAUSE_LABEL_COLOR = "0e7c86";

const CAUSE_LINE_RE =
  /^\s*[*_`>#-]*\s*(?:CAUSE\s*[:=]\s*[*_`]*\s*(.*?)|[Cc][Aa][Uu][Ss][Ee]\s*[:=]\s*[*_`]*\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+))\s*[*_`]*\s*$/;

export function normaliseCause(raw) {
  const words = String(raw || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .split("-")
    .filter(Boolean)
    .slice(0, CAUSE_WORDS);
  let slug = words.join("-");
  if (slug.length > CAUSE_MAX) slug = slug.slice(0, CAUSE_MAX).replace(/-+$/, "");
  return slug || null;
}

export function splitCause(text) {
  const lines = String(text || "").split(/\r?\n/);
  let cause = null;
  const kept = [];
  for (const line of lines) {
    const m = CAUSE_LINE_RE.exec(line);
    if (m) {
      cause = normaliseCause(m[1] || m[2]) || cause;
      continue;
    }
    kept.push(line);
  }
  while (kept.length && !kept[kept.length - 1].trim()) kept.pop();
  return { text: kept.join("\n").trim(), cause };
}

export function causeLabel(slug) {
  return `${CAUSE_LABEL_PREFIX}${slug}`;
}

export async function ensureCauseLabel(env, slug) {
  try {
    const res = await githubRequest(env, "POST", "/labels", {
      name: causeLabel(slug),
      color: CAUSE_LABEL_COLOR,
      description: "Reports the AI traced to the same likely cause",
    });
    if (res.status === 201 || res.status === 422) return causeLabel(slug);
    console.log(`github: cause label not created (status ${res.status})`);
    return null;
  } catch (_) {
    return null;
  }
}

export async function openIssuesWithLabel(env, label, limit = 20) {
  try {
    const query = `?state=open&per_page=${limit}&labels=${encodeURIComponent(label)}`;
    const res = await githubRequest(env, "GET", `/issues${query}`);
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data.filter((i) => i && !i.pull_request).map((i) => i.number).filter(Number.isInteger) : [];
  } catch (_) {
    return [];
  }
}

export async function hubFor(db, slug) {
  try {
    const row = await db.prepare(SQL.getCause).bind(slug).first();
    return row && row.hub_issue ? row.hub_issue : null;
  } catch (_) {
    return null;
  }
}

export async function linkToHub(env, slug, number, now) {
  try {
    const db = env.DB;
    const row = await db.prepare(SQL.getCause).bind(slug).first();
    if (!row) {
      await db.prepare(SQL.insertCause).bind(slug, number, now).run();
      return { hub: number, commented: false };
    }
    await db.prepare(SQL.bumpCause).bind(slug, now).run();
    if (row.hub_issue === number) return { hub: number, commented: false };
    const state = await issueState(env, row.hub_issue);
    if (state !== "open") {
      await db.prepare(SQL.setCauseHub).bind(slug, number).run();
      return { hub: number, commented: false };
    }
    const result = await createComment(env, row.hub_issue, `Related report with the same likely cause: #${number}`);
    if (result.gone) {
      await db.prepare(SQL.setCauseHub).bind(slug, number).run();
      return { hub: number, commented: false };
    }
    return { hub: row.hub_issue, commented: true };
  } catch (_) {
    console.log("causes: could not link the issue to its cause hub");
    return { hub: null, commented: false };
  }
}
