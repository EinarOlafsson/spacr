"""Stage selected G2 evidence without executing producers or changing source artifacts."""

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit, urlunsplit


BASE = "reports/graphstyle/g2"
SELECTION = ".codex-audit/evidence-selection.txt"
EXTRAS = ".codex-audit/g2_publication_extras.tsv"
VALIDATION = f"{BASE}/validation.md"
ROUTE_REVIEW = ".codex-audit/g2_route_review.md"
REMOVED = ".codex-audit/g2_removed_route_ids.json"
ROUTE_REPORT = ".codex-audit/g2_routes_report.json"
PLAN = "reports/graphstyle/routes/initial_plan.json"
DEFERRED = "reports/graphstyle/routes/deferred_g2.json"
FIXED = {
    SELECTION: "470e96a21efbacd77ccbb24fc1c0879f8ee809ea8461f72f8df9b0c6472486e0",
    VALIDATION: "535997b6004394f2e8dcf198f336f00f1feaf48158f4dcba4a7cb32ca22fe804",
    ROUTE_REVIEW: "25bf2ee5dde8cf06ed755421b88486cbb8224fd2d75fc6a7e5b758aa4066c8fc",
    REMOVED: "1533c221ea4a902652592102aae61816b368007f5da5c885da686ec5354993b2",
    ROUTE_REPORT: "5330334b58b37eeb6b9366f4ce3ee19060a51a2ef1190223d9468963abe782d8",
    PLAN: "cc41aef6994ecf2d4968b194564631cfc1d70e021c56bfbbe56f86910f079e79",
}
GENERATED = {f"{BASE}/{name}" for name in (
    "publication-provenance.json", "publication-inputs.sha256",
    "publication-files.txt", "publication.sha256")}
LINK = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*(<[^>\n]+>|[^\s)]+)(\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)")
CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
ARTIFACT = re.compile(r"\.(?:png|pdf|json|csv|npy|npz|html|md|log|txt|xml|sha256)$", re.I)
PREPARATION = (
    "Standalone review notes and the new test receipt still have local audit links: "
    "the proposed publication selection in `.codex-audit/evidence-selection.txt` "
    "requires curated review copies under `reports/graphstyle/g2/reviews/`, with "
    "original hashes and historical scope recorded, and byte-exact receipts under `checks/`. "
    "Local links must then be rewritten; omitted historical captures must remain clearly "
    "labelled local audit rather than broken published links. The plan also selects bounded "
    "renderer scripts and omits repeated raw arrays and caches. No copies are made by this draft update. "
)


def require(condition, message):
    """Stop before publishing a successful manifest when a contract is not met."""
    if not condition:
        raise ValueError(message)


def digest(path):
    """Hash a file in bounded memory without interpreting its contents."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(value):
    """Require a normalized repository-relative path with no traversal."""
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts,
            f"Unsafe relative path: {value!r}")
    require(path.as_posix() == value and "\n" not in value and "\r" not in value,
            f"Noncanonical relative path: {value!r}")
    return value


def source_file(root, name):
    """Reject missing files, symlinks and escaped source paths."""
    path = root / relative(name)
    require(path.is_file() and path.resolve().is_relative_to(root), f"Missing source: {name}")
    require(not any(part.is_symlink() for part in (path, *path.parents)
                    if part.is_relative_to(root)), f"Symlink source: {name}")
    return path


def selection_rows(root, name):
    """Parse the fixed TSV without globbing or expanding the selected evidence."""
    rows = []
    for number, line in enumerate(source_file(root, name).read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        require(fields[0] in ("KEEP", "COPY", "CURATE"), f"Unknown action {name}:{number}")
        require(len(fields) == (2 if fields[0] == "KEEP" else 3), f"Malformed {name}:{number}")
        action, source = fields[:2]
        destination = source if action == "KEEP" else fields[2]
        relative(source)
        relative(destination)
        require(destination.startswith(BASE + "/"), f"Destination outside evidence scope: {destination}")
        require(action != "CURATE" or source.endswith(".md"), "Only Markdown may be curated")
        rows.append({"action": action, "source": source, "destination": destination})
    return rows


def destinations(rows):
    """Build exact source mappings and the set of selected directory ancestors."""
    mapping = {}
    files = set()
    for row in rows:
        require(row["source"] not in mapping, f"Duplicate source: {row['source']}")
        require(row["destination"] not in files, f"Duplicate destination: {row['destination']}")
        mapping[row["source"]] = row["destination"]
        files.add(row["destination"])
    require(not files & GENERATED, "Generated output collides with selected input")
    files.update(GENERATED)
    directories = {parent.as_posix() for name in files for parent in PurePosixPath(name).parents
                   if parent.as_posix().startswith(BASE)}
    return mapping, files, directories


def canonical_target(raw, source, root):
    """Resolve a reference in its original context without basename or version substitution."""
    parsed = urlsplit(raw.strip("<>"))
    if parsed.scheme or parsed.netloc:
        return None, parsed
    path = unquote(parsed.path)
    if not path:
        return source, parsed
    if path.startswith(str(root) + "/"):
        path = path[len(str(root)) + 1:]
    elif path.startswith("/"):
        return None, parsed
    elif not path.startswith(("reports/", ".codex-audit/", "spacr/", "tests/", "tools/")):
        path = posixpath.join(posixpath.dirname(source), path)
    return posixpath.normpath(path).rstrip("/"), parsed


def published_target(raw, source, root, mapping, files, directories):
    """Return only an exact selected file or a directory containing selected files."""
    canonical, parsed = canonical_target(raw, source, root)
    target = mapping.get(canonical, canonical)
    return (target if target in files or target in directories else None), canonical, parsed


def literal_reference(raw, source, root, mapping, files, directories):
    """Resolve an explicit evidence-directory prefix, retaining its exact original version."""
    first = raw.split("/", 1)[0]
    if first and first not in (".", "..") and (root / BASE / "evidence" / first).is_dir():
        raw = f"{BASE}/evidence/{raw}"
    return published_target(raw, source, root, mapping, files, directories)


def relocate(target, parsed, destination):
    """Relativize an already verified target while retaining its query and fragment."""
    path = posixpath.relpath(target, posixpath.dirname(destination))
    return urlunsplit(("", "", quote(path, safe="/._-"), parsed.query, parsed.fragment))


def update_index(text, route_summary):
    """Apply the authorized publication title, scope, footer and final result changes."""
    old_title = "# G2 validation — draft evidence index"
    old_intro = ("**DRAFT, 2026-11-23.** This indexes the retained before/after evidence selected by the root reviewer. "
        "Acceptance below concerns the captured synthetic fixtures, not product completion, every caller, or the latest "
        "unvalidated shared-source revision. The immutable before source is G1 checkpoint "
        "`9510143823dcc1b984ecc522058a1e3f15e985d8`; candidate receipts bind actual working-file hashes. "
        "Equal Git HEADs do not establish equal source bytes.")
    require(text.count(old_title) == text.count(old_intro) == 1, "Frozen index title/scope changed")
    text = text.replace(old_title, "# G2a validation — selected evidence")
    text = text.replace(old_intro, "**2026-11-23.** This indexes the retained before/after evidence selected by the root reviewer. "
        "The code source for this partial G2a handoff is `d6c4e6a9615c55f91276c0a88035a914bf2c9784`. "
        "Acceptance below concerns the captured synthetic fixtures; it does not establish behavior for unexercised callers, "
        "native platforms, locales or interaction states, or whole-G2 final integration. The immutable before source is "
        "G1 checkpoint `9510143823dcc1b984ecc522058a1e3f15e985d8`; each candidate receipt retains the actual "
        "working-file hashes of its recorded pass. Equal Git HEADs do not establish equal source bytes.")
    require(text.count(PREPARATION) == 1, "Frozen preparation footer changed")
    text = text.replace(PREPARATION, "")
    old = "This index does not remove any of the 160 STEP_A deferrals or treat static reachability as visual certification."
    require(text.count(old) == 1, "Frozen route-scope sentence changed")
    text = text.replace(old, "The final route disposition removes exactly 78 converted entries; "
        "82 STEP_A and all 44 STEP_B entries remain deferred. Static reachability is not visual certification.")
    text += ("\n\n**Final bounded route and source checks.** Root job 1688 passed all three unchanged normal "
        "route-guard tests after the 78 exact removals ([JUnit](checks/route-guard/junit.xml), "
        "[stdout](checks/route-guard/stdout.log), [resources](checks/route-guard/resource.txt)). "
        "Job 1686 passed all six unchanged source/hash-comment/raw-tabular guards "
        "([JUnit](checks/source-guards/junit.xml), [stdout](checks/source-guards/stdout.log), "
        "[resources](checks/source-guards/resource.txt)). The [route review](reviews/g2_route_review.md), "
        "[exact removed IDs](reviews/g2_removed_route_ids.json), and original "
        "[pre-removal route report](reviews/g2_routes_report.json) retain the scope and source binding. "
        "The pinned plan and remaining deferred values are unchanged. Measure categorical replay, raw/transport "
        "boundaries, external drawing branches and native/tutorial/WS dependencies remain explicit. "
        "This is partial G2a acceptance, not whole-G2 completion.\n\n"
        "Root job 1689 also passed the scoped English runtime-admission check "
        "([result](checks/english-runtime-v3/stdout.log), [resources](checks/english-runtime-v3/resource.txt)). "
        "Its receipt records exactly two normal runtime scanner calls and 21 remaining admission gaps "
        "owned by the shared style lane. This is a tools/handoff result and does not close G1 integration "
        "or imply a new scientific run.\n")
    require(route_summary["removed"] == 78 and route_summary["retained"] == 126,
            "Unexpected final route counts")
    return text


def curate(text, row, root, mapping, files, directories, source_sha, route_summary):
    """Retain observations while relocating exact selected links and labeling omitted paths."""
    changes = []
    if row["source"] == VALIDATION:
        text = update_index(text, route_summary)
        changes.append({"kind": "authorized_index_title_scope_footer_and_final_results"})
    kept_tokens = []

    def link(match):
        """Rewrite one actual Markdown link without changing its referenced pass."""
        marker, label, raw, title = match.groups()
        target, canonical, parsed = published_target(raw, row["source"], root, mapping, files, directories)
        if parsed.scheme in ("http", "https", "mailto", "data") or parsed.netloc:
            replacement = match.group(0)
        elif target is not None:
            replacement = f"{marker}[{label}]({relocate(target, parsed, row['destination'])}{title or ''})"
            changes.append({"kind": "selected_link", "original": raw, "destination": target})
        else:
            historical = canonical or raw.strip("<>")
            replacement = f"{label} (local-only historical artifact: `{historical}`)"
            changes.append({"kind": "omitted_local_link", "original": raw, "path": historical})
        token = f"\x00G2LINK{len(kept_tokens)}\x00"
        kept_tokens.append(replacement)
        return token

    def literal(match):
        """Label artifact path literals without interpreting scientific values as paths."""
        raw = match.group(1)
        if not ARTIFACT.search(raw) and not raw.startswith(("reports/graphstyle/g2/evidence/", ".codex-audit/")):
            return match.group(0)
        if "{theme}" in raw:
            references = [(theme, literal_reference(raw.replace("{theme}", theme), row["source"],
                root, mapping, files, directories)) for theme in ("light", "dark")]
            if all(reference[0] is not None for _, reference in references):
                changes.append({"kind": "selected_explicit_theme_pattern", "original": raw,
                                "destinations": [reference[0] for _, reference in references]})
                links = ", ".join(f"[{theme}]({relocate(reference[0], reference[2], row['destination'])})"
                                  for theme, reference in references)
                return f"{match.group(0)} ({links})"
            changes.append({"kind": "context_relative_pattern", "original": raw})
            return match.group(0)
        target, canonical, parsed = literal_reference(raw, row["source"], root, mapping, files, directories)
        if target is not None:
            changes.append({"kind": "selected_path_literal", "original": raw, "destination": target})
            return f"[{raw}]({relocate(target, parsed, row['destination'])})"
        explicit = raw.startswith(("reports/", ".codex-audit/", "/"))
        exists = canonical and (root / canonical).resolve().is_relative_to(root) and (root / canonical).exists()
        if not explicit and not exists:
            changes.append({"kind": "context_relative_path", "original": raw})
            return match.group(0)
        changes.append({"kind": "local_path_literal", "original": raw, "path": canonical or raw})
        return f"{match.group(0)} (local-only historical reference)"

    output = []
    fence = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            delimiter = stripped[:3]
            fence = None if fence == delimiter else delimiter if fence is None else fence
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue
        require(not re.match(r"^\s*\[[^]]+\]:", line), "Unreviewed reference-style Markdown link")
        output.append(CODE.sub(literal, LINK.sub(link, line)))
    text = "".join(output)
    for index, replacement in enumerate(kept_tokens):
        text = text.replace(f"\x00G2LINK{index}\x00", replacement)
    require("\x00" not in text, "Unresolved link placeholder")
    header = ("> Curated publication copy. Original source: `" + row["source"] + "`; original SHA256: `" +
        source_sha + "`. Original observations retain their recorded historical scope. "
        "Only selected artifacts are linked; omitted artifacts are labeled local-only and are never "
        "retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original "
        "review context, which can include local-only omitted outputs; they do not select a later artifact. "
        "The validation index identifies the final selected subsets.\n\n")
    return header + text, changes


class HTMLReferences(HTMLParser):
    """Collect actual HTML asset/navigation references without executing embedded content."""

    def __init__(self):
        """Initialize the parser and exact reference list."""
        super().__init__(convert_charrefs=True)
        self.references = []

    def handle_starttag(self, tag, attrs):
        """Record resource-bearing attributes from each opening element."""
        for key, value in attrs:
            if key in ("href", "src", "poster") and value:
                self.references.append(value)
            require(key != "srcset", "Unreviewed HTML srcset in selected evidence")


def validate_links(stage, files, directories):
    """Reject every local Markdown or HTML link outside the selected publication set."""
    records = []
    for name in sorted(files):
        if not name.endswith((".md", ".html")):
            continue
        text = (stage / name).read_text()
        if name.endswith(".html"):
            parser = HTMLReferences()
            parser.feed(text)
            parser.close()
            links = parser.references
        else:
            links = []
            fence = None
            for line in text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith(("```", "~~~")):
                    delimiter = stripped[:3]
                    fence = None if fence == delimiter else delimiter if fence is None else fence
                elif fence is None:
                    require(not re.match(r"^\s*\[[^]]+\]:", line), f"Unreviewed reference link in {name}")
                    links.extend(match.group(3) for match in LINK.finditer(line))
                    require(not re.search(r"<(?:a|img|link|script)\b", line, re.I), f"Unreviewed HTML in {name}")
        for raw in links:
            parsed = urlsplit(raw.strip("<>"))
            if parsed.scheme in ("http", "https", "mailto", "data") or parsed.netloc:
                records.append({"document": name, "kind": "embedded" if parsed.scheme == "data" else "external_unfetched"})
                continue
            require(not parsed.scheme and not parsed.path.startswith("/"), f"Absolute local link in {name}: {raw}")
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parsed.path))) if parsed.path else name
            require(target in files or target in directories, f"Unselected local link in {name}: {raw}")
            require((stage / target).exists(), f"Missing staged link target in {name}: {raw}")
            records.append({"document": name, "target": target, "kind": "selected_local",
                            "fragment": parsed.fragment, "fragment_validation": "not_requested"})
    return records


def route_receipts(root):
    """Bind the new index claim to exact IDs, retained entries and successful JUnit receipts."""
    removed = json.loads(source_file(root, REMOVED).read_text())
    plan = json.loads(source_file(root, PLAN).read_text())
    current = json.loads(source_file(root, DEFERRED).read_text())
    initial = plan["maximum_deferred"]["2"]
    require(len(removed) == len(set(removed)) == 78, "Removal list must contain exactly 78 distinct IDs")
    require(set(initial) - set(current) == set(removed) and len(current) == 126,
            "Deferred changes differ from the accepted removal list")
    require(all(initial.get(key) == value for key, value in current.items()), "Retained deferred values changed")
    require(sum(key.startswith("spacr.plot.") for key in current) == 44, "STEP_B entries changed")
    require({"spacr.measure._wound_half_closure_figure", "spacr.measure._viability_controls_figure"} <= current.keys(),
            "Measure replay holds were removed")
    receipts = {}
    for folder, count in (("route-guard", 3), ("source-guards", 6)):
        prefix = f".codex-audit/checks/{folder}"
        require(source_file(root, f"{prefix}/exit-code.txt").read_text().strip() == "0", f"Failed {folder}")
        document = ET.parse(source_file(root, f"{prefix}/junit.xml")).getroot()
        cases = document.findall(".//testcase")
        require(len(cases) == count and not document.findall(".//failure") and not document.findall(".//error")
                and not document.findall(".//skipped"), f"Unexpected JUnit result: {folder}")
        receipts[folder] = {"tests": len(cases), "exit_code": 0}
    runtime_prefix = ".codex-audit/checks/english-runtime-v3"
    require(source_file(root, f"{runtime_prefix}/exit-code.txt").read_text().strip() == "0", "Failed runtime admission")
    runtime = json.loads(source_file(root, f"{runtime_prefix}/stdout.log").read_text())
    require(runtime.get("passed") is True and runtime.get("runtime_scanner_calls") == 2
            and runtime.get("remaining_admission_gaps") == 21, "Runtime admission result changed")
    receipts["english-runtime-v3"] = {"passed": True, "runtime_scanner_calls": 2,
                                     "remaining_admission_gaps": 21, "exit_code": 0}
    return {"removed": 78, "retained": 126, "step_a_retained": 82, "step_b_retained": 44, "checks": receipts}


def write_bytes(path, value):
    """Create a staged file exclusively so existing evidence can never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def main():
    """Copy the fixed selection into a fresh stage, validate links and write final hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    stage = args.output_root.absolute()
    require(stage == stage.resolve() and not stage.exists(), "Output root must be a new real directory")
    require(stage != root and not root.is_relative_to(stage), "Stage cannot contain the source root")
    rows = selection_rows(root, SELECTION)
    require(len(rows) == 755 and sum(row["action"] == "KEEP" for row in rows) == 722,
            "Fixed selection counts changed")
    rows += selection_rows(root, EXTRAS)
    mapping, files, directories = destinations(rows)
    helper = Path(__file__).resolve()
    require(helper == source_file(root, ".codex-audit/curate_g2_evidence.py"), "Run the source-bound curator")
    inputs = sorted({row["source"] for row in rows} | set(FIXED) | {EXTRAS, DEFERRED})
    before = {name: digest(source_file(root, name)) for name in inputs}
    for name, expected in FIXED.items():
        require(before[name] == expected, f"Frozen input changed: {name}")
    route_summary = route_receipts(root)
    stage.mkdir(parents=True, exist_ok=False)
    artifact_records = []
    for row in rows:
        source = source_file(root, row["source"])
        target = stage / row["destination"]
        curated = row["action"] == "CURATE" or row["source"] == VALIDATION
        changes = []
        if curated:
            text, changes = curate(source.read_text(), row, root, mapping, files, directories,
                                   before[row["source"]], route_summary)
            write_bytes(target, text.encode())
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
            os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        result_sha = digest(target)
        require(curated or result_sha == before[row["source"]], f"Copy changed bytes: {row['source']}")
        artifact_records.append({**row, "effective_action": "CURATE" if curated else row["action"],
            "source_sha256": before[row["source"]], "artifact_sha256": result_sha,
            "size_bytes": target.stat().st_size, "curation_changes": changes})
    link_records = validate_links(stage, files - GENERATED, directories)
    after = {name: digest(source_file(root, name)) for name in inputs}
    require(after == before, "An input changed during curation")
    provenance = {"schema": 1, "source_root": str(root), "selection_sha256": before[SELECTION],
        "helper_sha256": before[".codex-audit/curate_g2_evidence.py"],
        "inputs_unchanged": True, "no_producers_executed": True, "route_disposition": route_summary,
        "artifacts": artifact_records, "link_validation": {"passed": True, "records": link_records,
        "scope": "Markdown/HTML local file and directory targets must be selected; external URLs and fragments are not fetched or certified"},
        "limitations": ["Original JSON/text receipts may reference unselected historical artifacts and retain their original bytes.",
            "Raw array duplicates, caches and superseded outputs remain local; this is a review selection, not a complete replay dataset.",
            "No rendering, scientific computation, tests, browser session or source modification was performed by this curator."]}
    write_bytes(stage / f"{BASE}/publication-provenance.json", (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
    write_bytes(stage / f"{BASE}/publication-inputs.sha256", "".join(f"{before[name]}  {name}\n" for name in inputs).encode())
    write_bytes(stage / f"{BASE}/publication-files.txt", "".join(f"{name}\n" for name in sorted(files)).encode())
    manifest = f"{BASE}/publication.sha256"
    write_bytes(stage / manifest, "".join(f"{digest(stage / name)}  {name}\n" for name in sorted(files - {manifest})).encode())
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    require(actual == files, "Staged publication set differs from the exact selection")
    require(all(digest(source_file(root, name)) == value for name, value in before.items()), "Input changed before completion")
    print(json.dumps({"output_root": str(stage), "files": len(files), "curated": sum(
        row["effective_action"] == "CURATE" for row in artifact_records), "local_links": sum(
        row["kind"] == "selected_local" for row in link_records), "manifest_sha256": digest(stage / manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
