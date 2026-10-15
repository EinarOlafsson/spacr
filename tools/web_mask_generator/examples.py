"""Bounded public HF examples selected only from an operator-reviewed test list.

This does not infer that a training image is a held-out test image. The operator
must review family/source/split/rights and pin actual public bytes first.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import re
import secrets
import tempfile
from urllib.parse import quote

from spacr.qt.i18n import tr

from .inputs import DEFAULT_MAX_FILE_BYTES, describe_upload
from .registry import model_families


class ExampleStore:
    def __init__(self, records, *, job_parent=None):
        if not isinstance(records, list) or len(records) > 10000:
            raise ValueError(tr("Approved examples must be a bounded list."))
        families = {entry["family"] for entry in model_families()}
        self.records = []
        self.job_parent = job_parent
        allowed = {"family", "repo_id", "revision", "filename", "sha256", "bytes", "credit", "split"}
        for record in records:
            if not isinstance(record, dict) or set(record) != allowed:
                raise ValueError(tr("An approved example has incomplete or unsupported fields."))
            record = dict(record)
            path = PurePosixPath(str(record["filename"]))
            valid = (isinstance(record["family"], str) and record["family"] in families and record["split"] == "test"
                     and isinstance(record["repo_id"], str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", record["repo_id"])
                     and all(part not in (".", "..") for part in record["repo_id"].split("/"))
                     and isinstance(record["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", record["revision"])
                     and isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                     and isinstance(record["bytes"], int) and not isinstance(record["bytes"], bool) and 0 < record["bytes"] <= DEFAULT_MAX_FILE_BYTES
                     and isinstance(record["credit"], str) and record["credit"].strip()
                     and isinstance(record["filename"], str) and str(path) == record["filename"]
                     and path.name not in ("", ".", "..") and not path.is_absolute() and ".." not in path.parts and "\\" not in record["filename"])
            if not valid:
                raise ValueError(tr("An approved example needs a valid family, immutable HF dataset revision, test split, size, SHA-256 and credit."))
            self.records.append(record)
        self.records = tuple(self.records)

    def available(self, family):
        return any(record["family"] == family for record in self.records)

    @contextmanager
    def random_image(self, family):
        """Fetch one public reviewed test image; never accept a visitor URL/path."""
        import requests

        choices = [record for record in self.records if record["family"] == family]
        if not choices:
            raise ValueError(tr("No approved test image is configured for this model class."))
        record = dict(secrets.choice(choices))
        base = "https://huggingface.co/datasets/" + record["repo_id"]
        tail = quote(record["filename"], safe="/")
        url = base + "/resolve/" + record["revision"] + "/" + tail
        with tempfile.TemporaryDirectory(prefix="spacr-web-example-", dir=self.job_parent) as directory:
            image = Path(directory) / PurePosixPath(record["filename"]).name
            digest = hashlib.sha256()
            size = 0
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    with image.open("xb") as outgoing:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            size += len(chunk)
                            if size > record["bytes"]:
                                raise ValueError(tr("The public example exceeds its approved byte count."))
                            digest.update(chunk)
                            outgoing.write(chunk)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(tr("The public example does not match its approved size and SHA-256."))
            description = describe_upload(image)
            record["source_url"] = base + "/blob/" + record["revision"] + "/" + tail
            yield {"image": image, "source": record, "description": description}
