"""spaCR model upload endpoint — a Hugging Face Space that owns the token.

WHY THIS EXISTS. spaCR's Model Zoo has an Add button that offers to publish a
model to a shared collection. Doing that from the user's machine would mean
one of two things:

  1. shipping a write token inside spaCR, which anyone who installs the
     package can read out of it, and which can DELETE everything in the
     repository it can write to; or
  2. requiring every contributor to have their own Hugging Face account.

Neither is what we want. So the token lives HERE, in this Space's secrets,
and never leaves the server. spaCR POSTs a file and its scorecard; this
validates the request and commits it. A contributor needs no account.

DEPLOYING
  1. Create a Space with the Gradio SDK. On a free account that means
     ZeroGPU, which is fine -- see the _zero_gpu_probe note below. CPU basic
     needs a PRO subscription and is not required.
  2. Add a secret named HF_TOKEN: a FINE-GRAINED token with write access to
     UPLOAD_REPO and nothing else. Not an account-wide write token -- if this
     Space is ever compromised, the blast radius should be one repository.
  3. Add a secret named UPLOAD_REPO if it differs from the default below.
  4. Push these three files to the Space.

MODERATION. Everything lands under `staging/` and nothing is promoted
automatically. A public write endpoint WILL eventually receive something
unwanted -- junk, something copyrighted, or something malicious -- and the
owner of the account carries that. Review staging before promoting, and keep
the review a human step.
"""
import hashlib
import json
import os
import re
import time

import gradio as gr
from huggingface_hub import HfApi

UPLOAD_REPO = os.environ.get("UPLOAD_REPO", "einarolafsson/user-models")
TOKEN = os.environ.get("HF_TOKEN")
MAX_BYTES = 2_500_000_000          # a cpsam checkpoint is ~1.2 GB
ALLOWED_SUFFIXES = (".pth", ".pt", ".safetensors", ".CP_model")
RATE = {}                          # ip -> [timestamps]
RATE_LIMIT, RATE_WINDOW = 3, 3600  # uploads per IP per hour

api = HfApi(token=TOKEN)

# ZeroGPU refuses to start a Space with no @spaces.GPU function ("No
# @spaces.GPU function detected during startup"), and a free-tier Gradio Space
# is ZeroGPU unless the account has PRO. This endpoint needs no GPU at all --
# it hashes a file and makes an HTTPS call -- so this exists purely to satisfy
# that check. It is never called, and it costs nothing: ZeroGPU attaches a GPU
# only while a decorated function is actually running.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zero_gpu_probe():
        """Present so ZeroGPU will start this Space. Deliberately unused."""
        return "ok"
except Exception:                                            # noqa: BLE001
    pass


def _slug(text):
    out = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return out or "model"


def _rate_ok(who):
    now = time.time()
    seen = [t for t in RATE.get(who, []) if now - t < RATE_WINDOW]
    RATE[who] = seen
    if len(seen) >= RATE_LIMIT:
        return False
    seen.append(now)
    return True


def upload(file, name, kind, trained_on, scorecard_json, contact, request: gr.Request):
    """Validate, then commit to staging. Returns a message for the client."""
    if not TOKEN:
        return "error: this endpoint is not configured (no token)."
    who = getattr(request, "client", None)
    who = getattr(who, "host", "unknown") if who else "unknown"
    if not _rate_ok(who):
        return f"error: rate limit is {RATE_LIMIT} uploads per hour."
    if file is None:
        return "error: no file."
    path = file if isinstance(file, str) else file.name
    if not str(path).endswith(ALLOWED_SUFFIXES):
        return f"error: only {', '.join(ALLOWED_SUFFIXES)} are accepted."
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        return f"error: {size} bytes is over the {MAX_BYTES} limit."
    if not str(name or "").strip():
        return "error: a model name is required."

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    sha = digest.hexdigest()

    try:
        card = json.loads(scorecard_json or "{}")
    except Exception:
        return "error: the scorecard is not valid JSON."

    folder = f"staging/{_slug(name)}-{sha[:8]}"
    meta = dict(name=name, kind=kind or "cellpose", trained_on=trained_on,
                sha256=sha, size_bytes=size, contact=contact or "",
                uploaded=time.strftime("%Y-%m-%d %H:%M:%S"),
                uploaded_by_ip_hash=hashlib.sha256(who.encode()).hexdigest()[:16],
                scorecard=card, reviewed=False)
    try:
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=f"{folder}/{os.path.basename(path)}",
                        repo_id=UPLOAD_REPO, repo_type="model")
        api.upload_file(path_or_fileobj=json.dumps(meta, indent=2).encode(),
                        path_in_repo=f"{folder}/submission.json",
                        repo_id=UPLOAD_REPO, repo_type="model")
    except Exception as exc:
        return f"error: {exc}"
    return (f"ok: https://huggingface.co/{UPLOAD_REPO}/tree/main/{folder} "
            f"(sha256 {sha}) — held for review before it appears in the zoo.")


demo = gr.Interface(
    fn=upload,
    inputs=[gr.File(label="model file"), gr.Textbox(label="name"),
            gr.Textbox(label="kind", value="cellpose"),
            gr.Textbox(label="trained on"),
            gr.Textbox(label="scorecard (JSON)", value="{}"),
            gr.Textbox(label="contact (optional)")],
    outputs=gr.Textbox(label="result"),
    title="spaCR model upload",
    description=("Publishes a model to the shared spaCR collection. "
                 "Submissions are held for review before they appear in the "
                 "Model Zoo."),
)

# Launched at import time and blocking. Two things bite here: the Space image
# force-installs its OWN gradio, so requirements.txt must not pin one (a pin
# fails the build with ResolutionImpossible); and SSR mode returns from
# launch() and lets the process exit, which shows up as RUNTIME_ERROR with
# "Stopping Node.js server" in the log.
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False,
            prevent_thread_lock=True)

# Block explicitly. launch() returns on this Gradio, and a Space whose process
# exits right after "Running on local URL" is reported as RUNTIME_ERROR.
try:
    demo.block_thread()
except Exception:                                            # noqa: BLE001
    import threading
    threading.Event().wait()
