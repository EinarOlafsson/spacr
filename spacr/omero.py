"""OMERO import / export — a spaCR project out of a server, and results back in.

A large share of imaging labs never see a filesystem. Their images live in an
OMERO server, they are addressed by *id*, and the only way anyone looks at a
result is the right-hand panel in OMERO.web. spaCR, by contrast, is built
around a flat folder of Yokogawa-named TIFFs. This module is the bridge, in
both directions, and it is deliberately two directions rather than one: a
one-way copy out of OMERO leaves the answers stranded in a folder nobody on
the microscope side will ever open.

What it does
------------

**Import.** :func:`import_dataset` and :func:`import_plate` take an OMERO
Dataset or Plate id and write a spaCR source folder:

    plate1_A01_T0001F001L01A01Z01C01.tif
    plate1_A01_T0001F001L01A01Z01C02.tif
    plate1_A01_T0001F002L01A01Z01C01.tif
    ...

That is :func:`spacr.convert.target_name` verbatim — the exact shape
``spacr.utils._get_regex('cellvoyager', 'tif')`` parses — so the output folder
is handed straight to Mask/Measure with ``metadata_type='cellvoyager'`` and
nothing else. **No new layout is invented here.** The one thing that must
survive the trip is the plate geometry, because spaCR keys every measurement,
every plot and every well-level statistic off ``plateID`` / ``rowID`` /
``columnID`` / ``fieldID``; :func:`well_position` is where OMERO's 0-based
``(row, column)`` becomes spaCR's ``('r1', 'c1')`` and the well name ``A01``.

**Export.** :func:`export_map_annotation`, :func:`export_file_annotation` and
:func:`export_tag_annotation` push spaCR's own output back onto the OMERO
objects it came from: a key/value MapAnnotation per image or per well (what
OMERO.web actually renders), a FileAnnotation for the results CSV or the
figure PDF, and a TagAnnotation for a categorical verdict.

Why the logic is not in the adapter
-----------------------------------
Everything that is easy to get wrong here is *arithmetic and string
formatting*, not RPC: the row/column mapping, the filename, the id parsing,
the float formatting, the decision to replace rather than append. All of it
lives in pure functions over plain data — :func:`well_position`,
:func:`plane_filename`, :func:`parse_object_ref`, :func:`format_map_value`,
:func:`measurement_pairs`, :func:`plan_annotation` — and the ``BlitzGateway``
calls are confined to a thin layer that only walks containers and moves
bytes. That is why this module has a real test suite on a machine with no
OMERO server and no ``omero-py`` installed: the fake gateway in
``tests/test_omero.py`` implements about a dozen methods, and everything worth
testing is reachable through it.

Every adapter function takes ``gateway`` as its first argument. There is no
module-level connection and no implicit session — a caller that wants one
calls :func:`connect`, and a test passes its own object.

The well mapping, stated
------------------------
OMERO's ``WellWrapper.getRow()`` and ``.getColumn()`` are **0-based**.
:func:`well_position` adds one and hands the result to
:func:`spacr.schema.well_id`, which is spaCR's own definition of a well name.
Consequences worth writing down because they are the ones that get fumbled:

* row 0 -> ``A``, row 7 -> ``H`` (a 96-well plate), row 15 -> ``P`` (384).
* row 25 -> ``Z``.
* **row 26 -> ``AA``**, not ``[`` (which is what ``chr(65 + 26)`` gives) and
  not an ``IndexError`` (which is what ``string.ascii_uppercase[26]`` gives).
  This is bijective base 26, and it is not a hypothetical: a 1536-well plate
  has 32 rows and runs ``A``..``Z``, ``AA``..``AF``. Column 47 -> ``48``, for
  the same plate, so nothing here caps the column at 24 either.
* the column is zero padded to two digits (``A01``, never ``A1``) because
  spaCR's strict Yokogawa regex is ``[A-Z]\\d{2}``.

Sizes and units
---------------
``ImageWrapper.getPixelSizeX()`` returns a *length object* with a unit, not a
number of micrometres. This module never assumes µm: :func:`pixel_size_from`
carries the value and the unit symbol together in a :class:`PixelSize` and
writes both into the import sidecar. A pixel size recorded in nm that is read
back as µm is a 1000x error in every area in the database and it is silent.

What is written alongside the images
------------------------------------
Two sidecars, in the destination folder, both of which spaCR itself is happy
to ignore:

``omero_import.csv``
    one row per plane: the target filename, the OMERO image id and name, the
    spaCR keys (``plateID``/``rowID``/``columnID``/``fieldID``/``prc``), the
    channel index and its OMERO name, ``z``/``t``, the pixel size and its
    unit, and where the well came from (see ``well_source`` below).
``omero_import.json``
    one object per import: server host and port, container kind/id/name,
    plate token, counts, channel names, pixel size, the spaCR version and a
    UTC timestamp.

**Neither ever contains a password or a session key.** That is asserted by a
test, along with the fact that :class:`OmeroConnection` will not print one
either.

Not downloading a 100 GB plate to answer "what is in it"
--------------------------------------------------------
:func:`inspect_container` reports the image count, the wells, the dimensions
and the channel names **without ever calling ``getPlane``** — a fact the test
suite pins by counting calls on the fake gateway. The importers additionally
take ``limit`` (stop after N OMERO images) and ``dry_run`` (build the complete
plan, list every filename that would be written, touch no pixels).

Replace or append, and what is never done
-----------------------------------------
Annotations are written under a spaCR-owned namespace
(:data:`NAMESPACE_ROOT`), which is what makes a second run able to find its
own previous output instead of piling up a fifth copy of the same key/value
table.

**The default is REPLACE, implemented as update-in-place, and it is the safe
option rather than a compromise:** an existing MapAnnotation whose namespace
is exactly spaCR's has its value overwritten. Nothing is created, nothing is
unlinked, nothing is deleted, and no annotation outside spaCR's namespaces is
so much as read for its value. :data:`APPEND` is available and explicit for
anyone who wants a history of runs on the object.

Three consequences are stated rather than hidden:

* **This module never deletes anything.** There is no call to
  ``deleteObjects`` in it, under any mode, for any object. If a previous
  ``APPEND`` run left three copies, ``REPLACE`` updates the oldest and
  *reports* the rest in :attr:`AnnotationResult.duplicates` — removing them is
  a decision for a human with the OMERO.web UI, not for an importer.
* **FileAnnotations always append.** The bytes of an OriginalFile cannot be
  rewritten in place through the gateway, and the alternative — delete the old
  attachment — is exactly the destructive behaviour ruled out above. The
  namespace plus the timestamp in the description identify the newest.
* **TagAnnotations are never edited.** In OMERO a tag is a *shared* object;
  renaming the tag linked to this image renames it on every other object in
  the group that carries it. So :func:`export_tag_annotation` links a tag when
  the verdict is new, does nothing when the identical verdict is already
  linked, and reports the previous tag when the verdict has *changed* — it
  never unlinks and never renames.

Connection settings and secrets
-------------------------------
:func:`connection_settings` reads arguments first and the environment second:
``OMERO_HOST``, ``OMERO_PORT``, ``OMERO_USER``, ``OMERO_PASSWORD`` (or
``OMERO_PASS``), ``OMERO_SESSION_KEY``, ``OMERO_GROUP``, ``OMERO_SECURE``.
:class:`OmeroConnection` is frozen and redacts both secrets from ``repr()``,
``str()`` and :meth:`OmeroConnection.redacted`, and no log record in this
module carries one.

The optional dependency
-----------------------
``omero-py`` is an extra: ``pip install "spacr[omero]"``. It is deliberately
**not** part of ``spacr[all]``, because it depends on ``zeroc-ice``, a
compiled C++ Ice runtime whose wheels lag Python releases and which otherwise
needs a C++ toolchain (see the comment beside the extra in ``setup.py``).

So ``import spacr.omero`` must work without it, and it does: the import is
function-local, behind :func:`require_omero`, and a missing install produces
one actionable sentence naming ``pip install "spacr[omero]"`` rather than a
``ModuleNotFoundError`` six frames deep inside Ice's own import machinery.
This follows :data:`spacr.qt._QT_MISSING_MESSAGE` and
:func:`spacr.anndata_export.require_anndata`.

Two details of that guard are worth knowing:

* **A missing ``Ice`` counts as a missing omero extra.** A half-built
  ``zeroc-ice`` is the single most likely way this fails in the field, and
  ``No module named 'Ice'`` mentions neither OMERO nor spaCR.
  :func:`missing_omero_message` says what happened.
* **This module is called ``spacr/omero.py`` and does not shadow the
  third-party ``omero`` package.** Absolute imports have been the default
  since Python 3, so a module inside the ``spacr`` package that asks for
  ``omero`` gets the top-level distribution, not its own sibling. That is
  verified rather than assumed — see
  ``tests/test_omero.py::test_spacr_omero_does_not_shadow_the_third_party_package``,
  which puts a decoy ``omero`` package on ``sys.path`` and checks which one
  arrives — because a self-import here would be silent and would look like a
  broken OMERO install.

The import is written as
``importlib.import_module(OMERO_GATEWAY_MODULE)`` rather than as a literal
``import omero.gateway`` for that same reason: the one place the name is
resolved is a named constant next to the guard that checks what came back,
instead of a bare statement in a file of the same name. The cost is that
``tests/test_declared_dependencies_match_imports.py`` — which reads import
*statements* out of the AST — cannot see it, exactly as it cannot see
``umap``. That blind spot is pinned from the other side by
``tests/test_omero.py::test_omero_py_is_reached_through_a_string_literal_and_must_not_be_removed``,
so the next dependency census that reads "omero-py: unused" finds the answer
instead of deleting the extra.
"""
from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import mimetypes
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)

from . import schema
from .convert import target_name
from .tiff_io import write_tiff
from .version import get_version

__all__ = [
    "ACTION_CREATE",
    "ACTION_UNCHANGED",
    "ACTION_UPDATE",
    "ANNOTATION_MODES",
    "APPEND",
    "AnnotationPlan",
    "AnnotationResult",
    "ContainerListing",
    "DEFAULT_PORT",
    "ENV_VARS",
    "ExportResult",
    "FLOAT_FORMAT",
    "ICE_MODULES",
    "ImageInfo",
    "ImportResult",
    "MAP_CSV_COLUMNS",
    "MAX_KEY_CHARS",
    "MAX_MAP_PAIRS",
    "MAX_VALUE_CHARS",
    "MISSING_TEXT",
    "NAMESPACE_ROOT",
    "NAMESPACE_VERSION",
    "NAN_DROP",
    "NAN_KEEP",
    "NAN_POLICIES",
    "NS_FILE",
    "NS_MEASUREMENTS",
    "NS_TAG",
    "NS_WELL_SUMMARY",
    "OBJECT_TYPES",
    "OMERO_EXTRA",
    "OMERO_EXTRA_MODULES",
    "OMERO_GATEWAY_MODULE",
    "OMERO_MISSING_MESSAGE",
    "OmeroConnection",
    "OmeroConnectionError",
    "OmeroContainerError",
    "OmeroError",
    "OmeroExtraMissing",
    "OmeroIdError",
    "OmeroRef",
    "OmeroWellError",
    "PRIORITY_KEYS",
    "PixelSize",
    "PlanePlan",
    "REPLACE",
    "SECRET_PLACEHOLDER",
    "SIDECAR_CSV",
    "SIDECAR_JSON",
    "SPACR_NAMESPACES",
    "TRUNCATION_KEY",
    "WellPosition",
    "connect",
    "connection_settings",
    "export_file_annotation",
    "export_map_annotation",
    "export_plate_summaries",
    "export_tag_annotation",
    "format_map_value",
    "have_omero",
    "import_container",
    "import_dataset",
    "import_plate",
    "inspect_container",
    "is_missing",
    "is_spacr_namespace",
    "list_spacr_annotations",
    "measurement_pairs",
    "missing_omero_message",
    "omero_indices",
    "parse_object_id",
    "parse_object_ref",
    "pixel_size_from",
    "plan_annotation",
    "plan_tag",
    "plane_filename",
    "plate_token",
    "require_omero",
    "summarise_rows",
    "well_from_image_name",
    "well_position",
    "well_summary_pairs",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------

#: The ``setup.py`` extra that provides ``omero-py``.
OMERO_EXTRA = "omero"

#: The module actually imported. Held as a constant, and reached through
#: :func:`importlib.import_module`, because this file is itself called
#: ``omero.py``: keeping the one place the name is resolved next to the guard
#: that checks the result is worth more than the brevity of a bare statement.
OMERO_GATEWAY_MODULE = "omero.gateway"

#: Modules that only ``pip install "spacr[omero]"`` brings in. ``omero`` and
#: ``omero_version`` come from ``omero-py``; ``Ice``, ``IcePy`` and
#: ``Glacier2`` come from ``zeroc-ice`` underneath it, and are here because a
#: half-built Ice is the single most likely way this fails on a real machine.
OMERO_EXTRA_MODULES = frozenset(
    {"omero", "omero_version", "Ice", "IcePy", "Glacier2"})

#: The subset of :data:`OMERO_EXTRA_MODULES` that belongs to ``zeroc-ice``
#: rather than to ``omero-py``, and therefore earns the extra paragraph in
#: :func:`missing_omero_message`.
ICE_MODULES = frozenset({"Ice", "IcePy", "Glacier2"})

OMERO_MISSING_MESSAGE = """\
Talking to an OMERO server needs the optional `omero` extra, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[omero]"

It is deliberately not part of `spacr[all]`: omero-py depends on zeroc-ice, a
compiled C++ runtime, so it is only installed when you ask for it.\
"""

#: Appended to :data:`OMERO_MISSING_MESSAGE` when the module that could not be
#: imported belongs to Ice rather than to omero-py — that failure looks like a
#: bug and is almost always an incomplete build.
ICE_NOTE = """

`{module}` comes from zeroc-ice, not from omero-py. If the install above fails
while building it, you need a C++ toolchain and the Ice development headers,
or a Python version zeroc-ice publishes a wheel for.\
"""


class OmeroExtraMissing(ImportError):
    """``omero-py`` (or the Ice runtime under it) is not installed.

    An :class:`ImportError` subclass, so a caller already guarding with
    ``except ImportError`` keeps working and the actionable message — not a
    traceback through Ice's import machinery — is what reaches the user.
    """


def missing_omero_message(module: str) -> str:
    """Return the full "install the extra" text naming ``module``.

    :param module: the top-level module that could not be imported, e.g.
        ``'omero'`` or ``'Ice'``.
    :returns: :data:`OMERO_MISSING_MESSAGE` formatted for ``module``, with
        :data:`ICE_NOTE` appended when ``module`` belongs to zeroc-ice.
    """
    text = OMERO_MISSING_MESSAGE.format(module=module)
    if module in ICE_MODULES:
        text += ICE_NOTE.format(module=module)
    return text


def _missing_omero_extra(exc: ImportError) -> Optional[str]:
    """Identify the omero-extra module whose absence raised ``exc``.

    Only failures naming a module from :data:`OMERO_EXTRA_MODULES` count:
    anything else is a genuine bug and must keep its traceback rather than be
    reported as a missing install. Mirrors
    :func:`spacr.qt._missing_qt_extra`, including its two-step fallback.

    :param exc: the ``ImportError`` raised while importing the gateway.
    :returns: the top-level module name to name in the hint, or ``None`` when
        ``exc`` has nothing to do with the extra.
    """
    # ModuleNotFoundError sets `.name` to the module that was not found; a
    # failed `from omero.gateway import BlitzGateway` sets it to the submodule.
    root = (getattr(exc, "name", None) or "").split(".", 1)[0]
    if root in OMERO_EXTRA_MODULES:
        return root
    # Import hooks and hand-raised ImportErrors leave `.name` unset, so fall
    # back to the message text before giving up on the friendly path.
    text = str(exc)
    for module in sorted(OMERO_EXTRA_MODULES):
        if re.search(rf"\b{re.escape(module)}\b", text):
            return module
    return None


def require_omero() -> Any:
    """Import and return :mod:`omero.gateway`, or raise a message worth reading.

    :returns: the imported ``omero.gateway`` module, which is where
        ``BlitzGateway``, ``MapAnnotationWrapper`` and ``TagAnnotationWrapper``
        live.
    :raises OmeroExtraMissing: when ``omero-py`` (or the Ice runtime beneath
        it) is not installed. The message names
        ``pip install "spacr[omero]"``.
    :raises ImportError: unchanged, when the failure is a real bug inside a
        module that *is* installed.
    """
    try:
        gateway = importlib.import_module(OMERO_GATEWAY_MODULE)
    except ImportError as exc:
        module = _missing_omero_extra(exc)
        if module is None:
            raise
        raise OmeroExtraMissing(missing_omero_message(module)) from exc
    # A silent self-import would look exactly like a broken OMERO install, so
    # it is checked rather than argued about. It cannot happen through normal
    # absolute-import resolution; it can happen if something has put this
    # package's own directory on sys.path ahead of site-packages.
    origin = getattr(gateway, "__file__", "") or ""
    if os.path.realpath(origin) == os.path.realpath(__file__):
        raise OmeroExtraMissing(
            f"{OMERO_GATEWAY_MODULE!r} resolved to spacr's own "
            f"{__file__!r}, not to omero-py. Something has put spaCR's "
            f"package directory on sys.path ahead of site-packages; remove "
            f"it, then " + missing_omero_message("omero"))
    return gateway


def have_omero() -> bool:
    """Report whether the ``omero`` extra can be imported, without raising.

    :returns: ``True`` when :func:`require_omero` would succeed.
    """
    try:
        require_omero()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OmeroError(ValueError):
    """Base class for every refusal this module makes.

    A :class:`ValueError` rather than a bespoke hierarchy root: every one of
    these is "the input does not describe something I can act on", and a
    caller that already handles ``ValueError`` around a conversion step keeps
    working.
    """


class OmeroIdError(OmeroError):
    """An object id could not be read, or names the wrong kind of object."""


class OmeroConnectionError(OmeroError):
    """Connection settings are incomplete, or the server refused the login."""


class OmeroContainerError(OmeroError):
    """The id resolved to nothing, or to a container with nothing usable in it."""


class OmeroWellError(OmeroError):
    """A well's row/column indices are not a position on a plate."""


# ---------------------------------------------------------------------------
# The namespace
# ---------------------------------------------------------------------------

#: The root of every namespace spaCR writes.
#:
#: OMERO namespaces are conventionally host-like strings (``openmicroscopy.org
#: /omero/bulk_annotations``), and the useful property of one is that a
#: curious user can paste it into a browser and land somewhere that explains
#: what wrote it. spaCR's project URL does exactly that, which is worth more
#: than a prettier invented domain that resolves to nothing.
NAMESPACE_ROOT = "github.com/EinarOlafsson/spacr"

#: Bumped only if the *meaning* of a spaCR annotation changes, so that a run
#: of a newer spaCR does not silently overwrite an older run's annotation with
#: differently-defined keys. It is part of every namespace below.
NAMESPACE_VERSION = "1"

#: Per-image measurements (a MapAnnotation on an Image).
NS_MEASUREMENTS = f"{NAMESPACE_ROOT}/{NAMESPACE_VERSION}/measurements"
#: Per-well summary (a MapAnnotation on a Well).
NS_WELL_SUMMARY = f"{NAMESPACE_ROOT}/{NAMESPACE_VERSION}/well-summary"
#: Attached artefacts — a results CSV, a figure PDF (a FileAnnotation).
NS_FILE = f"{NAMESPACE_ROOT}/{NAMESPACE_VERSION}/file"
#: A categorical verdict: hit / not hit, QC pass / fail (a TagAnnotation).
NS_TAG = f"{NAMESPACE_ROOT}/{NAMESPACE_VERSION}/verdict"

#: Every namespace this module will write to. Anything outside this set is
#: read-only as far as spaCR is concerned, and is never even inspected for its
#: value.
SPACR_NAMESPACES = frozenset(
    {NS_MEASUREMENTS, NS_WELL_SUMMARY, NS_FILE, NS_TAG})


def is_spacr_namespace(namespace: Optional[str]) -> bool:
    """Report whether ``namespace`` is one spaCR owns and may write to.

    The test is exact equality against :data:`SPACR_NAMESPACES`, not a prefix
    match: a prefix match would claim ``github.com/EinarOlafsson/spacr-fork/
    1/measurements`` as spaCR's own and overwrite somebody else's annotation.

    :param namespace: an annotation namespace, possibly ``None`` (OMERO's
        default namespace, which spaCR never writes to).
    :returns: ``True`` when spaCR owns it.
    """
    return namespace in SPACR_NAMESPACES


# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------

#: OMERO's default SSL port.
DEFAULT_PORT = 4064

#: What is printed instead of a secret.
SECRET_PLACEHOLDER = "***"

#: Setting -> the environment variables consulted for it, in priority order.
#: ``OMERO_PASS`` is included because that is the spelling the OMERO CLI and
#: the ezomero ecosystem use; ``OMERO_PASSWORD`` wins when both are set.
ENV_VARS: Dict[str, Tuple[str, ...]] = {
    "host": ("OMERO_HOST",),
    "port": ("OMERO_PORT",),
    "username": ("OMERO_USER", "OMERO_USERNAME"),
    "password": ("OMERO_PASSWORD", "OMERO_PASS"),
    "session_key": ("OMERO_SESSION_KEY", "OMERO_SESSIONKEY"),
    "group": ("OMERO_GROUP",),
    "secure": ("OMERO_SECURE",),
}

_TRUE_TEXT = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE_TEXT = frozenset({"0", "false", "no", "off", "n", "f"})


@dataclass(frozen=True, repr=False)
class OmeroConnection:
    """Everything needed to open a session, with the secrets kept out of sight.

    Frozen, so a settings object cannot be mutated halfway through a run, and
    ``repr``-suppressed, because the single most common way a password reaches
    a log file, a crash report or a notebook cell is somebody printing the
    object that holds it. :meth:`redacted` is the form that is safe to write
    down.

    :param host: server hostname. Required.
    :param port: server port, default :data:`DEFAULT_PORT`.
    :param username: OMERO user name. Required for password auth, optional
        with a session key.
    :param password: the password. Never printed.
    :param session_key: an existing session uuid, used instead of a password.
        Never printed.
    :param secure: whether to keep the connection encrypted after login.
        Default ``True``.
    :param group: optional OMERO group to switch into after connecting.
    """

    host: str
    port: int = DEFAULT_PORT
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    session_key: Optional[str] = field(default=None, repr=False)
    secure: bool = True
    group: Optional[str] = None

    @property
    def auth_mode(self) -> str:
        """Return ``'session'`` or ``'password'`` — which credential is used.

        A session key wins when both are present, because that is what
        ``BlitzGateway`` itself does with them.
        """
        return "session" if self.session_key else "password"

    def redacted(self) -> Dict[str, Any]:
        """Return a plain dict of the settings, safe to log or serialise.

        Both secrets are replaced by :data:`SECRET_PLACEHOLDER` when present
        and by ``None`` when absent, so the shape of the record does not
        itself leak whether a password was supplied... it does say *which*
        credential was used, which is diagnostic and not a secret.

        :returns: a JSON-serialisable dict with no credential in it.
        """
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "secure": self.secure,
            "group": self.group,
            "auth_mode": self.auth_mode,
            "password": SECRET_PLACEHOLDER if self.password else None,
            "session_key": SECRET_PLACEHOLDER if self.session_key else None,
        }

    def describe(self) -> str:
        """Return a one-line human description, with no secret in it.

        :returns: e.g. ``omero.example.org:4064 as jdoe (password, secure)``.
        """
        who = self.username or "<session>"
        bits = [self.auth_mode, "secure" if self.secure else "INSECURE"]
        if self.group:
            bits.append(f"group={self.group}")
        return f"{self.host}:{self.port} as {who} ({', '.join(bits)})"

    def __repr__(self) -> str:
        """Return a repr that names the secrets without containing them."""
        return (
            f"OmeroConnection(host={self.host!r}, port={self.port!r}, "
            f"username={self.username!r}, secure={self.secure!r}, "
            f"group={self.group!r}, "
            f"password={SECRET_PLACEHOLDER if self.password else None!r}, "
            f"session_key="
            f"{SECRET_PLACEHOLDER if self.session_key else None!r})"
        )

    __str__ = __repr__


def _env_lookup(env: Mapping[str, str], key: str) -> Optional[str]:
    for name in ENV_VARS[key]:
        value = env.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _parse_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    raise OmeroConnectionError(
        f"{source}={value!r} is not a yes/no value. Use one of "
        f"{sorted(_TRUE_TEXT)} or {sorted(_FALSE_TEXT)}.")


def connection_settings(
    host: Optional[str] = None,
    *,
    port: Optional[Union[int, str]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    session_key: Optional[str] = None,
    secure: Optional[Union[bool, str]] = None,
    group: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> OmeroConnection:
    """Build an :class:`OmeroConnection` from arguments, then the environment.

    Arguments always win; anything left unset is looked up in ``env`` through
    :data:`ENV_VARS`. Reading the environment matters more here than it looks:
    it is how a password stays out of a notebook, out of a settings CSV and
    out of the shell history, which is the same reason ``OMERO_PASSWORD``
    exists at all.

    :param host: server hostname; falls back to ``OMERO_HOST``.
    :param port: server port; falls back to ``OMERO_PORT``, then
        :data:`DEFAULT_PORT`.
    :param username: falls back to ``OMERO_USER`` / ``OMERO_USERNAME``.
    :param password: falls back to ``OMERO_PASSWORD`` / ``OMERO_PASS``.
    :param session_key: falls back to ``OMERO_SESSION_KEY``; used instead of a
        password when present.
    :param secure: falls back to ``OMERO_SECURE``, then ``True``.
    :param group: falls back to ``OMERO_GROUP``.
    :param env: the environment to read. Defaults to :data:`os.environ`;
        passing a plain dict is how the tests avoid touching the real one.
    :returns: a validated, frozen :class:`OmeroConnection`.
    :raises OmeroConnectionError: when the host is missing, the port is not a
        usable TCP port, no credential was supplied, or a password was given
        with no user name to go with it.
    """
    env = os.environ if env is None else env

    host = host if host else _env_lookup(env, "host")
    if not host or not str(host).strip():
        raise OmeroConnectionError(
            "no OMERO host. Pass host=... or set the OMERO_HOST environment "
            "variable.")
    host = str(host).strip()

    raw_port = port if port is not None else _env_lookup(env, "port")
    if raw_port is None or str(raw_port).strip() == "":
        resolved_port = DEFAULT_PORT
    else:
        try:
            resolved_port = int(str(raw_port).strip())
        except (TypeError, ValueError):
            raise OmeroConnectionError(
                f"port {raw_port!r} is not an integer.") from None
    if not 1 <= resolved_port <= 65535:
        raise OmeroConnectionError(
            f"port {resolved_port} is not a TCP port (1-65535). OMERO's "
            f"default is {DEFAULT_PORT}.")

    username = username if username else _env_lookup(env, "username")
    password = password if password else _env_lookup(env, "password")
    session_key = session_key if session_key else _env_lookup(env, "session_key")
    group = group if group else _env_lookup(env, "group")

    raw_secure = secure if secure is not None else _env_lookup(env, "secure")
    resolved_secure = (True if raw_secure is None
                       else _parse_bool(raw_secure, source="secure"))

    if not password and not session_key:
        raise OmeroConnectionError(
            "no OMERO credential. Pass password=... or session_key=..., or "
            "set OMERO_PASSWORD or OMERO_SESSION_KEY. (The value is never "
            "printed, logged or written to a sidecar.)")
    if password and not session_key and not username:
        raise OmeroConnectionError(
            "a password was supplied with no user name. Pass username=... or "
            "set OMERO_USER.")

    return OmeroConnection(
        host=host,
        port=resolved_port,
        username=str(username) if username else None,
        password=str(password) if password else None,
        session_key=str(session_key) if session_key else None,
        secure=resolved_secure,
        group=str(group) if group else None,
    )


def _default_gateway_factory(settings: OmeroConnection) -> Any:
    """Build a real ``BlitzGateway`` from ``settings`` (needs the extra)."""
    gateway_module = require_omero()
    blitz = gateway_module.BlitzGateway
    if settings.session_key:
        return blitz(
            settings.username, settings.session_key,
            host=settings.host, port=settings.port, secure=settings.secure)
    return blitz(
        settings.username, settings.password,
        host=settings.host, port=settings.port, secure=settings.secure)


def connect(
    settings: Optional[OmeroConnection] = None,
    *,
    gateway_factory: Optional[Any] = None,
    **overrides: Any,
) -> Any:
    """Open a session and return the connected gateway.

    ``settings`` may be omitted entirely, in which case one is built from
    ``**overrides`` and the environment through :func:`connection_settings`.

    The caller owns the returned object and is responsible for closing it
    (``gateway.close()``); this module deliberately does not hold a
    module-level connection, so nothing here can leak a session between two
    unrelated runs.

    :param settings: a prepared :class:`OmeroConnection`, or ``None``.
    :param gateway_factory: callable taking the settings and returning an
        unconnected gateway. Defaults to building a real ``BlitzGateway``,
        which is the only line in this module that needs the extra.
    :param overrides: passed to :func:`connection_settings` when ``settings``
        is ``None``.
    :returns: the connected gateway object.
    :raises OmeroConnectionError: when the server refuses the login.
    :raises OmeroExtraMissing: when the extra is not installed and no
        ``gateway_factory`` was supplied.
    """
    if settings is None:
        settings = connection_settings(**overrides)
    factory = gateway_factory or _default_gateway_factory

    # Everything logged about a connection goes through `describe()`, which
    # cannot contain a credential.
    LOGGER.info("connecting to OMERO at %s", settings.describe())
    gateway = factory(settings)

    connected = gateway.connect()
    if connected is False or connected is None:
        raise OmeroConnectionError(
            f"OMERO refused the connection to {settings.describe()}. Check "
            f"the host, the port and the credential (the credential itself is "
            f"never shown here).")
    if settings.group and hasattr(gateway, "setGroupNameForSession"):
        gateway.setGroupNameForSession(settings.group)
    LOGGER.info("connected to OMERO at %s", settings.describe())
    return gateway


# ---------------------------------------------------------------------------
# Object ids
# ---------------------------------------------------------------------------

#: The OMERO container types this module knows how to name. Case is
#: normalised to OMERO's own capitalisation, which is what ``getObject``
#: expects.
OBJECT_TYPES: Tuple[str, ...] = (
    "Project", "Dataset", "Screen", "Plate", "Well", "Image")

_TYPE_BY_LOWER = {name.lower(): name for name in OBJECT_TYPES}

#: ``Dataset:123``, ``dataset-123``, and the ``?show=dataset-123`` fragment of
#: an OMERO.web URL — the three forms a user actually has on the clipboard.
_REF = re.compile(
    r"(?:^|[?&/])(?:show=)?(?P<kind>[A-Za-z]+)[:\-](?P<id>[+-]?\d+)\s*$")
_BARE = re.compile(r"^\s*(?P<id>[+-]?\d+)\s*$")


@dataclass(frozen=True)
class OmeroRef:
    """A parsed reference to one OMERO object.

    :param kind: the container type in OMERO's capitalisation
        (``'Dataset'``), or ``None`` when the input was a bare number and the
        caller must say what it is.
    :param object_id: the positive integer id.
    :param text: the original input, kept for error messages.
    """

    kind: Optional[str]
    object_id: int
    text: str

    def describe(self) -> str:
        """Return ``'Dataset:123'``, or ``'123'`` when the kind is unknown.

        :returns: a short human-readable reference.
        """
        return f"{self.kind}:{self.object_id}" if self.kind else str(self.object_id)


def parse_object_ref(value: Any) -> OmeroRef:
    """Parse an OMERO object reference into a kind and a positive id.

    Accepts, deliberately, every form a user is likely to paste:

    * ``123`` or ``'123'`` — the id alone, kind unknown;
    * ``'Dataset:123'``, ``'dataset-123'`` — id and kind together;
    * ``'https://omero.example.org/webclient/?show=plate-42'`` — what the
      OMERO.web address bar contains while you are looking at the plate.

    :param value: an int, or a string in one of the forms above, or an
        :class:`OmeroRef` (returned unchanged).
    :returns: an :class:`OmeroRef`.
    :raises OmeroIdError: for a negative or zero id, a non-numeric id, an
        empty value, a float, or a type name OMERO does not have. OMERO ids
        are positive int64 and 0 is never one.
    """
    if isinstance(value, OmeroRef):
        return value
    if isinstance(value, bool):
        # bool is an int; `True` is not an object id and must not become 1.
        raise OmeroIdError(f"{value!r} is not an OMERO object id.")
    if isinstance(value, int):
        return _checked_ref(None, value, str(value))
    if isinstance(value, float):
        raise OmeroIdError(
            f"{value!r} is a float; OMERO object ids are integers. Pass "
            f"int({value!r}) if that is what you meant.")
    if value is None:
        raise OmeroIdError(
            "no OMERO object id was given. Pass an id (123), a reference "
            "('Dataset:123') or an OMERO.web URL.")

    text = str(value).strip()
    if not text:
        raise OmeroIdError(
            "the OMERO object id is empty. Pass an id (123), a reference "
            "('Dataset:123') or an OMERO.web URL.")

    bare = _BARE.match(text)
    if bare:
        return _checked_ref(None, int(bare.group("id")), text)

    match = _REF.search(text)
    if not match:
        raise OmeroIdError(
            f"{value!r} is not an OMERO object reference. Expected an id "
            f"(123), a 'Type:id' reference ('Dataset:123'), or an OMERO.web "
            f"URL ending in '?show=dataset-123'.")
    kind = _TYPE_BY_LOWER.get(match.group("kind").lower())
    if kind is None:
        raise OmeroIdError(
            f"{match.group('kind')!r} in {value!r} is not an OMERO object "
            f"type. Known types: {', '.join(OBJECT_TYPES)}.")
    return _checked_ref(kind, int(match.group("id")), text)


def _checked_ref(kind: Optional[str], object_id: int, text: str) -> OmeroRef:
    if object_id <= 0:
        raise OmeroIdError(
            f"{text!r} gives object id {object_id}; OMERO ids are positive "
            f"integers, so there is nothing to fetch.")
    return OmeroRef(kind=kind, object_id=object_id, text=text)


def parse_object_id(value: Any, expect: Optional[str] = None) -> int:
    """Parse ``value`` to a positive OMERO id, checking the type it names.

    This is the guard that stops a Plate id being handed to the Dataset
    importer: the mistake is easy (both are integers, both exist on the same
    server) and the failure without this check is a confusing empty import
    rather than an error.

    :param value: anything :func:`parse_object_ref` accepts.
    :param expect: the required object type, e.g. ``'Dataset'``. A reference
        that names a *different* type is refused; a bare id, which names no
        type, is accepted and taken at the caller's word.
    :returns: the positive integer id.
    :raises OmeroIdError: for an unusable id, or when the reference names a
        type other than ``expect``.
    """
    ref = parse_object_ref(value)
    if expect is not None:
        wanted = _TYPE_BY_LOWER.get(str(expect).lower(), str(expect))
        if ref.kind is not None and ref.kind != wanted:
            raise OmeroIdError(
                f"{ref.describe()} is a {ref.kind}, but a {wanted} was "
                f"expected. A {ref.kind} id cannot be imported as a {wanted}; "
                f"use the matching importer, or pass the bare id {ref.object_id} "
                f"if you are certain it is a {wanted}.")
    return ref.object_id


# ---------------------------------------------------------------------------
# The well mapping — OMERO (0-based row, column) -> spaCR keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WellPosition:
    """One well, in both vocabularies at once.

    :param row_index: 1-based row (spaCR's convention). OMERO's row + 1.
    :param column_index: 1-based column. OMERO's column + 1.
    :param row_id: spaCR's ``rowID``, e.g. ``'r1'``.
    :param column_id: spaCR's ``columnID``, e.g. ``'c1'``.
    :param well: the well name, e.g. ``'A01'`` — zero padded to two column
        digits, which is what spaCR's strict Yokogawa regex requires.
    :param plate_format: the smallest standard plate this position fits
        (96, 384, 1536, ...), or ``None`` when it fits none. Reported, never
        enforced: a partial or non-standard plate is a real thing.
    """

    row_index: int
    column_index: int
    row_id: str
    column_id: str
    well: str
    plate_format: Optional[int] = None

    def prc(self, plate: str) -> str:
        """Return spaCR's ``prc`` well key for this position on ``plate``.

        :param plate: the ``plateID`` token.
        :returns: ``'plate1_r1_c1'``.
        """
        return schema.compose_prc(plate, self.row_id, self.column_id)


def well_position(row: Any, column: Any) -> WellPosition:
    """Map OMERO's 0-based ``(row, column)`` onto spaCR's well keys.

    This is the single most important line of the import, because everything
    downstream — every group-by, every plate heatmap, every well-level
    statistic — is keyed off ``rowID``/``columnID``, and a plate that comes in
    one row out is a plate whose controls are in the wrong place.

    Row letters are bijective base 26 via :func:`spacr.schema.well_id`, so::

        well_position(0, 0).well   == 'A01'
        well_position(25, 0).well  == 'Z01'
        well_position(26, 0).well  == 'AA01'      # not '[01', not IndexError
        well_position(31, 47).well == 'AF48'      # a 1536-well plate

    :param row: OMERO's ``WellWrapper.getRow()``, 0-based.
    :param column: OMERO's ``WellWrapper.getColumn()``, 0-based.
    :returns: a :class:`WellPosition`.
    :raises OmeroWellError: when either index is missing, not an integer, or
        negative. OMERO leaves both ``None`` on a well that was never placed
        on the plate, and silently treating that as row 0 would put an
        unplaced well in A01 alongside a real one.
    """
    row_index = _plate_index(row, "row")
    column_index = _plate_index(column, "column")
    return WellPosition(
        row_index=row_index,
        column_index=column_index,
        row_id=f"{schema.KEY_PREFIXES[schema.ROW_KEY]}{row_index}",
        column_id=f"{schema.KEY_PREFIXES[schema.COLUMN_KEY]}{column_index}",
        well=schema.well_id(row_index, column_index),
        plate_format=schema.plate_format_for(row_index, column_index),
    )


def _plate_index(value: Any, what: str) -> int:
    if value is None:
        raise OmeroWellError(
            f"the well has no {what} index (OMERO returned None). A well that "
            f"was never placed on the plate has no position, and guessing one "
            f"would put it on top of a real well.")
    if isinstance(value, bool) or not isinstance(value, int):
        # A float row index is not a rounding problem, it is a wrong object.
        try:
            as_int = int(str(value).strip())
        except (TypeError, ValueError):
            raise OmeroWellError(
                f"well {what} {value!r} is not an integer. OMERO's "
                f"get{what.capitalize()}() returns a 0-based int.") from None
    else:
        as_int = value
    if as_int < 0:
        raise OmeroWellError(
            f"well {what} {value!r} is negative. OMERO's indices are 0-based, "
            f"so the first {what} is 0.")
    return as_int + 1


def omero_indices(well: str) -> Tuple[int, int]:
    """Invert :func:`well_position`: ``'A01'`` -> ``(0, 0)``.

    The round trip is what makes the export direction possible — matching a
    spaCR per-well summary back onto the OMERO wells it came from.

    :param well: a well name (``'A01'``, ``'aa1'``, ``'AF48'``) or a spaCR
        ``(rowID, columnID)`` pair already joined, e.g. ``'r1_c1'`` is *not*
        accepted — pass the well name.
    :returns: ``(row, column)``, both 0-based, as OMERO reports them.
    :raises OmeroWellError: when ``well`` is not a well name.
    """
    try:
        row_id, column_id = schema.parse_well(well, strict=True)
    except Exception as exc:                       # schema raises WellParseError
        raise OmeroWellError(
            f"{well!r} is not a well name; expected something like 'A01'.") from exc
    row_index = schema.row_index(row_id)
    column_index = schema.column_index(column_id)
    if row_index is None or column_index is None:
        raise OmeroWellError(f"{well!r} is not a well name; expected 'A01'.")
    return row_index - 1, column_index - 1


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------

#: Characters kept in a plate token. Must agree with
#: ``spacr.convert._sanitise``; ``tests/test_omero.py`` asserts that it does.
_PLATE_UNSAFE = re.compile(r"[^A-Za-z0-9]+")

#: A well name embedded in an OMERO image name. Bounded on both sides so
#: ``C01`` in ``...L01A01Z01C01`` is not mistaken for well C1 — the token has
#: to be delimited by a separator or the end of the name.
_WELL_IN_NAME = re.compile(r"(?:^|[_\-. ])([A-Za-z]{1,2}\d{1,2})(?=$|[_\-. ])")


def plate_token(name: Any) -> str:
    """Reduce ``name`` to a plate token that survives spaCR's filename regex.

    Underscores are stripped along with every other non-alphanumeric
    character, and the reason is not cosmetic: the ``cellvoyager`` regex
    splits the plate from the well on ``_``, so a plate literally called
    ``my run`` would move the split point and misparse every well. This is the
    same rule as ``spacr.convert._sanitise``, which is the function that
    already governs the rest of spaCR's conversions.

    :param name: any label — an OMERO plate name, a dataset name, an id.
    :returns: a non-empty token of ``[A-Za-z0-9-]``, falling back to
        ``'plate'`` when nothing survives.
    """
    cleaned = _PLATE_UNSAFE.sub("-", str(name)).strip("-")
    return cleaned or "plate"


def plane_filename(plate: str, well: str, field_id: int, channel: int,
                   z: int = 1, t: int = 1) -> str:
    """Return the Yokogawa filename spaCR expects for one plane.

    A thin, deliberate delegation to :func:`spacr.convert.target_name`, so
    there is exactly one definition of the name in spaCR and an OMERO import
    cannot drift away from a folder conversion.

    :param plate: plate token; run it through :func:`plate_token` first.
    :param well: canonical well id, e.g. ``'A01'``.
    :param field_id: 1-based field (imaging site) id.
    :param channel: 1-based channel id.
    :param z: 1-based z-slice id.
    :param t: 1-based timepoint id.
    :returns: e.g. ``'plate1_A01_T0001F001L01A01Z01C01.tif'``.
    """
    return target_name(plate, well, field_id, channel, z=z, t=t)


def well_from_image_name(name: Any) -> Optional[str]:
    """Recover a well name from an OMERO image name, or return ``None``.

    Images in a *Dataset* have no plate geometry — a Dataset is a flat bag —
    so the only place a well can come from is the name the acquisition
    software gave the file. This reads the first delimited token that parses
    as a well within a 1536-well plate (32 rows, 48 columns).

    It is a guess, and it is treated as one: :func:`import_dataset` records
    ``well_source='name'`` or ``'sequence'`` per file in the sidecar CSV, and
    ``well_from_name=False`` turns it off entirely.

    :param name: the OMERO image name.
    :returns: a canonical well name (``'A01'``), or ``None`` when the name
        contains nothing that parses as one.
    """
    if name is None:
        return None
    for token in _WELL_IN_NAME.findall(str(name)):
        try:
            row_id, column_id = schema.parse_well(token, strict=True)
            row_index = schema.row_index(row_id)
            column_index = schema.column_index(column_id)
        except Exception:                          # not a well; try the next token
            continue
        if row_index and column_index and row_index <= 32 and column_index <= 48:
            return schema.well_id(row_index, column_index)
    return None


# ---------------------------------------------------------------------------
# Pixel size — a length, not a number of micrometres
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PixelSize:
    """A physical pixel size together with the unit it was recorded in.

    :param value: the magnitude, or ``None`` when the server has no calibration
        for this image (which is common, and is not an error).
    :param unit: the unit symbol as OMERO reports it — ``'MICROMETER'``,
        ``'µm'``, ``'nm'``. Carried verbatim; this module never converts,
        because a silent unit conversion is a silent factor of 1000.
    """

    value: Optional[float] = None
    unit: Optional[str] = None

    def __bool__(self) -> bool:
        """Return whether a magnitude is present."""
        return self.value is not None

    def describe(self) -> str:
        """Return ``'0.325 MICROMETER'``, or ``'unknown'``.

        :returns: a short human-readable pixel size.
        """
        if self.value is None:
            return "unknown"
        return f"{self.value:g} {self.unit}" if self.unit else f"{self.value:g}"


def pixel_size_from(length: Any) -> PixelSize:
    """Read a :class:`PixelSize` out of whatever ``getPixelSizeX()`` returned.

    omero-py returns different things depending on how it is called and how
    old the server is: a ``LengthI`` object (``getValue()`` / ``getSymbol()``
    or ``getUnit()``), a plain float (already in µm, by omero-py's own
    convention), or ``None`` for an uncalibrated image. All three are handled,
    and only the plain-float case assumes a unit — because in that case
    omero-py has already made the assumption itself, and this records that it
    did rather than pretending the unit is unknown.

    :param length: the value returned by ``getPixelSizeX``/``Y``/``Z``.
    :returns: a :class:`PixelSize`; ``PixelSize(None, None)`` when there is no
        calibration.
    """
    if length is None:
        return PixelSize()
    if isinstance(length, bool):
        return PixelSize()
    if isinstance(length, (int, float)):
        # omero-py's `getPixelSizeX()` with no `units=` argument returns a
        # float already converted to micrometres. Recording the unit it used
        # is the whole point of this function.
        return PixelSize(float(length), "MICROMETER")

    value = None
    for getter in ("getValue", "getvalue"):
        method = getattr(length, getter, None)
        if callable(method):
            try:
                value = method()
            except Exception:                      # a stub without the call
                value = None
            break
    if value is None:
        value = getattr(length, "value", None)
    if callable(value):
        value = None

    unit = None
    for getter in ("getSymbol", "getUnit"):
        method = getattr(length, getter, None)
        if callable(method):
            try:
                unit = method()
            except Exception:
                unit = None
            if unit is not None:
                break
    if unit is None:
        unit = getattr(length, "unit", None)
    if callable(unit):
        unit = None

    try:
        value = None if value is None else float(value)
    except (TypeError, ValueError):
        value = None
    return PixelSize(value, None if unit is None else str(unit))


# ---------------------------------------------------------------------------
# Walking containers (the adapter, kept as thin as it can be)
# ---------------------------------------------------------------------------

def _call(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the result of the first callable attribute in ``names``."""
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            try:
                return method()
            except Exception:                      # pragma: no cover - server
                return default
    return default


def _size(image: Any, name: str) -> int:
    value = _call(image, name, default=None)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    return value if value > 0 else 1


def _channel_names(image: Any) -> Tuple[str, ...]:
    channels = _call(image, "getChannels", default=None) or ()
    names: List[str] = []
    for index, channel in enumerate(channels):
        label = _call(channel, "getLabel", "getName", default=None)
        names.append(str(label) if label else str(index + 1))
    return tuple(names)


def _iter_dataset_images(dataset: Any) -> Iterator[Any]:
    for child in _call(dataset, "listChildren", "getChildren", default=()) or ():
        yield child


def _iter_wells(plate: Any) -> Iterator[Any]:
    """Yield the wells of a plate.

    ``PlateWrapper`` exposes both ``listChildren()`` and ``getWells()``
    depending on the omero-py version, so both are tried rather than betting
    on one.
    """
    wells = _call(plate, "getWells", "listChildren", default=None)
    for well in wells or ():
        yield well


def _iter_well_images(well: Any) -> Iterator[Any]:
    """Yield the images of a well, in WellSample (field) order."""
    samples = _call(well, "listChildren", "getWellSamples", default=None)
    if samples is None:
        count = _call(well, "countWellSample", default=0) or 0
        samples = []
        for index in range(int(count)):
            getter = getattr(well, "getWellSample", None)
            if callable(getter):
                samples.append(getter(index))
    for sample in samples or ():
        if sample is None:
            continue
        image = _call(sample, "getImage", default=None)
        yield image if image is not None else sample


def _resolve(gateway: Any, kind: str, object_id: int) -> Any:
    obj = gateway.getObject(kind, object_id)
    if obj is None:
        raise OmeroContainerError(
            f"{kind}:{object_id} does not exist, or is not visible to this "
            f"user in the current OMERO group. Check the id, and check that "
            f"you are in the group that owns it.")
    return obj


# ---------------------------------------------------------------------------
# Listing / inspecting, without fetching pixels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageInfo:
    """What is known about one OMERO image without reading a single pixel.

    :param image_id: the OMERO image id.
    :param name: the OMERO image name.
    :param size_x: width in pixels.
    :param size_y: height in pixels.
    :param size_z: number of z-slices.
    :param size_c: number of channels.
    :param size_t: number of timepoints.
    :param channels: the channel names OMERO holds, in channel order.
    :param pixel_size_x: the physical pixel size in x, with its unit.
    :param pixel_size_y: the physical pixel size in y, with its unit.
    :param pixel_size_z: the z step, with its unit.
    :param well: the well this image sits in, for a Plate; ``None`` in a
        Dataset, which has no plate geometry.
    :param field_id: the 1-based field (WellSample) index, for a Plate.
    """

    image_id: int
    name: str
    size_x: int = 0
    size_y: int = 0
    size_z: int = 1
    size_c: int = 1
    size_t: int = 1
    channels: Tuple[str, ...] = ()
    pixel_size_x: PixelSize = field(default_factory=PixelSize)
    pixel_size_y: PixelSize = field(default_factory=PixelSize)
    pixel_size_z: PixelSize = field(default_factory=PixelSize)
    well: Optional[str] = None
    field_id: Optional[int] = None

    @property
    def n_planes(self) -> int:
        """Return how many 2-D planes this image would import as."""
        return self.size_z * self.size_c * self.size_t


@dataclass(frozen=True)
class ContainerListing:
    """The answer to "what is in this container", with no pixels fetched.

    :param kind: ``'Dataset'`` or ``'Plate'``.
    :param object_id: the container's OMERO id.
    :param name: the container's OMERO name.
    :param images: one :class:`ImageInfo` per image found.
    :param wells: the well names present, for a Plate; empty for a Dataset.
    :param unplaced_wells: how many wells had no row/column and were skipped.
    """

    kind: str
    object_id: int
    name: str
    images: Tuple[ImageInfo, ...] = ()
    wells: Tuple[str, ...] = ()
    unplaced_wells: int = 0

    @property
    def n_images(self) -> int:
        """Return the number of images in the container."""
        return len(self.images)

    @property
    def n_planes(self) -> int:
        """Return how many 2-D TIFFs a full import would write."""
        return sum(image.n_planes for image in self.images)

    @property
    def channels(self) -> Tuple[str, ...]:
        """Return the channel names of the first image, or ``()``."""
        return self.images[0].channels if self.images else ()

    def describe(self) -> str:
        """Return a multi-line summary suitable for printing.

        :returns: the container, its images, their dimensions and channels,
            and the number of TIFFs an import would write.
        """
        lines = [f"{self.kind}:{self.object_id}  {self.name!r}",
                 f"  images  : {self.n_images}"]
        if self.wells:
            lines.append(
                f"  wells   : {len(self.wells)} "
                f"({self.wells[0]}..{self.wells[-1]})")
        if self.unplaced_wells:
            lines.append(f"  unplaced: {self.unplaced_wells} well(s) skipped")
        if self.images:
            first = self.images[0]
            lines.append(
                f"  size    : {first.size_x} x {first.size_y}, "
                f"z={first.size_z}, c={first.size_c}, t={first.size_t}")
            lines.append(f"  channels: {', '.join(first.channels) or '<unnamed>'}")
            lines.append(f"  pixel   : {first.pixel_size_x.describe()}")
        lines.append(f"  import  : would write {self.n_planes} TIFF(s)")
        return "\n".join(lines)


def _image_info(image: Any, *, well: Optional[str] = None,
                field_id: Optional[int] = None) -> ImageInfo:
    return ImageInfo(
        image_id=int(_call(image, "getId", default=0) or 0),
        name=str(_call(image, "getName", default="") or ""),
        size_x=_size(image, "getSizeX"),
        size_y=_size(image, "getSizeY"),
        size_z=_size(image, "getSizeZ"),
        size_c=_size(image, "getSizeC"),
        size_t=_size(image, "getSizeT"),
        channels=_channel_names(image),
        pixel_size_x=pixel_size_from(_call(image, "getPixelSizeX")),
        pixel_size_y=pixel_size_from(_call(image, "getPixelSizeY")),
        pixel_size_z=pixel_size_from(_call(image, "getPixelSizeZ")),
        well=well,
        field_id=field_id,
    )


def inspect_container(gateway: Any, ref: Any, *,
                      kind: Optional[str] = None) -> ContainerListing:
    """Report what a Dataset or Plate contains, **without fetching pixels**.

    This exists so that "what is in plate 4711?" costs a metadata query rather
    than a 100 GB download. Nothing in this function or anything it calls
    touches ``getPrimaryPixels`` or ``getPlane``, and the test suite asserts
    that by counting calls on the fake gateway.

    :param gateway: a connected ``BlitzGateway`` (or anything with the same
        ``getObject`` surface).
    :param ref: an id, a ``'Plate:4711'`` reference, or an OMERO.web URL.
    :param kind: the object type, when ``ref`` is a bare id. Defaults to the
        type named by ``ref``; one of the two must say.
    :returns: a :class:`ContainerListing`.
    :raises OmeroIdError: for an unusable id, or when ``kind`` and ``ref``
        disagree, or when neither says what the object is.
    :raises OmeroContainerError: when the id resolves to nothing.
    """
    parsed = parse_object_ref(ref)
    resolved_kind = _TYPE_BY_LOWER.get(str(kind).lower()) if kind else parsed.kind
    if kind and parsed.kind and resolved_kind != parsed.kind:
        raise OmeroIdError(
            f"{parsed.describe()} names a {parsed.kind}, but kind={kind!r} "
            f"was passed. Say it once.")
    if resolved_kind is None:
        raise OmeroIdError(
            f"{parsed.text!r} is a bare id, so it does not say what it is. "
            f"Pass kind='Dataset' or kind='Plate', or use a reference like "
            f"'Plate:{parsed.object_id}'.")
    if resolved_kind not in ("Dataset", "Plate"):
        raise OmeroIdError(
            f"inspect_container handles Dataset and Plate; {resolved_kind} is "
            f"neither.")

    container = _resolve(gateway, resolved_kind, parsed.object_id)
    name = str(_call(container, "getName", default="") or "")

    if resolved_kind == "Dataset":
        images = tuple(_image_info(image) for image in _iter_dataset_images(container))
        return ContainerListing(kind="Dataset", object_id=parsed.object_id,
                                name=name, images=images)

    images_list: List[ImageInfo] = []
    wells: List[str] = []
    unplaced = 0
    for well in _iter_wells(container):
        try:
            position = well_position(_call(well, "getRow"), _call(well, "getColumn"))
        except OmeroWellError:
            unplaced += 1
            continue
        wells.append(position.well)
        for index, image in enumerate(_iter_well_images(well), start=1):
            if image is None:
                continue
            images_list.append(_image_info(image, well=position.well, field_id=index))
    return ContainerListing(
        kind="Plate", object_id=parsed.object_id, name=name,
        images=tuple(images_list), wells=tuple(sorted(set(wells))),
        unplaced_wells=unplaced)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

#: The per-plane sidecar written next to the TIFFs.
SIDECAR_CSV = "omero_import.csv"
#: The per-container sidecar written next to the TIFFs.
SIDECAR_JSON = "omero_import.json"

#: Columns of :data:`SIDECAR_CSV`, in order. Named so that the OMERO identity
#: and spaCR's own keys sit side by side in one file — which is the thing that
#: makes an import auditable a year later.
MAP_CSV_COLUMNS: Tuple[str, ...] = (
    "target", "omero_image_id", "omero_image_name",
    "plateID", "rowID", "columnID", "fieldID", "well", "prc", "prcf",
    "channel", "channel_name", "z", "t",
    "size_x", "size_y",
    "pixel_size_x", "pixel_size_y", "pixel_size_unit",
    "well_source",
)


@dataclass(frozen=True)
class PlanePlan:
    """One 2-D plane that would be, or was, written.

    :param filename: the target filename (not a path — the destination folder
        is the caller's).
    :param image_id: the OMERO image the plane comes from.
    :param image_name: that image's OMERO name.
    :param well: the well name, e.g. ``'A01'``.
    :param position: the full :class:`WellPosition` for that well.
    :param field_id: 1-based field id.
    :param channel: 1-based channel id.
    :param channel_name: the channel's OMERO name.
    :param z: 1-based z index.
    :param t: 1-based timepoint index.
    :param info: the source image's :class:`ImageInfo`.
    :param well_source: ``'plate'`` (from the well's row/column), ``'name'``
        (parsed out of the image name) or ``'sequence'`` (assigned in listing
        order because nothing else said).
    """

    filename: str
    image_id: int
    image_name: str
    well: str
    position: WellPosition
    field_id: int
    channel: int
    channel_name: str
    z: int
    t: int
    info: ImageInfo
    well_source: str

    def csv_row(self, plate: str) -> Dict[str, Any]:
        """Return this plane as a :data:`MAP_CSV_COLUMNS` row.

        :param plate: the ``plateID`` token used for the filenames.
        :returns: a dict keyed by :data:`MAP_CSV_COLUMNS`.
        """
        return {
            "target": self.filename,
            "omero_image_id": self.image_id,
            "omero_image_name": self.image_name,
            "plateID": plate,
            "rowID": self.position.row_id,
            "columnID": self.position.column_id,
            "fieldID": f"{schema.KEY_PREFIXES[schema.FIELD_KEY]}{self.field_id}",
            "well": self.well,
            "prc": self.position.prc(plate),
            "prcf": schema.compose_prcf(
                plate, self.position.row_id, self.position.column_id,
                self.field_id),
            "channel": self.channel,
            "channel_name": self.channel_name,
            "z": self.z,
            "t": self.t,
            "size_x": self.info.size_x,
            "size_y": self.info.size_y,
            "pixel_size_x": self.info.pixel_size_x.value,
            "pixel_size_y": self.info.pixel_size_y.value,
            "pixel_size_unit": self.info.pixel_size_x.unit,
            "well_source": self.well_source,
        }


@dataclass(frozen=True)
class ImportResult:
    """What an import did, or (with ``dry_run``) what it would have done.

    :param kind: ``'Dataset'`` or ``'Plate'``.
    :param object_id: the container's OMERO id.
    :param name: the container's OMERO name.
    :param plate: the ``plateID`` token every filename was built with.
    :param dst: the destination folder.
    :param planned: every plane the import covered, in write order.
    :param written: the filenames actually written.
    :param skipped: filenames that already existed and were left alone.
    :param dry_run: whether pixels were fetched at all.
    :param limited: whether ``limit`` stopped the walk before the end.
    :param n_images: how many OMERO images were visited.
    """

    kind: str
    object_id: int
    name: str
    plate: str
    dst: str
    planned: Tuple[PlanePlan, ...] = ()
    written: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()
    dry_run: bool = False
    limited: bool = False
    n_images: int = 0

    def describe(self) -> str:
        """Return a short human summary of the import.

        :returns: a multi-line string naming the container, the plate token,
            the counts and the destination.
        """
        verb = "would write" if self.dry_run else "wrote"
        lines = [
            f"{self.kind}:{self.object_id} {self.name!r} -> {self.dst}",
            f"  plate token : {self.plate}",
            f"  images      : {self.n_images}"
            + ("  (stopped early: limit reached)" if self.limited else ""),
            f"  planes      : {len(self.planned)}",
            f"  {verb:<12}: {len(self.written)}",
        ]
        if self.skipped:
            lines.append(f"  skipped     : {len(self.skipped)} (already present)")
        return "\n".join(lines)


def _plane_plans(info: ImageInfo, plate: str, well: str, position: WellPosition,
                 field_id: int, well_source: str) -> Iterator[PlanePlan]:
    for t in range(1, info.size_t + 1):
        for z in range(1, info.size_z + 1):
            for c in range(1, info.size_c + 1):
                channel_name = (info.channels[c - 1]
                                if c - 1 < len(info.channels) else str(c))
                yield PlanePlan(
                    filename=plane_filename(plate, well, field_id, c, z=z, t=t),
                    image_id=info.image_id,
                    image_name=info.name,
                    well=well,
                    position=position,
                    field_id=field_id,
                    channel=c,
                    channel_name=channel_name,
                    z=z,
                    t=t,
                    info=info,
                    well_source=well_source,
                )


def _write_planes(image_by_id: Mapping[int, Any],
                  plans: Sequence[PlanePlan], dst: Path,
                  overwrite: bool) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    written: List[str] = []
    skipped: List[str] = []
    pixels_cache: Dict[int, Any] = {}
    for plan in plans:
        target = dst / plan.filename
        if target.exists() and not overwrite:
            skipped.append(plan.filename)
            continue
        pixels = pixels_cache.get(plan.image_id)
        if pixels is None:
            image = image_by_id[plan.image_id]
            pixels = _call(image, "getPrimaryPixels", default=None)
            if pixels is None:
                raise OmeroContainerError(
                    f"image {plan.image_id} has no pixels; it may still be "
                    f"importing on the server.")
            pixels_cache[plan.image_id] = pixels
        # OMERO's getPlane is (theZ, theC, theT), all 0-based.
        plane = pixels.getPlane(plan.z - 1, plan.channel - 1, plan.t - 1)
        write_tiff(target, plane)
        written.append(plan.filename)
    return tuple(written), tuple(skipped)


def _write_sidecars(dst: Path, result_kind: str, object_id: int, name: str,
                    plate: str, plans: Sequence[PlanePlan],
                    listing: ContainerListing, settings: Optional[OmeroConnection],
                    dry_run: bool, limit: Optional[int]) -> None:
    with open(dst / SIDECAR_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MAP_CSV_COLUMNS))
        writer.writeheader()
        for plan in plans:
            writer.writerow(plan.csv_row(plate))

    first = listing.images[0] if listing.images else None
    # `settings.redacted()` rather than `settings`: this file is written into
    # a project folder that gets copied, zipped and shared.
    payload = {
        "spacr_version": get_version(),
        "imported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "server": (settings.redacted() if settings is not None else None),
        "container": {"kind": result_kind, "id": object_id, "name": name},
        "plate": plate,
        "namespace": NAMESPACE_ROOT,
        "n_images": listing.n_images,
        "n_planes": len(plans),
        "wells": list(listing.wells),
        "channels": list(listing.channels),
        "pixel_size": {
            "x": first.pixel_size_x.value if first else None,
            "y": first.pixel_size_y.value if first else None,
            "z": first.pixel_size_z.value if first else None,
            "unit": first.pixel_size_x.unit if first else None,
        },
        "dry_run": bool(dry_run),
        "limit": limit,
    }
    with open(dst / SIDECAR_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_dst(dst: Union[str, os.PathLike]) -> Path:
    path = Path(dst)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_limit(limit: Optional[int]) -> None:
    """Refuse a limit that would import nothing while looking like a success."""
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise OmeroError(f"limit={limit!r} is not an integer or None.")
    if limit < 1:
        raise OmeroError(
            f"limit={limit} would import nothing. Use limit=None for the whole "
            f"container, or dry_run=True to see what it holds without "
            f"fetching pixels.")


#: Columns per row when :func:`import_dataset` has to invent well names. Twelve
#: is a 96-well plate read row-major, which is the layout a bag of images most
#: often came off. It is a *labelling*, not a claim about the physical plate.
SEQUENCE_COLUMNS = 12


def import_dataset(gateway: Any, ref: Any, dst: Union[str, os.PathLike], *,
                   plate: Optional[str] = None,
                   limit: Optional[int] = None,
                   dry_run: bool = False,
                   overwrite: bool = False,
                   well_from_name: bool = True,
                   settings: Optional[OmeroConnection] = None) -> ImportResult:
    """Import an OMERO **Dataset** into a spaCR source folder.

    A Dataset is a flat bag of images with no plate geometry, so the wells
    have to come from somewhere. Two sources, in order, and which one was used
    is recorded per file in :data:`SIDECAR_CSV`:

    ``'name'``
        the image name contains a delimited well token (see
        :func:`well_from_image_name`). Turned off with
        ``well_from_name=False``.
    ``'sequence'``
        nothing said, so wells are handed out row-major in listing order —
        ``A01``, ``A02``, ... — one per image. This is a *labelling*, not a
        claim about the physical plate, and it is what makes the images
        loadable by spaCR at all.

    :param gateway: a connected gateway.
    :param ref: a Dataset id, ``'Dataset:123'``, or an OMERO.web URL.
    :param dst: destination folder; created if missing.
    :param plate: the ``plateID`` token. Defaults to the Dataset's own name,
        run through :func:`plate_token`.
    :param limit: stop after this many OMERO images. The cheap way to try an
        import against a container you have not seen.
    :param dry_run: plan everything, write the sidecars, fetch no pixels.
    :param overwrite: rewrite TIFFs that already exist. Off by default, so a
        re-run resumes rather than redoing the download.
    :param well_from_name: whether to look for a well in the image name.
    :param settings: the connection these images came from, recorded
        (redacted) in :data:`SIDECAR_JSON`.
    :returns: an :class:`ImportResult`.
    :raises OmeroIdError: when ``ref`` names something other than a Dataset —
        a Plate id passed here is refused rather than half-imported.
    :raises OmeroContainerError: when the id resolves to nothing or the
        Dataset has no images.
    """
    _check_limit(limit)
    object_id = parse_object_id(ref, expect="Dataset")
    container = _resolve(gateway, "Dataset", object_id)
    name = str(_call(container, "getName", default="") or "")
    plate_name = plate_token(plate if plate else (name or f"dataset{object_id}"))
    dst_path = _prepare_dst(dst)

    infos: List[ImageInfo] = []
    plans: List[PlanePlan] = []
    image_by_id: Dict[int, Any] = {}
    field_by_well: Dict[str, int] = {}
    limited = False
    sequence = 0

    for image in _iter_dataset_images(container):
        if limit is not None and len(infos) >= limit:
            limited = True
            break
        info = _image_info(image)
        well = well_from_image_name(info.name) if well_from_name else None
        source = "name"
        if well is None:
            row, column = divmod(sequence, SEQUENCE_COLUMNS)
            well = schema.well_id(row + 1, column + 1)
            sequence += 1
            source = "sequence"
        position = well_position(*omero_indices(well))
        field_by_well[well] = field_by_well.get(well, 0) + 1
        infos.append(info)
        image_by_id[info.image_id] = image
        plans.extend(_plane_plans(info, plate_name, well, position,
                                  field_by_well[well], source))

    if not infos:
        raise OmeroContainerError(
            f"Dataset:{object_id} {name!r} contains no images, so there is "
            f"nothing to import.")

    listing = ContainerListing(kind="Dataset", object_id=object_id, name=name,
                               images=tuple(infos))
    written: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()
    if not dry_run:
        written, skipped = _write_planes(image_by_id, plans,
                                         dst_path, overwrite)
    _write_sidecars(dst_path, "Dataset", object_id, name, plate_name, plans,
                    listing, settings, dry_run, limit)
    LOGGER.info("imported Dataset:%s into %s (%d plane(s))",
                object_id, dst_path, len(written))
    return ImportResult(kind="Dataset", object_id=object_id, name=name,
                        plate=plate_name, dst=str(dst_path),
                        planned=tuple(plans), written=written, skipped=skipped,
                        dry_run=dry_run, limited=limited, n_images=len(infos))


def import_plate(gateway: Any, ref: Any, dst: Union[str, os.PathLike], *,
                 plate: Optional[str] = None,
                 limit: Optional[int] = None,
                 dry_run: bool = False,
                 overwrite: bool = False,
                 settings: Optional[OmeroConnection] = None) -> ImportResult:
    """Import an OMERO **Plate** into a spaCR source folder.

    A Plate is the case spaCR was built for: every image already knows its
    well, and the well knows its row and column. Those indices are carried
    across by :func:`well_position` and become the well name in the filename
    and ``rowID``/``columnID`` in the sidecar, so a plate map drawn in spaCR
    lines up with the plate map in OMERO.web.

    Fields are the WellSample order within each well, 1-based, which is what
    OMERO means by an imaging site.

    Wells with no row/column (present in the Plate but never placed) are
    **skipped and counted**, never defaulted to A01.

    :param gateway: a connected gateway.
    :param ref: a Plate id, ``'Plate:4711'``, or an OMERO.web URL.
    :param dst: destination folder; created if missing.
    :param plate: the ``plateID`` token. Defaults to the Plate's own name.
    :param limit: stop after this many OMERO images.
    :param dry_run: plan everything, write the sidecars, fetch no pixels.
    :param overwrite: rewrite TIFFs that already exist.
    :param settings: the connection, recorded (redacted) in the JSON sidecar.
    :returns: an :class:`ImportResult`.
    :raises OmeroIdError: when ``ref`` names something other than a Plate.
    :raises OmeroContainerError: when the id resolves to nothing or the Plate
        has no placed wells with images.
    """
    _check_limit(limit)
    object_id = parse_object_id(ref, expect="Plate")
    container = _resolve(gateway, "Plate", object_id)
    name = str(_call(container, "getName", default="") or "")
    plate_name = plate_token(plate if plate else (name or f"plate{object_id}"))
    dst_path = _prepare_dst(dst)

    infos: List[ImageInfo] = []
    plans: List[PlanePlan] = []
    image_by_id: Dict[int, Any] = {}
    wells: List[str] = []
    unplaced = 0
    limited = False

    for well in _iter_wells(container):
        if limited:
            break
        try:
            position = well_position(_call(well, "getRow"), _call(well, "getColumn"))
        except OmeroWellError:
            unplaced += 1
            continue
        wells.append(position.well)
        for field_id, image in enumerate(_iter_well_images(well), start=1):
            if image is None:
                continue
            if limit is not None and len(infos) >= limit:
                limited = True
                break
            info = _image_info(image, well=position.well, field_id=field_id)
            infos.append(info)
            image_by_id[info.image_id] = image
            plans.extend(_plane_plans(info, plate_name, position.well, position,
                                      field_id, "plate"))

    if not infos:
        raise OmeroContainerError(
            f"Plate:{object_id} {name!r} has no placed well containing an "
            f"image ({unplaced} well(s) had no row/column), so there is "
            f"nothing to import.")

    listing = ContainerListing(kind="Plate", object_id=object_id, name=name,
                               images=tuple(infos),
                               wells=tuple(sorted(set(wells))),
                               unplaced_wells=unplaced)
    written: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()
    if not dry_run:
        written, skipped = _write_planes(image_by_id, plans,
                                         dst_path, overwrite)
    _write_sidecars(dst_path, "Plate", object_id, name, plate_name, plans,
                    listing, settings, dry_run, limit)
    LOGGER.info("imported Plate:%s into %s (%d plane(s))",
                object_id, dst_path, len(written))
    return ImportResult(kind="Plate", object_id=object_id, name=name,
                        plate=plate_name, dst=str(dst_path),
                        planned=tuple(plans), written=written, skipped=skipped,
                        dry_run=dry_run, limited=limited, n_images=len(infos))


def import_container(gateway: Any, ref: Any, dst: Union[str, os.PathLike], *,
                     kind: Optional[str] = None, **kwargs: Any) -> ImportResult:
    """Import a Dataset or a Plate, dispatching on what ``ref`` says it is.

    :param gateway: a connected gateway.
    :param ref: ``'Plate:4711'``, ``'Dataset:123'``, an OMERO.web URL, or a
        bare id together with ``kind``.
    :param dst: destination folder.
    :param kind: ``'Dataset'`` or ``'Plate'``, required when ``ref`` is bare.
    :param kwargs: passed to :func:`import_dataset` or :func:`import_plate`.
    :returns: an :class:`ImportResult`.
    :raises OmeroIdError: when the kind is unknown, unsupported, or contradicts
        ``ref``.
    """
    parsed = parse_object_ref(ref)
    resolved = _TYPE_BY_LOWER.get(str(kind).lower()) if kind else parsed.kind
    if kind and parsed.kind and resolved != parsed.kind:
        raise OmeroIdError(
            f"{parsed.describe()} names a {parsed.kind}, but kind={kind!r} "
            f"was passed. Say it once.")
    if resolved == "Dataset":
        return import_dataset(gateway, parsed, dst, **kwargs)
    if resolved == "Plate":
        return import_plate(gateway, parsed, dst, **kwargs)
    if resolved is None:
        raise OmeroIdError(
            f"{parsed.text!r} is a bare id, so it does not say what it is. "
            f"Pass kind='Dataset' or kind='Plate'.")
    raise OmeroIdError(
        f"importing a {resolved} is not supported; spaCR imports a Dataset or "
        f"a Plate.")


# ---------------------------------------------------------------------------
# Export: measurements -> key/value pairs
# ---------------------------------------------------------------------------

#: Keep a missing value as an explicit ``NaN`` entry. The default.
NAN_KEEP = "keep"
#: Drop the key entirely when its value is missing.
NAN_DROP = "drop"
#: Both NaN policies.
NAN_POLICIES: Tuple[str, ...] = (NAN_KEEP, NAN_DROP)

#: What a missing value renders as under :data:`NAN_KEEP`.
MISSING_TEXT = "NaN"

#: How floats are rendered. Six significant digits: enough to distinguish two
#: real measurements, short enough that a column of them is readable in
#: OMERO.web's panel, and it drops trailing zeros so ``3.0`` becomes ``3``.
FLOAT_FORMAT = "{:.6g}"

#: The most key/value pairs one MapAnnotation will carry, *including* the
#: truncation notice. See :func:`measurement_pairs` for why there is a cap.
MAX_MAP_PAIRS = 50

#: Keys longer than this are truncated with an ellipsis.
MAX_KEY_CHARS = 128
#: Values longer than this are truncated with an ellipsis.
MAX_VALUE_CHARS = 255

#: The key that says the table was cut short, and where the rest is.
TRUNCATION_KEY = "spacr_truncated"

#: Keys promoted to the top of every annotation, in this order, when present.
#: They are what makes the panel readable: the identity of the thing first,
#: then how many objects it is about, then the measurements.
PRIORITY_KEYS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "well", "prc", "prcf", "prcfo",
    "object_label", "condition", "n_objects",
    "count_cell", "count_nucleus", "count_pathogen",
)


def is_missing(value: Any) -> bool:
    """Report whether ``value`` counts as missing for an annotation.

    ``None`` and NaN are missing. So is any object whose ``!=`` against itself
    is undefined — that is how pandas' ``NA``/``NaT`` sentinels behave, and
    treating an "I refuse to say" sentinel as present would write the string
    ``<NA>`` into an OMERO panel.

    :param value: anything.
    :returns: ``True`` when the value is missing.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def format_map_value(value: Any, *, float_format: str = FLOAT_FORMAT,
                     max_chars: int = MAX_VALUE_CHARS) -> str:
    """Render ``value`` as the **string** an OMERO map annotation stores.

    OMERO map values are strings — there is no numeric type in the model — so
    the formatting decision is not cosmetic, it is the only thing that decides
    what a reader sees and what a re-parser gets back. The rules, all of them:

    * ``bool`` first, before the numeric branch, because ``True`` is an
      ``int`` and would otherwise render as ``1``. ``numpy.bool_`` needs its
      own check for the same reason: it is not a ``bool``, and it does answer
      ``__index__``.
    * integers render exactly: ``str(int(value))``, never through
      ``float_format``, so a 12-digit object id does not become ``1.23457e+11``.
    * floats go through :data:`FLOAT_FORMAT` — six significant digits, with
      trailing zeros dropped, so ``3.0`` is ``'3'`` and ``1.23456789e-5`` is
      ``'1.23457e-05'``.
    * ``inf`` and ``-inf`` render as ``'inf'`` / ``'-inf'`` rather than being
      hidden, because in spaCR they mean a ratio with a zero denominator and
      that is worth seeing.
    * missing values (see :func:`is_missing`) render as :data:`MISSING_TEXT`.
      Dropping them instead is what :data:`NAN_DROP` is for, at the
      :func:`measurement_pairs` level; here the value has to become *some*
      string.
    * anything else is ``str()``, stripped, newlines collapsed to spaces
      (a newline inside a map value breaks the panel's layout), and truncated
      to ``max_chars`` with a trailing ``…``.

    :param value: the value to render.
    :param float_format: the format string used for non-integral floats.
    :param max_chars: the length at which the text is truncated.
    :returns: a string, always.
    """
    if is_missing(value):
        return MISSING_TEXT
    if isinstance(value, bool):
        return "True" if value else "False"
    if getattr(getattr(value, "dtype", None), "kind", None) == "b":
        # np.bool_ is not a bool and does answer __index__, so without this it
        # would render as '1'. The dtype kind is the exact question to ask.
        return "True" if bool(value) else "False"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float_format.format(value)
    if not isinstance(value, (str, bytes)):
        # numpy scalars and Decimal reach here: they behave like numbers but
        # are instances of neither builtin type. `__index__` is the exact
        # question "is this an integer?" and is what distinguishes np.int64
        # from np.float64 without a dtype lookup.
        try:
            if hasattr(value, "__index__"):
                return str(int(value))
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isinf(number):
                return "inf" if number > 0 else "-inf"
            if math.isnan(number):
                return MISSING_TEXT
            return float_format.format(number)
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _clean_key(key: Any) -> str:
    text = " ".join(str(key).split())
    if len(text) > MAX_KEY_CHARS:
        text = text[: MAX_KEY_CHARS - 1] + "…"
    return text


def measurement_pairs(row: Mapping[str, Any], *,
                      columns: Optional[Sequence[str]] = None,
                      priority: Sequence[str] = PRIORITY_KEYS,
                      max_pairs: int = MAX_MAP_PAIRS,
                      nan_policy: str = NAN_KEEP,
                      extra: Optional[Mapping[str, Any]] = None,
                      float_format: str = FLOAT_FORMAT
                      ) -> Tuple[Tuple[str, str], ...]:
    """Project one measurement row onto the key/value pairs OMERO stores.

    Takes plain data — a ``dict``, or ``df.iloc[i].to_dict()`` — so that the
    projection can be tested without a DataFrame, a database or a server.

    **Why there is a cap.** A spaCR object table has 300-600 columns. A
    MapAnnotation with 400 entries renders in OMERO.web as a 400-row scrolling
    table in a 300 px panel, which nobody reads and which makes the panel
    useless for the annotations that *were* worth showing. So the annotation
    is a *summary*, capped at ``max_pairs`` (:data:`MAX_MAP_PAIRS` = 50 by
    default) and ordered so the useful keys are the ones that survive:
    ``priority`` keys first in the order given, then the row's own order.
    When anything is cut, the last pair is :data:`TRUNCATION_KEY`, saying how
    many were dropped and that the full table is the CSV attached under
    :data:`NS_FILE`. The full table is never *only* in the annotation.

    The result always has at most ``max_pairs`` entries, notice included.

    :param row: the measurement row, as a mapping.
    :param columns: restrict to these columns, in this order (after the
        priority keys). ``None`` uses every key in ``row``.
    :param priority: keys promoted to the front when present.
    :param max_pairs: the hard cap, including the truncation notice.
    :param nan_policy: :data:`NAN_KEEP` (default — a missing value becomes the
        string ``'NaN'``, so "measured and missing" is distinguishable from
        "never computed") or :data:`NAN_DROP` (the key is omitted).
    :param extra: pairs merged in *after* ``row``, overriding it — used for
        provenance such as the spaCR version.
    :param float_format: passed to :func:`format_map_value`.
    :returns: a tuple of ``(key, value)`` string pairs.
    :raises OmeroError: for an unknown ``nan_policy`` or ``max_pairs`` < 2
        (a cap of one leaves no room for both a measurement and a notice).
    """
    if nan_policy not in NAN_POLICIES:
        raise OmeroError(
            f"nan_policy={nan_policy!r} is not one of {NAN_POLICIES}.")
    if max_pairs < 2:
        raise OmeroError(
            f"max_pairs={max_pairs} leaves no room for a measurement and the "
            f"truncation notice; use at least 2.")

    merged: Dict[str, Any] = dict(row)
    if extra:
        merged.update(extra)

    if columns is None:
        wanted = list(merged)
    else:
        wanted = [c for c in columns if c in merged]

    ordered: List[str] = [k for k in priority if k in merged and k in wanted]
    seen = set(ordered)
    for key in wanted:
        if key not in seen:
            ordered.append(key)
            seen.add(key)

    rendered: List[Tuple[str, str]] = []
    for key in ordered:
        value = merged[key]
        if nan_policy == NAN_DROP and is_missing(value):
            continue
        rendered.append((_clean_key(key),
                         format_map_value(value, float_format=float_format)))

    if len(rendered) <= max_pairs:
        return tuple(rendered)

    kept = rendered[: max_pairs - 1]
    dropped = len(rendered) - len(kept)
    kept.append((
        TRUNCATION_KEY,
        f"showing {len(kept)} of {len(rendered)} values ({dropped} not shown); "
        f"the full table is attached as a CSV in namespace {NS_FILE}"))
    return tuple(kept)


def summarise_rows(rows: Sequence[Mapping[str, Any]], *,
                   columns: Optional[Sequence[str]] = None,
                   ) -> Dict[str, Any]:
    """Reduce many object rows to one per-well summary, in plain Python.

    The mean of each numeric column over the rows that actually have a value
    there, plus ``n_objects``. Missing values are *excluded from their own
    column's mean* rather than zero-filled: in spaCR a NaN is usually
    structural (a ``pathogen_*`` column is NaN for a cell with no pathogen in
    it), and zero-filling turns "no pathogen" into "a pathogen of zero size".
    A column with no usable value in any row is reported as ``None``, which
    :func:`measurement_pairs` renders as ``NaN``.

    Non-numeric columns are carried through when every row agrees on the
    value — that is how ``plateID``, ``rowID``, ``columnID`` and ``condition``
    reach the well annotation — and dropped when they disagree, because a
    single well cannot have two plate ids and picking one would be a guess.

    :param rows: the per-object rows for one well.
    :param columns: restrict to these columns; ``None`` uses the union of the
        rows' keys.
    :returns: a plain dict, ready for :func:`measurement_pairs`.
    """
    rows = list(rows)
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    else:
        keys = list(columns)

    summary: Dict[str, Any] = {"n_objects": len(rows)}
    for key in keys:
        numbers: List[float] = []
        others: List[Any] = []
        for row in rows:
            if key not in row:
                continue
            value = row[key]
            if is_missing(value):
                continue
            if isinstance(value, bool) or isinstance(value, str):
                others.append(value)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                others.append(value)
                continue
            if math.isnan(number):
                continue
            numbers.append(number)
        if numbers and not others:
            summary[key] = sum(numbers) / len(numbers)
        elif others and not numbers:
            unique = {str(v) for v in others}
            summary[key] = others[0] if len(unique) == 1 else None
        elif not numbers and not others:
            summary[key] = None
        else:
            # Mixed text and numbers in one column: there is no honest
            # summary, so say nothing rather than average half of it.
            summary[key] = None
    return summary


def well_summary_pairs(rows: Sequence[Mapping[str, Any]], *,
                       columns: Optional[Sequence[str]] = None,
                       **kwargs: Any) -> Tuple[Tuple[str, str], ...]:
    """Summarise object rows and project them onto key/value pairs.

    :func:`summarise_rows` followed by :func:`measurement_pairs`, which is the
    combination the per-well export always wants.

    :param rows: the per-object rows for one well.
    :param columns: restrict the summary to these columns.
    :param kwargs: passed to :func:`measurement_pairs`.
    :returns: a tuple of ``(key, value)`` string pairs.
    """
    return measurement_pairs(summarise_rows(rows, columns=columns), **kwargs)


# ---------------------------------------------------------------------------
# Export: the replace-or-append decision
# ---------------------------------------------------------------------------

#: Update spaCR's own previous annotation in place. The default.
REPLACE = "replace"
#: Always create a new annotation, keeping the old one.
APPEND = "append"
#: Both modes.
ANNOTATION_MODES: Tuple[str, ...] = (REPLACE, APPEND)

#: A new annotation is created and linked.
ACTION_CREATE = "create"
#: An existing spaCR annotation is overwritten in place.
ACTION_UPDATE = "update"
#: Nothing to do — what is already there is what would be written.
ACTION_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class AnnotationPlan:
    """What an export is about to do, decided before anything is touched.

    :param action: :data:`ACTION_CREATE`, :data:`ACTION_UPDATE` or
        :data:`ACTION_UNCHANGED`.
    :param annotation_id: the annotation to update, for
        :data:`ACTION_UPDATE`/:data:`ACTION_UNCHANGED`; ``None`` for a create.
    :param namespace: the namespace the annotation lives in.
    :param duplicates: other spaCR annotations already in that namespace,
        left untouched and reported.
    :param reason: one sentence explaining the choice, for the result and the
        log.
    """

    action: str
    namespace: str
    annotation_id: Optional[int] = None
    duplicates: Tuple[int, ...] = ()
    reason: str = ""


def _own(existing: Iterable[Sequence[Any]], namespace: str) -> List[Sequence[Any]]:
    """Return the entries whose namespace is exactly ``namespace``."""
    return [entry for entry in existing
            if len(entry) >= 2 and entry[1] == namespace]


def plan_annotation(existing: Iterable[Sequence[Any]], namespace: str,
                    mode: str = REPLACE) -> AnnotationPlan:
    """Decide whether to overwrite spaCR's previous annotation or add one.

    Pure: ``existing`` is a sequence of ``(annotation_id, namespace)`` pairs,
    which is all the decision needs, so the rule is testable without a server.

    Under :data:`REPLACE` — the default — an annotation whose namespace is
    *exactly* ``namespace`` is updated in place. Nothing is created, nothing
    is unlinked and **nothing is deleted**; a re-run therefore leaves one
    annotation rather than a growing pile, and an annotation in any other
    namespace (a colleague's, OMERO's own bulk annotations) is never even
    considered. When more than one already exists — only possible after an
    :data:`APPEND` run or a concurrent writer — the oldest (lowest id) is
    updated and the rest are reported in ``duplicates`` rather than removed,
    because deleting an annotation somebody may have linked elsewhere is not a
    decision an exporter gets to make.

    :param existing: ``(id, namespace)`` pairs already on the target.
    :param namespace: the spaCR namespace being written.
    :param mode: :data:`REPLACE` or :data:`APPEND`.
    :returns: an :class:`AnnotationPlan`.
    :raises OmeroError: for an unknown mode, or a namespace spaCR does not own
        (this module refuses to write outside :data:`SPACR_NAMESPACES`, which
        is the guarantee that makes "replace" safe).
    """
    _check_mode_and_namespace(mode, namespace)
    if mode == APPEND:
        return AnnotationPlan(
            action=ACTION_CREATE, namespace=namespace,
            reason="mode=append: a new annotation is added and nothing "
                   "existing is touched.")

    mine = sorted(_own(existing, namespace), key=lambda entry: entry[0])
    if not mine:
        return AnnotationPlan(
            action=ACTION_CREATE, namespace=namespace,
            reason=f"no annotation in {namespace} yet.")
    oldest = mine[0]
    duplicates = tuple(int(entry[0]) for entry in mine[1:])
    reason = f"updating annotation {oldest[0]} in {namespace} in place."
    if duplicates:
        reason += (f" {len(duplicates)} further annotation(s) in the same "
                   f"namespace were left alone: {list(duplicates)}. Remove "
                   f"them in OMERO.web if they are unwanted; spaCR does not "
                   f"delete annotations.")
    return AnnotationPlan(action=ACTION_UPDATE, namespace=namespace,
                          annotation_id=int(oldest[0]),
                          duplicates=duplicates, reason=reason)


def plan_tag(existing: Iterable[Sequence[Any]], namespace: str, text: str,
             mode: str = REPLACE) -> AnnotationPlan:
    """Decide what to do about a categorical verdict tag.

    Tags get their own rule, and the reason is a property of OMERO rather than
    a preference: **a TagAnnotation is a shared object**. The tag reading
    ``hit`` on this image is very likely the same row in the database as the
    tag reading ``hit`` on two hundred other images, so editing its text to
    say ``not hit`` would silently relabel all of them. Updating in place is
    therefore not available here, whatever the mode.

    What happens instead:

    * the identical verdict is already linked -> :data:`ACTION_UNCHANGED`,
      nothing is written and the export is idempotent;
    * no spaCR verdict is linked -> :data:`ACTION_CREATE`;
    * a *different* spaCR verdict is linked -> :data:`ACTION_CREATE` for the
      new one, and the old one is reported in ``duplicates`` and left in
      place. Two verdicts on one object is visibly wrong in OMERO.web, which
      is better than silently rewriting history.

    :param existing: ``(id, namespace, text)`` triples already on the target.
    :param namespace: the spaCR verdict namespace.
    :param text: the verdict being written.
    :param mode: :data:`REPLACE` or :data:`APPEND`; :data:`APPEND` skips even
        the idempotence check.
    :returns: an :class:`AnnotationPlan`.
    :raises OmeroError: for an unknown mode or a namespace spaCR does not own.
    """
    _check_mode_and_namespace(mode, namespace)
    if mode == APPEND:
        return AnnotationPlan(
            action=ACTION_CREATE, namespace=namespace,
            reason="mode=append: a new tag is linked unconditionally.")

    mine = sorted(_own(existing, namespace), key=lambda entry: entry[0])
    same = [entry for entry in mine if len(entry) >= 3 and str(entry[2]) == str(text)]
    if same:
        return AnnotationPlan(
            action=ACTION_UNCHANGED, namespace=namespace,
            annotation_id=int(same[0][0]),
            duplicates=tuple(int(e[0]) for e in mine if e[0] != same[0][0]),
            reason=f"the verdict {text!r} is already linked; nothing to do.")
    stale = tuple(int(entry[0]) for entry in mine)
    reason = f"linking the verdict {text!r}."
    if stale:
        reason += (f" A tag is a shared object, so the previous verdict(s) "
                   f"{list(stale)} were left linked rather than renamed — "
                   f"unlink them in OMERO.web if they are wrong.")
    return AnnotationPlan(action=ACTION_CREATE, namespace=namespace,
                          duplicates=stale, reason=reason)


def _check_mode_and_namespace(mode: str, namespace: str) -> None:
    if mode not in ANNOTATION_MODES:
        raise OmeroError(
            f"mode={mode!r} is not one of {ANNOTATION_MODES}.")
    if not is_spacr_namespace(namespace):
        raise OmeroError(
            f"{namespace!r} is not a spaCR namespace. This module only writes "
            f"to {sorted(SPACR_NAMESPACES)}, which is what makes replacing an "
            f"annotation safe: it can only ever replace its own.")


# ---------------------------------------------------------------------------
# Export: the adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationResult:
    """What one annotation write did.

    :param action: :data:`ACTION_CREATE`, :data:`ACTION_UPDATE` or
        :data:`ACTION_UNCHANGED`.
    :param namespace: the namespace written.
    :param annotation_id: the annotation's id, when the server gave one.
    :param n_pairs: how many key/value pairs were written (map annotations).
    :param duplicates: other spaCR annotations in the same namespace, left
        untouched.
    :param reason: the plan's explanation, carried through.
    """

    action: str
    namespace: str
    annotation_id: Optional[int] = None
    n_pairs: int = 0
    duplicates: Tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ExportResult:
    """What a multi-target export did.

    :param namespace: the namespace written.
    :param results: one :class:`AnnotationResult` per target, keyed in
        ``targets`` order.
    :param targets: the target labels (well names, image ids) in order.
    :param missing: labels that were asked for but not found on the container.
    """

    namespace: str
    results: Tuple[AnnotationResult, ...] = ()
    targets: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()

    def describe(self) -> str:
        """Return a one-line summary of the export.

        :returns: counts of created, updated and unchanged annotations.
        """
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.action] = counts.get(result.action, 0) + 1
        parts = [f"{count} {action}" for action, count in sorted(counts.items())]
        text = f"{self.namespace}: {', '.join(parts) or 'nothing to do'}"
        if self.missing:
            text += f"; {len(self.missing)} target(s) not found: {list(self.missing)}"
        return text


def list_spacr_annotations(target: Any) -> Tuple[Tuple[int, Optional[str]], ...]:
    """Return the ``(id, namespace)`` of every spaCR annotation on ``target``.

    ``listAnnotations()`` is deliberately called **without** a namespace
    filter and the filtering is done here. The filter is what decides whether
    an annotation gets written to, and that decision must not depend on a
    server honouring a query parameter: a server that ignored ``ns=`` would
    otherwise hand back a stranger's annotation and this module would update
    it.

    :param target: any OMERO object wrapper with ``listAnnotations()``.
    :returns: ``(id, namespace)`` pairs, only for namespaces in
        :data:`SPACR_NAMESPACES`.
    """
    found: List[Tuple[int, Optional[str]]] = []
    for annotation in _call(target, "listAnnotations", default=()) or ():
        namespace = _call(annotation, "getNs", default=None)
        if is_spacr_namespace(namespace):
            found.append((int(_call(annotation, "getId", default=0) or 0), namespace))
    return tuple(found)


def _annotations_with_ns(target: Any, namespace: str) -> List[Tuple[int, Any, Any]]:
    """Return ``(id, namespace, wrapper)`` for annotations in ``namespace``."""
    out: List[Tuple[int, Any, Any]] = []
    for annotation in _call(target, "listAnnotations", default=()) or ():
        ns = _call(annotation, "getNs", default=None)
        if ns == namespace:
            out.append((int(_call(annotation, "getId", default=0) or 0),
                        ns, annotation))
    return out


def _make_annotation(gateway: Any, wrapper_name: str,
                     annotation_factory: Optional[Any]) -> Any:
    if annotation_factory is not None:
        return annotation_factory(wrapper_name, gateway)
    gateway_module = require_omero()
    return getattr(gateway_module, wrapper_name)(gateway)


def export_map_annotation(gateway: Any, target: Any,
                          pairs: Sequence[Sequence[Any]], *,
                          namespace: str = NS_MEASUREMENTS,
                          mode: str = REPLACE,
                          annotation_factory: Optional[Any] = None,
                          ) -> AnnotationResult:
    """Write spaCR measurements onto an OMERO object as key/value pairs.

    A MapAnnotation is what OMERO.web renders in the right-hand panel, which
    makes it the only annotation an OMERO user reliably *sees*. Build the
    pairs with :func:`measurement_pairs` (per object) or
    :func:`well_summary_pairs` (per well) — both already do the string
    conversion OMERO requires and the truncation the panel requires.

    :param gateway: a connected gateway; used only to construct the wrapper.
    :param target: the Image, Well, Dataset or Plate wrapper to annotate.
    :param pairs: the key/value pairs. Values are cast to ``str`` here as a
        last line of defence — OMERO's model has no numeric map value.
    :param namespace: which spaCR namespace to write. Must be one spaCR owns.
    :param mode: :data:`REPLACE` (default: update spaCR's own previous
        annotation in place) or :data:`APPEND`.
    :param annotation_factory: callable ``(wrapper_name, gateway)`` returning
        a new annotation wrapper. Defaults to the real
        ``omero.gateway.MapAnnotationWrapper``; the tests pass their own.
    :returns: an :class:`AnnotationResult`.
    :raises OmeroError: for an unknown mode or a foreign namespace.
    """
    clean = tuple((str(key), str(value)) for key, value in pairs)
    existing = _annotations_with_ns(target, namespace)
    plan = plan_annotation([(e[0], e[1]) for e in existing], namespace, mode)

    if plan.action == ACTION_UPDATE:
        wrapper = next(e[2] for e in existing if e[0] == plan.annotation_id)
        wrapper.setValue(list(clean))
        wrapper.save()
        LOGGER.info("OMERO map annotation %s: %s", plan.annotation_id, plan.reason)
        return AnnotationResult(action=ACTION_UPDATE, namespace=namespace,
                                annotation_id=plan.annotation_id,
                                n_pairs=len(clean), duplicates=plan.duplicates,
                                reason=plan.reason)

    wrapper = _make_annotation(gateway, "MapAnnotationWrapper", annotation_factory)
    wrapper.setNs(namespace)
    wrapper.setValue(list(clean))
    wrapper.save()
    target.linkAnnotation(wrapper)
    annotation_id = _call(wrapper, "getId", default=None)
    LOGGER.info("OMERO map annotation created in %s: %s", namespace, plan.reason)
    return AnnotationResult(
        action=ACTION_CREATE, namespace=namespace,
        annotation_id=None if annotation_id is None else int(annotation_id),
        n_pairs=len(clean), duplicates=plan.duplicates, reason=plan.reason)


def export_tag_annotation(gateway: Any, target: Any, text: str, *,
                          namespace: str = NS_TAG,
                          mode: str = REPLACE,
                          annotation_factory: Optional[Any] = None,
                          ) -> AnnotationResult:
    """Attach a categorical verdict — ``'hit'``, ``'QC fail'`` — as a tag.

    See :func:`plan_tag` for why a tag is never edited in place: it is a
    shared object, and renaming it would relabel every other object carrying
    it. This is idempotent for an unchanged verdict and additive (with a
    report) for a changed one, and it never unlinks or deletes.

    :param gateway: a connected gateway.
    :param target: the object to tag.
    :param text: the verdict.
    :param namespace: the spaCR verdict namespace.
    :param mode: :data:`REPLACE` (default) or :data:`APPEND`.
    :param annotation_factory: callable ``(wrapper_name, gateway)``; defaults
        to ``omero.gateway.TagAnnotationWrapper``.
    :returns: an :class:`AnnotationResult`.
    :raises OmeroError: for an unknown mode, a foreign namespace, or an empty
        verdict (a tag with no text is invisible in OMERO.web).
    """
    verdict = str(text).strip()
    if not verdict:
        raise OmeroError(
            "an empty verdict would create a tag with no text, which is "
            "invisible in OMERO.web. Pass the verdict, e.g. 'hit'.")

    existing = _annotations_with_ns(target, namespace)
    triples = [(e[0], e[1], _call(e[2], "getValue", "getTextValue", default=None))
               for e in existing]
    plan = plan_tag(triples, namespace, verdict, mode)

    if plan.action == ACTION_UNCHANGED:
        LOGGER.info("OMERO verdict unchanged: %s", plan.reason)
        return AnnotationResult(action=ACTION_UNCHANGED, namespace=namespace,
                                annotation_id=plan.annotation_id,
                                duplicates=plan.duplicates, reason=plan.reason)

    wrapper = _make_annotation(gateway, "TagAnnotationWrapper", annotation_factory)
    wrapper.setNs(namespace)
    wrapper.setValue(verdict)
    wrapper.save()
    target.linkAnnotation(wrapper)
    annotation_id = _call(wrapper, "getId", default=None)
    LOGGER.info("OMERO verdict %r linked: %s", verdict, plan.reason)
    return AnnotationResult(
        action=ACTION_CREATE, namespace=namespace,
        annotation_id=None if annotation_id is None else int(annotation_id),
        duplicates=plan.duplicates, reason=plan.reason)


def export_file_annotation(gateway: Any, target: Any,
                           path: Union[str, os.PathLike], *,
                           namespace: str = NS_FILE,
                           mimetype: Optional[str] = None,
                           description: Optional[str] = None,
                           ) -> AnnotationResult:
    """Attach a results CSV or a figure PDF to an OMERO object.

    The counterpart to the truncated MapAnnotation: the panel gets the fifty
    numbers worth reading, this gets the whole table.

    **File annotations always append**, and that is the non-destructive
    choice rather than an oversight. The bytes of an OriginalFile cannot be
    rewritten through the gateway, so "replace" would have to mean *delete the
    previous attachment* — deleting evidence of an earlier run, which is
    exactly what this module refuses to do anywhere. Previous spaCR file
    annotations are reported in :attr:`AnnotationResult.duplicates`, and the
    description carries a UTC timestamp so the newest is identifiable.

    :param gateway: a connected gateway; ``createFileAnnfromLocalFile`` is
        called on it.
    :param target: the Dataset, Plate, Image or Well to attach to.
    :param path: the local file.
    :param namespace: the spaCR file namespace.
    :param mimetype: overrides the type guessed from the extension.
    :param description: overrides the generated description.
    :returns: an :class:`AnnotationResult` with ``action=create``.
    :raises OmeroError: for a foreign namespace, or a path that is not a file.
    """
    _check_mode_and_namespace(APPEND, namespace)
    local = Path(path)
    if not local.is_file():
        raise OmeroError(f"{local} is not a file, so there is nothing to attach.")

    guessed = mimetype or mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    desc = description or f"spaCR {get_version()} results, written {stamp}"

    previous = tuple(entry[0] for entry in _annotations_with_ns(target, namespace))
    wrapper = gateway.createFileAnnfromLocalFile(
        str(local), mimetype=guessed, ns=namespace, desc=desc)
    target.linkAnnotation(wrapper)
    annotation_id = _call(wrapper, "getId", default=None)
    reason = f"attached {local.name} ({guessed}) in {namespace}."
    if previous:
        reason += (f" {len(previous)} earlier spaCR attachment(s) {list(previous)} "
                   f"were left in place: file contents cannot be replaced "
                   f"through the gateway and spaCR does not delete.")
    LOGGER.info("OMERO file annotation: %s", reason)
    return AnnotationResult(
        action=ACTION_CREATE, namespace=namespace,
        annotation_id=None if annotation_id is None else int(annotation_id),
        duplicates=previous, reason=reason)


def export_plate_summaries(gateway: Any, ref: Any,
                           summaries: Mapping[str, Sequence[Mapping[str, Any]]],
                           *,
                           namespace: str = NS_WELL_SUMMARY,
                           mode: str = REPLACE,
                           annotation_factory: Optional[Any] = None,
                           **pair_kwargs: Any) -> ExportResult:
    """Write one per-well MapAnnotation per well of an OMERO Plate.

    This is the half that closes the loop: spaCR measured the plate, and the
    numbers land back on the wells they came from, where the person at the
    microscope will actually see them. Wells are matched by *name*
    (``'A01'``), which round-trips through :func:`well_position` and
    :func:`omero_indices`, so the mapping used on the way in is the mapping
    used on the way out.

    :param gateway: a connected gateway.
    :param ref: a Plate id, ``'Plate:4711'``, or an OMERO.web URL.
    :param summaries: ``{well_name: [object_row, ...]}``. The rows are
        summarised with :func:`summarise_rows`; pass a single already-summarised
        row as a one-element list to skip the aggregation.
    :param namespace: the spaCR well-summary namespace.
    :param mode: :data:`REPLACE` (default) or :data:`APPEND`.
    :param annotation_factory: callable ``(wrapper_name, gateway)``.
    :param pair_kwargs: passed to :func:`measurement_pairs`.
    :returns: an :class:`ExportResult`; wells named in ``summaries`` that the
        Plate does not have are listed in :attr:`ExportResult.missing` rather
        than raising, because a plate map with an extra control well in it is
        a normal thing to hand in.
    :raises OmeroIdError: when ``ref`` does not name a Plate.
    :raises OmeroContainerError: when the Plate does not exist.
    """
    object_id = parse_object_id(ref, expect="Plate")
    container = _resolve(gateway, "Plate", object_id)

    wanted = {str(name).strip().upper(): rows for name, rows in summaries.items()}
    results: List[AnnotationResult] = []
    labels: List[str] = []
    seen: set = set()

    for well in _iter_wells(container):
        try:
            position = well_position(_call(well, "getRow"), _call(well, "getColumn"))
        except OmeroWellError:
            continue
        rows = wanted.get(position.well)
        if rows is None:
            continue
        seen.add(position.well)
        pairs = well_summary_pairs(rows, **pair_kwargs)
        results.append(export_map_annotation(
            gateway, well, pairs, namespace=namespace, mode=mode,
            annotation_factory=annotation_factory))
        labels.append(position.well)

    missing = tuple(sorted(set(wanted) - seen))
    if missing:
        LOGGER.warning(
            "Plate:%s has no well(s) %s; their summaries were not written.",
            object_id, list(missing))
    return ExportResult(namespace=namespace, results=tuple(results),
                        targets=tuple(labels), missing=missing)
