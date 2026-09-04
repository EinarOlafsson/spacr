"""One choice a biologist recognises, in front of fifty-three they do not.

`organelle` is the most over-configured object class in spaCR: 53 settings
reach the mask pipeline, and a user who knows they are imaging lysosomes has
to know what a ridge filter is, which of `organelle_method`'s seven values
are legal for which morphology, and what `organelle_network_threshold` does
when they are not segmenting a network.

WHAT THIS IS NOT. It is not a taxonomy compiled into a pipeline. The nine
cell-biology categories do not map one-to-one onto the four
``organelle_morphology`` values:

    'spots'      punctate (vesicles, lipid droplets)
    'network'    filamentous (mitochondria, ER tubules)
    'irregular'  solid blobby (Golgi, lysosomes)
    'ring'       hollow (endosomes, autophagosomes)

"Vesicular" is a CELL-BIOLOGY category -- a membrane-bound compartment that
carries cargo -- and spots/ring/network/irregular is an IMAGE-APPEARANCE
category: what the segmentation has to find. They do not nest, because the
same biological family looks different at different sizes. A 200 nm
transport vesicle is a diffraction-limited dot. A 2 um vacuole is a visible
ring. Both are Vesicular.

So the mapping is not ``type -> morphology``. It is

    (type, expected size) -> morphology

and hard-coding one morphology per type would be wrong for half the entries
in the request's own list, silently: the user picks Vesicular, gets a spot
detector, and their lysosomes come out as rings of holes.

WHAT THIS IS. A named PRESET that sets several settings at once and says
what it set and why. Choosing a type fills the advanced settings with
recommended values and leaves every one of them editable. Nothing here
overrides a value the user chose; :func:`preset_for` returns a
recommendation and :func:`apply_preset` fills only what the user has not
set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

#: The value that reproduces today's behaviour exactly. It is the default so
#: that no existing settings CSV changes meaning on load: a file written
#: before `organelle_type` existed has no opinion about it, and 'custom'
#: recommends nothing.
DEFAULT_TYPE = "custom"

#: Organelle slots whose setting captions remain materialized in every
#: runtime translation catalog. This is the four-slot legacy catalog
#: contract, not the number a new form should display: fresh forms correctly
#: start with zero organelles, while slots five through twenty-six reuse the
#: primary slot's source-bound translation at runtime.
CATALOGUED_ORGANELLE_SLOTS = 4

#: Above this diameter, in PIXELS, a round compartment's lumen is resolvable
#: and it images as a ring rather than a filled dot.
#:
#: The number is a working threshold, not a physical constant, and it is
#: named here so it can be argued with rather than buried in a branch. At a
#: typical 60x/1.4 confocal sampling of ~100 nm/px, 15 px is ~1.5 um -- an
#: object whose membrane and lumen are several pixels apart. Below it the
#: PSF fills the middle and a hollow vesicle is indistinguishable from a
#: solid one, so asking for a ring detector produces holes that are not
#: there.
RING_RESOLVABLE_PX = 15


@dataclass(frozen=True)
class OrganelleType:
    """One named preset: what it is, what it looks like, what to run.

    :param label: what the user picks.
    :param members: the structures included under this name.
        Kept verbatim so the list a biologist recognises is the list they
        see.
    :param morphology: the `organelle_morphology` this maps to, or None when
        SIZE DECIDES -- see :attr:`size_split`.
    :param size_split: ``(small_morphology, large_morphology)``, used when
        ``morphology`` is None. The split point is
        :data:`RING_RESOLVABLE_PX`.
    :param method: the recommended `organelle_method`, which must be legal
        for the resulting morphology.
    :param params: further recommended settings.
    :param caveat: what is honestly weak about this preset. Shown to the
        user rather than hidden, because a preset that quietly does
        something adjacent to what its name says is worse than one that
        admits it.
    """

    label: str
    members: Tuple[str, ...]
    method: str
    morphology: Optional[str] = None
    size_split: Optional[Tuple[str, str]] = None
    params: Mapping[str, object] = field(default_factory=dict)
    caveat: str = ""

    def morphology_for(self, diameter_px: Optional[float]) -> Optional[str]:
        """The morphology this type implies at ``diameter_px``.

        :returns: one of the four morphologies, or None for ``custom``,
            which deliberately recommends nothing.
        """
        if self.morphology is not None:
            return self.morphology
        if self.size_split is None:
            return None
        small, large = self.size_split
        if diameter_px is None:
            return small
        return large if float(diameter_px) >= RING_RESOLVABLE_PX else small


# ---------------------------------------------------------------------------
# The table. One row per category the maintainer listed, with the rows where
# SIZE DECIDES marked, and the rows with no dedicated detector said out loud.
# ---------------------------------------------------------------------------

ORGANELLE_TYPES: Dict[str, OrganelleType] = {
    "custom": OrganelleType(
        label="Custom (no preset)",
        members=(),
        method="",
        morphology=None,
        caveat="Recommends nothing and changes nothing. This is the default "
               "so a settings file written before organelle_type existed "
               "keeps its exact meaning.",
    ),

    # -- size decides -------------------------------------------------------
    "vesicular": OrganelleType(
        label="Vesicular",
        members=("vacuoles", "autophagosomes", "transport vesicles",
                 "secretory vesicles", "early endosomes", "lysosomes"),
        # Split, and this is the row that proves the mapping is not
        # one-to-one: a transport vesicle is a dot and a vacuole is a ring,
        # and both are on the maintainer's own Vesicular list.
        size_split=("spots", "ring"),
        method="log",
        params={"organelle_watershed_spots": True,
                "organelle_log_min_sigma": 1,
                "organelle_log_max_sigma": 6},
        caveat="Size decides the detector here, not the name. Below "
               f"{RING_RESOLVABLE_PX} px diameter these are dots; above it "
               "their lumen resolves and they are rings. LYSOSOMES are the "
               "exception on this list -- they image as solid blobby, so "
               "set organelle_morphology to 'irregular' for those.",
    ),
    "spherical": OrganelleType(
        label="Spherical",
        members=("nucleus", "nucleolus", "chloroplasts", "swollen "
                 "mitochondria", "macromolecular condensates"),
        size_split=("spots", "irregular"),
        method="otsu",
        params={"organelle_fill_holes": 64},
        caveat="Small condensates are dots; a nucleus or chloroplast is a "
               "large solid object and thresholds better than it blob-"
               "detects.",
    ),

    # -- one morphology each ------------------------------------------------
    "punctate": OrganelleType(
        label="Punctate",
        members=("peroxisomes", "lipid droplets", "ribosomes",
                 "PML nuclear bodies", "vaults"),
        morphology="spots",
        method="log",
        params={"organelle_watershed_spots": True,
                "organelle_log_min_sigma": 1,
                "organelle_log_max_sigma": 5,
                "organelle_log_threshold": 0.01},
        caveat="",
    ),
    "filamentous": OrganelleType(
        label="Filamentous",
        members=("microtubules", "F-actin filaments",
                 "intermediate filaments"),
        morphology="network",
        method="ridge",
        params={"organelle_ridge_filter": "frangi",
                "organelle_ridge_sigmas": [1, 2, 3],
                "organelle_skeletonize": True},
        caveat="",
    ),
    "tubular": OrganelleType(
        label="Tubular",
        members=("smooth endoplasmic reticulum",
                 "healthy mitochondrial networks", "sorting endosomes",
                 "trans-Golgi network"),
        morphology="network",
        method="ridge",
        params={"organelle_ridge_filter": "sato",
                "organelle_ridge_sigmas": [1, 2, 4],
                "organelle_skeletonize": False},
        caveat="Sorting endosomes on this list are only tubular where they "
               "are genuinely tubulated; rounded ones belong under "
               "Vesicular.",
    ),
    "reticular": OrganelleType(
        label="Reticular",
        members=("endoplasmic reticulum meshwork",
                 "interconnected mitochondrial reticula"),
        morphology="network",
        method="hysteresis",
        params={"organelle_hysteresis_low": 0.2,
                "organelle_hysteresis_high": 0.6,
                "organelle_skeletonize": False},
        caveat="Hysteresis rather than a ridge filter because what matters "
               "in a meshwork is CONNECTIVITY -- a dual threshold keeps a "
               "faint strand that links two bright ones, and a per-pixel "
               "tubeness score drops it.",
    ),
    "cisternal": OrganelleType(
        label="Cisternal",
        members=("rough endoplasmic reticulum sheets", "Golgi stacks",
                 "nuclear envelope"),
        morphology="irregular",
        method="adaptive",
        params={"organelle_adaptive_block_size": 51,
                "organelle_adaptive_offset": 5},
        caveat="THERE IS NO SHEET DETECTOR. A cisterna imaged in the plane "
               "of the sheet is a solid patch and 'irregular' is right; "
               "imaged edge-on it is a line and looks filamentous. If your "
               "sheets come out fragmented, try 'network'.",
    ),

    # -- no dedicated detector, and the file says so ------------------------
    "toroidal": OrganelleType(
        label="Toroidal",
        members=("condensing or stressed mitochondria", "midbodies",
                 "nuclear pore complexes"),
        morphology="ring",
        method="dog",
        params={"organelle_ring_sigma_inner": 1.0,
                "organelle_ring_sigma_outer": 3.0,
                "organelle_ring_fill_method": "flood"},
        caveat="THERE IS NO TOROID DETECTOR. This is 'ring' plus a shape "
               "filter, which finds the hole but does not test that it is "
               "closed. Nuclear pore complexes are far below the "
               "diffraction limit and will image as dots whatever is set "
               "here -- use Punctate for those.",
    ),
    "crescent": OrganelleType(
        label="Crescent",
        members=("phagophores or early autophagosomes",
                 "specialized yeast nucleoli"),
        morphology="ring",
        method="dog",
        params={"organelle_ring_sigma_inner": 1.0,
                "organelle_ring_sigma_outer": 4.0,
                "organelle_ring_min_prominence": 0.05,
                "organelle_ring_fill_method": "convex"},
        caveat="THERE IS NO CRESCENT DETECTOR. This is 'ring' with the "
               "prominence lowered so an OPEN arc still registers, and a "
               "convex fill so the unclosed side does not leak. A phagophore "
               "that has sealed is an autophagosome and belongs under "
               "Vesicular.",
    ),
}

#: Display order for the picker: no-op first, followed by the two presets that
#: select a detector by object size and then the remaining morphologies.
TYPE_ORDER: Tuple[str, ...] = (
    "custom", "punctate", "vesicular", "spherical", "filamentous",
    "tubular", "reticular", "cisternal", "toroidal", "crescent",
)

#: Which `organelle_method` values are legal for each morphology. Mirrors
#: `spacr.object._validate_organelle_settings`, which raises on a bad pair
#: before any image is loaded. Duplicated here ONLY so this module's own
#: tests can prove every preset it ships is legal -- the validator remains
#: the authority at run time.
LEGAL_METHODS: Dict[str, Tuple[str, ...]] = {
    "spots": ("otsu", "adaptive", "log", "dog", "cellpose"),
    "network": ("otsu", "adaptive", "ridge", "hysteresis", "cellpose",
                "unet"),
    "irregular": ("otsu", "adaptive", "cellpose"),
    "ring": ("otsu", "adaptive", "dog", "log", "cellpose"),
}


def known_types() -> Tuple[str, ...]:
    """Every `organelle_type`, in the order the picker shows them."""
    return TYPE_ORDER


def resolve_type(name: Optional[str]) -> OrganelleType:
    """The preset called ``name``.

    :raises ValueError: for an unknown name, listing the known ones.
        Falling back to 'custom' would mean a typo silently segmented with
        different settings than the user asked for.
    """
    key = str(name or DEFAULT_TYPE).strip().lower()
    if key not in ORGANELLE_TYPES:
        raise ValueError(
            f"organelle_type={name!r} is not one of {list(TYPE_ORDER)}")
    return ORGANELLE_TYPES[key]


def preset_for(name: Optional[str],
               diameter_px: Optional[float] = None) -> Dict[str, object]:
    """What this type RECOMMENDS. It does not apply anything.

    :param name: organelle type name resolved by :func:`resolve_type`; blank
        or ``custom`` yields no recommended settings, while an unknown name
        raises instead of silently changing the segmentation.
    :param diameter_px: the value of `organelle_diameter`. It is half the
        mapping: the same type is a dot at one size and a ring at another.
    :returns: settings to their recommended values, empty for 'custom'.
    """
    preset = resolve_type(name)
    morphology = preset.morphology_for(diameter_px)
    if morphology is None:
        return {}

    out: Dict[str, object] = {"organelle_morphology": morphology}
    # The method has to be legal for the morphology that SIZE just chose,
    # not for the one the table lists first. A Vesicular preset recommending
    # 'log' is fine for spots and fine for ring; one recommending 'ridge'
    # would raise the moment the user's diameter tipped it into a ring.
    if preset.method and preset.method in LEGAL_METHODS[morphology]:
        out["organelle_method"] = preset.method
    else:
        out["organelle_method"] = LEGAL_METHODS[morphology][0]
    out.update(dict(preset.params))
    return out


def apply_preset(settings: Mapping[str, object],
                 *, explain: bool = False) -> Dict[str, object]:
    """Fill unset organelle settings from the selected preset.

    Parameters
    ----------
    settings : mapping
        Run settings containing ``organelle_type`` and, when relevant,
        ``organelle_diameter``. The input mapping is not modified.
    explain : bool, default=False
        Print the selected morphology, values filled by the preset, values
        retained from ``settings``, and any preset caveat.

    Returns
    -------
    dict
        Copy of ``settings`` with missing or ``None`` preset keys filled.
        Existing non-``None`` values are preserved so users can adjust the
        recommended method or thresholds.

    Raises
    ------
    ValueError
        If ``organelle_type`` does not name a known preset.
    """
    out = dict(settings)
    name = out.get("organelle_type", DEFAULT_TYPE)
    preset = resolve_type(name)
    recommended = preset_for(name, out.get("organelle_diameter"))

    applied, kept = {}, {}
    for key, value in recommended.items():
        if key in out and out[key] is not None:
            kept[key] = out[key]
        else:
            out[key] = value
            applied[key] = value

    if explain and preset.label:
        _explain(preset, out.get("organelle_diameter"), applied, kept)
    return out


def _explain(preset: OrganelleType, diameter, applied, kept) -> None:
    """Print how an organelle preset affected resolved settings.

    :param preset: Resolved organelle preset being explained.
    :param diameter: Requested organelle diameter shown for size-split presets.
    :param applied: Setting values supplied by the preset.
    :param kept: Existing setting values retained instead of preset values.
    :returns: ``None``.
    """
    if not applied and not kept:
        return
    print(f"organelle_type = {preset.label}")
    if preset.members:
        print(f"  covers: {', '.join(preset.members)}")
    if preset.size_split:
        small, large = preset.size_split
        print(f"  size decides the detector: below {RING_RESOLVABLE_PX} px "
              f"'{small}', at or above it '{large}'; "
              f"organelle_diameter is {diameter}")
    for key, value in sorted(applied.items()):
        print(f"  set    {key} = {value!r}")
    for key, value in sorted(kept.items()):
        print(f"  KEPT   {key} = {value!r} (yours, not overridden)")
    if preset.caveat:
        print(f"  note: {preset.caveat}")


#: Settings that stay in the plain "Organelle" category: the ones a
#: biologist recognises without knowing how segmentation works. Everything
#: else moves to "Organelle advanced" while remaining visible and editable.
BASIC_SETTINGS: Tuple[str, ...] = (
    "organelle_channel",
    "organelle_type",
    "organelle_diameter",
    "organelle_min_area",
    "organelle_max_area",
    "organelle_mask_within_cells",
    "organelle_remove_border",
)


def is_basic(setting: str) -> bool:
    """True when ``setting`` belongs in the plain Organelle category."""
    return str(setting) in BASIC_SETTINGS


# ---------------------------------------------------------------------------
# THE SLOTS. How many organelles a run has, and what each one's keys are
# called.
# ---------------------------------------------------------------------------
# An organelle SLOT is one segmented object with its own channel, its own
# type preset and its own copy of every detection setting. How many exist is
# a setting -- :data:`NUMBER_OF_ORGANELLES` -- and the keys of each slot are
# GENERATED from it rather than written out, so two gives two slots and seven
# gives seven.
#
# LOWERING THE NUMBER HIDES SLOTS, IT DOES NOT DELETE THEM, which is why
# there are two answers to "which slots are there" below.
# :func:`active_organelle_roles` is what a panel shows and
# :func:`declared_organelle_roles` is what a settings dict must keep: a file
# written at seven and opened at two shows two and carries seven, so the
# values ride along untouched and putting the number back brings the old
# answers with it. A count that erased what it could not currently show
# would punish a user for trying a smaller number, which is the opposite of
# a setting worth exploring. `spacr.settings` generates its type, tooltip
# and category registries for every slot :data:`MAX_ORGANELLES` allows, for
# the same reason: a hidden slot still has to be readable.

#: The setting that decides how many organelle slots a run has.
NUMBER_OF_ORGANELLES = "number_of_organelles"

#: The most slots that can be named, and why there is a limit at all.
#:
#: A slot's role name IS the prefix of every key it owns, and the prefixes
#: are LETTERED: slot 1 is the original ``organelle``, slots 2..26 take a
#: single letter -- ``organelleb`` through ``organellez`` -- and slots 27 and
#: up CARRY into two letters, ``organelleaa`` onward. Digits cannot be used,
#: because object types are embedded directly in underscore-separated object
#: keys: ``organelle_2`` cannot round-trip through a ``prcfo`` key and
#: ``organelle2`` is ambiguous with the object LABELLED 2.
#:
#: THE CEILING IS NOT THE ALPHABET ANY MORE. It was 26 because the lettering
#: stopped at ``z``; the carry makes it arbitrary, and this number is now a
#: bound rather than a limitation. It is kept rather than removed for the
#: reason :func:`organelle_roles` gives: a settings file asking for more
#: should be TOLD which of its keys stopped existing rather than silently
#: clamped.
#:
#: 702 is where two letters run out (26 + 26x26). Three letters would give
#: 18,278 and cost a tuple that size at import for slots nobody has asked
#: for; raising it is one edit if anyone ever does.
MAX_ORGANELLES = 702

#: How many slots a settings file that says nothing has.
#:
#: NONE. A run has the organelles it says it has, and a form that opens
#: showing four unconfigured slots adds four
#: settings and two categories of noise on the busiest screen in the tool.
#:
#: A FILE THAT CARRIES ORGANELLE VALUES IS NOT SAYING "NONE", though, and it
#: was written before the count existed. `organelle_count` infers the count
#: from the slots such a file actually holds rather than reading this, so an
#: old settings file still means exactly what it meant. This number is what
#: a file with no organelle keys AT ALL gets, which is a file that is not
#: asking for any.
DEFAULT_NUMBER_OF_ORGANELLES = 0


def organelle_role(number: int) -> str:
    """The key prefix owned by slot ``number``, counting from one.

    :param number: the slot as the user counts it -- 1 is Organelle 1.
    :returns: ``'organelle'`` for slot 1, ``'organelle<letter>'`` for slots
        2..26, and a CARRIED suffix from 27 up -- ``organelleaa`` onward.
        This is the prefix every one of that slot's settings carries.
    :raises ValueError: outside ``1..MAX_ORGANELLES``, naming the bound.

    SLOTS 1..26 ARE BYTE-IDENTICAL to what they have always been, which is
    how the arbitrary count is reached without migrating anything: no
    measurement database, settings CSV or run journal moves, because none of
    the names they contain change.

    ``organellea`` IS NEVER MINTED -- slot 1 is the bare word -- so a
    single-letter suffix can never be confused with the first letter of a
    carried one, and :data:`_ROLE_MATCH` (longest first) does the rest.
    """
    index = int(number)
    if not 1 <= index <= MAX_ORGANELLES:
        raise ValueError(
            f"organelle slot {number!r} is outside 1..{MAX_ORGANELLES}; "
            f"slots are lettered and carry past 'z', so {MAX_ORGANELLES} is "
            "where two letters run out")
    if index == 1:
        return "organelle"
    if index <= 26:
        return f"organelle{chr(ord('a') + index - 1)}"
    return f"organelle{_carried_suffix(index - 27)}"


def _carried_suffix(offset: int) -> str:
    """The ``offset``-th suffix past ``z``: 0 is ``aa``, 676 is ``aaa``.

    Fixed-width base-26 blocks rather than a bijective count, so every
    suffix of a given length is used before the next length starts and the
    ordering a reader would guess is the one they get.
    """
    length = 2
    while offset >= 26 ** length:
        offset -= 26 ** length
        length += 1
    letters = []
    for _ in range(length):
        offset, remainder = divmod(offset, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def organelle_roles(count: int = MAX_ORGANELLES) -> Tuple[str, ...]:
    """The prefixes of the first ``count`` slots, in slot order.

    :param count: how many slots. Zero is legal and means a run with no
        organelle at all -- most runs -- and returns an empty tuple.
    :raises ValueError: for a count above :data:`MAX_ORGANELLES`. Silently
        clamping would let a settings file ask for thirty slots and get
        twenty-six without being told which of its keys stopped existing;
        :func:`organelle_count` is where a value read from a file is
        clamped, and it says so.
    """
    return tuple(organelle_role(index)
                 for index in range(1, max(int(count), 0) + 1))


#: Every slot spaCR can name. What the settings registries are generated for.
ALL_ORGANELLE_ROLES: Tuple[str, ...] = organelle_roles(MAX_ORGANELLES)

#: Longest first, because ``'organelle'`` is a prefix of every other role:
#: shortest-first would read ``organelleb_channel`` as the primary slot.
_ROLE_MATCH: Tuple[str, ...] = tuple(
    sorted(ALL_ORGANELLE_ROLES, key=len, reverse=True))


def organelle_number(role: str) -> int:
    """The one-based slot number a role prefix stands for.

    :param role: ``'organelle'``, ``'organelleb'``, ...
    :raises ValueError: for anything that is not a slot prefix.
    """
    name = str(role)
    if name == "organelle":
        return 1
    suffix = name[len("organelle"):] if name.startswith("organelle") else ""
    if suffix and suffix.isalpha() and suffix.islower():
        if len(suffix) == 1:
            if "b" <= suffix <= "z":
                return ord(suffix) - ord("a") + 1
        else:
            # The inverse of `_carried_suffix`: earlier lengths are used up
            # before this one starts, so their totals are added back.
            offset = 0
            for length in range(2, len(suffix)):
                offset += 26 ** length
            for char in suffix:
                offset = offset * 26 + (ord(char) - ord("a"))
            return offset + 27
    raise ValueError(
        f"{role!r} is not an organelle role; expected 'organelle', "
        "'organelleb'..'organellez', then 'organelleaa' onward")


def organelle_slot_label(role: str) -> str:
    """What the user calls a slot: ``Organelle 1``, ``Organelle 2``, ..."""
    return f"Organelle {organelle_number(role)}"


def organelle_role_of(key: str) -> Optional[str]:
    """Which slot a settings key belongs to, or None.

    :param key: any settings key. ``'organelle_channel'`` belongs to slot 1
        and ``'organelleb_channel'`` to slot 2; ``'summarize_organelles_by'``
        and ``'number_of_organelles'`` belong to no slot, because they are
        decisions about the organelles collectively rather than settings OF
        one.
    """
    text = str(key)
    if not text.startswith("organelle"):
        return None
    # THE ROLE IS READ OFF THE KEY, not searched for among every role there
    # could be. Scanning `_ROLE_MATCH` was O(roles x keys), and with the
    # ceiling raised from 26 to 702 that is a million string comparisons to
    # answer a question about one settings dict -- 4 ms per call, on a path
    # the settings form takes repeatedly. A key is `<role>_<question>`, so
    # the role is the text before the first underscore and `organelle_number`
    # is what decides whether it is a real one.
    head, separator, _rest = text.partition("_")
    if not separator and head != text:
        return None
    try:
        organelle_number(head)
    except ValueError:
        return None
    return head


def slot_setting(key: str, role: str) -> str:
    """One slot's spelling of a primary ``organelle_*`` key.

    :param key: a primary key, e.g. ``'organelle_diameter'``.
    :param role: the prefix to translate it into. Any object's prefix is
        accepted, not only a slot's: the same translation answers "what is
        this decision called for the pathogen", and the settings tables use
        it that way. What is checked is the KEY, because translating
        something that is not a primary organelle setting produces a key no
        reader has ever heard of.
    :raises ValueError: if ``key`` is not a primary organelle setting.
    """
    text = str(key)
    if not text.startswith("organelle_"):
        raise ValueError(f"not a primary organelle setting: {key!r}")
    return f"{role}_{text[len('organelle_'):]}"


def primary_setting(key: str) -> str:
    """The primary ``organelle_*`` spelling of one slot's key.

    The inverse of :func:`slot_setting`. A key belonging to no slot is
    returned unchanged, so a caller can run a whole settings dict through it.
    """
    text = str(key)
    role = organelle_role_of(text)
    if role is None or text == role:
        return text
    return f"organelle_{text[len(role) + 1:]}"


def organelle_count(settings: Mapping[str, object]) -> int:
    """How many slots ``settings`` asks for, clamped to what can exist.

    :param settings: a run settings mapping. A missing, blank or
        unparseable value means :data:`DEFAULT_NUMBER_OF_ORGANELLES` --
        a settings file written before the count existed is not making a
        claim about it, and refusing to open one over a typo in a number
        would lose the whole file.
    :returns: an integer in ``0..MAX_ORGANELLES``.
    """
    raw = settings.get(NUMBER_OF_ORGANELLES) if settings else None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _count_implied_by_the_slots(settings)
    try:
        count = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return _count_implied_by_the_slots(settings)
    return max(0, min(count, MAX_ORGANELLES))


def _count_implied_by_the_slots(settings: Mapping[str, object]) -> int:
    """How many slots a file that never named a count is actually using.

    :param settings: a run settings mapping.
    :returns: the number of slots that carry a value, in ``0..MAX``.

    THE DEFAULT IS NONE, and a file written before the count existed is not
    claiming to have none -- it is not making a claim at all. Reading the
    slots it carries is what keeps such a file meaning what it meant when
    the default was four: a file with four organelle channels still gets
    four, and one with none gets none.

    A SLOT COUNTS WHEN IT HOLDS SOMETHING. A key present but empty is a
    placeholder the panel wrote, not a slot the run uses, so the highest
    slot with a real value decides -- gaps included, because slot three
    existing means slots one and two do.
    """
    if not settings:
        return 0
    # ONE PASS OVER THE KEYS, not one pass per role. See
    # `organelle_role_of` for why: the roles are no longer a short list.
    highest = 0
    for key, value in settings.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        role = organelle_role_of(key)
        if role is None:
            continue
        try:
            highest = max(highest, organelle_number(role))
        except ValueError:            # pragma: no cover - role_of validated it
            continue
    return min(highest, MAX_ORGANELLES)


def active_organelle_roles(
        settings: Mapping[str, object]) -> Tuple[str, ...]:
    """The slots ``settings`` currently has, in slot order.

    What a panel shows. A slot outside this tuple is HIDDEN, not gone: its
    keys are still typed, still in the settings dict and still written back
    out, which is what makes lowering the number reversible.
    """
    return organelle_roles(organelle_count(settings))


def declared_organelle_roles(
        settings: Mapping[str, object]) -> Tuple[str, ...]:
    """Every slot ``settings`` has to keep values for, in slot order.

    The active slots, PLUS any further slot the mapping already carries a key
    for. That union is the whole of "lowering it hides them and keeps their
    values": a file written at seven and opened at two declares seven, so the
    defaults machinery leaves slots three to seven exactly as it found them
    instead of dropping them on the way back out.
    """
    active = active_organelle_roles(settings)
    present = {organelle_role_of(key) for key in (settings or {})}
    present.discard(None)
    highest = max((organelle_number(role) for role in present), default=0)
    return organelle_roles(max(len(active), highest))


def organelle_slot_is_active(key: str,
                             settings: Mapping[str, object]) -> bool:
    """Whether ``key``'s slot is one of the ones this run has.

    True for every key that belongs to no slot, so a caller can use it as a
    filter over a whole settings dict without having to know which keys are
    organelle settings.
    """
    role = organelle_role_of(key)
    return role is None or role in active_organelle_roles(settings)
