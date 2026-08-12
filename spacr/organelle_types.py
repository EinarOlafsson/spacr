"""One choice a biologist recognises, in front of fifty-three they do not.

`organelle` is the most over-configured object class in spaCR: 53 settings
reach the mask pipeline, and a user who knows they are imaging lysosomes has
to know what a ridge filter is, which of `organelle_method`'s seven values
are legal for which morphology, and what `organelle_network_threshold` does
when they are not segmenting a network.

WHAT THIS IS NOT. It is not a taxonomy compiled into a pipeline. The
maintainer asked for nine cell-biology categories and asked, in the same
breath, to verify rather than assume that they map onto the four
`organelle_morphology` values. They do not:

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
    :param members: the structures the maintainer listed under this name.
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

#: The order the picker shows them in: no-op first, then the two where size
#: decides, then the rest as the maintainer listed them.
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
    """Fill the organelle settings this type recommends, WITHOUT overriding.

    The rule the instruction sets: "PRESET, DO NOT OVERRIDE. Choosing a type
    fills the advanced settings with its recommended values and leaves them
    editable. A user who then changes `organelle_method` keeps that change;
    the type does not silently reassert itself."

    So a key already present in ``settings`` is never touched. The caller
    owns what it set; this fills gaps.

    :param settings: the run's settings. Not modified.
    :param explain: print what the preset chose and why. The preset is only
        an improvement over 53 knobs if the user can see what it did.
    :returns: a new dict.
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
#: else moves to "Organelle advanced" -- STILL VISIBLE AND STILL EDITABLE.
#: Hiding a setting that remains in the settings dict is how a run gets a
#: value nobody can see, which is exactly how this project acquired eleven
#: phantom settings (instruction 61).
BASIC_SETTINGS: Tuple[str, ...] = (
    "organelle_channel",
    "organelle_type",
    "organelle_diameter",
    "organelle_min_size",
    "organelle_max_size",
    "organelle_mask_within_cells",
    "organelle_remove_border",
)


def is_basic(setting: str) -> bool:
    """True when ``setting`` belongs in the plain Organelle category."""
    return str(setting) in BASIC_SETTINGS
