"""Stable, lightweight entry points for scripted spaCR workflows.

The pipeline modules remain importable for backward compatibility. This
module is the smaller interface recommended for new scripts: typed
configuration objects expand through the same defaults used by the GUI, and
the heavy image stack is imported only when a run starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

PathInput = Union[str, PathLike[str]]
SourceInput = Union[PathInput, Sequence[PathInput]]

__all__ = ["MaskConfig", "MeasureConfig", "run_mask", "run_measure"]


def _source_value(value: SourceInput) -> Union[str, list[str]]:
    """Return a settings-compatible source path or list of paths."""
    if isinstance(value, (str, PathLike)):
        return str(value)
    return [str(item) for item in value]


def _merge_extra(settings: MutableMapping[str, Any],
                 extra: Mapping[str, Any]) -> None:
    """Merge advanced settings while protecting the typed fields."""
    overlap = sorted(set(settings) & set(extra))
    if overlap:
        names = ", ".join(overlap)
        raise ValueError(
            f"extra repeats typed setting(s): {names}. Set those values on "
            "the configuration object instead."
        )
    settings.update(extra)


@dataclass(frozen=True)
class MaskConfig:
    """Configuration for image preparation and Cellpose segmentation.

    Parameters
    ----------
    src:
        Plate folder, or a sequence of plate folders, containing the source
        microscopy images.
    cell_channel, nucleus_channel, pathogen_channel:
        Zero-based intensity channel used for each segmentation target. Leave
        an object at ``None`` when it should not be segmented. At least one is
        required.
    cell_diameter, nucleus_diameter, pathogen_diameter:
        Expected object diameters in pixels. ``None`` lets spaCR estimate the
        diameter from representative fields.
    channels:
        Intensity channels retained in the merged field arrays.
    magnification:
        Objective magnification recorded with the run.
    pipeline_style:
        ``"v1"`` for the stable disk-based pipeline or ``"v2"`` for the
        experimental streaming path.
    extra:
        Advanced settings not represented by typed fields. Keys that repeat a
        typed field are refused so one script cannot contain two answers.

    :ivar test_mode: run a small representative subset and enable diagnostic
        output before committing compute to the complete plate.
    :ivar dry_run: validate the source, settings, and planned outputs without
        loading a model, running segmentation, or writing files.

    Examples
    --------
    >>> config = MaskConfig(
    ...     "/data/plate01", cell_channel=0, nucleus_channel=1,
    ...     cell_diameter=60, nucleus_diameter=20,
    ... )
    >>> settings = config.to_settings()
    >>> settings["src"]
    '/data/plate01'
    """

    src: SourceInput
    cell_channel: Optional[int] = None
    nucleus_channel: Optional[int] = None
    pathogen_channel: Optional[int] = None
    cell_diameter: Optional[float] = None
    nucleus_diameter: Optional[float] = None
    pathogen_diameter: Optional[float] = None
    channels: Sequence[int] = (0, 1, 2, 3)
    magnification: float = 20
    pipeline_style: str = "v1"
    test_mode: bool = False
    dry_run: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_settings(self) -> dict[str, Any]:
        """Return the complete settings dictionary consumed by Mask.

        Returns
        -------
        dict
            A fresh dictionary containing spaCR defaults and this
            configuration's overrides.

        Raises
        ------
        ValueError
            If no segmentation channel is selected, the pipeline style is
            unknown, or ``extra`` repeats a typed setting.
        """
        if all(value is None for value in (
                self.cell_channel, self.nucleus_channel,
                self.pathogen_channel)):
            raise ValueError("select at least one segmentation channel")
        if self.pipeline_style not in {"v1", "v2"}:
            raise ValueError("pipeline_style must be 'v1' or 'v2'")
        values: dict[str, Any] = {
            "src": _source_value(self.src),
            "cell_channel": self.cell_channel,
            "nucleus_channel": self.nucleus_channel,
            "pathogen_channel": self.pathogen_channel,
            "cell_diameter": self.cell_diameter,
            "nucleus_diameter": self.nucleus_diameter,
            "pathogen_diameter": self.pathogen_diameter,
            "channels": list(self.channels),
            "magnification": self.magnification,
            "pipeline_style": self.pipeline_style,
            "test_mode": self.test_mode,
            "dry_run": self.dry_run,
        }
        _merge_extra(values, self.extra)
        from .settings import set_default_settings_preprocess_generate_masks
        return set_default_settings_preprocess_generate_masks(values)


@dataclass(frozen=True)
class MeasureConfig:
    """Configuration for object measurement and crop generation.

    Parameters
    ----------
    src:
        A ``merged`` folder, or sequence of folders, produced by Mask.
    cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim:
        Plane index holding each label mask. Use ``None`` for object types not
        present in the merged arrays.
    channels:
        Intensity planes measured for every selected object.
    crop_mode:
        Object types for which PNG crops are written when ``save_png`` is
        true.
    png_channel_mapping:
        Mapping from ``r``, ``g`` and ``b`` to source intensity planes.
    extra:
        Advanced settings not represented by typed fields.

    :ivar save_png: write per-object PNG crops and register their paths for
        downstream annotation, classification, and image plots.
    :ivar test_mode: measure a small sample of merged fields with diagnostic
        plotting enabled before processing the complete dataset.
    :ivar dry_run: validate the measurement plan and return its problems
        without loading the measurement pipeline or writing files.
    :ivar resume: continue an interrupted run after validating completed
        fields and clearing any partial database rows before retrying them.
    """

    src: SourceInput
    cell_mask_dim: Optional[int] = 4
    nucleus_mask_dim: Optional[int] = 5
    pathogen_mask_dim: Optional[int] = 6
    channels: Sequence[int] = (0, 1, 2, 3)
    crop_mode: Sequence[str] = ("cell",)
    save_png: bool = True
    png_channel_mapping: Mapping[str, int] = field(
        default_factory=lambda: {"r": 2, "g": 1, "b": 0})
    test_mode: bool = False
    dry_run: bool = False
    resume: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_settings(self) -> dict[str, Any]:
        """Return the complete settings dictionary consumed by Measure."""
        if all(value is None for value in (
                self.cell_mask_dim, self.nucleus_mask_dim,
                self.pathogen_mask_dim)):
            raise ValueError("select at least one mask plane to measure")
        values: dict[str, Any] = {
            "src": _source_value(self.src),
            "cell_mask_dim": self.cell_mask_dim,
            "nucleus_mask_dim": self.nucleus_mask_dim,
            "pathogen_mask_dim": self.pathogen_mask_dim,
            "channels": list(self.channels),
            "crop_mode": list(self.crop_mode),
            "save_png": self.save_png,
            "png_channel_mapping": dict(self.png_channel_mapping),
            "test_mode": self.test_mode,
            "dry_run": self.dry_run,
            "resume": self.resume,
        }
        _merge_extra(values, self.extra)
        from .settings import get_measure_crop_settings
        return get_measure_crop_settings(values)


def run_mask(config: Union[MaskConfig, Mapping[str, Any]]) -> Any:
    """Run Mask from a typed configuration or an existing settings mapping.

    :param config: typed mask configuration or compatible settings mapping.

    A configuration with ``dry_run=True`` returns preflight problems and
    writes nothing. A normal run returns ``None`` after writing its outputs.
    """
    settings = config.to_settings() if isinstance(config, MaskConfig) else dict(config)
    from .core import preprocess_generate_masks
    return preprocess_generate_masks(settings)


def run_measure(config: Union[MeasureConfig, Mapping[str, Any]]) -> Any:
    """Run Measure from a typed configuration or existing settings mapping.

    :param config: typed measurement configuration or settings mapping.

    A configuration with ``dry_run=True`` returns preflight problems and
    writes nothing. A normal run returns ``None`` after writing its outputs.
    """
    settings = (
        config.to_settings() if isinstance(config, MeasureConfig)
        else dict(config)
    )
    from .measure import measure_crop
    return measure_crop(settings)
