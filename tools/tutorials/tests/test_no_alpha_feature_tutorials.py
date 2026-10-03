"""No tutorial teaches or shows an alpha feature.

The maintainer's rule (2026-09-26): nothing under Preferences -> Show alpha
features gets a tutorial. The one exception is the setting itself, which a
Preferences lesson may mention. Every registered name in
``spacr.settings.ALPHA_FEATURES`` (settings keys, dropdown values, widget
object names and their visible labels, module keys, Model Zoo keys and
family stems) is searched for in every lesson source and the published
English catalog: narration, objectives, titles, prerequisites, example
files and visual names alike.

Promoting a feature out of alpha deletes its registry entry, which lifts
the ban here with no other edit.
"""
import json
import re
from pathlib import Path

import pytest

from spacr import settings as spacr_settings

REPO = Path(__file__).resolve().parents[3]
LESSONS = sorted((REPO / "tools" / "tutorials" / "lessons").glob("*.json"))
PUBLISHED_CATALOG = REPO / "docs" / "source" / "_extra" / "tutorials" / "catalog" / "lessons_en.json"

#: Fixed on-screen text of each registered alpha widget, so a lesson that
#: names the button or menu entry rather than its object name is caught too.
#: An empty tuple means the widget has no fixed text (a live status label).
#: Every registered widget must appear here: a new alpha widget fails
#: ``test_every_alpha_widget_has_its_visible_text_listed`` until it is added.
WIDGET_LABELS = {
    "ActivationCounterfactualViewer": ("Counterfactuals…",),
    "AnnotateFieldQCButton": ("Field QC…",),
    "AnnotateFieldQCDialog": ("Field quality labels",),
    "AnnotateSimilarityOptions": ("Similar crops",),
    "ConvertPlateBarcodeLinkage": ("Plate barcode linkage",),
    "EmbeddingsLabelsButton": (),
    "EmbeddingsUseForLabel": (),
    "EmbeddingsUseForPicker": (),
    "EmbeddingsUseForRun": (),
    "MakeMasksSam2Button": ("SAM2 tracking",),
    "MakeMasksUncertaintyEnsembleSetting": ("Uncertainty ensemble model",),
    "MakeMasksVirtualStainApply": ("Apply virtual stain",),
    "MaskVirtualStainApply": ("Apply virtual stain",),
    "QCClassifierCard": ("Image QC classifier",),
    # Alpha organism pages: their Home tile and page titles.
    "PlasmodiumOrganismPage": ("Plasmodium spp.",),
    "CandidaOrganismPage": ("Candida spp.",),
    "TrypanosomaOrganismPage": ("Trypanosoma spp.",),
    "LeishmaniaOrganismPage": ("Leishmania spp.",),
    "GiardiaOrganismPage": ("Giardia duodenalis",),
    "VirusOrganismPage": ("Virus infection",),
    "MammalianOrganismPage": ("Mammalian cells",),
    # Alpha "Load test data…" buttons share their text with every non-alpha
    # module's button of the same name, and Embeddings' "Labels…", "Use
    # for:" and "Run" are ordinary words; the object names still guard them.
    "ControlChartTestDataButton": (),
    "ConvertTestDataButton": (),
    "DataManagerTestDataButton": (),
    "EmbeddingsTestDataButton": (),
    "ExternalMasksTestDataButton": (),
    "FeatureExplorerTestDataButton": (),
    "LayerViewerTestDataButton": (),
    "OutliersTestDataButton": (),
    "PipelineGraphTestDataButton": (),
    "PowerTestDataButton": (),
    "ProjectBrowserTestDataButton": (),
    "TrellisTestDataButton": (),
    "MakeMasksUseInMaskGeneration": ("Use in Mask generation",),
    # The settings category title is listed with the button (501), so
    # narration naming either is caught.
    "PlaqueEstimateScaleTime": ("Estimate scale / time (experimental)",
                                "Experimental growth estimates"),
    "PlaqueEstimateScaleTimeNote": ("Published RH/HFF reference",),
    "DistributedAllocatedGpus": ("segment batches on every GPU allocated",),
    "MaskGpuProgress": (),
    "MeasureConfluencyToggle": ("Confluency",),
    "MakeMasksRoisButton": ("ROIs", "QuPath GeoJSON", "ImageJ RoiSet",
                            "ImageJ RoiSets", "COCO JSON"),
    "WatchFolderProgress": ("watch folder",),
    "ControlChartHitPanel": ("Score hits", "SSMD", "B-score", "robust z",
                             "Call hits by", "Hit threshold"),
    "ControlChartHitsSection": (),
    "ControlChartExportHits": ("Export hits",),
    "MeasureWoundToggle": ("Wound",),
    "AnnotateBlindToggle": ("Blind",),
    "MakeMasksBlindToggle": ("Blind",),
    "CloudSourceBrowse": ("Browse cloud storage",),
    "MakeMasksPromptCategory": ("Segment by prompt", "Prompt with micro-SAM"),
    # "Like this" is left out: it matches ordinary narration ("a plot like
    # this"); the object name still guards the button.
    "AnnotateFindSimilar": (),
    "EmbeddingsFoundationLabel": ("Foundation model",),
    "EmbeddingsFoundationPicker": ("None (use the backbone)",),
    "EmbeddingsWellMilButton": ("Learn from well labels",),
    "EmbeddingsDinoPretrainButton": ("Pretrain on these crops",),
    "CellposeWorkbenchVirtualStain": ("Virtual staining",),
    "FigureIntegrityCheck": ("Check figure integrity on export",),
    "AnalysisLockButton": ("Lock analysis",),
    # Run notifications (Preferences). One-word row labels shared with
    # ordinary text (Email, Desktop, When, From, To) are left out.
    "NotifyTabHelp": ("spaCR can tell you when a long run finishes or fails",),
    "NotifyRunsEnabled": ("Notify me",),
    "NotifyRunsWhen": ("When a run finishes or fails", "Only when a run fails"),
    "NotifyRunsMinMinutes": ("Runs longer than",),
    "NotifyDesktop": (),
    "NotifyEmail": (),
    "NotifySmtpHost": ("SMTP server",),
    "NotifySmtpPort": ("SMTP port",),
    "NotifySmtpSecurity": (),
    "NotifySmtpUser": ("SMTP user name",),
    "NotifySmtpPassword": ("SMTP password",),
    "NotifyEmailFrom": (),
    "NotifyEmailTo": (),
    "NotifySlack": ("Slack",),
    "NotifySlackWebhook": ("Slack webhook",),
    "NotifyNtfy": ("ntfy",),
    "NotifyNtfyServer": ("ntfy server",),
    "NotifyNtfyTopic": ("ntfy topic",),
    "NotifyNtfyToken": ("ntfy access token",),
    "NotifySendTest": ("Send a test",),
    "NotifyForgetSecrets": ("Forget saved secrets",),
    "NotifyTestResult": (),
    "NotifyTeams": ("Microsoft Teams",),
    "NotifyTeamsWebhook": ("Teams webhook",),
    "NotifyWebhook": ("Webhook",),
    "NotifyWebhookUrl": ("Webhook address",),
    "NotifyWebhookToken": ("Webhook access token",),
    "ReportArchivePackage": ("Archive package",),
    "ReportZenodoDeposit": ("Deposit on Zenodo",),
    "RunHistoryExportWorkflow": ("Export workflow",),
    "ControlChartChemistry": ("Cluster similarity", "no compound table"),
    "ControlChartChemistrySection": ("Structures and SAR",),
    "PowerArrayedPlanner": ("Arrayed-assay planner", "Pilot table"),
    "ControlChartAnomaly": ("Score anomalies against the negative control",
                            "Outlier quantile", "Known hits",
                            "Export anomalies"),
    "ControlChartAnomalySection": ("Anomalies",),
    # With its ellipsis: bare "uncertainty" is ordinary statistics narration.
    "MakeMasksUncertaintyButton": ("Uncertainty…",),
    "MakeMasksUncertaintySetting": ("Uncertainty…",),
    # Item 508: Image enhancement's hand-off to Mask generation.
    "MakeMasksUseInMaskGeneration": ("Use in Mask generation",),
    # Plugin catalogue (Preferences). Its one-word buttons (List, Open,
    # Uninstall) and table headers are left out as ordinary words.
    "PluginCatalogueHelp": ("Browse a catalogue of community plugins",),
    "PluginCatalogueSource": ("Catalogue file, folder or address",),
    "PluginCatalogueLoad": (),
    "PluginCatalogueTable": (),
    "PluginCatalogueInstall": ("Install or update",),
    "PluginCatalogueUninstall": (),
    "PluginCatalogueOpen": (),
    "PluginCatalogueStatus": (),
    "MapBarcodesSpatialToggle": ("Spatial transcriptomics",),
    "MapBarcodesSpatialCard": ("Spatial transcriptomics",),
}

#: The setting itself. Allowed only in a Preferences lesson.
TOGGLE_TERMS = ("Show alpha features", "alpha features", "show_alpha_features",
                "Show alpha species", "alpha species", "show_alpha_species")

#: Organism pages moved under Show alpha species on 2026-10-03 whose lessons
#: were published before the move; their retirement is routed to the
#: tutorial lane, so they are not counted until it lands.
PUBLISHED_SPECIES_LESSONS = spacr_settings._SPECIES_WITH_PUBLISHED_LESSONS
_PUBLISHED_SPECIES_PAGES = frozenset({"PlasmodiumOrganismPage", "CandidaOrganismPage"})


def _registry_terms():
    """(term, what) pairs for everything registered as alpha."""
    terms = []
    registries = (*spacr_settings.ALPHA_FEATURES.items(),
                  *spacr_settings.ALPHA_SPECIES.items())
    for number, entry in registries:
        for key in entry.get("settings", ()) or ():
            terms.append((key, f"{number} setting {key}"))
            if "_" in key:
                terms.append((key.replace("_", " "), f"{number} setting {key}"))
        for key, values in (entry.get("choices") or {}).items():
            for value in values:
                terms.append((value, f"{number} {key} choice {value}"))
        for name in entry.get("widgets", ()) or ():
            if name in _PUBLISHED_SPECIES_PAGES:
                continue
            terms.append((name, f"{number} widget {name}"))
            for label in WIDGET_LABELS.get(name, ()):
                terms.append((label, f"{number} widget {name} ({label!r})"))
        for key in entry.get("apps", ()) or ():
            if key in PUBLISHED_SPECIES_LESSONS:
                continue
            terms.append((key, f"{number} module {key}"))
        for key in entry.get("models", ()) or ():
            terms.append((key, f"{number} model {key}"))
            terms.append((key.split("_")[0], f"{number} model family {key.split('_')[0]}"))
    return sorted(set(terms))


def _pattern(term):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])",
                      re.IGNORECASE)


def _strings(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


def _is_preferences_lesson(lesson):
    return any("preferences" in str(lesson.get(field) or "").lower()
               for field in ("id", "slug", "app_key", "host_app_key"))


def alpha_references(lesson, *, alpha_apps=()):
    """Every place ``lesson`` names an alpha feature, as readable strings."""
    found = []
    if lesson.get("app_key") in alpha_apps or lesson.get("host_app_key") in alpha_apps:
        found.append(f"app_key: alpha module {lesson.get('app_key')}")
    terms = [(_pattern(term), what) for term, what in _registry_terms()]
    toggle = [_pattern(term) for term in TOGGLE_TERMS]
    for where, text in _strings(lesson):
        for pattern, what in terms:
            if pattern.search(text):
                found.append(f"{where}: {what}")
        if not _is_preferences_lesson(lesson):
            if any(pattern.search(text) for pattern in toggle):
                found.append(f"{where}: the Show alpha features setting outside Preferences")
    return found


def _alpha_apps():
    return frozenset(spacr_settings._alpha_names("apps")) - PUBLISHED_SPECIES_LESSONS


def test_the_registry_is_not_empty_so_this_guard_is_not_vacuous():
    assert spacr_settings.ALPHA_FEATURES
    assert _registry_terms()


def test_every_alpha_widget_has_its_visible_text_listed():
    missing = sorted(set(spacr_settings._alpha_names("widgets")) - set(WIDGET_LABELS))
    assert not missing, (
        "Add the fixed on-screen text of these alpha widgets to WIDGET_LABELS "
        f"(an empty tuple if they have none): {missing}")


@pytest.mark.parametrize("path", LESSONS, ids=[path.stem for path in LESSONS])
def test_no_lesson_teaches_or_shows_an_alpha_feature(path):
    lesson = json.loads(path.read_text(encoding="utf-8"))
    found = alpha_references(lesson, alpha_apps=_alpha_apps())
    assert not found, (
        f"{path.name} references alpha features, which get no tutorials:\n  "
        + "\n  ".join(found))


#: Published references already fixed in the lesson source and waiting for
#: the next publication, as {lesson id: {exact finding}}. Each entry must
#: still be found in the published catalog: once the corrected lesson is
#: published, the stale entry fails the test until it is removed.
PUBLISHED_PENDING_REPUBLICATION = {
    # 24_plaque scene 15 named 501's "Experimental growth estimates"; the
    # source narration drops it and the re-recorded lesson awaits publication.
    "24_plaque": {".scenes[14].narration: 501 widget PlaqueEstimateScaleTime "
                  "('Experimental growth estimates')"},
}


def test_the_published_catalog_teaches_no_alpha_feature():
    catalog = json.loads(PUBLISHED_CATALOG.read_text(encoding="utf-8"))
    found, stale = {}, {}
    for lesson in catalog["lessons"]:
        hits = set(alpha_references(lesson, alpha_apps=_alpha_apps()))
        pending = PUBLISHED_PENDING_REPUBLICATION.get(lesson.get("id"), set())
        if hits - pending:
            found[lesson.get("id")] = sorted(hits - pending)
        if pending - hits:
            stale[lesson.get("id")] = sorted(pending - hits)
    assert not found, found
    assert not stale, ("Published and no longer found; remove from "
                       f"PUBLISHED_PENDING_REPUBLICATION: {stale}")


def test_pending_republication_entries_are_fixed_in_the_source():
    for lesson_id in PUBLISHED_PENDING_REPUBLICATION:
        path = REPO / "tools" / "tutorials" / "lessons" / f"{lesson_id}.json"
        lesson = json.loads(path.read_text(encoding="utf-8"))
        assert not alpha_references(lesson, alpha_apps=_alpha_apps()), lesson_id


#: A fixed registry for the scanner's own tests, so they do not change as
#: real features are promoted out of alpha.
EXAMPLE_REGISTRY = {
    426: {'settings': ('timeflows_model',),
          'choices': {'timelapse_mode': ('timeflows',)}},
    541: {'settings': ('confluency',), 'widgets': ('MeasureConfluencyToggle',)},
    545: {'widgets': ('MakeMasksRoisButton',)},
    551: {'models': ('stardist_v1',)},
    900: {'apps': ('example_alpha_module',)},
}


@pytest.fixture
def example_registry(monkeypatch):
    monkeypatch.setattr(spacr_settings, "ALPHA_FEATURES", EXAMPLE_REGISTRY)


def test_the_scanner_catches_each_kind_of_reference(example_registry):
    lesson = {"id": "99_example", "app_key": "measure", "scenes": [
        {"visual": "07_confluency_overlay", "narration": "Press Confluency."},
        {"narration": "Choose TimeFlows as the tracking mode."},
        {"narration": "Open ROIs… and export COCO JSON."},
        {"narration": "Pick a StarDist model."},
        {"narration": "Turn on Show alpha features in Preferences."},
    ]}
    found = "\n".join(alpha_references(lesson))
    for expected in ("confluency", "timeflows", "MakeMasksRoisButton",
                     "model family stardist", "outside Preferences"):
        assert expected in found
    clean = {"id": "98_example", "scenes": [
        {"narration": "Watch the console and check alpha at zero point zero five."},
        {"narration": "Open Hit List to rank the regression hits."}]}
    assert alpha_references(clean) == []


def test_an_alpha_module_lesson_is_caught_by_its_app_key(example_registry):
    lesson = {"id": "99_example", "app_key": "example_alpha_module", "scenes": []}
    assert "app_key: alpha module example_alpha_module" in alpha_references(
        lesson, alpha_apps=_alpha_apps())


def test_a_preferences_lesson_may_mention_the_toggle_but_nothing_behind_it(example_registry):
    lesson = {"id": "90_preferences", "app_key": "preferences", "scenes": [
        {"narration": "Show alpha features reveals features still in testing; "
                      "leave it off unless you are asked to try one."}]}
    assert alpha_references(lesson) == []
    lesson["scenes"].append({"narration": "With it on, Measure gains Confluency."})
    found = alpha_references(lesson)
    assert found and all(item.startswith(".scenes[1].narration") for item in found)
    assert any("confluency" in item.lower() for item in found)
