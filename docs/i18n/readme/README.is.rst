|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Skjöl
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Gagnvirkt kennsluefni
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-útgáfa
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 til 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Prófunarsafn
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt-viðmót
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Frumkóði á GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-mál
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial-leyfi
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Nýjustu uppsetningarforrit
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge-uppskrift

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

Tungumál: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**

spaCR aðgreinir og mælir stakar frumur í afkastamiklum smásjármyndum, tengir hverja frumu við gRNA-ið sem hún fékk og greinir frá því hvaða gen breyttu svipgerðinni. Plötumyndir og FASTQ-raðir eru inntak; mælingar fyrir hvert viðfang, þjálfaðir flokkarar, áhrifastærðir fyrir hverja leiðarsameind og hvert gen og forgangsraðaður niðurstöðulisti eru úttak.

Fyrir myndgreindar samsettar CRISPR-skimanir nær þetta yfir allt verkflæðið. Ef þú ert með afkastamiklar smásjármyndir en enga skimun er hægt að keyra aðgreiningu, mælingar, merkingar og flokkun sjálfstætt.

Myndir, grímur, myndúrklippur, mælingar, merkingar, spár, strikamerki og brunnaauðkenni eru geymd í einu SQLite-verkefni, þannig að rekja má niðurstöðugildi aftur til viðfangsins sem það kom frá.

Keyrðu spaCR sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa. Báðar leiðir nota sömu einingar og CUDA er virkjað sjálfkrafa þegar einingin styður það.


Yfirlit yfir verkflæðið
-----------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

**Data**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

**Results & QC**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|

**Explore**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|

**Assays**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Design**

|App_experiment_design|\ |App_power|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Opna API-skjölin fyrir Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Format Converter
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Import Project
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir External Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Plate Queue
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Batch Runner
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Distributed Jobs
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Database Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Data Manager
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Plate Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Training Runs
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Run History
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Report
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Investigate Hit
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Pipeline Graph
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Prediction Profiler
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir QC Dashboard
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Lineage
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Layer Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Tabulate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: Opna API-skjölin fyrir Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir macOS 11+ (Intel og Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg

.. spacr-workflow-end

Veldu verkflæðiseiningu til að opna API-síðu hennar. Taflan sýnir öll önnur forrit í sömu flokkum og röð og á upphafssíðu spaCR.


Setja upp spaCR
---------------

Skjáborðsforrit
~~~~~~~~~~~~~~~~~~~

Stöðvarstöðvarnar innihalda persónulegt Python umhverfi, þannig að konda og núverandi Python uppsetningu er ekki nauðsynlegt.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir 64-bita Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Eldri spaCR-uppsetningarforrit
   :target: ../../source/installers.rst
.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Opna smásjárgagnasafnið í BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Opna prófunargagnasafnið á Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito

.. spacr-installer-links-end

Fyrstu þremur tákn leyfja núverandi útgáfu. spaCR táknin opnar fullkomið installer arkívu. Installer tengsl og verslun filnames eru uppfærdur af útgáfur vinnuflu; fyrri installerir eru enn í sama útgáfa arkíva.

Í Linux skaltu gera skrána sem var sótt keyranlega og keyra hana:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Á macOS, opna ``.pkg``. Núverandi beta er ekki notarið; ef Gatekeeper blokkir það, velja **System Settings → Privacy & Security → Open Anyway**.

Sjá `Installer leiðbeiningar <../../source/installer_guide.rst>`_ til að uppgötva, deinstalla, offline og vandamálið.

Python-uppsetning
~~~~~~~~~~~~~~~~~~~

Python 3.12 hefur breiðustu val af ókeypis vísindalegar pakka:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR styður Python **3.9 til 3.14**, að undanskildu Python 3.14.1 sem torchvision styður ekki. Mælt er með Linux fyrir CUDA-verkflæði; macOS og Windows eru einnig studd.

Slepptu Qt á þjóni, reikniklasa eða CI-keyrsluumhverfi:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Opinlegri samsetningar eru settar sérstakt, t.d. ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` og ``spacr[czi,nd2,lif]``. Sjá `Uppsetningu leiðbeiningar <../../source/installer_guide.rst>`_ fyrir fullkomna útgáfur og Python-version samskipti tól.

Skipanalínuskipanir
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Set ``SPACR_LOG_LEVEL=DEBUG`` þegar ákvarðanir. Rotating logs eru skrifað á ``~/.spacr/logs/spacr.log``.


Það sem hægt er að gera
-----------------------

Meirihluti skrefna fylgja sex mólum:

- **Mask** hlutgreinir frumur, frumukjarna, sýkla og frumulíffæri með Cellpose.
- **Measure** skrifar lögunar-, styrkleika-, áferðar-, rúm- og samstaðsetningareiginleika ásamt myndúrklippum viðfanga í SQLite.
- **Annotate** merkir myndúrklippur í lyklaborðsstýrðu hnitaneti og styður biðraðir virks náms.
- **Classify** þjálfar líkön byggð á myndum eða mælingum og skráir frammistöðu á fráteknum gögnum með hverjum varðpunkti.
- **Map Barcodes** varpar FASTQ-lestrum á brunna og gRNA og veitir gæðamat fyrir magn, árekstra og þekju.
- **Regression** metur áhrif leiðarsameinda, gena, skilyrða og viðmiða með líkanafjölskyldum sem henta samfelldum gildum, hlutföllum og talningum.

Saminn verkefni getur einnig hönnuð plötur, uppgötvað styrk, rétt batch áhrif, athuga afgræðslu gæði, rannsaka tengda plöt og gróðurs, útvarpa AnnData, endurheimta ábreytt vinnu og skráð settingar bak hverra niðurstöðu.

Veldu næsta síðu með því sem þú vilt gera:

- `Samskiptaþjálfunar <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 leiðbeiningar vinnufluðum frá uppsetningu í gegnum hit rannsóknir.
- `Python API snemma byrjun <../../source/python_api.rst>`_ — hlaupa og staðfest pipelines frá skriptum, notebooks eða klúster.
- `Leikstjóri <../../source/features.rst>`_ — hæfileika, fullnægjandi og valfrjáls tengsl.
- `Heilluð API reference <https://einarolafsson.github.io/spacr/api/index.html>`_ — stuðlað innfangspunktur eftir verkefni, með fullkomna mótum tengslum einn hærra.
- `Sjálf tungumál og þýðingu leiðbeining <../../source/localization.rst>`_ — samskipti tungumál, kontext hjálp og vísindaleg útleiðslu.

Tungumál og þýðingar
~~~~~~~~~~~~~~~~~~~~~~

Viðmótið styður tíu tungumál í leiðsögn og stillingum. AI- og LIVE-stýringar, lýsingar á einingum og yfirfarin samhengishjálp eru einnig þýdd. Skiptu um tungumál undir **spaCR → Stillingar → Tungumál** án endurræsingar. Annálar, slóðir, gagnagrunnsgildi og mælingar eru aldrei þýdd; vísindaleg úttök haldast á viðurkenndri ensku. Sjá `stefnu um samhengishjálp <../../source/localization.rst#contextual-help>`_.

Hreyfimyndaleiðbeiningar fyrir stillingar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stillingar með sjónræna skýringu bjóða upp á **Animation**-stýringu í verkfæraábendingunni. Skoðaðu `myndasafn stillingahreyfimynda <https://einarolafsson.github.io/spacr/setting_animations.html>`_ eða `skrá stillingahreyfimynda <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Gögn
----

Viðmiðunargagnasöfn
~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Opna raðgreiningargagnasafnið hjá NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Opna spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Opna bioRxiv-forprentið
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1


Framlög og aðstoð
------------------------

Vel afmarkaðar villuskýrslur og beiðnir um eiginleika eru velkomnar í `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar bilun er tilkynnt skaltu láta spaCR-útgáfu, stýrikerfi, Python-útgáfu, einingastillingar og viðeigandi bút úr annál fylgja. ``spacr-doctor`` safnar flestum þessara upplýsinga sjálfkrafa.

Leyfi
~~~~~~~~~

Frumkóði núverandi þróunargreinar er aðgengilegur samkvæmt `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Notkun í atvinnuskyni krefst sérstaks leyfis frá höfundarréttarhafa. Útgefnar útgáfur til og með spaCR 1.4.9.9 eru áfram tiltækar samkvæmt MIT-leyfinu sem fylgdi þeim útgáfum.

Kennsluefni
~~~~~~~~~~~

`Gagnvirka spaCR-kennslusafnið <https://einarolafsson.github.io/spacr/tutorials/>`_ inniheldur talsettar og textaðar leiðbeiningar um uppsetningu og hvert verkflæði: 73 kennslustundir með 50 röddum á átta tungumálum.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR nýtist við rannsóknina skaltu vitna í:

Olafsson EB, *et al.* A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem *T. gondii* modulator ESCRT subversion.

`Bioregl fyrirframskrift <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Programvarparkíf <https://doi.org/10.5281/zenodo.21343317>`_

Þakkir
~~~~~~~~~~~~~~~

spaCR byggir á opnum vísindahugbúnaði, meðal annars NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch og Qt. Sjá `upplýsingar um þýðingarlíkön <../TRANSLATION_MODELS.md>`_ fyrir líkönin sem voru notuð við gerð fjöltyngdra skjala og viðmótsskráa.
