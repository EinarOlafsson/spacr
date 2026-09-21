|Platforms| |Python| |Qt| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: दस्तावेज़
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: इंटरैक्टिव ट्यूटोरियल
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI संस्करण
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 से 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: परीक्षण समूह
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt इंटरफ़ेस
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub स्रोत
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub समस्याएँ
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause लाइसेंस
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv प्रीप्रिंट
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: नवीनतम इंस्टॉलर
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge संस्करण
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge डाउनलोड
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge की नवीनतम रिलीज़ की तिथि
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI डाउनलोड
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS और Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: spaCR को उद्धृत करें
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: पिछले 30 पूरे दिनों में spaCR की PyPI डाउनलोड रैंकिंग

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

भाषाएँ: `🌐 हिन्दी ▾ <README.md>`_

.. spacr-language-picker-end

**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**

spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रति-वस्तु फीनोटाइप को अनुक्रमण से प्राप्त गाइड प्रचुरता के साथ एकीकृत करता है और अनुमान लगाता है कि कौन-से जीन फीनोटाइपिक परिवर्तनों से जुड़े हैं। प्लेट छवियों और FASTQ रीड से शुरू करके, यह प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव अनुमान तथा प्राथमिकता के अनुसार क्रमित हिट सूची बनाता है।

segmentation, measurement, annotation और classification modules भी एक sequencing arm के बिना चलता है।

छवियाँ, मास्क, क्रॉप, मापन, एनोटेशन, भविष्यवाणियाँ, बारकोड और वेल पहचानकर्ता एक ही SQLite परियोजना में रहते हैं।

यह डेस्कटॉप एप्लिकेशन के रूप में, या वर्कस्टेशन, सर्वर या क्लस्टर पर हेडलेस रूप में चलता है।

हार्डवेयर समर्थन
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


spaCR इंस्टॉल करें
-----------------

डेस्कटॉप एप्लिकेशन
~~~~~~~~~~~~~~~~~~~

इंस्टॉलर अपने स्वयं के Python को जोड़ते हैं. कॉन्डा की आवश्यकता नहीं है.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11 के लिए spaCR 1.5.0.6 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.6/SpaCR-1.5.0.6-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel और Apple Silicon) के लिए spaCR 1.5.0.6 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.6/SpaCR-1.5.0.6-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-बिट Linux के लिए spaCR 1.5.0.6 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.6/SpaCR-1.5.0.6-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: spaCR के पुराने इंस्टॉलर
   :target: ../../source/installers.rst

.. spacr-installer-links-end

पहले तीन आइकन वर्तमान रिलीज डाउनलोड करते हैं. spaCR आईकन पूरे इंस्टॉलर संग्रहालय को खोलता है. इंस्टॉलर लिंक और संस्करण फ़ाइल नाम जारी कार्यप्रवाह द्वारा अद्यतन किए जाते हैं; पिछले इंस्टोलर एक ही रिलीज़ संग्रहीत में रहते हैं.

Linux पर डाउनलोड की गई फ़ाइल को निष्पादन योग्य बनाएँ और चलाएँ:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

macOS पर, ``.pkg`` खोलें. वर्तमान बीटा नोटिस नहीं किया गया है; यदि Gatekeeper इसे ब्लॉक करता है, तो **सिस्टम सेटिंग्स → गोपनीयता और सुरक्षा → किसी भी तरह से खोलें** का चयन करें.

अद्यतन, अनइंस्टॉल, ऑफ़लाइन और समस्या हल करने के लिए निर्देशों के लिए `इंस्टॉलर गाइड <../../source/installer_guide.rst>`_ देखें।

PyPI से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~

PyPI रिलीज़ के लिए, Conda वातावरण के भीतर pip से spaCR इंस्टॉल करें। Python 3.12 में वैकल्पिक वैज्ञानिक पैकेजों की सबसे व्यापक उपलब्धता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR, Python **3.9 through 3.14** का समर्थन करता है, Python 3.14.1 को छोड़कर, जिसे torchvision बाहर रखता है। सबसे भारी CUDA और ROCm कार्यप्रवाहों के लिए Linux की अनुशंसा की जाती है; macOS और Windows भी समर्थित हैं, और दोनों अपने GPU का उपयोग करते हैं — macOS, Metal के माध्यम से, जो Apple Silicon और Intel Mac में लगे AMD कार्डों को कवर करता है, और Windows, CUDA या DirectML के माध्यम से।

सर्वर, क्लस्टर या CI रनर पर Qt को छोड़ दें:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

वैकल्पिक एकीकरण अलग से स्थापित किए जाते हैं, उदाहरण के लिए ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` और ``spacr[czi,nd2,lif]``. पूर्ण अतिरिक्त और Python संस्करण संगतता तालिका में `स्थापना गाइड <../../source/installer_guide.rst>`_ देखें.

conda-forge से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~~~~~~~~

आधिकारिक conda-forge पैकेज सक्रिय वातावरण में spaCR और उसकी डेस्कटॉप निर्भरताएँ इंस्टॉल करता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

सोर्स कोड से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~~~~~

रिपॉज़िटरी को क्लोन करें और उसे एडिटेबल मोड में इंस्टॉल करें, ताकि आपकी वर्किंग कॉपी *ही* इंस्टॉल किया गया पैकेज हो और बदलाव दोबारा इंस्टॉल किए बिना लागू हो जाएँ::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

यह ``main`` को क्लोन करता है, जो डिफ़ॉल्ट ब्रांच है और जिसमें नवीनतम रिलीज़ रहती है। विकास ``nightly`` पर होता है; इसके बजाय वह ब्रांच क्लोन करने के लिए ``--branch nightly`` जोड़ें। किसी विशिष्ट रिलीज़ के लिए::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

बाद के बदलाव लाने के लिए, क्लोन के अंदर से चलाएँ::

    git pull
    pip install -e .

दूसरी पंक्ति की ज़रूरत तभी होती है जब निर्भरताएँ या एंट्री पॉइंट बदले हों; Python कोड इसके बिना भी लागू हो जाता है। अगर pull करने के बाद भी कोई कमांड पुराना कोड चलाता है, तो ``spacr-doctor`` बताता है कि आपके पाथ पर असल में कौन-सा ``spacr`` है; आम तौर पर यही कारण होता है।

सोर्स कोड से इंस्टॉलेशन (हल्का)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

योगदानकर्ताओं को इतिहास चाहिए; केवल spaCR चलाने के लिए इनमें से कोई एक तरीका चुनें, जिन्हें 2026-09-15 को ``packaging/measure_clone_forms.sh`` से मापा गया::

    # One commit instead of every version: 540 MB downloaded, 69 s.
    # No history, so no git log, no git blame and no git bisect.
    # git pull still works, but stays shallow until git fetch --unshallow.
    git clone --depth 1 https://github.com/EinarOlafsson/spacr.git
    cd spacr && pip install -e .

    # Only the files spaCR runs from: 81 MB on disk, 39 s. No history
    # either, and no docs, tests, tools or example data.
    # --with-docs, --with-tests and --with-translations put those back;
    # --dir, --branch, --no-install and --help do the obvious things.
    # packaging/source_install_excludes.txt lists every skipped path.
    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

पूरा क्लोन 1186 MB के चेकआउट के लिए 5.8 GB डाउनलोड करता है। उस क्लोन में ``--filter=blob:none`` जोड़ने से कोई बचत नहीं होती: चेकआउट वैसे भी सारे blob ले आता है।


कमांड-लाइन प्रवेश बिंदु
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
   spacr-download --list                      # what example data exists
   spacr-download measure annotate            # fetch example sets by name
   spacr-make-masks --folder DIR              # curate masks as a resumable queue
   spacr-make-masks --folder DIR --order easy --limit 50

समस्या निवारण के समय ``SPACR_LOG_LEVEL=DEBUG`` सेट करें। रोटेटिंग लॉग ``~/.spacr/logs/spacr.log`` में लिखे जाते हैं।

``spacr-run --list`` उन मॉड्यूलों की सूची दिखाता है जिनके पास ग्राफ़िकल इंटरफ़ेस के बिना चलाने के लिए कमांड-लाइन प्रवेश बिंदु हैं। केवल GUI में उपलब्ध एनोटेशन, क्यूरेशन, तुलना और अन्वेषण मॉड्यूल इस सूची में शामिल नहीं होते।


मुख्य वर्कफ़्लो
-------------

मुख्य कार्यप्रवाह में छह मॉड्यूल हैं:

- **Mask** Cellpose से कोशिकाओं, नाभिकों, रोगजनकों और कोशिकांगों का विभाजन करता है।
- **Measure** आकृति-विज्ञान, तीव्रता, टेक्सचर, स्थानिक और सह-स्थानीकरण विशेषताओं के साथ ऑब्जेक्ट क्रॉप को SQLite में लिखता है।
- **Annotate** कीबोर्ड से संचालित ग्रिड में क्रॉप को लेबल करता है और सक्रिय-अधिगम कतारों का समर्थन करता है।
- **Classify** छवि- या मापन-आधारित मॉडल प्रशिक्षित करता है और प्रत्येक चेकपॉइंट के साथ होल्ड-आउट डेटा पर प्रदर्शन दर्ज करता है।
- **Map Barcodes** FASTQ रीड को वेल और gRNA से मैप करता है तथा प्रचुरता, टकराव और कवरेज का QC प्रदान करता है।
- **Regression** सतत, भिन्नात्मक और गणना प्रतिक्रियाओं के अनुकूल मॉडल परिवारों से गाइड, जीन, स्थिति और नियंत्रण प्रभावों का अनुमान लगाता है।

spaCR मॉड्यूल
-------------

.. spacr-workflow-begin

मुख्य
^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

डेटा
^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

उपकरण
^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

एसे
^^^

Quantitative readouts for biological assays.

| |Module_analyze_plaques|\ |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Mask API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Measure API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Annotate API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Classify API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Map Barcodes API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Regression API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Import API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_embeddings| image:: ../../../spacr/resources/icons/workflow/apps/embeddings.png
   :width: 16.0%
   :alt: Embeddings API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Run Compare API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Experiment Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Power / Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Dose–Response API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: QC API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Make Masks API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Align & Stitch API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Image UMAP API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Gate Editor API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Graph Builder API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Plaque Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Recruitment API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Invasion Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Replication Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

spaCR के साथ आने वाले सभी मॉड्यूल, उसी क्रम में जिसमें होम स्क्रीन उन्हें दिखाती है: पहले छह पाइपलाइन मॉड्यूल, फिर बाकी सब। किसी मॉड्यूल का API पृष्ठ खोलने के लिए उसकी टाइल चुनें।

हर टूल का विवरण `सुविधा गाइड <../../source/features.rst>`_ में देखें।

अन्य संसाधन
~~~~~~~~~~~~~~~

- `इंटरैक्टिव ट्यूटोरियल <https://einarolafsson.github.io/spacr/tutorials/>`_ — स्थापना से हिट जांच के माध्यम से 73 निर्देशित कार्यप्रवाह।
- `Python API त्वरित प्रारंभ <../../source/python_api.rst>`_ - स्क्रिप्ट, नोटबुक या एक क्लस्टर से पाइपलाइन चलाएं और वैध करें।
- `सुविधा गाइड <../../source/features.rst>`_ — क्षमताओं, परिपक्वता और वैकल्पिक एकीकरण।
- `शुद्ध API संदर्भ <https://einarolafsson.github.io/spacr/api/index.html>`_ - कार्य के आधार पर समर्थित प्रवेश बिंदु, पूर्ण मॉड्यूल संदर्भ एक स्तर गहरा है।
- `भाषा और अनुवाद गाइड <../../source/localization.rst>`_ — इंटरफ़ेस भाषाएं, संदर्भ सहायता और वैज्ञानिक-आउटपुट नीति।

भाषा और अनुवाद
~~~~~~~~~~~~~~~~~~~~~~

इंटरफ़ेस नेविगेशन और प्राथमिकताओं में दस भाषाओं का समर्थन करता है। AI और LIVE नियंत्रण, मॉड्यूल विवरण और समीक्षित संदर्भ सहायता भी अनुवादित हैं। पुनः आरंभ किए बिना **spaCR → प्राथमिकताएँ → भाषा** में भाषा बदलें। लॉग, पथ, डेटाबेस मान और मापन कभी अनुवादित नहीं होते; वैज्ञानिक आउटपुट मानक अंग्रेज़ी में रहता है। `संदर्भ-सहायता नीति <../../source/localization.rst#contextual-help>`_ देखें।

नौ गैर-अंग्रेजी कैटलॉग मशीन-ग्रेड किए जाते हैं और तकनीकी रूप से समीक्षा की जाती है, प्रत्येक भाषा के एक स्वदेशी बोलने वाले द्वारा अंत तक पढ़ने के बजाय. `समीक्षा स्कोप <../REVIEW_SCOPE_2026-09-04.md>`_ रिकॉर्ड किस भाषाओं में एक मानव पास था, कितने शरीर को कवर किया गया था, और प्रत्येक शब्द अंग्रेजी में निर्णय के अनुसार छोड़ दिया गया था.

एनिमेटेड सेटिंग मार्गदर्शन
~~~~~~~~~~~~~~~~~~~~~~~~~

दृश्य व्याख्या वाली सेटिंग के टूलटिप में **Animation** नियंत्रण मिलता है। `सेटिंग एनिमेशन गैलरी <https://einarolafsson.github.io/spacr/setting_animations.html>`_ या `सेटिंग एनिमेशन रजिस्ट्री <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ देखें।

डेटा
----

संदर्भ डेटासेट
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: BioStudies माइक्रोस्कोपी डेटासेट खोलें
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Hugging Face परीक्षण डेटासेट खोलें
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: NCBI अनुक्रमण डेटासेट खोलें
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: spaCRPower खोलें
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: bioRxiv प्रीप्रिंट खोलें
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

जानवरों का मॉडल
~~~~~~~~~~~~~~~

spaCR प्रशिक्षित मॉडलों का एक कैटलॉग भेजता है और उन्हें मांग पर पकड़ता है. होम स्क्रीन से ब्राउज़ करने और स्थापित करने के लिए **मॉडल Zoo** खोलें, या सेटिंग्स फ़ाइल में एक कुंजी नामित करें - ``pathogen_model: toxoplasma_pv_v1`` - और मॉडल को डाउनलोड किया जाता है और पहली बार जांच की जाती है. प्रत्येक प्रकाशित प्रविष्टि में एक SHA-256 होता है; एक के बिना एक प्रविष्ट को अस्वीकार कर दिया जाता है, बल्कि स्थापित किया जा सकता है, क्योंकि एक ट्रिगर या प्रतिस्थापित चेकपॉइंट को वास्तविक से नहीं बताया जा सकता।

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 229 images from 2 datasets, 104 round-1 and 125 newly curated
     - F1 0.864 against 0.713 for stock cpsam on 11 held-out in-house wells, at IoU 0.5; literature hold-out pending
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_plaque_v2``
       (Cellpose-SAM (cpsam_v2))
     - 488 curated fields across four domains -- 298 wells cropped from published figures, 96 phone-camera wells, 67 PFA and 27 methanol-fixed whole-well microscope scans; 27,582 plaques
     - not scored against stock; on 81 held-out fields it ties round 3 on literature (0.819 vs 0.820) and beats it by 0.166 on phone-camera wells (0.415 vs 0.249)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993 on its own held-out split; on the test set shared with v2 it scores mAP50 0.8838, against v2's 0.9457
   * - ``toxoplasma_well_detector_v2``
       (YOLO26n (ultralytics 8.4.155))
     - plate images and literature figures; 1,070 train / 254 val / 129 test, split by PMC article so no paper is in two sets; training data at einarolafsson/toxoplasma-plaque-well-detector-dataset
     - mAP50 0.9457 and mAP50-95 0.8341 against v1's (yolo_welldetect_v3.pt) 0.8838 and 0.7630 on the SAME test set; stock YOLO has no plaque-well class, so v1 is the baseline
   * - ``toxoplasma_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOST CELL MASK channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5
   * - ``toxoplasma_pv_v2``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 556 curated images accumulated over five rounds
     - F1 0.817 +/- 0.036 by 5-fold cross-validation over 619 pairs; ~0.86 against 0.713 for stock on the 11 in-house held-out wells
   * - ``toxoplasma_pv_v3``
       (Cellpose-SAM (cpsam_v2))
     - the 556 curated PV fields of round 5, split 437 train / 108 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset
     - F1 0.860 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.803 against 0.505
   * - ``live_cell_v1``
       (Cellpose-SAM (cpsam_v2))
     - 11,007 transmitted-light fields from 14 public datasets, split by acquisition 6,778 train / 2,030 validation / 2,199 test; training data at einarolafsson/live-cell-segmentation-dataset
     - on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell
   * - ``nuclei_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - nuclei predicted from the HOST CELL MASK channel alone; 453 well-grouped held-out fields, hosts HFF/HeLa/THP1
     - F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5
   * - ``cell_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - the HOST CELL outline predicted from the Hoechst (nuclear) channel alone; 2,578 training fields and 451 held-out test fields, split by well so no well is on both sides
     - F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569
   * - ``toxoplasma_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOECHST channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5

.. spacr-model-zoo-end

ऊपर दिए गए प्रत्येक आंकड़े को उन छवियों पर मापा जाता है जिन्हें मॉडल ने प्रशिक्षण में कभी नहीं देखा है।

**सटीकता** यह है कि एक मॉडल के वस्तुओं में से कितने वास्तविक हैं; **रिकॉर्ड** यह कि वास्तविक वस्तुएं कितनी हैं. वे विपरीत दिशाओं में विफल होते हैं: खराब परिशुद्धता प्लेटों का आविष्कार करती है, खराब रिकॉल उन्हें याद करती है.

**F1** दोनों संयुक्त हैं, और यह उद्धृत किया जाता है क्योंकि प्रत्येक अकेले ट्रिविल रूप से खेला जाता है - निकट-पूर्ण सटीकता के लिए एक अविश्वसनीय प्लेक की रिपोर्ट करें, या करीब-पूर्ण पुनरावृत्ति के लिए प्रत्येक अंधेरे ब्लॉब. जो आप बेहतर खो देंगे, यह अनुमान पर निर्भर करता है, और गिनती आमतौर पर बेहतर है अति-कवाना द्वारा सेवा की जाती है: प्लेक्स मॉडल को 0.858 की परिभाषा में स्वीकार किया गया था और 0.811 को 0.939 और 0.631.

**IoU**, यूनियन के माध्यम से पारगमन, यह है कि कितना एक अनुमानित वस्तु और वास्तविक एक ओवरपॉप, वे एक साथ कवर क्षेत्र द्वारा विभाजित है. यह नियंत्रक है कि बाकी के खिलाफ पढ़ा जाता है, इसलिए एक स्कोर इसका सीमा के बिना कुछ भी नहीं है: "F1 0.864 में IoU 0.5" एक वैक्यूल की गिनती करता है जैसा कि पाया जाता है जब दोनों आउटलिन अपने संयुक्त क्षेत्र के आधे से अधिक सहमत होते हैं।

**mAP50** और **map50-95** डिटेक्टर से संबंधित हैं. पहला पूछता है कि क्या बर्तन पाए गए हैं; दूसरा इसे 0.5 से 0.95 तक के दस सीमाओं के माध्यम से दोहराता है, इसलिए यह भी पूछा जाता है कि प्रत्येक बॉक्स को कितनी ठोस रूप से खींचा जाता है. उनके बीच का अंतर स्थान है, नहीं पहचान।

**Cross-validated**, एक **SD** के साथ, स्कोर का मतलब है कि विभिन्न विभाजनों पर तीन रनों का औसत है और एसडी यह है कि वे कितनी दूर चले गए हैं. एक विभाजन भाग्यशाली हो सकता है: इस मॉडल का साहित्यिक आंकड़ा एक 19 अच्छी विभाजन पर 0.834 है और सभी तीनों के बीच 0.806 है.

मॉडलों को उनके लेखक के स्वयं के Hugging Face खाते पर होस्ट किया जाता है, इसलिए एक में योगदान करने का मतलब किसी और के लेखन एक्सेस को सौंपना नहीं है. ``spacr.model_zoo`` का ``publish_model`` अपलोड करता है और जोड़ने के लिए कैटलॉग पंक्ति प्रिंट करता है.


प्रदर्शन का निदान
----------------------

हार्डवेयर रिपोर्ट बनाएँ और उसे प्रदर्शन-संबंधी इश्यू के साथ संलग्न करें::

    python tools/spacr_hardware_report.py

``~/.spacr/reports`` पर बचत करें और मार्ग प्रिंट करें. ``--quick`` लंबे संदर्भ संकेतों को स्काइप करता है; ``--out PATH`` स्थान निर्धारित करता है.

कोई परियोजना डेटा नहीं पढ़ता. टाइम आयात, संख्यात्मक पुस्तकालय, खिड़की निर्माण और एनीमेशन. रिपोर्ट प्रोसेसर-आर्किटेक्चर इमुलेशन (एक x86_64 Python Apple सिलिकॉन पर निर्माण) और NumPy के BLAS कार्यान्वयन।

कमांड लाइन संदर्भ
----------------------

नीचे दिए गए प्रत्येक कमांड को ``pip install spacr`` द्वारा स्थापित किया जाता है. उनमें से सभी ``--help`` को स्वीकार करते हैं.

आवेदन शुरू करने के लिए
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` मॉडल सेटअप स्क्रीनिंग को स्काइप करता है, जो अन्यथा एक अप्रत्याशित नौकरी को अवरुद्ध करेगा।

``spacr-qt`` और ``spacr-nightly`` ``spacr`` के सहयोगी हैं।

जब spaCR शुरू नहीं होता है
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` प्रत्येक चेक पर एक पंक्ति प्रिंट करता है, जिसमें प्रत्येक विफलता के लिए एक कमांड चलाया जाता है. यह यह भी रिपोर्ट करता है कि ``spacr`` मार्ग पर है, जो कि एक पुराने संपादित करने योग्य स्थापना की छाया है.

``safespacr`` प्रत्येक प्राथमिकता को अपने डिफ़ॉल्ट के रूप में पढ़ता है और पृष्ठभूमि, एनीमेशन, वर्बस लॉगिंग और प्री-लोड को मजबूर करता है. जब एक सहेजा गया प्राथमिकता लॉन्च को तोड़ देती है तो इसका उपयोग करें. यह स्थायी रूप से कुछ भी नहीं बदलता है.

मॉड्यूल बेहोश रूप से चलाएं
~~~~~~~~~~~~~~~~~~~~~~~~~~

कोई Qt, कोई डिस्प्ले नहीं - क्लस्टर, सर्वर और सीआई के लिए।

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` उसी सेटिंग्स को पढ़ता है जो चलता है और रिपोर्ट करता है कि क्या खो रहा है, विरोधाभासी है या कुछ भी नहीं बता रहा है।

``spacr-run --list`` केवल एक शीर्ष के बिना प्रवेश बिंदु के साथ मॉड्यूल दिखाता है; नोटिस, चिकित्सा और अन्वेषण इंटरैक्टिव हैं और अनदेखा किया गया है।

बाद में एक दौड़ की जांच करें
~~~~~~~~~~~~~~~~~~~~~~~~~~~

प्रत्येक रन ``~/.spacr/runs`` में रिकॉर्ड किया जाता है, जिसमें इसकी सेटिंग्स, हैश किए गए इनपुट, आउटपुट्स, चेतावनी, संस्करण और बीज होते हैं।

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

डेटा की जांच और स्थापना
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

परिवेश
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

रूटिंग लॉग ``~/.spacr/logs/spacr.log`` में लिखे जाते हैं. उस फ़ाइल को एक बग रिपोर्ट में जोड़ें.


योगदान और सहायता
------------------------

बग रिपोर्ट और स्पष्ट रूप से सीमित फ़ीचर अनुरोध `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ के माध्यम से भेजें। किसी विफलता की रिपोर्ट करते समय spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग्स और संबंधित लॉग अंश शामिल करें। ``spacr-doctor`` इनमें से अधिकांश जानकारी एकत्र करता है; प्रदर्शन संबंधी समस्या की रिपोर्ट करते समय हार्डवेयर रिपोर्ट भी शामिल करें।

लाइसेंस
~~~~~~~~~

spaCR is released under the `BSD 3 क्लास लाइसेंस <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

यदि spaCR प्रकाशित कार्य में योगदान देता है, तो एक उद्धरण का मूल्यांकन किया जाता है और लाइसेंस की शर्त नहीं है - नीचे `spaCR का संदर्भ`_ देखें।

ट्यूटोरियल
~~~~~~~~~

`इंटरैक्टिव spaCR ट्यूटोरियल लाइब्रेरी <https://einarolafsson.github.io/spacr/tutorials/>`_ में स्थापना और प्रत्येक ऐप कार्यप्रवाह के वर्णित तथा कैप्शनयुक्त मार्गदर्शन हैं: आठ भाषाओं में 50 आवाज़ों के साथ 73 पाठ।

spaCR का संदर्भ
~~~~~~~~~~~~~~

यदि spaCR आपके शोध में योगदान देता है, तो इसका उद्धरण दें:

Olafsson EB, *et al.* एक संयोजित छवि-आधारित CRISPR स्क्रीनिंग EAF1 को *T. gondii* के रूप में पहचानती है ESCRT उप-विवाद का मॉड्यूलर।

`BioRxiv प्रीप्रिंट <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `सॉफ्टवेयर संग्रह <https://doi.org/10.5281/zenodo.21343316>`_

आभार
~~~~~~~~~~~~~~~

spaCR NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch और Qt सहित मुक्त वैज्ञानिक सॉफ़्टवेयर पर आधारित है। बहुभाषी दस्तावेज़ और इंटरफ़ेस कैटलॉग तैयार करने में उपयोग किए गए मॉडल के लिए `अनुवाद मॉडल श्रेय <../TRANSLATION_MODELS.md>`_ देखें।
