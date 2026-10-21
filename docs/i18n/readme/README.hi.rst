|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

वेबसाइट: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: spaCR वेबसाइट

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API दस्तावेज़

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: इंटरैक्टिव ट्यूटोरियल

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI संस्करण

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 से 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Codecov परीक्षण कवरेज

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: परीक्षण समूह

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt इंटरफ़ेस

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub समस्याएँ

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause लाइसेंस

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv प्रीप्रिंट

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge संस्करण

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge की नवीनतम रिलीज़ की तिथि

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: GHCR पर Docker इमेज

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI डाउनलोड

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS और Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: spaCR को उद्धृत करें

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: पिछले 30 पूरे दिनों में spaCR की PyPI डाउनलोड रैंकिंग

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/intro-1.desktop.svg"><img src="../../readme_images/hi/intro-1.svg" alt="← पिछला"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/intro-2.desktop.svg"><img src="../../readme_images/hi/intro-2.svg" alt="अगला →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-language-picker.desktop.svg"><img src="../../readme_images/hi/generated-language-picker.svg" alt="भाषाएँ: हिन्दी"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/hi/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/spacr.desktop.svg"><img src="../../readme_images/hi/spacr.svg" alt="spaCR. CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण। spaCR माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रति-वस्तु फीनोटाइप को अनुक्रमण से प्राप्त गाइड प्रचुरता के साथ एकीकृत करता है और अनुमान लगाता है कि कौन-से जीन फीनोटाइपिक परिवर्तनों से जुड़े हैं। प्लेट छवियों और FASTQ रीड से शुरू करके, यह प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव अनुमान तथा प्राथमिकता के अनुसार क्रमित हिट सूची बनाता है। segmentation, measurement, annotation और classification modules भी एक sequencing arm के बिना चलता है। Make Masks सेगमेंटेशन मास्क सुधारता है और YOLO निर्यात के लिए Box टूल से स्वतंत्र, क्लास-लेबल वाले आयतों का एनोटेशन करता है। ये बॉक्स स्रोत इमेज या मास्क बदले बिना अपने लेबल और इतिहास रखते हैं। हर टूल का विवरण सुविधा गाइड में देखें। छवियाँ, मास्क और क्रॉप डिस्क पर फ़ाइलें हैं। मापन, एनोटेशन और मॉडल पूर्वानुमान परियोजना के SQLite डेटाबेस measurements.db में तालिकाएँ हैं। यह डेस्कटॉप एप्लिकेशन के रूप में, या वर्कस्टेशन, सर्वर या क्लस्टर पर हेडलेस रूप में चलता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/try-spacr.desktop.svg"><img src="../../readme_images/hi/try-spacr.svg" alt="spaCR आज़माएँ."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/hi/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/hardware-support.desktop.svg"><img src="../../readme_images/hi/hardware-support.svg" alt="हार्डवेयर समर्थन."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/hi/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-hardware-1.desktop.svg"><img src="../../readme_images/hi/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. समर्थित (स्थिर) लागू (बीटा) केवल CPU समर्थन."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/hi/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-spacr.desktop.svg"><img src="../../readme_images/hi/install-spacr.svg" alt="spaCR इंस्टॉल करें. इंस्टॉलेशन का तरीका चुनें: डेस्कटॉप एप्लिकेशन: Linux, macOS या Windows के लिए इंस्टॉलर, जिसमें अपना Python शामिल है। pip: Conda परिवेश में python -m pip install spacr चलाएँ। conda-forge: Conda परिवेश में conda install conda-forge::spacr चलाएँ। Docker: Linux कंटेनर में कमांड-लाइन पाइपलाइन, CPU इमेज या NVIDIA GPU इमेज से:"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.5 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.5-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/hi/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-spacr-2.desktop.svg"><img src="../../readme_images/hi/install-spacr-2.svg" alt="Docker गाइड (nightly, अंग्रेज़ी में) में GPU सेटअप, डेटा माउंट और पाइपलाइन कमांड बताए गए हैं।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/desktop-application.desktop.svg"><img src="../../readme_images/hi/desktop-application.svg" alt="डेस्कटॉप एप्लिकेशन. इंस्टॉलर अपने स्वयं के Python को जोड़ते हैं. कॉन्डा की आवश्यकता नहीं है."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="Windows 10/11 के लिए spaCR 1.5.1.5 डाउनलोड करें"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="macOS 11+ (Intel और Apple Silicon) के लिए spaCR 1.5.1.5 डाउनलोड करें"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="64-बिट Linux के लिए spaCR 1.5.1.5 डाउनलोड करें"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/hi/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/desktop-application-2.desktop.svg"><img src="../../readme_images/hi/desktop-application-2.svg" alt="Linux पर डाउनलोड की गई फ़ाइल को निष्पादन योग्य बनाएँ और चलाएँ:"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/hi/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/desktop-application-3.desktop.svg"><img src="../../readme_images/hi/desktop-application-3.svg" alt="macOS पर, .pkg खोलें. वर्तमान बीटा नोटिस नहीं किया गया है; यदि Gatekeeper इसे ब्लॉक करता है, तो सिस्टम सेटिंग्स → गोपनीयता और सुरक्षा → किसी भी तरह से खोलें का चयन करें. इंस्टॉलेशन, अपडेट, दोबारा इंस्टॉल करने, अनइंस्टॉल करने, ऑफ़लाइन उपयोग और समस्या निवारण के निर्देशों के लिए nightly शाखा की वर्तमान इंस्टॉलेशन गाइड (अंग्रेज़ी में) देखें। वर्कस्टेशन और सर्वर संबंधी सुझावों तथा GPU संगतता तालिकाओं के लिए सिस्टम आवश्यकताएँ देखें।"></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/hi/desktop-application-3-link-1.svg" alt="nightly शाखा की वर्तमान इंस्टॉलेशन गाइड (अंग्रेज़ी में)"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/hi/desktop-application-3-link-2.svg" alt="सिस्टम आवश्यकताएँ"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/pypi-installation.desktop.svg"><img src="../../readme_images/hi/pypi-installation.svg" alt="PyPI से इंस्टॉलेशन. PyPI रिलीज़ के लिए, Conda वातावरण के भीतर pip से spaCR इंस्टॉल करें। Python 3.12 में वैकल्पिक वैज्ञानिक पैकेजों की सबसे व्यापक उपलब्धता है:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/hi/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/pypi-installation-2.desktop.svg"><img src="../../readme_images/hi/pypi-installation-2.svg" alt="spaCR, Python 3.9 through 3.14 का समर्थन करता है, Python 3.14.1 को छोड़कर, जिसे torchvision बाहर रखता है। सबसे भारी CUDA और ROCm कार्यप्रवाहों के लिए Linux की अनुशंसा की जाती है; macOS और Windows भी समर्थित हैं, और दोनों अपने GPU का उपयोग करते हैं — macOS, Metal के माध्यम से, जो Apple Silicon और Intel Mac में लगे AMD कार्डों को कवर करता है, और Windows, CUDA या DirectML के माध्यम से। मानक इंस्टॉलेशन में Qt डेस्कटॉप इंटरफ़ेस शामिल है। सर्वर, क्लस्टर या CI रनर पर इसे खोले बिना कमांड-लाइन पाइपलाइन चलाएँ:"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/hi/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/pypi-installation-3.desktop.svg"><img src="../../readme_images/hi/pypi-installation-3.svg" alt="वैकल्पिक एकीकरण अलग से स्थापित किए जाते हैं, उदाहरण के लिए spacr[zarr], spacr[omero],spacr[napari] और spacr[czi,nd2,lif]. पूर्ण अतिरिक्त और Python संस्करण संगतता तालिका में स्थापना गाइड (nightly, अंग्रेज़ी में) देखें."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/conda-forge-installation.desktop.svg"><img src="../../readme_images/hi/conda-forge-installation.svg" alt="conda-forge से इंस्टॉलेशन. आधिकारिक conda-forge पैकेज सक्रिय वातावरण में spaCR और उसकी डेस्कटॉप निर्भरताएँ इंस्टॉल करता है:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source.desktop.svg"><img src="../../readme_images/hi/install-from-source.svg" alt="सोर्स कोड से इंस्टॉलेशन. रिपॉज़िटरी को क्लोन करें और उसे एडिटेबल मोड में इंस्टॉल करें, ताकि आपकी वर्किंग कॉपी ही इंस्टॉल किया गया पैकेज हो और बदलाव दोबारा इंस्टॉल किए बिना लागू हो जाएँ:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source-2.desktop.svg"><img src="../../readme_images/hi/install-from-source-2.svg" alt="यह main को क्लोन करता है, जो डिफ़ॉल्ट ब्रांच है और जिसमें नवीनतम रिलीज़ रहती है। विकास nightly पर होता है; इसके बजाय वह ब्रांच क्लोन करने के लिए --branch nightly जोड़ें। किसी विशिष्ट रिलीज़ के लिए:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source-3.desktop.svg"><img src="../../readme_images/hi/install-from-source-3.svg" alt="बाद के बदलाव लाने के लिए, क्लोन के अंदर से चलाएँ:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source-4.desktop.svg"><img src="../../readme_images/hi/install-from-source-4.svg" alt="निर्भरताएँ या प्रवेश बिंदु बदलने पर फिर से इंस्टॉल करें। Python कोड में बदलाव सीधे लागू होते हैं; spacr-doctor सक्रिय इंस्टॉलेशन की पहचान करता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source-light.desktop.svg"><img src="../../readme_images/hi/install-from-source-light.svg" alt="सोर्स कोड से इंस्टॉलेशन (हल्का). योगदानकर्ताओं को संस्करण इतिहास चाहिए; केवल spaCR चलाने के लिए नीचे दिए विकल्पों में से चुनें। nightly के 05302fd5c कमिट की माप 2026-10-07 को packaging/measure_clone_forms.sh से की गई:"></picture></a></div>

.. code-block:: bash

   # One commit instead of every version: 2048 MB downloaded, 140 s.
   # No history, so no git log, no git blame and no git bisect.
   # git pull still works, but stays shallow until git fetch --unshallow.
   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

   # Runtime files: 104 MB on disk, 6 s (Git: 38 MB; files: 66 MB).
   # No history, docs, tests, tools, features or example data.
   # --with-docs, --with-tests and --with-translations put those back;
   # --dir, --branch, --no-install and --help do the obvious things.
   # packaging/source_install_excludes.txt lists every skipped path.
   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/hi/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/install-from-source-light-2.desktop.svg"><img src="../../readme_images/hi/install-from-source-light-2.svg" alt="nightly के पूर्ण क्लोन ने 9.25 GiB डाउनलोड किया। उथले क्लोन में --filter=blob:none जोड़ने से चेकआउट छोटा नहीं होता: उसका Git ऑब्जेक्ट भंडार अब भी 2032 MB है। Git आवश्यकता पड़ने पर बाद में लाए गए ऑब्जेक्ट की सूचना नहीं देता, इसलिए कुल डाउनलोड नहीं मापा गया। nightly की संस्करण-नियंत्रित फ़ाइलें चेकआउट में 958 MB लेती हैं (2026-10-20 को मापी गईं), जिसमें Git इतिहास शामिल नहीं है। डाउनलोड का आकार और समय शाखा के अनुसार बदलते हैं।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/command-line-entry-points.desktop.svg"><img src="../../readme_images/hi/command-line-entry-points.svg" alt="कमांड-लाइन प्रवेश बिंदु."></picture></a></div>

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

.. raw:: html

   <div><a href="../../readme_text/hi/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/hi/command-line-entry-points-2.svg" alt="spacr-run --list उन मॉड्यूलों की सूची दिखाता है जिनके पास ग्राफ़िकल इंटरफ़ेस के बिना चलाने के लिए कमांड-लाइन प्रवेश बिंदु हैं। केवल GUI में उपलब्ध एनोटेशन, क्यूरेशन, तुलना और अन्वेषण मॉड्यूल इस सूची में शामिल नहीं होते।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/core-workflow.desktop.svg"><img src="../../readme_images/hi/core-workflow.svg" alt="मुख्य वर्कफ़्लो. मुख्य कार्यप्रवाह में छह मॉड्यूल हैं: Mask Cellpose से कोशिकाओं, नाभिकों, रोगजनकों और कोशिकांगों का विभाजन करता है। Measure आकृति-विज्ञान, तीव्रता, टेक्सचर, स्थानिक और सह-स्थानीकरण विशेषताओं के साथ ऑब्जेक्ट क्रॉप को SQLite में लिखता है। Annotate कीबोर्ड से संचालित ग्रिड में क्रॉप को लेबल करता है और सक्रिय-अधिगम कतारों का समर्थन करता है। Classify छवि- या मापन-आधारित मॉडल प्रशिक्षित करता है और प्रत्येक चेकपॉइंट के साथ होल्ड-आउट डेटा पर प्रदर्शन दर्ज करता है। Map Barcodes FASTQ रीड को वेल और gRNA से मैप करता है तथा प्रचुरता, टकराव और कवरेज का QC प्रदान करता है। Regression सतत, भिन्नात्मक और गणना प्रतिक्रियाओं के अनुकूल मॉडल परिवारों से गाइड, जीन, स्थिति और नियंत्रण प्रभावों का अनुमान लगाता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/spacr-modules.desktop.svg"><img src="../../readme_images/hi/spacr-modules.svg" alt="spaCR मॉड्यूल."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/hi/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-workflow-1.desktop.svg"><img src="../../readme_images/hi/generated-workflow-1.svg" alt="मुख्य. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/hi/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-workflow-2.desktop.svg"><img src="../../readme_images/hi/generated-workflow-2.svg" alt="डेटा. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/hi/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-workflow-3.desktop.svg"><img src="../../readme_images/hi/generated-workflow-3.svg" alt="उपकरण. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/hi/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-workflow-4.desktop.svg"><img src="../../readme_images/hi/generated-workflow-4.svg" alt="जीव. जीव-विशिष्ट छवि विश्लेषण और परिमाणात्मक परीक्षण परिणाम।"></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="Mask API खोलें"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="Measure API खोलें"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="Annotate API खोलें"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="Classify API खोलें"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="Map Barcodes API खोलें"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="Regression API खोलें"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="Import API खोलें"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="Embeddings API खोलें"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="Run Compare API खोलें"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="Experiment Design API खोलें"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="Power / Design API खोलें"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="Dose–Response API खोलें"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="QC API खोलें"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="Make Masks API खोलें"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="Align &amp; Stitch API खोलें"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="Image UMAP API खोलें"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="Gate Editor API खोलें"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="Graph Builder API खोलें"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="Toxoplasma API खोलें"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/hi/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/spacr-modules-2.desktop.svg"><img src="../../readme_images/hi/spacr-modules-2.svg" alt="होम स्क्रीन पर टाइल वाले सभी मॉड्यूल उसी स्क्रीन के क्रम में दिए गए हैं: पहले पाइपलाइन के छह मॉड्यूल, फिर बाकी। किसी मॉड्यूल का API पृष्ठ खोलने के लिए उसकी टाइल चुनें।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources.desktop.svg"><img src="../../readme_images/hi/other-resources.svg" alt="अन्य संसाधन."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources-2-1.desktop.svg"><img src="../../readme_images/hi/other-resources-2-1.svg" alt="इंटरैक्टिव ट्यूटोरियल — इंस्टॉलेशन से लेकर स्क्रीनिंग में मिले हिट की जाँच तक के निर्देशित कार्यप्रवाह।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources-2-2.desktop.svg"><img src="../../readme_images/hi/other-resources-2-2.svg" alt="Python API त्वरित प्रारंभ - स्क्रिप्ट, नोटबुक या एक क्लस्टर से पाइपलाइन चलाएं और वैध करें।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources-2-3.desktop.svg"><img src="../../readme_images/hi/other-resources-2-3.svg" alt="सुविधा गाइड — क्षमताओं, परिपक्वता और वैकल्पिक एकीकरण।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources-2-4.desktop.svg"><img src="../../readme_images/hi/other-resources-2-4.svg" alt="शुद्ध API संदर्भ - कार्य के आधार पर समर्थित प्रवेश बिंदु, पूर्ण मॉड्यूल संदर्भ एक स्तर गहरा है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-resources-2-5.desktop.svg"><img src="../../readme_images/hi/other-resources-2-5.svg" alt="भाषा और अनुवाद गाइड — इंटरफ़ेस भाषाएं, संदर्भ सहायता और वैज्ञानिक-आउटपुट नीति।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/language-translation.desktop.svg"><img src="../../readme_images/hi/language-translation.svg" alt="भाषा और अनुवाद. इंटरफ़ेस नेविगेशन और प्राथमिकताओं में दस भाषाओं का समर्थन करता है। AI और LIVE नियंत्रण, मॉड्यूल विवरण और समीक्षित संदर्भ सहायता भी अनुवादित हैं। पुनः आरंभ किए बिना spaCR → प्राथमिकताएँ → भाषा में भाषा बदलें। लॉग, पथ, डेटाबेस मान और मापन कभी अनुवादित नहीं होते; वैज्ञानिक आउटपुट मानक अंग्रेज़ी में रहता है। संदर्भ-सहायता नीति देखें।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/animated-setting-guidance.desktop.svg"><img src="../../readme_images/hi/animated-setting-guidance.svg" alt="एनिमेटेड सेटिंग मार्गदर्शन. दृश्य व्याख्या वाली सेटिंग के टूलटिप में Animation नियंत्रण मिलता है। सेटिंग एनिमेशन गैलरी या सेटिंग एनिमेशन रजिस्ट्री देखें।"></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/hi/animated-setting-guidance-link-1.svg" alt="सेटिंग एनिमेशन गैलरी"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/hi/animated-setting-guidance-link-2.svg" alt="सेटिंग एनिमेशन रजिस्ट्री"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/data.desktop.svg"><img src="../../readme_images/hi/data.svg" alt="डेटा. संदर्भ डेटासेट."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="BioStudies माइक्रोस्कोपी डेटासेट खोलें"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="Hugging Face परीक्षण डेटासेट खोलें"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="NCBI अनुक्रमण डेटासेट खोलें"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="spaCRPower खोलें"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="bioRxiv प्रीप्रिंट खोलें"></a>

.. raw:: html

   <div><a href="../../readme_text/hi/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/model-zoo.desktop.svg"><img src="../../readme_images/hi/model-zoo.svg" alt="मॉडल लाइब्रेरी. spaCR प्रशिक्षित मॉडलों का एक कैटलॉग भेजता है और उन्हें मांग पर पकड़ता है. होम स्क्रीन से ब्राउज़ करने और स्थापित करने के लिए मॉडल Zoo खोलें, या सेटिंग्स फ़ाइल में एक कुंजी नामित करें - pathogen_model: toxoplasma_pv_v1 - और मॉडल को डाउनलोड किया जाता है और पहली बार जांच की जाती है. प्रत्येक प्रकाशित प्रविष्टि में एक SHA-256 होता है; एक के बिना एक प्रविष्ट को अस्वीकार कर दिया जाता है, बल्कि स्थापित किया जा सकता है, क्योंकि एक ट्रिगर या प्रतिस्थापित चेकपॉइंट को वास्तविक से नहीं बताया जा सकता।"></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v5.svg"><img src="../../models/cards/toxoplasma_plaque_v5.phone.svg" alt="Score card of Plaques v5 (toxoplasma_plaque_v5, default): F1 0.816 on 153 held-out fields against 0.227 for stock cpsam_v2. F1 0.816 against stock cpsam_v2&#x27;s 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1.svg" alt="पुराने मॉडल."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-7.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-1-link-8.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-2.svg" alt="विकल्प."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-2-link-3.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-2-link-3.svg" alt="live_cell_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/hi/generated-model-zoo-3.svg" alt="मास्क जनरेटर"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/hi/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/diagnosing-performance.desktop.svg"><img src="../../readme_images/hi/diagnosing-performance.svg" alt="प्रदर्शन का निदान. हार्डवेयर रिपोर्ट बनाएँ और उसे प्रदर्शन-संबंधी इश्यू के साथ संलग्न करें:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/hi/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/hi/diagnosing-performance-2.svg" alt="~/.spacr/reports पर बचत करें और मार्ग प्रिंट करें. --quick लंबे संदर्भ संकेतों को स्काइप करता है; --out PATH स्थान निर्धारित करता है. प्रोजेक्ट का कोई डेटा नहीं पढ़ता। इम्पोर्ट, संख्यात्मक लाइब्रेरी, विंडो निर्माण और एनीमेशन में लगने वाला समय मापता है, और Apple Silicon पर x86_64 इम्यूलेशन तथा NumPy के BLAS कार्यान्वयन की जानकारी देता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/command-line-reference.desktop.svg"><img src="../../readme_images/hi/command-line-reference.svg" alt="कमांड लाइन संदर्भ. नीचे दिए गए प्रत्येक कमांड को pip install spacr द्वारा स्थापित किया जाता है. उनमें से सभी --help को स्वीकार करते हैं."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/launching-the-application.desktop.svg"><img src="../../readme_images/hi/launching-the-application.svg" alt="आवेदन शुरू करने के लिए."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/hi/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/hi/when-spacr-will-not-start.svg" alt="जब spaCR शुरू नहीं होता है."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/hi/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/hi/when-spacr-will-not-start-2.svg" alt="spacr-doctor प्रत्येक जाँच के लिए एक पंक्ति प्रिंट करता है, और प्रत्येक विफलता के लिए चलाने का एक कमांड देता है। यह यह भी बताता है कि पथ पर कौन-सा spacr है। safespacr हर प्राथमिकता को उसके डिफ़ॉल्ट मान से पढ़ता है और पृष्ठभूमि, एनिमेशन, विस्तृत लॉगिंग और प्रीलोडिंग बंद कर देता है। जब कोई सहेजी गई प्राथमिकता प्रोग्राम को शुरू होने से रोके, तब इसका उपयोग करें। सुरक्षित मोड में सहेजी गई प्राथमिकताएँ सामान्य रूप से लिखी जाती हैं और बनी रहती हैं।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/running-modules-headlessly.desktop.svg"><img src="../../readme_images/hi/running-modules-headlessly.svg" alt="मॉड्यूल बेहोश रूप से चलाएं. कोई Qt, कोई डिस्प्ले नहीं - क्लस्टर, सर्वर और सीआई के लिए।"></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/hi/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/hi/running-modules-headlessly-2.svg" alt="validate वही सेटिंग्स पढ़ता है जो रन पढ़ता, और हर समस्या को, जैसे कोई छूटी हुई सेटिंग या ऐसा पथ जो मौजूद नहीं है, सुधार के साथ सूचीबद्ध करता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/inspecting-a-run.desktop.svg"><img src="../../readme_images/hi/inspecting-a-run.svg" alt="किसी रन की जाँच. प्रत्येक रन ~/.spacr/runs में रिकॉर्ड किया जाता है, जिसमें इसकी सेटिंग्स, हैश किए गए इनपुट, आउटपुट्स, चेतावनी, संस्करण और बीज होते हैं।"></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/hi/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/hi/auditing-data-and-installation.svg" alt="डेटा की जांच और स्थापना."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/hi/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/environment.desktop.svg"><img src="../../readme_images/hi/environment.svg" alt="परिवेश."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/hi/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/environment-2.desktop.svg"><img src="../../readme_images/hi/environment-2.svg" alt="रूटिंग लॉग ~/.spacr/logs/spacr.log में लिखे जाते हैं. उस फ़ाइल को एक बग रिपोर्ट में जोड़ें."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/contributing-and-support.desktop.svg"><img src="../../readme_images/hi/contributing-and-support.svg" alt="योगदान और सहायता. बग रिपोर्ट और स्पष्ट रूप से सीमित फ़ीचर अनुरोध GitHub Issues के माध्यम से भेजें। किसी विफलता की रिपोर्ट करते समय spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग्स और संबंधित लॉग अंश शामिल करें। spacr-doctor इनमें से अधिकांश जानकारी एकत्र करता है; प्रदर्शन संबंधी समस्या की रिपोर्ट करते समय हार्डवेयर रिपोर्ट भी शामिल करें।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/licensing.desktop.svg"><img src="../../readme_images/hi/licensing.svg" alt="लाइसेंस. spaCR is released under the BSD 3 क्लास लाइसेंस. यदि spaCR प्रकाशित कार्य में योगदान देता है, तो एक उद्धरण का मूल्यांकन किया जाता है और लाइसेंस की शर्त नहीं है - नीचे spaCR का संदर्भ देखें।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/tutorials.desktop.svg"><img src="../../readme_images/hi/tutorials.svg" alt="ट्यूटोरियल. spaCR की इंटरैक्टिव ट्यूटोरियल लाइब्रेरी में इंस्टॉलेशन और मॉड्यूल के उपयोग की चरण-दर-चरण जानकारी है। हर पाठ में उपलब्ध नैरेशन और भाषाएँ दी गई हैं।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/citing-spacr.desktop.svg"><img src="../../readme_images/hi/citing-spacr.svg" alt="spaCR का संदर्भ. यदि spaCR आपके शोध में योगदान देता है, तो इसका उद्धरण दें: Olafsson EB, et al. एक संयोजित छवि-आधारित CRISPR स्क्रीनिंग EAF1 को T. gondii के रूप में पहचानती है ESCRT उप-विवाद का मॉड्यूलर।"></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/hi/citing-spacr-2-1.svg" alt="BioRxiv प्रीप्रिंट"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/hi/citing-spacr-2-2.svg" alt="सॉफ्टवेयर संग्रह"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr.svg" alt="अन्य कार्य उद्धरण spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr-2-1.svg" alt="टॉक्सोप्लाज्मा गोंडी में चयापचय अनुकूलन और पोषक तत्वों के अधिग्रहण: अवशोषण मार्ग-अनावश्यक मुट्ठी से अंतर्दृष्टि।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr-2-2.svg" alt="IRE1α मैक्रोफैग fungicidal गतिविधि को बढ़ावा देने के लिए phagosomal कैल्शियम प्रवाह को बढ़ाता है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr-2-3.svg" alt="टॉक्सोप्लाज्मा जीआरए 8 होस्ट एससीआरटी सामान प्रोटीन एलजी-2 को शामिल करता है और परजीवी के मेटाबोलिक अखंडता के लिए आवश्यक है।"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/hi/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/hi/other-work-citing-spacr-2-5.svg" alt="spaCR: CRISPR-Cas9 स्क्रीनिंग (प्रिंट संस्करण 1) के अंतरिक्ष प्रजनन विश्लेषण।"></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/hi/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/hi/acknowledgments.desktop.svg"><img src="../../readme_images/hi/acknowledgments.svg" alt="आभार. spaCR NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch और Qt सहित मुक्त वैज्ञानिक सॉफ़्टवेयर पर आधारित है। बहुभाषी दस्तावेज़ और इंटरफ़ेस कैटलॉग तैयार करने में उपयोग किए गए मॉडल के लिए अनुवाद मॉडल श्रेय देखें।"></picture></a></div>
