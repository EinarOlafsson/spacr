# Item 545 — actual QuPath GUI acceptance

PASSED on source 5014fb90f71302246782e099aad0159a04294be7. spaCR source and shared fixture hashes still match the fixture receipt. No repository changes, commits, global installations, user-display access or GPU work occurred.

The official stable QuPath v0.7.0 Linux artifact is 256,194,072 bytes, below the authorized 300MiB limit. Its SHA-256 matched the digest in the official GitHub release API: 165e27a0731d58ba039e9d0d34a54acf896bb92e81922370df295cb85418ce53. Retained official metadata: official-release.json. Release: https://github.com/qupath/qupath/releases/tag/v0.7.0

The normal packaged GUI launcher ran in a private Xvfb display with JavaFX software rendering, private HOME/preferences/temp/cache, hidden CUDA/HIP/ROCR, 2G Java heap and observed cgroup MemoryMax=4G/MemorySwapMax=0. It opened the real spaCR-exported field.tif, then used InteractiveObjectImporter.promptToImportObjectsFromFile to import spacr-field.geojson into its visible viewer. It did not substitute a headless parser for GUI acceptance. The official launcher and GUI startup-script mechanism are documented/implemented at:

- https://qupath.readthedocs.io/en/latest/docs/advanced/command_line.html
- https://github.com/qupath/qupath/blob/v0.7.0/qupath-gui-fx/src/main/java/qupath/lib/gui/QuPathGUI.java
- https://github.com/qupath/qupath/blob/v0.7.0/qupath-gui-fx/src/main/java/qupath/lib/gui/commands/InteractiveObjectImporter.java

All eight native annotations matched expected names, object IDs, classes, UUIDs, bounds and areas. Native JTS geometry was valid, and all 49,152 per-object pixel-centre comparisons matched the original cell/nucleus masks exactly (zero differences). This includes holes, multipart objects, touching pieces and label65535. QuPath saved imported.qpdata. The final screenshot visibly shows “Annotation list (8)”, all eight names/classes, the image and geometry overlays. Both the native scene screenshot and independently captured whole-Xvfb screenshot are retained and were visually inspected by the AI agent; no human signoff is claimed.

Final scope exited0 and left no QuPath process. verification.json and gui-receipt.json retain exact checks; evidence.sha256 binds inputs, scripts and screenshots. Initial harness failures are retained: missing optional groovy-json imports, then Groovy map-property lookup ambiguity. The final harness uses bundled Gson and explicit map.get access. A first passing capture had a clipped display; its receipt remains in attempt4-passed-cropped-display, followed by the clearer final capture. Neither application nor export code was modified to pass.

Proposed closure evidence: together with the separate successful native Fiji GUI receipt and existing exact spaCR GeoJSON/RoiSet/COCO round-trips, this satisfies item545's external-application opening criterion for these tested installed versions. Alpha status and the existing Make Masks save behavior remain as documented.
