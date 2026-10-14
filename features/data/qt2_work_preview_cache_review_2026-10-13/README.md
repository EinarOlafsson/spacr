# Qt2 preview-cache source reviews

Home reports cached/uncached repaint ratio 0.7524828279 against the unchanged inclusive 0.75 bound on CI run 38488240863. Home owns the isolated investigation and repair. These two source reviews and delivery audit do not establish the failure cause, a cache defect, regression, flakiness or successful optimization.

The benchmark measures minimum GUI-thread CPU from 12 warmed synchronous repaints in each of nine rounds per cache mode, then compares separate medians. Common backdrop/Qt work is included. Fixed 300+300+50 ms setup waits differ from the 3000 ms settling used in another heartbeat test. Cache configuration does not measure cache hits or pixel correctness.

The live-preview source selects DeviceCoordinateCache, replaces items for new pictures and delegates transform regeneration to Qt. No missing cache setup or stale-picture defect was established. Runtime cache behavior and performance remain unobserved by Work. Conditional diagnostic questions are retained for Home; no threshold/source/test, GUI, GPU or CI action occurred.

Root checked the three artifact hashes, six current source hashes, their exact eb72 Git blobs and the original Home observation blob. Official Qt documentation links in the source review describe general API behavior, not the runner's actual behavior.
