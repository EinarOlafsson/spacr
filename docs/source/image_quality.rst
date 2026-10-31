Image quality before segmentation
=================================

From **Home → Core → Mask**, open the **Image Quality** settings category
before running segmentation. The same settings apply when Mask's **Timelapse**
switch is on. Screening measures the raw,
unnormalized image channels. Start in ``report`` mode to inspect acquisition
problems before deciding which fields to exclude.

The policy
----------

* ``image_qc_mode``: ``off`` (default), ``report`` (flag fields but process
  them), or ``exclude`` (skip flagged fields downstream).
* ``image_qc_channels``: acquisition-channel identifiers to screen; an empty
  list screens every stored raw channel. V1 uses zero-based array channels;
  V2 uses mapped acquisition-channel identifiers. Threshold dictionaries use
  the same identifiers.
* ``image_qc_min_focus``: channel-to-threshold mapping for minimum Laplacian
  variance. A stack uses its best plane's score; an out-of-focus plane alone
  does not reject an otherwise usable stack. Choose thresholds from controls
  acquired with the same intensity scale and optics.
* ``image_qc_max_saturation``: channel-to-threshold mapping for the maximum
  fraction of saturated pixels, from zero to one.
* ``image_qc_saturation_level``: channel-to-intensity mapping for the
  acquisition saturation ceiling. For example, a 12-bit acquisition stored
  in ``uint16`` normally needs its acquisition ceiling rather than the
  ``uint16`` ceiling. Integer data otherwise use the dtype maximum; floating
  data require an explicit level when a saturation threshold is enabled.
* ``image_qc_max_nonfinite``: maximum fraction of nonfinite pixels, from zero
  to one; defaults to zero.

Focus thresholds use raw intensity units squared. These are acquisition
checks: low object counts do not cause exclusion. The brightest observed
pixel is never substituted for a calibrated saturation level.

Optional classifier
-------------------

``image_qc_classifier`` is off by default. When enabled with screening in
``report`` or ``exclude`` mode, it adds per-channel flags for out-of-focus,
saturated, debris, bubble and empty images. It runs on the CPU and uses a
pinned, checksum-verified checkpoint trained on synthetic planted-defect
fields; it does not train the built-in model during screening. Missing weights
are acquired through the model cache when downloads are allowed. With
``SPACR_MODEL_DOWNLOAD=off``, provide verified cached weights or set
``image_qc_classifier_model`` to a compatible saved classifier. Synthetic
training and checksum verification do not establish accuracy on your images;
inspect representative controls in ``report`` mode before excluding fields.

``image_qc_classifier_threshold`` defaults to 0.5; a class score at or above
it adds a flag. The built-in model's scores are shifted per class so that 0.5
corresponds to its stored per-class cut-off; a fine-tuned model's scores are
used unchanged. ``image_qc_classifier_labels`` accepts a table with ``field``
and ``label`` columns, plus an optional ``channel`` column. Use ``good`` or
one or more of ``out_of_focus``, ``saturated``, ``debris``, ``bubble`` and
``empty``, separated by semicolons. At least four matched raw fields are
required to fine-tune; ten or more also produce cross-validated precision and
recall in ``qc/image_qc_benchmark.csv``. The tuned classifier is saved as
``qc/image_qc_model.pt``. These results describe the supplied labels and
fields, not independent validation on other acquisitions.

Review and outputs
------------------

Open **Home → Data → QC** and choose **Review image quality** for a local
thumbnail gallery.
Flagged fields appear first, with at most 64 previews. Display contrast is
stretched for inspection; the metrics still use raw intensities. Stack
thumbnails are maximum projections.

The project's ``qc`` directory contains:

* ``image_quality.csv``: every screened field and channel, its focus variance,
  saturation level/fraction, nonfinite fraction, status and reasons;
* ``image_quality.json``: the exact policy and excluded field identities;
* ``image_quality.html``: the review gallery.

Exclusion retains the original inputs. Segmentation, merge and Measure respect
the saved exclusions; excluded fields are not represented as successful
zero-object detections.

Revisiting an analyzed project
--------------------------------

New exclusions are refused if the affected fields already have retained
measurement rows. Use ``report`` to inspect that project without changing its
inclusion policy, or run the exclusion policy in a fresh project. Existing
measurements and the previous policy are retained when this check refuses a
change. Turning screening ``off`` clears the active exclusion policy.

The Python entry points are :func:`spacr.image_quality.quality_policy`,
:func:`spacr.image_quality.assess_image` and
:func:`spacr.image_quality.screen_fields`.
