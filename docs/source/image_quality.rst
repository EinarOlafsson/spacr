Image quality before segmentation
=================================

From **Home → Mask**, open **Image Quality** before running segmentation.
The same settings are available in Timelapse. Screening measures the raw,
unnormalized image channels. Start in ``report`` mode to inspect acquisition
problems before deciding which fields to exclude.

The policy
----------

* ``image_qc_mode``: ``off`` (default), ``report`` (flag fields but process
  them), or ``exclude`` (skip flagged fields downstream).
* ``image_qc_channels``: acquisition-channel indices to screen; an empty
  list screens every available channel.
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

Review and outputs
------------------

Open **QC Dashboard → Review image quality** for a local thumbnail gallery.
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
