Train/test leakage audit
========================

Classifier accuracy is invalid when related crops occur on both sides of an
evaluation boundary. spaCR now audits the permanent ``train/``/``test/``
split, the ordinary train/validation holdout, every outer and inner CV
boundary, and the CV partition as a whole before fitting.

What is verified
----------------

The audit checks:

* the same path or crop identity never crosses a boundary;
* exported augmentations such as ``_rot90``, ``_flip_h`` and ``_aug3`` remain
  with their source object;
* byte-identical crops are detected by streaming SHA-256 even if renamed;
* the requested ``cv_group_by`` identity (field, well or plate) stays intact;
* every CV sample is held out exactly once;
* no related family is assigned to more than one held-out fold; and
* related crops do not carry conflicting class labels.

Filename identities follow spaCR's
``<plate>_<well>_<field>_..._<object>.png`` crop convention. With
``leakage_require_identity=True`` (the default), a filename that cannot prove
its requested identity fails the audit; an unknown relationship is not
reported as independent.

Ordinary validation is now group-aware too. ``cv_group_by='well'`` uses the
same group-stratified partitioner as CV and selects the candidate fold closest
to ``val_split`` and the full dataset's class distribution. Augmentation is
applied only after the split and only to training.

Run the audit directly
----------------------

.. code-block:: bash

   spacr-leakage /data/classifier_dataset
   spacr-leakage /data/classifier_dataset --group-by plate \
       --output /tmp/leakage.json

The command prints JSON and exits ``0`` for a verified split, ``1`` when
leakage or unverifiable identities are found, and ``2`` when the dataset
cannot be audited. ``--no-content-hash`` reduces I/O but cannot detect renamed
copies. ``--allow-unverifiable`` is intended for diagnosing legacy datasets;
it does not make their performance estimate trustworthy.

Classify settings
-----------------

``leakage_audit_train_test``
   Audit ``train/`` against ``test/`` before fitting. Default ``True``.

``leakage_hash_content``
   Stream SHA-256 for copy/rename detection. Default ``True``.

``leakage_require_identity``
   Fail when protected identity or content cannot be verified. Default
   ``True``.

``evaluation_fail_on_leakage``
   Stop before fitting when an audit fails. Default ``True``. Setting it to
   ``False`` records a failed audit but cannot make the resulting metric valid.

Reports are written beside the classifier run as
``train_test_leakage_audit.json`` and
``train_validation_leakage_audit.json``. CV audit records are also included
in the Classifier Evaluation workbench's ``leakage.json``.

API
---

.. automodule:: spacr.classifier_evaluation
   :members: LeakageReport, FoldLeakageAudit, audit_split_leakage, audit_cv_folds, audit_dataset_splits, write_leakage_audit
