"""Record a bounded native CV run with the explicit prepared-split workaround."""
from capture_barcode_saved_plots import launch
from stage_lesson import DEFAULT_STAGE


if __name__ == '__main__':
    raise SystemExit(launch('classify_merged', 'classify_canonical_existing',
                           ['--run', '--settings-tour', '--ai-controls',
                            '--classifier-existing-split', str(DEFAULT_STAGE / 'classify_canonical_split')],
                           300, stage=DEFAULT_STAGE / 'classify_canonical_capture'))
