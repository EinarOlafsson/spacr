"""Apply only explicit, English-pinned Map Barcodes editorial translations."""
import argparse
from pathlib import Path

from apply_translation_review import promote
from stage_lesson import read


def apply(stage, languages):
    bundle = read(Path(__file__).parent / 'lessons/reviews/12_map_barcodes.reviewed.json')
    for language in languages:
        record = dict(bundle['translations'][language], language=language,
                      lesson=bundle['lesson'], english_sha256=bundle['english_sha256'],
                      review=bundle['review'])
        promote(record, Path(stage))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--languages', nargs='+', required=True)
    args = parser.parse_args()
    apply(args.stage, args.languages)
