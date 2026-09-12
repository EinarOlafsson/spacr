"""Promote the source-pinned Embeddings translations, never other lessons."""
from pathlib import Path

from apply_translation_review import promote, SPOKEN, CAPTION_ONLY
from stage_lesson import DEFAULT_STAGE, read


def main():
    root = Path(__file__).resolve().parent
    reviews = [read(path) for path in sorted((root / 'lessons/reviews').glob('77_embeddings.*.json'))]
    if {review['language'] for review in reviews} != SPOKEN | CAPTION_ONLY or len(reviews) != 13:
        raise ValueError('Require all thirteen retained non-English languages')
    for review in reviews:
        if review['lesson'] != '77_embeddings':
            raise ValueError('This promoter may only update Embeddings')
        promote(review, DEFAULT_STAGE)


if __name__ == '__main__':
    main()
