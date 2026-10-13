"""Explain whole-catalog caption failures before launching a browser."""


def validate_caption_structure(english, captions, language):
    """Match the player's all-lesson caption contract, not only the target."""
    translated = {lesson['id']: lesson for lesson in captions['lessons']}
    for source in english['lessons']:
        identity = source['id']
        lesson = translated.get(identity)
        if lesson is None:
            raise ValueError(f'{language} captions lack {identity}')
        scenes = lesson.get('scenes')
        if not isinstance(scenes, list) or len(scenes) != len(source['scenes']):
            raise ValueError(f'{language} caption scene count differs for {identity}')
        if not all(isinstance(scene.get('narration'), str) and scene['narration'].strip() for scene in scenes):
            raise ValueError(f'{language} caption text is empty or invalid for {identity}')
