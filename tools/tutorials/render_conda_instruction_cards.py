#!/usr/bin/env python3
"""Add explicit Conda command-reference cards using the existing visual style."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, read, write


def main():
    capture = DEFAULT_STAGE / 'captures/conda_release_native'
    provenance = read(capture / 'provenance.json')
    if not provenance.get('completed_capture') or provenance['installed_identity']['version'] != '1.5.0.4':
        raise ValueError('These cards describe the verified channel version, not a newer checkout')
    path = DEFAULT_STAGE.parent / 'tools/render_install_keyframes.py'
    spec = importlib.util.spec_from_file_location('existing_conda_style', path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    cards = [
        ('07_create_conda', 'Create a new Conda environment', [
            ('Open a Conda-enabled shell; choose an unused environment name',
             'conda create -n spacr-conda -c conda-forge python=3.12')],
         'A Conda-compatible manager must already be installed. Do not overwrite an existing environment.'),
        ('08_activate_conda', 'Activate and check the selected environment', [
            ('Activate the new environment', 'conda activate spacr-conda'),
            ('Confirm where its interpreter belongs', 'python -c "import sys; print(sys.prefix)"')],
         'The recorded verification uses a longer private prefix. Its environment is genuinely separate.'),
        ('09_install_conda', 'Select conda-forge explicitly', [
            ('Run inside the environment you selected', 'conda install conda-forge::spacr'),
            ('Check the channel and installed version', 'conda list spacr')],
         'The recorded solve used only conda-forge and pinned 1.5.0.4, the channel version on 2026-09-10.'),
        ('10_current_release_choice', 'Choose the release you need', [
            ('Conda channel at this recording', 'spaCR 1.5.0.4 — older Home layout'),
            ('Current PyPI and desktop installer at this recording', 'spaCR 1.5.0.5 — new nested module layout')],
         'See the pip or Platform installers lesson for the newer release. Recheck channel versions later.'),
    ]
    frames = read(capture / 'frames.json')
    for name, title, rows, note in cards:
        target = capture / (name + '.png')
        if target.exists():
            raise FileExistsError('Earlier cards are retained; do not overwrite them')
        image, draw = style.canvas(title, 'Command reference — not recorded terminal output')
        draw.rounded_rectangle(style.box((150, 190, 1770, 925)), radius=style.scaled(19),
                               fill=style.PANEL, outline=style.PANEL_LINE, width=2)
        for index, (label, command) in enumerate(rows):
            if draw.textlength(command, font=style.COMMAND) > style.scaled(1500):
                raise ValueError('Conda command would be clipped')
            top = 255 + index * 200
            style.draw_text(draw, (200, top), label, style.SMALL, style.MUTED)
            style.draw_text(draw, (200, top + 55), command, style.COMMAND, style.TEXT)
        if draw.textlength(note, font=style.SMALL) > style.scaled(1430):
            raise ValueError('Conda note would be clipped')
        style.note(draw, note)
        image.save(target)
        frames[name] = dict(image=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            buttons=[], kind='generated_command_reference_not_terminal_output')
    write(capture / 'frames.json', frames)
    provenance['generated_instruction_cards'] = [row[0] for row in cards]
    write(capture / 'provenance.json', provenance)
    print('Four explicit Conda reference cards created; native captures unchanged.')


if __name__ == '__main__':
    main()
