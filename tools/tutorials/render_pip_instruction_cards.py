#!/usr/bin/env python3
"""Reuse the established installation-card style, explicitly as instructions.

These code-rendered cards are not screenshots or fabricated terminal output.
The accompanying verification and GUI frames come from the real installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--capture-name', default='pip_installed_visible_window')
    args = parser.parse_args()
    capture = args.stage / 'captures' / args.capture_name
    provenance = read(capture / 'provenance.json')
    if not provenance.get('completed_capture'):
        raise ValueError('Real installation verification must finish first')
    style_path = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/tools/render_install_keyframes.py')
    spec = importlib.util.spec_from_file_location('existing_installation_style', style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    cards = [
        ('07_create_environment', 'Create an isolated environment', [
            ('Linux / macOS — Python 3.12 must already be installed', 'python3.12 -m venv .venv'),
            ('Windows — with the Python launcher', 'py -3.12 -m venv .venv')],
         'Choose a new environment folder. Do not overwrite an existing analysis environment.'),
        ('08_activate_environment', 'Activate the environment in your shell', [
            ('Linux / macOS — bash or zsh', 'source .venv/bin/activate'),
            ('Windows — PowerShell', '.\\.venv\\Scripts\\Activate.ps1'),
            ('Windows — Command Prompt', '.venv\\Scripts\\activate.bat')],
         'PowerShell policy may block activation; use Command Prompt or the environment Python directly.'),
        ('09_install_package', 'Install the public Python package', [
            ('Inside the selected environment', 'python -m pip install --upgrade pip'),
            ('Reproduce the release verified in this lesson', 'python -m pip install "spacr==1.5.0.5"')],
         'Current base dependencies include Qt. Scientific packages can require several GB and minutes.'),
        ('10_launch_commands', 'Launch the installed desktop application', [
            ('Normal launch — first-run questions may appear', 'spacr'),
            ('Explicit setup skip used only for this bounded verification', 'spacr --no-setup')],
         'Use the same activated environment. Home and module tutorials cover the application workflows.'),
        ('11_update_intentionally', 'Update deliberately, then verify again', [
            ('Close spaCR and activate the environment you mean to update', 'python -m pip install --upgrade spacr'),
            ('Check dependencies after the update', 'python -m pip check'),
            ('Check the installation and your hardware', 'spacr-doctor')],
         'Record the installed version with the analysis. An upgrade can change dependencies and results.'),
    ]
    frames = read(capture / 'frames.json')
    for name, title, rows, note in cards:
        target = capture / (name + '.png')
        if target.exists():
            raise FileExistsError('Instruction cards are not overwritten; use a new capture')
        image, draw = style.canvas(title, 'Command reference — not recorded terminal output')
        draw.rounded_rectangle(style.box((150, 190, 1770, 925)), radius=style.scaled(19),
                               fill=style.PANEL, outline=style.PANEL_LINE, width=2)
        for index, (label, command) in enumerate(rows):
            top = 235 + index * 168
            style.draw_text(draw, (200, top), label, style.SMALL, style.MUTED)
            style.draw_text(draw, (200, top + 53), command, style.COMMAND, style.TEXT)
        # Reject clipped instructions instead of silently squeezing their text.
        for label, command in rows:
            if draw.textlength(command, font=style.COMMAND) > style.scaled(1500):
                raise ValueError('A command does not fit the existing card style')
            if draw.textlength(label, font=style.SMALL) > style.scaled(1500):
                raise ValueError('An instruction label does not fit the card')
        if draw.textlength(note, font=style.SMALL) > style.scaled(1430):
            raise ValueError('The instruction note does not fit the card')
        style.note(draw, note)
        image.save(target)
        frames[name] = dict(image=target.name,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            buttons=[], kind='generated_command_reference_not_terminal_output')
    write(capture / 'frames.json', frames)
    provenance['generated_instruction_cards'] = [card[0] for card in cards]
    provenance['instruction_style_source'] = str(style_path)
    write(capture / 'provenance.json', provenance)
    print('Five explicit command-reference cards added; original native screenshots unchanged.')


if __name__ == '__main__':
    main()
