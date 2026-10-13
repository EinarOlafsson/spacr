"""Render explicit command references for the merged installation lesson.

These are instructions, not terminal output or executed Windows/macOS recordings.
The verified installation identity is retained separately from these cards.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
from pathlib import Path
from stage_lesson import REPO, read, write


def cards():
    """Three alternative installation routes and their separate removal commands."""
    return [
        ('00_install_routes', 'Install spaCR: choose one route', [
            ('Python virtual environment', 'Python 3.12 → .venv → pip install spacr'),
            ('Conda environment with the PyPI package', 'Conda → spacr-pip → pip install spacr'),
            ('Conda environment with the conda-forge package', 'Conda → spacr-conda → conda-forge')],
         'Choose one route. Conda-forge packages cannot be installed into a Python virtual environment.'),
        ('pip_03_install', 'Install with pip in the active environment', [
            ('Upgrade pip first', 'pip install --upgrade pip'),
            ('Install the published spaCR package', 'pip install spacr')],
         'The package includes Qt desktop dependencies. Wait for the scientific and desktop packages to finish.'),
        ('pip_04_check', 'Verify the installed pip package', [
            ('Print the version and package path', 'python -c "import spacr; print(spacr.__version__); print(spacr.__file__)"'),
            ('Check installed dependency requirements', 'python -m pip check')],
         'Keep the installed version with your analysis notes. A nightly source checkout may have newer interface features.'),
        ('pip_06_update', 'Update a pip installation', [
            ('Close spaCR and activate its environment', 'pip install --upgrade spacr'),
            ('Check dependencies after the update', 'python -m pip check')],
         'Use pip for both pip routes, including spaCR installed with pip inside a Conda environment.'),
        ('pip_07_uninstall', 'Remove the pip-installed spaCR package', [
            ('Close spaCR and activate the environment', 'python -m pip uninstall spacr'),
            ('Review and accept pip’s removal prompt', 'This removes spaCR, not every dependency or your projects')],
         'Command reference. Package removal and full environment removal are separate operations.'),
        ('pip_08_remove_environment', 'Remove only your dedicated tutorial environment', [
            ('Linux / macOS: deactivate, then remove the dedicated folder', 'deactivate; rm -r .venv'),
            ('Windows PowerShell: deactivate, then remove the dedicated folder', 'deactivate; Remove-Item -Recurse -Force .venv'),
            ('Conda environment created only for pip-installed spaCR', 'conda deactivate; conda env remove -n spacr-pip')],
         'Confirm the folder or environment name first. Keep microscopy projects and analysis results elsewhere.'),
        ('conda_01_create', 'Install from conda-forge in a new Conda environment', [
            ('Solve Python and spaCR together', 'conda create -n spacr-conda -c conda-forge python=3.12 spacr'),
            ('Use a different name if this environment already exists', 'Review Conda’s package plan before accepting it')],
         'This is a Conda environment, not .venv. The channel release can differ from PyPI and nightly.'),
        ('conda_02_plan', 'Review the conda-forge package plan', [
            ('Conda lists the selected package versions and downloads', 'Review the plan, then accept Conda’s own prompt'),
            ('Wait for download and installation completion', 'A solved plan alone is not a completed installation')],
         'Instruction reference, not a reconstructed package plan or terminal output.'),
        ('conda_03_activate', 'Activate and check the conda-forge installation', [
            ('Activate the environment you created', 'conda activate spacr-conda'),
            ('Check the installed version and channel', 'conda list spacr'),
            ('Check dependencies, then launch from this same environment', 'spacr-doctor; spacr')],
         'Follow diagnostic guidance relevant to your work. Close spaCR before updating or removing its environment.'),
        ('conda_04_update', 'Update from conda-forge', [
            ('Close spaCR and activate spacr-conda', 'conda update -c conda-forge spacr'),
            ('Confirm the available installed version', 'conda list spacr')],
         'An update may report that the requested packages are already installed. Check the actual version.'),
        ('conda_05_uninstall', 'Remove the Conda package or its dedicated environment', [
            ('Package removal: activate spacr-conda, then review the removal plan', 'conda remove spacr'),
            ('Alternative: remove the whole environment created for this tutorial', 'conda deactivate; conda env remove -n spacr-conda')],
         'Choose one removal scope. Keep microscopy projects and analysis results separately.'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    proof = read(args.verification / 'provenance.json')
    if (not proof.get('completed_capture') or not proof.get('installed_identity', {}).get('version')
            or args.destination.exists()):
        raise ValueError('Use completed public installation evidence and a new card destination')
    style_path = REPO / 'tools/tutorials/authoring/tools/render_install_keyframes.py'
    spec = importlib.util.spec_from_file_location('installation_card_style', style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    prepared = []
    for name, title, rows, note in cards():
        image, draw = style.canvas(title, 'Command reference — not recorded terminal output')
        draw.rounded_rectangle(style.box((150, 190, 1770, 925)), radius=style.scaled(19),
                               fill=style.PANEL, outline=style.PANEL_LINE, width=2)
        for index, (label, command) in enumerate(rows):
            if (draw.textlength(command, font=style.COMMAND) > style.scaled(1500)
                    or draw.textlength(label, font=style.SMALL) > style.scaled(1500)):
                raise ValueError(f'Instruction would be clipped: {name}')
            top = 235 + index * 168
            style.draw_text(draw, (200, top), label, style.SMALL, style.MUTED)
            style.draw_text(draw, (200, top + 53), command, style.COMMAND, style.TEXT)
        if draw.textlength(note, font=style.SMALL) > style.scaled(1430):
            raise ValueError(f'Instruction note would be clipped: {name}')
        style.note(draw, note)
        prepared.append((name, image))
    args.destination.mkdir(parents=True)
    frames = {}
    for name, image in prepared:
        output = args.destination / (name + '.png')
        image.save(output)
        frames[name] = {'image': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                        'buttons': [], 'kind': 'generated_command_reference_not_terminal_output'}
    write(args.destination / 'frames.json', frames)
    write(args.destination / 'provenance.json', {'completed_capture': True,
        'scope': 'Generated command references; no platform installation or removal execution claim',
        'verification_capture': str(args.verification.resolve()),
        'verification_sha256': hashlib.sha256((args.verification / 'provenance.json').read_bytes()).hexdigest(),
        'verified_public_installation': proof['installed_identity'],
        'style_source_sha256': hashlib.sha256(style_path.read_bytes()).hexdigest(),
        'author_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'published': False})
    print(f'{len(frames)} explicit command references; no reconstructed program output')


if __name__ == '__main__':
    main()
