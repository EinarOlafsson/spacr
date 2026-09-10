#!/usr/bin/env python3
"""Reuse the existing card style; distinguish platform guidance from capture."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil

from stage_lesson import DEFAULT_STAGE, REPO, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-name', required=True)
    args = parser.parse_args()
    if Path(args.capture_name).name != args.capture_name or args.capture_name in {'.', '..'}:
        parser.error('Choose one private capture directory name')
    capture = DEFAULT_STAGE / 'captures' / args.capture_name
    provenance = read(capture / 'provenance.json')
    if not provenance.get('completed_capture') or provenance['installed_identity']['version'] != '1.5.0.5':
        raise ValueError('The real public-installer verification must finish first')
    style_path = DEFAULT_STAGE.parent / 'tools/render_install_keyframes.py'
    spec = importlib.util.spec_from_file_location('existing_installer_style', style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    cards = [
        ('09_windows_guidance', 'Windows: select the online setup executable', [
            ('Official release asset — source-reviewed guidance, not a Windows recording',
             'spaCR-1.5.0.5-Windows-Online-Setup.exe'),
            ('Current installer source enables automatic acceleration by default',
             'Untick acceleration only when you want the CPU-only option')],
         'A private Python runtime is downloaded. Windows installation was not executed on this Linux host.'),
        ('10_macos_guidance', 'macOS: the universal online package', [
            ('Official package for Intel and Apple silicon — not a macOS recording',
             'spaCR-1.5.0.5-macOS-Universal-Online.pkg'),
            ('The app launcher starts per-user setup if its runtime is not yet installed',
             'Open spaCR from Applications; allow time for downloads')],
         'Verify the official source before approving OS security prompts. Do not disable system protections.'),
        ('11_linux_commands', 'Linux x86-64: run the verified download', [
            ('From the folder containing the official downloaded installer',
             'chmod +x spaCR-1.5.0.5-Linux-x86_64-Online.run'),
            ('Normal launch uses automatic accelerator selection',
             './spaCR-1.5.0.5-Linux-x86_64-Online.run'),
            ('An explicit CPU-only choice is available',
             './spaCR-1.5.0.5-Linux-x86_64-Online.run --torch-backend cpu')],
         'Our isolated run supplied private paths, --skip-system-deps and --no-launch. These are instructions.'),
        ('12_logs_and_versions', 'Keep installation evidence with the analysis', [
            ('Inside the chosen private installation root', 'install.log'),
            ('Recorded backend and optional choices', 'install-profile.json'),
            ('Diagnostic command from the installed runtime', 'spacr-doctor')],
         'Review logs for private paths or data before sharing. Keep the version used for an ongoing analysis.'),
    ]
    frames = read(capture / 'frames.json')
    source = DEFAULT_STAGE / 'captures/installation_sources_centred_elements'
    name = '04_github_current_assets'
    original = read(source / 'frames.json')[name]
    incoming = source / original['image']
    if hashlib.sha256(incoming.read_bytes()).hexdigest() != original['sha256']:
        raise ValueError('Previously recorded official release screenshot changed')
    target = capture / '08_official_release.png'
    if target.exists():
        raise FileExistsError('Earlier release frames are retained')
    shutil.copyfile(incoming, target)
    frames['08_official_release'] = dict(original, image=target.name,
        reused_native_capture=str(incoming))
    for name, title, rows, note in cards:
        target = capture / (name + '.png')
        if target.exists():
            raise FileExistsError('Earlier cards are retained')
        image, draw = style.canvas(title, 'Reference guidance — not recorded platform installation output')
        draw.rounded_rectangle(style.box((150, 190, 1770, 925)), radius=style.scaled(19),
                               fill=style.PANEL, outline=style.PANEL_LINE, width=2)
        for index, (label, command) in enumerate(rows):
            if (draw.textlength(command, font=style.COMMAND) > style.scaled(1500)
                    or draw.textlength(label, font=style.SMALL) > style.scaled(1500)):
                raise ValueError('Installer instruction would be clipped')
            top = 235 + index * 168
            style.draw_text(draw, (200, top), label, style.SMALL, style.MUTED)
            style.draw_text(draw, (200, top + 53), command, style.COMMAND, style.TEXT)
        if draw.textlength(note, font=style.SMALL) > style.scaled(1430):
            raise ValueError('Installer note would be clipped')
        style.note(draw, note)
        image.save(target)
        frames[name] = dict(image=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            buttons=[], kind='generated_reference_not_platform_recording')
    write(capture / 'frames.json', frames)
    provenance['generated_instruction_cards'] = [row[0] for row in cards]
    provenance['reused_official_release_capture'] = dict(path=str(incoming), sha256=original['sha256'])
    provenance['platform_guidance_source_files'] = [dict(path=name,
        sha256=hashlib.sha256((REPO / name).read_bytes()).hexdigest()) for name in [
            'packaging/online/spacr_online_installer.nsi',
            'packaging/online/build_macos_online.sh',
            'packaging/online/install_spacr_unix.sh']]
    write(capture / 'provenance.json', provenance)
    print('Four reference cards and one reused genuine release-page screenshot; native frames unchanged.')


if __name__ == '__main__':
    main()
