#!/usr/bin/env python3
"""Render Docker instructions with the existing 4K installation card style."""
from pathlib import Path
import argparse
import hashlib
import re
import sys
from stage_lesson import REPO, write

sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
import render_install_keyframes as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--version', required=True, help='Exact available spaCR container image tag')
    a = parser.parse_args()
    a.output = a.destination
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', a.version):
        parser.error('Expected an exact published version')
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output
    image,draw=r.canvas('Install Docker for spaCR','Official setup links  •  Linux x86-64 containers')
    r.panel(draw,label='Setup links and image requirements')
    for y,title,url in [(330,'Docker Desktop','https://docs.docker.com/get-started/get-docker/'),(455,'Docker Engine on Linux','https://docs.docker.com/engine/install/'),(580,'Available spaCR versions','https://github.com/EinarOlafsson/spacr/pkgs/container/spacr')]:
     r.draw_text(draw,(215,y),title,r.BODY,r.TEXT)
     r.draw_text(draw,(215,y+42),url,r.SMALL,r.CYAN)
    r.note(draw,'CPU: no GPU required. Desktop GUI: use the spaCR desktop installer.')
    image.save(out/'docker_01_install.png')
    scenes=[{'visual':'docker_01_install','image':'keyframes/docker_01_install.png','focus':None}]
    base=f'ghcr.io/einarolafsson/spacr:{a.version}'
    cards=[('docker_02_image_check','Check an exact spaCR image','Commands to run  •  compare the reported version with the image tag',[
     f'IMAGE={base}','docker pull "$IMAGE"','docker run --rm "$IMAGE" spacr --version','docker run --rm "$IMAGE" spacr-run --list'],3,'Use a version listed on the official container package page.'),
    ('docker_03_mount_and_run','Run a pipeline with mounted data','Linux Bash commands  •  mount images and exported settings',[
     'mkdir -p screen "$HOME/.cellpose/models"','docker run --rm --user "$(id -u):$(id -g)" \\',
     '  -v "$PWD/screen:/data" \\','  -v "$HOME/.cellpose/models:/models" \\',
     '  "$IMAGE" spacr-run measure \\','  --settings /data/settings/measure_settings.csv'],5,'Settings use /data paths. Save outputs under /data to keep them.'),
    ('docker_04_gpu','Optional NVIDIA GPU processing','CUDA 12.4  •  NVIDIA host driver 550+ and configured Container Toolkit',[
     f'GPU_IMAGE={base}-cuda12.4','docker run --rm --gpus all \\',
     '  --user "$(id -u):$(id -g)" \\',
     '  -v "$PWD/screen:/data" -v "$HOME/.cellpose/models:/models" \\',
     '  "$GPU_IMAGE" spacr-run mask \\',
     '  --settings /data/settings/gen_mask_settings.csv'],0,'Prepare the host driver and Container Toolkit using their official guides.')]

    for visual,title,subtitle,commands,active,note in cards:
     image,draw=r.canvas(title,subtitle)
     r.panel(draw,label='Documented commands to run')
     for index,command in enumerate(commands):
      if r.COMMAND.getlength(command)>r.scaled(1454):raise ValueError('Command exceeds normal card text width')
      top=315+index*78
      if index==active:
       draw.rounded_rectangle(r.box((195,top-3,1722,top+67)),radius=r.scaled(7),fill='#101c2a')
      prompt='>' if command.startswith('  ') else '$'
      r.draw_text(draw,(218,top+32),prompt,r.COMMAND,r.GREEN,anchor='lm')
      r.draw_text(draw,(268,top+32),command,r.COMMAND,r.TEXT,anchor='lm')
     r.note(draw,note)
     image.save(out/(visual+'.png'))
     focus=r.focus_box((195,315+active*78-3,1527,70))
     scenes.append({'visual':visual,'image':f'keyframes/{visual}.png','focus':focus})
    frames = {row['visual']: {'image': Path(row['image']).name,
        'sha256': hashlib.sha256((out / Path(row['image']).name).read_bytes()).hexdigest(),
        'buttons': [], 'kind': 'generated_command_reference_not_terminal_output'} for row in scenes}
    write(out / 'frames.json', frames)
    write(out / 'provenance.json', {'completed_capture': True,
        'scope': 'Docker setup and pipeline instructions; no container execution claim',
        'image_version': a.version, 'published': False,
        'sources': ['https://docs.docker.com/get-started/get-docker/',
            'https://docs.docker.com/engine/install/',
            'https://github.com/EinarOlafsson/spacr/pkgs/container/spacr']})
    print('Prepared4 instructional Docker cards; no Docker commands executed:',out,flush=True)


if __name__ == '__main__':
    main()
