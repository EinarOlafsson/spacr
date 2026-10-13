"""Compare a trained Cellpose checkpoint with the weights it started from (CPU only).

Reports how many parameter tensors changed and the largest absolute change.
A changed tensor shows the run updated the model; it says nothing about accuracy.
"""
import argparse
import json
from pathlib import Path

import torch


def compare(base, trained):
    before = torch.load(base, map_location='cpu', weights_only=True)
    after = torch.load(trained, map_location='cpu', weights_only=True)
    shared = sorted(set(before) & set(after))
    changed, largest = [], 0.0
    for name in shared:
        a, b = before[name], after[name]
        if not torch.is_floating_point(a) or a.shape != b.shape:
            continue
        delta = (b.float() - a.float()).abs().max().item()
        if delta > 0:
            changed.append(name)
            largest = max(largest, delta)
    return dict(base=str(base), trained=str(trained), tensors_compared=len(shared),
                tensors_changed=len(changed), largest_absolute_change=largest,
                only_in_base=len(set(before) - set(after)), only_in_trained=len(set(after) - set(before)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('trained', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = compare(args.base, args.trained)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
