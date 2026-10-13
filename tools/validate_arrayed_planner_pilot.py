"""Check the arrayed-assay planner (item 585) on a real pilot plate.

Reads the control wells of the four TSG101 screen plates (spaCR measurement
databases; columns c21-c24, sixteen rows, every plate one biological
replicate of the same layout) and treats them as a pilot:

* three readouts per cell: ``cell_area`` (continuous), the number of
  parasites in the cell (count; the pathogen table's objects per cell) and
  whether the cell holds more than one parasite (proportion);
* the negative-control wells are the pilot proper: their nested variance
  components are estimated, the planner's cheapest design is found for an
  effect of about a tenth of the mean, and that design is checked two ways:
  a parametric simulation of the planner's own model
  (:func:`spacr.sp_stats._simulate_arrayed_power`, the DONE WHEN) and a
  hierarchical resampling of the real fields
  (:func:`spacr.sp_stats._resample_arrayed_power`) that assumes no model.
  Resampling a finite pilot is optimistic when wells and fields are drawn
  distinct and pessimistic when drawn with replacement, so both are run and
  the planned power should fall between them;
* the search stops at designs the pilot can hold without repeating a well
  or field (:data:`LIMITS`), so the distinct resampling is possible;
* both control conditions together estimate the replicate-by-condition
  term. c21-c22 are the negative control on plates 2-4 and the positive on
  plate 1, where the layout was swapped (settled when the screen's figure
  was rebuilt); c23-c24 the other way round.

Usage::

    python tools/validate_arrayed_planner_pilot.py \
        --root /media/carruthers/mnt3/claude/toxoplasma_projects/tsg101_screen \
        --out features/data/585_real_pilot_check_2026-09-30.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from spacr.sp_stats import (_arrayed_power, _nested_variance_components,
                            _plan_arrayed_design, _resample_arrayed_power,
                            _simulate_arrayed_power)

CONTROL_COLUMNS = ('c21', 'c22', 'c23', 'c24')
READOUTS = (('cell_area', 'continuous', 0.10),
            ('parasites', 'count', 0.10),
            ('multiple_parasites', 'proportion', 0.10))
N_SIM = 2000
#: About three Monte Carlo standard errors of a power near 0.8 at N_SIM.
MC_MARGIN = 0.03
#: Search limits the resampling can honour: the pilot must hold every
#: design's wells and fields without repeating one (each plate has at least
#: 23 negative-control wells with 10 or more fields, so 2 x 11 fit).
LIMITS = {'max_replicates': 24, 'max_wells': 11, 'max_fields': 10}


def load_controls(root: Path) -> pd.DataFrame:
    """Read every control cell of the four plates with its three readouts.

    :param root: the screen folder holding ``plate1`` .. ``plate4``.
    :returns: one row per cell with ``plateID``, ``prc``, ``fieldID``,
        ``condition`` (``nc`` or ``pc``) and the readouts.
    """
    frames = []
    columns = ", ".join(f"'{c}'" for c in CONTROL_COLUMNS)
    for plate in range(1, 5):
        path = root / f'plate{plate}' / 'measurements' / 'measurements.db'
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
            cells = pd.read_sql(
                'SELECT object_label, plateID, rowID, columnID, prcf, '
                f'cell_area FROM cell WHERE columnID IN ({columns})', con)
            parasites = pd.read_sql(
                'SELECT cell_id, prcf FROM pathogen '
                f'WHERE columnID IN ({columns}) AND cell_id > 0', con)
        counts = (parasites.groupby(['prcf', 'cell_id']).size()
                  .rename('parasites').reset_index())
        cells = cells.merge(counts, how='left',
                            left_on=['prcf', 'object_label'],
                            right_on=['prcf', 'cell_id'])
        cells['parasites'] = cells['parasites'].fillna(0.0)
        first_pair = cells['columnID'].isin(('c21', 'c22'))
        negative = ~first_pair if plate == 1 else first_pair
        cells['condition'] = np.where(negative, 'nc', 'pc')
        frames.append(cells)
    data = pd.concat(frames, ignore_index=True)
    data['prc'] = data['plateID'] + '_' + data['rowID'] + '_' + data['columnID']
    data['fieldID'] = data['prcf']
    data['multiple_parasites'] = (data['parasites'] >= 2).astype(float)
    return data


def _finite(values: dict) -> dict:
    """Replace NaN with None so the receipt is strict JSON."""
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in values.items()}


def check(data: pd.DataFrame, n_sim: int = N_SIM) -> dict:
    """Plan and check a design for every readout and pairing.

    :param data: the control cells from :func:`load_controls`.
    :param n_sim: simulated experiments per check.
    :returns: the receipt body.
    """
    pilot = data[data['condition'] == 'nc']
    result = {'n_cells_nc': int(len(pilot)), 'n_cells_all': int(len(data)),
              'n_sim': n_sim, 'readouts': {}}
    for value, readout, fraction in READOUTS:
        comps = _nested_variance_components(pilot, value,
                                            replicate='plateID')
        both = _nested_variance_components(data, value, replicate='plateID',
                                           condition='condition')
        effect = round(fraction * comps['mean'], 6 if readout != 'continuous'
                       else 0)
        # The harmonic mean of cells per field, rounded so the parametric
        # simulation (whole cells) and the plan describe the same design.
        cells = int(round(comps['cells_per_field_effective']))
        entry = {'effect': effect, 'components_nc': _finite(
            {k: v for k, v in comps.items() if k != 'estimated'}),
                 'components_nc_pc': _finite(
            {k: v for k, v in both.items() if k != 'estimated'}),
                 'designs': {}}
        for paired in (False, True):
            designs = _plan_arrayed_design(comps, effect, power=0.8,
                                           paired=paired, readout=readout,
                                           cells=cells, **LIMITS)
            if designs.empty:
                entry['designs']['paired' if paired else 'unpaired'] = None
                continue
            best = designs.iloc[0]
            shape = dict(replicates=int(best.replicates),
                         wells=int(best.wells), fields=int(best.fields))
            simulated = _simulate_arrayed_power(
                comps, effect, **shape, cells=cells, paired=paired,
                readout=readout, n_sim=n_sim, seed=1)
            distinct = _resample_arrayed_power(
                pilot, value, effect, **shape, paired=paired,
                readout=readout, n_sim=n_sim, seed=2)
            repeated = _resample_arrayed_power(
                pilot, value, effect, **shape, paired=paired,
                readout=readout, n_sim=n_sim, seed=3, replace=True)
            with_rc = _arrayed_power(both, effect, **shape, cells=cells,
                                     paired=paired, readout=readout,
                                     baseline=comps['mean'])
            entry['designs']['paired' if paired else 'unpaired'] = {
                **shape, 'cells_per_field': cells,
                'planned_power': float(best.power),
                'simulated_power': simulated,
                'resampled_power_distinct': distinct,
                'resampled_power_with_replacement': repeated,
                'power_with_nc_pc_components': with_rc,
                'simulation_gap': simulated - float(best.power),
                'planned_within_resampling_bracket': bool(
                    repeated - MC_MARGIN <= float(best.power)
                    <= distinct + MC_MARGIN)}
        result['readouts'][f'{value} ({readout})'] = entry
    return result


def main() -> None:
    """Parse arguments, run the check and write the receipt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-sim', type=int, default=N_SIM)
    args = parser.parse_args()
    body = check(load_controls(args.root), n_sim=args.n_sim)
    body = {'item': 585, 'date': '2026-09-30',
            'source': str(args.root), 'plates': 4,
            'control_columns': list(CONTROL_COLUMNS), **body}
    args.out.write_text(json.dumps(body, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(body, indent=2))


if __name__ == '__main__':
    main()
