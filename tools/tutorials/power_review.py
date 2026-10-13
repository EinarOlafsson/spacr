"""Review a contradictory simulator headline; never approve sample-size advice."""
from power_evidence import check_recommendation

INPUTS = ('backend', 'method', 'class_neg_mu', 'class_neg_var', 'class_pos_mu',
    'class_pos_var', 'gene_abundance_alpha', 'gene_hit_rate',
    'imaging_n_cells_per_well_mu', 'imaging_n_cells_per_well_var', 'imaging_split',
    'min_cells_per_well', 'n_genes_in_library', 'n_reads_per_well',
    'n_wells_per_screen', 'pcr_factor_mu', 'pcr_factor_var', 'read_depth_cv',
    'sequencing_error_rate', 'sequencing_n_cells_per_well_lambda',
    'well_abundance_factor_mu', 'well_abundance_factor_var')


def review_baselines(records):
    """Require identical nominal inputs and distinct recorded random draws."""
    spec = records['spec']; cells = spec['cells_per_well']
    wells = spec['n_plates'] * spec['wells_per_plate']
    groups = [[r for r in records[name + '_scan']
               if r['imaging_n_cells_per_well_mu'] == cells and r['n_wells_per_screen'] == wells]
              for name in ('cells', 'wells')]
    if any(len(group) != spec['n_replicates'] for group in groups):
        raise ValueError('Both scans must contain every baseline replicate')
    reference = tuple(groups[0][0][key] for key in INPUTS)
    if any(tuple(row[key] for key in INPUTS) != reference for group in groups for row in group):
        raise ValueError('The allegedly same baseline has different nominal inputs')
    seeds = [{row['seed_used'] for row in group} for group in groups]
    if any(len(s) != spec['n_replicates'] for s in seeds) or seeds[0] & seeds[1]:
        raise ValueError('The same-design estimates must come from distinct recorded random draws')
    rejected = False
    try:
        check_recommendation(records['answer'], wells)
    except ValueError:
        rejected = True
    else:
        raise ValueError('This review requires the specifically observed contradictory headline')
    return dict(nominal_inputs_identical=True, independent_seed_groups=[sorted(s) for s in seeds],
                same_well_count_advice_rejected=rejected, recommendation_approved=False,
                calibrated_statistical_power=False)
