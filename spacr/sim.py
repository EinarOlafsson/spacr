
import os, random, warnings, traceback, sqlite3, shap, math, gc
from time import time, sleep
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import statsmodels.api as sm
from multiprocessing import cpu_count, Pool, Manager
from copy import deepcopy

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore RuntimeWarning

def generate_gene_list(number_of_genes, number_of_all_genes):
    """Return ``number_of_genes`` randomly-drawn gene indices without replacement.

    :param number_of_genes: Number of gene indices to draw.
    :param number_of_all_genes: Size of the pool ``[0, number_of_all_genes)``.
    :returns: List of drawn gene indices.
    """
    genes_ls = list(range(number_of_all_genes))
    random.shuffle(genes_ls)
    gene_list = genes_ls[:number_of_genes]
    return gene_list

# plate_map is a table with a row for each well, containing well metadata: plate_id, row_id, and column_id
def generate_plate_map(nr_plates):
    #print('nr_plates',nr_plates)
    """Return a 384-well plate map DataFrame spanning ``nr_plates`` plates.

    :param nr_plates: Number of plates to enumerate.
    :returns: DataFrame with ``plate_row_column``, ``plate_id``, ``row_id``,
        ``column_id`` columns (16 rows x 24 columns per plate).
    """
    plate_row_column = [f"{i+1}_{ir+1}_{ic+1}" for i in range(nr_plates) for ir in range(16) for ic in range(24)]
    df= pd.DataFrame({'plate_row_column': plate_row_column})
    df["plate_id"], df["row_id"], df["column_id"] = zip(*[r.split("_") for r in df['plate_row_column']])
    return df

def gini_coefficient(x):
    """Return the Gini coefficient of ``x`` via the pairwise absolute difference formula.

    :param x: 1-D array-like of non-negative values.
    :returns: Gini coefficient in ``[0, 1]``.
    """
    diffsum = np.sum(np.abs(np.subtract.outer(x, x)))
    return diffsum / (2 * len(x) ** 2 * np.mean(x))

def gini_gene_well(x):
    """Return the Gini coefficient of ``x`` using a memory-cheap upper-triangle sum.

    :param x: 1-D array-like of non-negative values.
    :returns: Gini coefficient in ``[0, 1]``; 0 is perfect equality, 1 is perfect inequality.
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def gini(x):
    """Return the Gini coefficient of ``x`` via the ranked-sum formulation.

    Reference: StatsDirect non-parametric methods
    (http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm).

    :param x: 1-D array-like; all values treated equally.
    :returns: Gini coefficient in ``[0, 1]``.
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x))  # ranks of x
    return 1 - (2 * (r * x).sum() + s) / (n * s)

def dist_gen(mean, sd, df):
    """Draw a length-``len(df)`` Poisson sample with gamma-distributed rates.

    :param mean: Mean of the gamma prior on the Poisson rate.
    :param sd: Standard deviation of the gamma prior.
    :param df: DataFrame whose length sets the sample size.
    :returns: Tuple ``(samples, length)`` where ``samples`` is a NumPy array of
        Poisson draws and ``length`` is ``len(df)``.
    """
    length = len(df)
    shape = (mean / sd) ** 2  # Calculate shape parameter
    scale = (sd ** 2) / mean  # Calculate scale parameter
    rate = np.random.gamma(shape, scale, size=length)  # Generate random rate from gamma distribution
    data = np.random.poisson(rate)  # Use the random rate for a Poisson distribution
    return data, length

def generate_gene_weights(positive_mean, positive_variance, df):
    """Draw ``len(df)`` gene weights from a Beta distribution matched to the given moments.

    :param positive_mean: Target mean of the Beta distribution in ``(0, 1)``.
    :param positive_variance: Target variance (must be feasible for the mean).
    :param df: DataFrame whose length sets the sample size.
    :returns: NumPy array of Beta-distributed weights.
    """
    # alpha and beta for positive distribution
    a1 = positive_mean*(positive_mean*(1-positive_mean)/positive_variance - 1)
    b1 = a1*(1-positive_mean)/positive_mean
    weights = np.random.beta(a1, b1, len(df))
    return weights

def normalize_array(arr):
    """Return ``arr`` min-max scaled into ``[0, 1]``.

    :param arr: Input NumPy array.
    :returns: Normalized array of the same shape.
    """
    min_value = np.min(arr)
    max_value = np.max(arr)
    normalized_arr = (arr - min_value) / (max_value - min_value)
    return normalized_arr

def generate_power_law_distribution(num_elements, coeff):
    """Return a normalized power-law probability vector of length ``num_elements``.

    :param num_elements: Length of the returned distribution.
    :param coeff: Positive exponent applied as ``i^-coeff``.
    :returns: NumPy array that sums to 1.
    """
    base_distribution = np.arange(1, num_elements + 1)
    powered_distribution = base_distribution ** -coeff
    normalized_distribution = powered_distribution / np.sum(powered_distribution)
    return normalized_distribution

# distribution generator function
def power_law_dist_gen(df, avg, well_ineq_coeff):
    """Return ``len(df)`` per-well values sampled from an average-scaled power-law.

    :param df: DataFrame of wells whose length sets the sample size.
    :param avg: Scale factor applied to each drawn probability.
    :param well_ineq_coeff: Power-law exponent (larger = more unequal).
    :returns: NumPy array of per-well quantities.
    """
    # Generate a power-law distribution for wells
    distribution = generate_power_law_distribution(len(df), well_ineq_coeff)
    dist = np.random.choice(distribution, len(df)) * avg
    return dist

# plates is a table with for each cell in the experiment with columns [plate_id, row_id, column_id, gene_id, is_active]
def run_experiment(plate_map, number_of_genes, active_gene_list, avg_genes_per_well, sd_genes_per_well, avg_cells_per_well, sd_cells_per_well, well_ineq_coeff, gene_ineq_coeff):
    """Simulate one cell-level screening experiment and return per-cell + summary tables.

    Draws per-well gene assignments from a power-law distribution and per-well
    cell counts from a gamma/Poisson mixture, then labels each cell active/inactive.

    :param plate_map: DataFrame of wells with plate/row/column identifiers.
    :param number_of_genes: Total number of genes in the pool.
    :param active_gene_list: Gene indices considered active (positive class).
    :param avg_genes_per_well: Mean genes-per-well before power-law scaling.
    :param sd_genes_per_well: Standard deviation of genes-per-well.
    :param avg_cells_per_well: Mean cells-per-well.
    :param sd_cells_per_well: Standard deviation of cells-per-well.
    :param well_ineq_coeff: Power-law exponent for well-level inequality.
    :param gene_ineq_coeff: Power-law exponent for gene-level inequality.
    :returns: Tuple ``(cell_df, genes_per_well_df, wells_per_gene_df, df_ls)``
        where ``df_ls`` contains per-well gene counts, per-gene well counts,
        per-well Gini values, per-gene Gini values, gene weights and well weights.
    """

    #generate primary distributions and genes
    cpw, _ = dist_gen(avg_cells_per_well, sd_cells_per_well, plate_map)
    gpw, _ = dist_gen(avg_genes_per_well, sd_genes_per_well, plate_map)
    genes = [*range(1, number_of_genes+1, 1)]
    
    #gene_weights = generate_power_law_distribution(number_of_genes, gene_ineq_coeff)
    gene_weights = {gene: weight for gene, weight in zip(genes, generate_power_law_distribution(number_of_genes, gene_ineq_coeff))} # Generate gene_weights as a dictionary        
    gene_weights_array = np.array(list(gene_weights.values())) # Convert the values to an array
    
    well_weights = generate_power_law_distribution(len(plate_map), well_ineq_coeff)
    
    gene_to_well_mapping = {}
    for gene in range(1, number_of_genes + 1):  # ensures gene-1 is within bounds
        if gene-1 < len(gpw):
            max_index = len(plate_map['plate_row_column'])  # this should be the number of choices available from plate_map
            num_samples = int(gpw[gene-1])
            if num_samples >= max_index:
                num_samples = max_index - 1  # adjust to maximum possible index
            gene_to_well_mapping[gene] = np.random.choice(plate_map['plate_row_column'], size=num_samples, replace=False, p=well_weights)
        else:
            break  # break the loop if gene-1 is out of bounds for gpw

    cells = []
    for i in [*range(0,len(plate_map))]:
        ciw = random.choice(cpw)
        present_genes = [gene for gene, wells in gene_to_well_mapping.items() if plate_map.loc[i, 'plate_row_column'] in wells] # Select genes present in the current well
        present_gene_weights = [gene_weights[gene] for gene in present_genes] # For sampling, filter gene_weights according to present_genes
        present_gene_weights /= np.sum(present_gene_weights)
        if present_genes:
            giw = np.random.choice(present_genes, int(gpw[i]), p=present_gene_weights)
            if len(giw) > 0:
                for _ in range(0,int(ciw)):
                    gene_nr = random.choice(giw)
                    cell = {
                        'plate_row_column': plate_map.loc[i, 'plate_row_column'],
                        'plate_id': plate_map.loc[i, 'plate_id'], 
                        'row_id': plate_map.loc[i, 'row_id'], 
                        'column_id': plate_map.loc[i, 'column_id'],
                        'genes_in_well': len(giw), 
                        'gene_id': gene_nr,
                        'is_active': int(gene_nr in active_gene_list)
                    }
                    cells.append(cell)
    
    cell_df = pd.DataFrame(cells)
    cell_df = cell_df.dropna()

    # calculate well, gene counts per well
    gene_counts_per_well = cell_df.groupby('plate_row_column')['gene_id'].nunique().sort_values().tolist()
    well_counts_per_gene = cell_df.groupby('gene_id')['plate_row_column'].nunique().sort_values().tolist()

    # Create DataFrames
    genes_per_well_df = pd.DataFrame(gene_counts_per_well, columns=['genes_per_well'])
    genes_per_well_df['rank'] = range(1, len(genes_per_well_df) + 1)
    wells_per_gene_df = pd.DataFrame(well_counts_per_gene, columns=['wells_per_gene'])
    wells_per_gene_df['rank'] = range(1, len(wells_per_gene_df) + 1)
    
    ls_ = []
    gini_ls = []
    for i,val in enumerate(cell_df['plate_row_column'].unique().tolist()):
        temp = cell_df[cell_df['plate_row_column']==val]
        x = temp['gene_id'].value_counts().to_numpy()
        gini_val = gini_gene_well(x)
        ls_.append(val)
        gini_ls.append(gini_val)
    gini_well = np.array(gini_ls)
    
    ls_ = []
    gini_ls = []
    for i,val in enumerate(cell_df['gene_id'].unique().tolist()):
        temp = cell_df[cell_df['gene_id']==val]
        x = temp['plate_row_column'].value_counts().to_numpy()
        gini_val = gini_gene_well(x)
        ls_.append(val)
        gini_ls.append(gini_val)
    gini_gene = np.array(gini_ls)
    df_ls = [gene_counts_per_well, well_counts_per_gene, gini_well, gini_gene, gene_weights_array, well_weights]
    return cell_df, genes_per_well_df, wells_per_gene_df, df_ls

def classifier(positive_mean, positive_variance, negative_mean, negative_variance, classifier_accuracy, df):
    """Assign a Beta-distributed score to each row of ``df`` with class-swap noise.

    :param positive_mean: Mean of the Beta distribution for active cells.
    :param positive_variance: Variance of the Beta for active cells.
    :param negative_mean: Mean of the Beta for inactive cells.
    :param negative_variance: Variance of the Beta for inactive cells.
    :param classifier_accuracy: Probability in ``[0, 1]`` that the correct
        Beta is used for the row's ``is_active`` label.
    :param df: DataFrame containing an ``is_active`` column.
    :returns: The input DataFrame with an added ``score`` column.
    """
    def calc_alpha_beta(mean, variance):
        """Return ``(alpha, beta)`` for a Beta with the given mean and variance."""
        if mean <= 0 or mean >= 1:
            raise ValueError("Mean must be between 0 and 1 exclusively.")
        max_variance = mean * (1 - mean)
        if variance <= 0 or variance >= max_variance:
            raise ValueError(f"Variance must be positive and less than {max_variance}.")
        
        alpha = mean * (mean * (1 - mean) / variance - 1)
        beta = alpha * (1 - mean) / mean
        return alpha, beta
    
    # Apply the beta distribution based on 'is_active' status with consideration for classifier error
    def get_score(is_active):
        """Return a Beta sample from the correct or incorrect class distribution."""
        if np.random.rand() < classifier_accuracy:  # With classifier_accuracy probability, choose the correct distribution
            return np.random.beta(a1, b1) if is_active else np.random.beta(a2, b2)
        else:  # With 1-classifier_accuracy probability, choose the incorrect distribution
            return np.random.beta(a2, b2) if is_active else np.random.beta(a1, b1)

    # Calculate alpha and beta for both distributions
    a1, b1 = calc_alpha_beta(positive_mean, positive_variance)
    a2, b2 = calc_alpha_beta(negative_mean, negative_variance)
    df['score'] = df['is_active'].apply(get_score)

    return df

def compute_roc_auc(cell_scores):
    """Return ROC-curve arrays and AUC for a DataFrame of ``is_active``/``score`` rows.

    :param cell_scores: DataFrame with columns ``is_active`` and ``score``.
    :returns: Dict with keys ``threshold``, ``tpr``, ``fpr``, ``roc_auc``.
    """
    fpr, tpr, thresh = roc_curve(cell_scores['is_active'], cell_scores['score'], pos_label=1)
    roc_auc = auc(fpr, tpr)
    cell_roc_dict = {'threshold':thresh,'tpr': tpr,'fpr': fpr, 'roc_auc':roc_auc}
    return cell_roc_dict

def compute_precision_recall(cell_scores):
    """Return precision/recall/F1/PR-AUC arrays for a DataFrame of ``is_active``/``score`` rows.

    :param cell_scores: DataFrame with columns ``is_active`` and ``score``.
    :returns: Dict with keys ``threshold``, ``precision``, ``recall``,
        ``f1_score``, ``pr_auc``.
    """
    pr, re, th = precision_recall_curve(cell_scores['is_active'], cell_scores['score'])
    th = np.insert(th, 0, 0)
    f1_score = 2 * (pr * re) / (pr + re)
    pr_auc = auc(re, pr)
    cell_pr_dict = {'threshold':th,'precision': pr,'recall': re, 'f1_score':f1_score, 'pr_auc': pr_auc}
    return cell_pr_dict

def get_optimum_threshold(cell_pr_dict):
    """Return the classification threshold that maximises F1 in a PR result dict.

    :param cell_pr_dict: Dict as returned by :func:`compute_precision_recall`.
    :returns: Threshold value that maximises the F1 score.
    """
    cell_pr_dict_df = pd.DataFrame(cell_pr_dict)
    max_x = cell_pr_dict_df.loc[cell_pr_dict_df['f1_score'].idxmax()]
    optimum = float(max_x['threshold'])
    return optimum

def update_scores_and_get_cm(cell_scores, optimum):
    """Add a per-threshold predicted-label column and return the confusion matrix.

    :param cell_scores: DataFrame with columns ``is_active`` and ``score``.
    :param optimum: Score threshold used to binarise predictions.
    :returns: Tuple ``(cell_scores, cell_cm)`` where ``cell_cm`` is a NumPy
        confusion matrix.
    """
    cell_scores[optimum] = cell_scores.score.map(lambda x: 1 if x >= optimum else 0)
    cell_cm = metrics.confusion_matrix(cell_scores.is_active, cell_scores[optimum])
    return cell_scores, cell_cm

def cell_level_roc_auc(cell_scores):
    """Compute cell-level ROC/PR metrics and confusion matrix at the F1-optimal threshold.

    :param cell_scores: DataFrame with columns ``is_active`` and ``score``.
    :returns: Tuple ``(cell_roc_dict_df, cell_pr_dict_df, cell_scores, cell_cm)``.
    """
    cell_roc_dict = compute_roc_auc(cell_scores)
    cell_pr_dict = compute_precision_recall(cell_scores)
    optimum = get_optimum_threshold(cell_pr_dict)
    cell_scores, cell_cm = update_scores_and_get_cm(cell_scores, optimum)
    cell_pr_dict['optimum'] = optimum
    cell_roc_dict_df = pd.DataFrame(cell_roc_dict)
    cell_pr_dict_df = pd.DataFrame(cell_pr_dict)
    return cell_roc_dict_df, cell_pr_dict_df, cell_scores, cell_cm

def generate_well_score(cell_scores):
    """Aggregate cell-level scores into per-well summary rows.

    :param cell_scores: DataFrame indexed by cells with ``plate_row_column``,
        ``is_active`` and ``gene_id`` columns.
    :returns: DataFrame indexed by ``plate_row_column`` with
        ``average_active_score``, ``gene_list``, and ``score`` columns.
    """
    # Compute mean and list of unique gene_ids
    well_score = cell_scores.groupby(['plate_row_column']).agg(
        average_active_score=('is_active', 'mean'),
        gene_list=('gene_id', lambda x: np.unique(x).tolist()))
    well_score['score'] = np.log10(well_score['average_active_score'] + 1)
    return well_score

def sequence_plates(well_score, number_of_genes, avg_reads_per_gene, sd_reads_per_gene, sequencing_error=0.01):
    """Simulate sequencing of every well and return per-well gene fraction and metadata.

    Each gene present in a well accrues a Poisson-distributed read count that
    may be reassigned to a random well with probability ``sequencing_error``.

    :param well_score: DataFrame with a ``gene_list`` column per well.
    :param number_of_genes: Number of distinct genes in the pool.
    :param avg_reads_per_gene: Mean of the per-gene read count distribution.
    :param sd_reads_per_gene: Standard deviation of that distribution.
    :param sequencing_error: Probability of assigning a read to the wrong well.
        Default ``0.01``.
    :returns: Tuple ``(gene_fraction_map, metadata)`` DataFrames indexed by well.
    """

    reads, _ = dist_gen(avg_reads_per_gene, sd_reads_per_gene, well_score)
    gene_names = [f'gene_{v}' for v in range(number_of_genes+1)]
    all_wells = well_score.index

    gene_counts_map = pd.DataFrame(np.zeros((len(all_wells), number_of_genes+1)), columns=gene_names, index=all_wells)
    sum_reads = []

    for _, row in well_score.iterrows():
        gene_list = row['gene_list']
        
        if gene_list:
            for gene in gene_list:
                gene_count = int(random.choice(reads))

                # Decide whether to introduce error or not
                error = np.random.binomial(1, sequencing_error)
                if error:
                    # Randomly select a different well
                    wrong_well = np.random.choice(all_wells)
                    gene_counts_map.loc[wrong_well, f'gene_{int(gene)}'] += gene_count
                else:
                    gene_counts_map.loc[_, f'gene_{int(gene)}'] += gene_count
        
        sum_reads.append(np.sum(gene_counts_map.loc[_, :]))

    gene_fraction_map = gene_counts_map.div(gene_counts_map.sum(axis=1), axis=0)
    gene_fraction_map = gene_fraction_map.fillna(0)
    
    metadata = pd.DataFrame(index=well_score.index)
    metadata['genes_in_well'] = gene_fraction_map.astype(bool).sum(axis=1)
    metadata['sum_fractions'] = gene_fraction_map.sum(axis=1)
    metadata['sum_reads'] = sum_reads

    return gene_fraction_map, metadata

#metadata['sum_reads'] = metadata['sum_fractions'].div(metadata['genes_in_well'])
def regression_roc_auc(results_df, active_gene_list, control_gene_list, alpha = 0.05, optimal=False):
    """Score regression hits against ground truth and compute ROC/PR metrics.

    Marks each gene as active/inactive/control, derives a hit cutoff from the
    control coefficients, and returns ROC/PR curves, a confusion matrix and
    per-run summary statistics.

    :param results_df: Regression output with ``gene``, ``coef`` and ``P>|t|``.
    :param active_gene_list: Gene indices considered truly active.
    :param control_gene_list: Gene indices used to derive the coefficient cutoff.
    :param alpha: Significance threshold applied to p-values. Default ``0.05``.
    :param optimal: When True, use the F1-optimal probability threshold instead
        of ``0.5`` for the final confusion matrix.
    :returns: Tuple ``(results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm,
        sim_stats)`` where ``sim_stats`` is a single-row DataFrame.
    """
    results_df = results_df.rename(columns={"P>|t|": "p"})

    # asign active genes a value of 1 and inactive genes a value of 0
    actives_list = ['gene_' + str(i) for i in active_gene_list]
    results_df['active'] = results_df['gene'].apply(lambda x: 1 if x in actives_list else 0)
    results_df['active'].fillna(0, inplace=True)
    
    #generate a colun to color control,active and inactive genes
    controls_list = ['gene_' + str(i) for i in control_gene_list]
    results_df['color'] = results_df['gene'].apply(lambda x: 'control' if x in controls_list else ('active' if x in actives_list else 'inactive'))
    
    #generate a size column and handdf.replace([np.inf, -np.inf], np.nan, inplace=True)le infinate and NaN values create a new column for -log(p)
    results_df['size'] = results_df['active']
    results_df['p'] = results_df['p'].clip(lower=0.0001)
    results_df['logp'] = -np.log10(results_df['p'])
    
    #calculate cutoff for hits based on randomly chosen 'control' genes
    control_df = results_df[results_df['color'] == 'control']
    control_mean = control_df['coef'].mean()
    #control_std = control_df['coef'].std()
    control_var = control_df['coef'].var()
    cutoff = abs(control_mean)+(3*control_var)
    
    #calculate discriptive statistics for active genes
    active_df = results_df[results_df['color'] == 'active']
    active_mean = active_df['coef'].mean()
    active_std = active_df['coef'].std()
    active_var = active_df['coef'].var()
    
    #calculate discriptive statistics for active genes
    inactive_df = results_df[results_df['color'] == 'inactive']
    inactive_mean = inactive_df['coef'].mean()
    inactive_std = inactive_df['coef'].std()
    inactive_var = inactive_df['coef'].var()
    
    #generate score column for hits and non hitts
    results_df['score'] = np.where(((results_df['coef'] >= cutoff) | (results_df['coef'] <= -cutoff)) & (results_df['p'] <= alpha), 1, 0)
    
    #calculate regression roc based on controll cutoff
    fpr, tpr, thresh = roc_curve(results_df['active'], results_df['score'])
    roc_auc = auc(fpr, tpr)
    reg_roc_dict_df = pd.DataFrame({'threshold':thresh, 'tpr': tpr, 'fpr': fpr, 'roc_auc':roc_auc})

    pr, re, th = precision_recall_curve(results_df['active'], results_df['score'])
    th = np.insert(th, 0, 0)
    f1_score = 2 * (pr * re) / (pr + re)
    pr_auc = auc(re, pr)
    reg_pr_dict_df = pd.DataFrame({'threshold':th, 'precision': pr, 'recall': re, 'f1_score':f1_score, 'pr_auc': pr_auc})

    optimal_threshold = reg_pr_dict_df['f1_score'].idxmax()
    if optimal:
        results_df[optimal_threshold] = results_df.score.apply(lambda x: 1 if x >= optimal_threshold else 0)
        reg_cm = confusion_matrix(results_df.active, results_df[optimal_threshold])
    else:
        results_df[0.5] = results_df.score.apply(lambda x: 1 if x >= 0.5 else 0)
        reg_cm = confusion_matrix(results_df.active, results_df[0.5])
    
    TN = reg_cm[0][0]
    FP = reg_cm[0][1]
    FN = reg_cm[1][0]
    TP = reg_cm[1][1]
    
    accuracy = (TP + TN) / (TP + FP + FN + TN)  # Accuracy
    sim_stats = {'optimal_threshold':optimal_threshold,
                 'accuracy': accuracy,
                 'prauc':pr_auc,
                 'roc_auc':roc_auc,
                 'inactive_mean':inactive_mean,
                 'inactive_std':inactive_std,
                 'inactive_var':inactive_var,
                 'active_mean':active_mean,
                 'active_std':active_std,
                 'active_var':active_var,
                 'cutoff':cutoff,
                 'TP':TP,
                 'FP':FP,
                 'TN':TN,
                 'FN':FN}
    
    return results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm, pd.DataFrame([sim_stats])

def plot_histogram(data, x_label, ax, color, title, binwidth=0.01, log=False):
    """Draw a Seaborn density histogram on ``ax`` for the given column.

    :param data: Data source passed to ``sns.histplot``.
    :param x_label: Column name plotted on the x-axis.
    :param ax: Matplotlib axes to draw into.
    :param color: Bar/fill color.
    :param title: Axes title.
    :param binwidth: Histogram bin width; falsy uses Seaborn's default.
    :param log: When True, apply a log scale to the y-axis.
    :returns: None.
    """
    if not binwidth:
        sns.histplot(data=data, x=x_label, ax=ax, color=color, kde=False, stat='density', 
                    legend=False, fill=True, element='step', palette='dark')
    else:
        sns.histplot(data=data, x=x_label, ax=ax, color=color, binwidth=binwidth, kde=False, stat='density', 
                    legend=False, fill=True, element='step', palette='dark')
    if log:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(x_label)

def plot_roc_pr(data, ax, title, x_label, y_label):
    """Plot a ROC or PR curve with a diagonal random-classifier reference line.

    :param data: DataFrame containing the ``x_label`` and ``y_label`` columns.
    :param ax: Matplotlib axes to draw into.
    :param title: Axes title.
    :param x_label: Column name plotted on the x-axis.
    :param y_label: Column name plotted on the y-axis.
    :returns: None.
    """
    ax.plot(data[x_label], data[y_label], color='black', lw=0.5)
    ax.plot([0, 1], [0, 1], color='black', lw=0.5, linestyle="--", label='random classifier')
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(loc="lower right")

def plot_confusion_matrix(data, ax, title):
    """Render a 2x2 confusion matrix as an annotated Seaborn heatmap.

    :param data: 2x2 NumPy confusion matrix ordered ``[[TN, FP], [FN, TP]]``.
    :param ax: Matplotlib axes to draw into.
    :param title: Axes title.
    :returns: None.
    """
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in data.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in data.flatten()/np.sum(data)]
    
    sns.heatmap(data, cmap='Blues', ax=ax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j+0.5, i+0.5, f'{group_names[i*2+j]}\n{group_counts[i*2+j]}\n{group_percentages[i*2+j]}',
                    ha="center", va="center", color="black")

    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])


def run_simulation(settings):
    """Run one end-to-end pooled-screen simulation and return every intermediate table.

    Composes :func:`generate_gene_list`, :func:`generate_plate_map`,
    :func:`run_experiment`, :func:`classifier`, cell/well aggregation,
    :func:`sequence_plates` and :func:`regression_roc_auc` into a single call.

    :param settings: Dict of simulation parameters (gene counts, distribution
        moments, sequencing error, classifier accuracy, ...).
    :returns: Tuple ``(cell_scores, cell_roc_dict_df, cell_pr_dict_df,
        cell_cm, well_score, gene_fraction_map, metadata, results_df,
        reg_roc_dict_df, reg_pr_dict_df, reg_cm, sim_stats,
        genes_per_well_df, wells_per_gene_df, dists)``.
    """
    #try:
    active_gene_list = generate_gene_list(settings['number_of_active_genes'], settings['number_of_genes'])
    control_gene_list = generate_gene_list(settings['number_of_control_genes'], settings['number_of_genes'])
    plate_map = generate_plate_map(settings['nr_plates'])

    #control_map = plate_map[plate_map['column_id'].isin(['c1', 'c2', 'c3', 'c23', 'c24'])] # Extract rows where 'column_id' is in [1,2,3,23,24]
    plate_map = plate_map[~plate_map['column_id'].isin(['c1', 'c2', 'c3', 'c23', 'c24'])] # Extract rows where 'column_id' is not in [1,2,3,23,24]

    cell_level, genes_per_well_df, wells_per_gene_df, dists = run_experiment(plate_map, settings['number_of_genes'], active_gene_list, settings['avg_genes_per_well'], settings['sd_genes_per_well'], settings['avg_cells_per_well'], settings['sd_cells_per_well'], settings['well_ineq_coeff'], settings['gene_ineq_coeff'])
    cell_scores = classifier(settings['positive_mean'], settings['positive_variance'], settings['negative_mean'], settings['negative_variance'], settings['classifier_accuracy'], df=cell_level)
    cell_roc_dict_df, cell_pr_dict_df, cell_scores, cell_cm = cell_level_roc_auc(cell_scores)
    well_score = generate_well_score(cell_scores)
    gene_fraction_map, metadata = sequence_plates(well_score, settings['number_of_genes'], settings['avg_reads_per_gene'], settings['sd_reads_per_gene'], sequencing_error=settings['sequencing_error'])
    x = gene_fraction_map
    y = np.log10(well_score['score']+1)
    x = sm.add_constant(x)
    #y = y.fillna(0)
    #x = x.fillna(0)
    #x['const'] = 0.0
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x)
    results_summary = model.summary()
    results_as_html = results_summary.tables[1].as_html()
    results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.rename_axis("gene").reset_index()
    results_df = results_df.iloc[1: , :]
    results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm, sim_stats = regression_roc_auc(results_df, active_gene_list, control_gene_list, alpha = 0.05, optimal=False)
    #except Exception as e:
    #    print(f"An error occurred while saving data: {e}")
    output = [cell_scores, cell_roc_dict_df, cell_pr_dict_df, cell_cm, well_score, gene_fraction_map, metadata, results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm, sim_stats, genes_per_well_df, wells_per_gene_df]
    del cell_scores, cell_roc_dict_df, cell_pr_dict_df, cell_cm, well_score, gene_fraction_map, metadata, results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm, sim_stats, genes_per_well_df, wells_per_gene_df
    gc.collect()
    return output, dists

def vis_dists(dists, src, v, i):
    """Save side-by-side histograms of the six per-run distributions in ``dists``.

    :param dists: Six arrays in order ``[genes/well, wells/gene, gini_well,
        gini_gene, gene_weights, well_weights]``.
    :param src: Output directory used by :func:`save_plot`.
    :param v: Variable label passed through to :func:`save_plot`.
    :param i: Simulation index passed through to :func:`save_plot`.
    :returns: None.
    """
    n_graphs = 6
    height_graphs = 4
    n=0
    width_graphs = height_graphs*n_graphs
    fig2, ax =plt.subplots(1,n_graphs, figsize = (width_graphs,height_graphs))
    names = ['genes/well', 'wells/gene', 'genes/well gini', 'wells/gene gini', 'gene_weights', 'well_weights']
    for index, dist in enumerate(dists):
        temp = pd.DataFrame(dist, columns = [f'{names[index]}'])
        sns.histplot(data=temp, x=f'{names[index]}', kde=False, binwidth=None, stat='count', element="step", ax=ax[n], color='teal', log_scale=False)
        n+=1
    save_plot(fig2, src, 'dists', i)
    plt.close(fig2)
    plt.figure().clear() 
    plt.cla() 
    plt.clf()
    del dists

    return

def visualize_all(output):
    """Render the full 13-panel diagnostic figure for one simulation output.

    :param output: The 14-element list returned by :func:`run_simulation` (all
        elements before ``dists``).
    :returns: The generated Matplotlib figure.
    """

    cell_scores = output[0]
    cell_roc_dict_df = output[1]
    cell_pr_dict_df = output[2]
    cell_cm = output[3]
    well_score = output[4]
    gene_fraction_map = output[5]
    metadata = output[6]
    results_df = output[7]
    reg_roc_dict_df = output[8]
    reg_pr_dict_df = output[9]
    reg_cm =output[10]
    sim_stats = output[11]
    genes_per_well_df = output[12]
    wells_per_gene_df = output[13]

    hline = -np.log10(0.05)
    n_graphs = 13
    height_graphs = 4
    n=0
    width_graphs = height_graphs*n_graphs

    fig, ax =plt.subplots(1,n_graphs, figsize = (width_graphs,height_graphs))

    #plot genes per well
    gini_genes_per_well = gini(genes_per_well_df['genes_per_well'].tolist())
    plot_histogram(genes_per_well_df, "genes_per_well", ax[n], 'slategray', f'gene/well (gini = {gini_genes_per_well:.2f})', binwidth=None, log=False)
    n+=1
    
    #plot wells per gene
    gini_wells_per_gene = gini(wells_per_gene_df['wells_per_gene'].tolist())
    plot_histogram(wells_per_gene_df, "wells_per_gene", ax[n], 'slategray', f'well/gene (Gini = {gini_wells_per_gene:.2f})', binwidth=None, log=False)
    #ax[n].set_xscale('log')
    n+=1
    
    #plot cell classification score by inactive and active
    active_distribution = cell_scores[cell_scores['is_active'] == 1] 
    inactive_distribution = cell_scores[cell_scores['is_active'] == 0]
    plot_histogram(active_distribution, "score", ax[n], 'slategray', 'Cell scores', log=False)#, binwidth=0.01, log=False)
    plot_histogram(inactive_distribution, "score", ax[n], 'teal', 'Cell scores', log=False)#, binwidth=0.01, log=False)

    legend_elements = [Patch(facecolor='slategray', edgecolor='slategray', label='Inactive'),
                   Patch(facecolor='teal', edgecolor='teal', label='Active')]
    
    ax[n].legend(handles=legend_elements, loc='upper right')


    ax[n].set_xlim([0, 1])
    n+=1
    
    #plot classifier cell predictions by inactive and active well average
    inactive_distribution_well = inactive_distribution.groupby(['plate_id', 'row_id', 'column_id'])['score'].mean().reset_index(name='score')
    active_distribution_well = active_distribution.groupby(['plate_id', 'row_id', 'column_id'])['score'].mean().reset_index(name='score')
    mixed_distribution_well = cell_scores.groupby(['plate_id', 'row_id', 'column_id'])['score'].mean().reset_index(name='score')

    plot_histogram(inactive_distribution_well, "score", ax[n], 'slategray', 'Well scores', log=False)#, binwidth=0.01, log=False)
    plot_histogram(active_distribution_well, "score", ax[n], 'teal', 'Well scores', log=False)#, binwidth=0.01, log=False)
    plot_histogram(mixed_distribution_well, "score", ax[n], 'red', 'Well scores', log=False)#, binwidth=0.01, log=False)
    
    legend_elements = [Patch(facecolor='slategray', edgecolor='slategray', label='Inactive'),
                   Patch(facecolor='teal', edgecolor='teal', label='Active'),
                   Patch(facecolor='red', edgecolor='red', label='Mixed')]
    
    ax[n].legend(handles=legend_elements, loc='upper right')

    ax[n].set_xlim([0, 1])
    #ax[n].legend()
    n+=1
    
    #plot ROC (cell classification)
    plot_roc_pr(cell_roc_dict_df, ax[n], 'ROC (Cell)', 'fpr', 'tpr')
    ax[n].plot([0, 1], [0, 1], color='black', lw=0.5, linestyle="--", label='random classifier')
    n+=1
    
    #plot Presision recall (cell classification)
    plot_roc_pr(cell_pr_dict_df, ax[n], 'Precision recall (Cell)', 'recall', 'precision')
    ax[n].set_ylim([-0.1, 1.1])
    ax[n].set_xlim([-0.1, 1.1])
    n+=1
    
    #Confusion matrix at optimal threshold
    plot_confusion_matrix(cell_cm, ax[n], 'Confusion Matrix Cell')
    n+=1
    
    #plot well score
    plot_histogram(well_score, "score", ax[n], 'teal', 'Well score', binwidth=0.005, log=True)
    #ax[n].set_xlim([0, 1])
    n+=1

    control_df = results_df[results_df['color'] == 'control']
    control_mean = control_df['coef'].mean()
    control_var = control_df['coef'].std()
    #control_var = control_df['coef'].var()
    cutoff = abs(control_mean)+(3*control_var)
    categories = ['inactive', 'control', 'active']
    colors = ['lightgrey', 'black', 'purple']
    
    for category, color in zip(categories, colors):
        df = results_df[results_df['color'] == category]
        ax[n].scatter(df['coef'], df['logp'], c=color, alpha=0.7, label=category)

    reg_lab = ax[n].legend(title='', frameon=False, prop={'size': 10})
    ax[n].add_artist(reg_lab)
    ax[n].axhline(hline, zorder = 0,c = 'k', lw = 0.5,ls = '--')
    ax[n].axvline(-cutoff, zorder = 0,c = 'k', lw = 0.5,ls = '--')
    ax[n].axvline(cutoff, zorder = 0,c = 'k', lw = 0.5,ls = '--')
    ax[n].set_title(f'Regression, threshold {cutoff:.3f}')
    ax[n].set_xlim([-1, 1.1])
    n+=1

    # error plot
    df = results_df[['gene', 'coef', 'std err', 'p']]
    df = df.sort_values(by = ['coef', 'p'], ascending = [True, False], na_position = 'first')
    df['rank'] = [*range(0,len(df),1)]
    
    #df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    #df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    #df['std err'] = pd.to_numeric(df['std err'], errors='coerce')
    #df['rank'] = df['rank'].astype(float)
    #df['coef'] = df['coef'].astype(float)
    #df['std err'] = df['std err'].astype(float)
    #epsilon = 1e-6  # A small constant to ensure std err is never zero
    #df['std err adj'] = df['std err'].replace(0, epsilon)

    ax[n].plot(df['rank'], df['coef'], '-', color = 'black')
    ax[n].fill_between(df['rank'], df['coef'] - abs(df['std err']), df['coef'] + abs(df['std err']), alpha=0.4, color='slategray')
    ax[n].set_title('Effect score error')
    ax[n].set_xlabel('rank')
    ax[n].set_ylabel('Effect size')
    n+=1

    #plot ROC (gene classification)
    plot_roc_pr(reg_roc_dict_df, ax[n], 'ROC (gene)', 'fpr', 'tpr')
    ax[n].legend(loc="lower right")
    n+=1
    
    #plot Presision recall (regression classification)
    plot_roc_pr(reg_pr_dict_df, ax[n], 'Precision recall (gene)', 'recall', 'precision')
    ax[n].legend(loc="lower right")
    n+=1
    
    #Confusion matrix at optimal threshold
    plot_confusion_matrix(reg_cm, ax[n], 'Confusion Matrix Reg')

    for n in [*range(0,n_graphs,1)]:
        ax[n].spines['top'].set_visible(False)
        ax[n].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    gc.collect()
    return fig

def create_database(db_path):
    """Ensure a SQLite database file exists at ``db_path``.

    :param db_path: Filesystem path for the SQLite database.
    :returns: None.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        #print(f"SQLite version: {sqlite3.version}")
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()

def append_database(src, table, table_name):
    """Append a DataFrame to ``<src>/simulations.db`` under ``table_name``.

    :param src: Directory containing (or that should contain) ``simulations.db``.
    :param table: DataFrame written with ``if_exists='append'``.
    :param table_name: Target table name in the SQLite database.
    :returns: None.
    """
    try:
        conn = sqlite3.connect(f'{src}/simulations.db', timeout=3600)
        table.to_sql(table_name, conn, if_exists='append', index=False)
    except sqlite3.OperationalError as e:
        print("SQLite error:", e)
    finally:
        conn.close()
    return

def save_data(src, output, settings, save_all=False, i=0, variable='all'):
    """Persist one simulation's output tables to a SQLite database under ``src``.

    In the default mode only a concatenated summary row (settings + sim_stats
    + Gini metrics) is appended to a ``simulations`` table. When ``save_all``
    is True, every intermediate table is written under its canonical name.

    :param src: Output directory containing ``simulations.db``.
    :param output: 14-element list from :func:`run_simulation`.
    :param settings: Simulation settings dict recorded as the first row.
    :param save_all: When True, write every intermediate table separately.
    :param i: Simulation index used to tag the summary row.
    :param variable: Name of the swept variable used for tagging.
    :returns: None.
    """
    try:
        if not save_all:
            src = f'{src}'
            os.makedirs(src, exist_ok=True)
        else:
            os.makedirs(src, exist_ok=True)

        settings_df = pd.DataFrame({key: [value] for key, value in settings.items()})
        output = [settings_df] + output
        table_names = ['settings', 'cell_scores', 'cell_roc', 'cell_precision_recall', 'cell_confusion_matrix', 'well_score', 'gene_fraction_map', 'metadata', 'regression_results', 'regression_roc', 'regression_precision_recall', 'regression_confusion_matrix', 'sim_stats', 'genes_per_well', 'wells_per_gene']

        if not save_all:
            gini_genes_per_well = gini(output[13]['genes_per_well'].tolist())
            gini_wells_per_gene = gini(output[14]['wells_per_gene'].tolist())
            indices_to_keep= [0,12] # Specify the indices to remove
            filtered_output = [v for i, v in enumerate(output) if i in indices_to_keep]
            df_concat = pd.concat(filtered_output, axis=1)
            df_concat['genes_per_well_gini'] = gini_genes_per_well
            df_concat['wells_per_gene_gini'] = gini_wells_per_gene
            df_concat['date'] = datetime.now()
            df_concat[f'variable_{variable}_sim_nr'] = i

            append_database(src, df_concat, 'simulations')
            del gini_genes_per_well, gini_wells_per_gene, df_concat

        if save_all:
            for i, df in enumerate(output):
                df = output[i]
                if table_names[i] == 'well_score':
                    df['gene_list'] = df['gene_list'].astype(str)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                append_database(src, df, table_names[i])
            del df
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
        print(traceback.format_exc())
    
    del output, settings_df
    return

def save_plot(fig, src, variable, i):
    """Save a Matplotlib figure to ``<src>/<variable>/<i>_figure.pdf``.

    :param fig: Figure to save.
    :param src: Root directory for outputs.
    :param variable: Sub-folder name (the swept variable label).
    :param i: Zero-padded simulation index used in the file name.
    :returns: None.
    """
    os.makedirs(f'{src}/{variable}', exist_ok=True)
    filename_fig = f'{src}/{variable}/{str(i)}_figure.pdf'
    fig.savefig(filename_fig, dpi=600, format='pdf', bbox_inches='tight')
    return
    
def run_and_save(i, settings, time_ls, total_sims):
    """Worker that runs one simulation, saves outputs, and appends its runtime.

    :param i: Simulation index (used for filenames and tagging).
    :param settings: Simulation settings dict.
    :param time_ls: Shared list receiving the elapsed time in seconds.
    :param total_sims: Total simulation count (used for progress display only).
    :returns: Tuple ``(i, sim_time, None)``.
    """
    #print(f'Runnings simulation with the following paramiters')
    #print(settings)
    settings['random_seed'] = False
    if settings['random_seed']:
        random.seed(42) # sims will be too similar with random seed
    src = settings['src']
    plot = settings['plot']
    v = settings['variable']
    start_time = time()  # Start time of the simulation
    #now = datetime.now() # get current date
    #date_string = now.strftime("%y%m%d") # format as a string in 'ddmmyy' format        
    date_string = settings['start_time']
    #try:
    output, dists = run_simulation(settings)
    sim_time = time() - start_time  # Elapsed time for the simulation
    settings['sim_time'] = sim_time
    src = os.path.join(f'{src}/{date_string}',settings['name'])
    save_data(src, output, settings, save_all=False, i=i, variable=v)
    if plot:
        vis_dists(dists,src, v, i)
        fig = visualize_all(output)
        save_plot(fig, src, v, i)
        plt.close(fig)
        plt.figure().clear() 
        plt.cla() 
        plt.clf()
        del fig
    del output, dists
    gc.collect()
    #except Exception as e:
    #    print(e, end='\r', flush=True)
    #    sim_time = time() - start_time
        #print(traceback.format_exc(), end='\r', flush=True)
    time_ls.append(sim_time)
    return i, sim_time, None
    
def validate_and_adjust_beta_params(sim_params):
    """Clamp per-run Beta variances so the requested mean/variance is feasible.

    :param sim_params: List of per-run parameter dicts with ``positive_mean``,
        ``negative_mean``, ``positive_variance``, ``negative_variance``.
    :returns: The same list with any infeasible variances capped to 99% of the
        theoretical maximum for the requested mean.
    """
    adjusted_params = []
    for params in sim_params:
        max_pos_variance = params['positive_mean'] * (1 - params['positive_mean'])
        max_neg_variance = params['negative_mean'] * (1 - params['negative_mean'])

        # Adjust positive variance
        if params['positive_variance'] >= max_pos_variance:
            print(f'changed positive variance from {params["positive_variance"]} to {max_pos_variance * 0.99}')
            params['positive_variance'] = max_pos_variance * 0.99  # Adjust to 99% of the maximum allowed variance

        # Adjust negative variance
        if params['negative_variance'] >= max_neg_variance:
            print(f'changed negative variance from {params["negative_variance"]} to {max_neg_variance * 0.99}')
            params['negative_variance'] = max_neg_variance * 0.99  # Adjust to 99% of the maximum allowed variance

        adjusted_params.append(params)
        
    return adjusted_params

def generate_paramiters(settings):
    """Expand a sweep-settings dict into one settings dict per (Cartesian) simulation.

    :param settings: Config dict where each swept key holds an iterable of values.
    :returns: Shuffled list of per-run settings dicts, already run through
        :func:`validate_and_adjust_beta_params`.
    """
    
    settings['positive_mean'] = [0.8]

    sim_ls = []
    for avg_genes_per_well in settings['avg_genes_per_well']:
        replicates = settings['replicates']
        for avg_cells_per_well in settings['avg_cells_per_well']:
            for classifier_accuracy in settings['classifier_accuracy']:
                for positive_mean in settings['positive_mean']:
                    for avg_reads_per_gene in settings['avg_reads_per_gene']:
                        for sequencing_error in settings['sequencing_error']:
                            for well_ineq_coeff in settings['well_ineq_coeff']:
                                for gene_ineq_coeff in settings['gene_ineq_coeff']:
                                    for nr_plates in settings['nr_plates']:
                                        for number_of_genes in settings['number_of_genes']:
                                            for number_of_active_genes in settings['number_of_active_genes']:
                                                for i in range(1, replicates+1):
                                                    sett = deepcopy(settings)
                                                    sett['avg_genes_per_well'] = avg_genes_per_well
                                                    sett['sd_genes_per_well'] = avg_genes_per_well / 2
                                                    sett['avg_cells_per_well'] = avg_cells_per_well
                                                    sett['sd_cells_per_well'] = avg_cells_per_well / 2
                                                    sett['classifier_accuracy'] = classifier_accuracy
                                                    sett['positive_mean'] = positive_mean
                                                    sett['negative_mean'] = 1-positive_mean
                                                    sett['positive_variance'] = (1-positive_mean)/2
                                                    sett['negative_variance'] = (1-positive_mean)/2
                                                    sett['avg_reads_per_gene'] = avg_reads_per_gene
                                                    sett['sd_reads_per_gene'] = avg_reads_per_gene / 2
                                                    sett['sequencing_error'] = sequencing_error
                                                    sett['well_ineq_coeff'] = well_ineq_coeff
                                                    sett['gene_ineq_coeff'] = gene_ineq_coeff
                                                    sett['nr_plates'] = nr_plates
                                                    sett['number_of_genes'] = number_of_genes
                                                    sett['number_of_active_genes'] = number_of_active_genes
                                                    sim_ls.append(sett)

    random.shuffle(sim_ls)
    sim_ls = validate_and_adjust_beta_params(sim_ls)
    print(f'Running {len(sim_ls)} simulations.')
    #for x in sim_ls: 
    #    print(x['positive_mean'])
    return sim_ls

def run_multiple_simulations(settings):
    """Fan out the sweep from :func:`generate_paramiters` across a process pool.

    Uses a ``multiprocessing.Pool`` with ``max_workers`` (or ``cpu_count()-4``)
    workers, prints a progress line, and drives each worker through
    :func:`run_and_save`.

    :param settings: Sweep-settings dict. Must include ``max_workers``.
    :returns: None.
    """

    now = datetime.now() # get current date
    start_time = now.strftime("%y%m%d") # format as a string in 'ddmmyy' format 
    settings['start_time'] = start_time

    sim_ls = generate_paramiters(settings)
    #print(f'Running {len(sim_ls)} simulations.')

    max_workers = settings['max_workers'] or cpu_count() - 4
    with Manager() as manager:
        time_ls = manager.list()
        total_sims = len(sim_ls)
        with Pool(max_workers) as pool:
            result = pool.starmap_async(run_and_save, [(index, settings, time_ls, total_sims) for index, settings in enumerate(sim_ls)])
            while not result.ready():
                try:
                    sleep(0.01)
                    sims_processed = len(time_ls)
                    average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                    time_left = (((total_sims - sims_processed) * average_time) / max_workers) / 60
                    print(f'Progress: {sims_processed}/{total_sims} Time/simulation {average_time:.3f}sec Time Remaining {time_left:.3f} min.', end='\r', flush=True)
                    gc.collect()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
            try:
                result.get()
            except Exception as e:
                print(e)
                print(traceback.format_exc())
            
def generate_integers(start, stop, step):
    """Return ``list(range(start, stop + 1, step))`` (inclusive upper bound)."""
    return list(range(start, stop + 1, step))

def generate_floats(start, stop, step):
    """Return an inclusive list of floats from ``start`` to ``stop`` with ``step`` spacing."""
    # Determine the number of decimal places in 'step'
    num_decimals = str(step)[::-1].find('.')
    
    current = start
    floats_list = []
    while current <= stop:
        # Round each float to the appropriate number of decimal places
        floats_list.append(round(current, num_decimals))
        current += step
    
    return floats_list

def remove_columns_with_single_value(df):
    """Return ``df`` without columns whose values are constant across rows.

    :param df: Source DataFrame.
    :returns: Copy of ``df`` with zero-variance columns dropped.
    """
    to_drop = [column for column in df.columns if df[column].nunique() == 1]
    return df.drop(to_drop, axis=1)

def read_simulations_table(db_path):
    """Return the ``simulations`` table from ``db_path`` as a DataFrame.

    :param db_path: Path to a SQLite database written by :func:`save_data`.
    :returns: DataFrame of the ``simulations`` table, or ``None`` on failure.
    """
    # Create a connection object using the connect function
    conn = sqlite3.connect(db_path)
    
    # Read the 'simulations' table into a pandas DataFrame
    try:
        df = pd.read_sql_query("SELECT * FROM simulations", conn)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Close the connection to SQLite database
        conn.close()
    
    return df

def plot_simulations(df, variable, x_rotation=None, legend=False, grid=False, clean=True, verbose=False):
    """Grid-plot PR-AUC vs ``variable`` for every unique combination of the other sweep dimensions.

    :param df: DataFrame containing ``prauc``, ``variable`` and the standard
        grouping columns (``number_of_active_genes``, ``avg_reads_per_gene``, ...).
    :param variable: Column plotted on the x-axis of each subplot.
    :param x_rotation: Degrees to rotate x-tick labels. ``None`` uses 45.
    :param legend: When True, show the per-subplot legend.
    :param grid: When True, draw grid lines.
    :param clean: When True, drop grouping columns whose values never vary.
    :param verbose: When True, annotate each subplot with its filter conditions.
    :returns: The generated Matplotlib figure.
    """
    
    grouping_vars = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                     'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                     'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    if clean:
        relevant_data = remove_columns_with_single_value(relevant_data)
    
    grouping_vars = [col for col in grouping_vars if col != variable]
    
    # Check if the necessary columns are present in the DataFrame
    required_columns = {variable, 'prauc'} | set(grouping_vars)
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"DataFrame must contain {missing_cols} columns")
        
    #if not dependent is None:
    
    # Get unique combinations of conditions from grouping_vars
    unique_combinations = df[grouping_vars].drop_duplicates()
    num_combinations = len(unique_combinations)

    # Determine the layout of the subplots
    num_rows = math.ceil(np.sqrt(num_combinations))
    num_cols = math.ceil(num_combinations / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows * num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (ax, (_, row)) in enumerate(zip(axes, unique_combinations.iterrows())):

        # Filter the DataFrame for the current combination of variables
        condition = {var: row[var] for var in grouping_vars}
        subset_df = df[df[grouping_vars].eq(row).all(axis=1)]
        
        # Group by 'variable' and calculate mean and std dev of 'prauc'
        grouped = subset_df.groupby(variable)['prauc'].agg(['mean', 'std'])
        grouped = grouped.sort_index()  # Sort by the variable for orderly plots

        # Plotting the mean of 'prauc' with std deviation as shaded area
        ax.plot(grouped.index, grouped['mean'], marker='o', linestyle='-', color='b', label='Mean PRAUC')
        ax.fill_between(grouped.index, grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], color='gray', alpha=0.5, label='Std Dev')

        # Setting plot labels and title
        title_details = ', '.join([f"{var}={row[var]}" for var in grouping_vars])
        ax.set_xlabel(variable)
        ax.set_ylabel('Precision-Recall AUC (PRAUC)')
        #ax.set_title(f'PRAUC vs. {variable} | {title_details}')
        ax.grid(grid)

        if legend:
            ax.legend()

        # Set x-ticks and rotate them as specified
        ax.set_xticks(grouped.index)
        ax.set_xticklabels(grouped.index, rotation=x_rotation if x_rotation is not None else 45)
        
        if verbose:
            verbose_text = '\n'.join([f"{var}: {val}" for var, val in condition.items()])
            ax.text(0.95, 0.05, verbose_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Hide any unused axes if there are any
    for ax in axes[idx+1:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
    
def plot_correlation_matrix(df, annot=False, cmap='inferno', clean=True):
    """Render a lower-triangular correlation heatmap of the standard sweep + metric columns.

    :param df: DataFrame containing sweep variables plus ``prauc``, ``roc_auc``
        and related outputs.
    :param annot: When True, write numeric correlations in each cell.
    :param cmap: Colormap name or object (overridden internally to a diverging
        palette).
    :param clean: When True, drop constant columns before computing correlations.
    :returns: The generated Matplotlib figure.
    """
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    grouping_vars = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                     'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                     'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    grouping_vars = grouping_vars + ['optimal_threshold', 'accuracy', 'prauc', 'roc_auc','genes_per_well_gini', 'wells_per_gene_gini']
    # 'inactive_mean', 'inactive_std', 'inactive_var', 'active_mean', 'active_std', 'inactive_var', 'cutoff', 'TP', 'FP', 'TN', 'FN', 

    if clean:
        df = remove_constant_columns(df)
        grouping_vars = [feature for feature in grouping_vars if feature in df.columns]

    # Subsetting the DataFrame to include only the relevant variables
    relevant_data = df[grouping_vars]
    
    if clean:
        relevant_data = remove_columns_with_single_value(relevant_data)
        
    # Calculating the correlation matrix
    corr_matrix = relevant_data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plotting the correlation matrix
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap=cmap, fmt=".2f", linewidths=.5, robust=True)
    #plt.title('Correlation Matrix with Heatmap')

    plt.tight_layout()
    plt.show()
    save_plot(fig, src='figures', variable='correlation_matrix', i=1)
    return fig

def plot_feature_importance(df, target='prauc', exclude=None, clean=True):
    """Train a RandomForestRegressor on sweep variables and plot the resulting importances.

    :param df: DataFrame with sweep columns and ``target``.
    :param target: Column predicted by the regressor. Default ``'prauc'``.
    :param exclude: Column name or list of columns to remove from the feature set.
    :param clean: When True, drop constant columns before fitting.
    :returns: The generated Matplotlib figure.
    """
    
    # Define the features for the model
    features = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                     'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                     'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    if clean:
        df = remove_columns_with_single_value(df)
        features = [feature for feature in features if feature in df.columns]
    
    # Remove excluded features if specified
    if isinstance(exclude, list):
        features = [feature for feature in features if feature not in exclude]
    elif exclude is not None:
        features = [feature for feature in features if feature != exclude]
    
    # Train the model
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(df[features], df[target])
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot horizontal bar chart
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(indices)), importances[indices], color="teal", align="center", alpha=0.6)
    plt.yticks(range(len(indices)), [features[i] for i in indices[::-1]])  # Invert y-axis to match the order
    plt.gca().invert_yaxis()  # Invert the axis to have the highest importance at the top
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    save_plot(fig, src='figures', variable='feature_importance', i=1)
    return fig

def calculate_permutation_importance(df, target='prauc', exclude=None, n_repeats=10, clean=True):
    """Fit a RandomForest and plot permutation-based feature importances for the sweep columns.

    :param df: DataFrame with sweep columns and ``target``.
    :param target: Column predicted by the regressor. Default ``'prauc'``.
    :param exclude: Column name or list of columns to remove from the feature set.
    :param n_repeats: Number of permutations per feature. Default ``10``.
    :param clean: When True, drop constant columns before fitting.
    :returns: The generated Matplotlib figure.
    """
    
    features = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    if clean:
        df = remove_columns_with_single_value(df)
        features = [feature for feature in features if feature in df.columns]
    
    if isinstance(exclude, list):
        for ex in exclude:
            features.remove(ex)
    if not exclude is None:
        features.remove(exclude)
    
    X = df[features]
    y = df[target]

    # Initialize a model (you could pass it as an argument if you'd like to use a different one)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)

    # Plotting
    sorted_idx = perm_importance.importances_mean.argsort()
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], color="teal", align="center", alpha=0.6)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([df.columns[i] for i in sorted_idx])
    ax.set_xlabel('Permutation Importance')
    plt.tight_layout()
    plt.show()
    save_plot(fig, src='figures', variable='permutation_importance', i=1)
    return fig
    
def plot_partial_dependences(df, target='prauc', clean=True):
    """Fit a GradientBoostingRegressor and plot partial dependences for every sweep feature.

    :param df: DataFrame with sweep columns and ``target``.
    :param target: Column predicted by the regressor. Default ``'prauc'``.
    :param clean: When True, drop constant columns before fitting.
    :returns: The generated Matplotlib figure.
    """
    
    features = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    if clean:
        df = remove_columns_with_single_value(df)
        features = [feature for feature in features if feature in df.columns]

    X = df[features]
    y = df[target]
    
    # Train a model
    model = GradientBoostingRegressor()
    model.fit(X, y)
    
    # Determine the number of rows and columns for subplots
    n_cols = 4  # Number of columns in subplot grid
    n_rows = (len(features) + n_cols - 1) // n_cols  # Calculate rows needed
    
    # Plot partial dependence
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle('Partial Dependence Plots', fontsize=20, y=1.03)
    
    # Flatten the array of axes if it's multidimensional
    axs = axs.flatten() if n_rows > 1 else [axs]
    
    for i, feature in enumerate(features):
        ax = axs[i]
        disp = PartialDependenceDisplay.from_estimator(model, X, features=[feature], ax=ax)
        ax.set_title(feature)  # Set title to the name of the feature

    # Hide unused axes if any
    for ax in axs[len(features):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    save_plot(fig, src='figures', variable='partial_dependences', i=1)
    return fig

def save_shap_plot(fig, src, variable, i):
    """Save a SHAP figure to ``<src>/<variable>/<i>_figure.pdf``."""
    import os
    os.makedirs(f'{src}/{variable}', exist_ok=True)
    filename_fig = f'{src}/{variable}/{str(i)}_figure.pdf'
    fig.savefig(filename_fig, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved figure as {filename_fig}")

def generate_shap_summary_plot(df,target='prauc', clean=True):
    """Fit a RandomForest and render a SHAP summary plot over the standard sweep features.

    :param df: DataFrame with sweep columns and ``target``.
    :param target: Column predicted by the regressor. Default ``'prauc'``.
    :param clean: When True, drop constant columns before fitting.
    :returns: The current Matplotlib figure (SHAP creates it as a side effect).
    """
    
    features = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
                'classifier_accuracy', 'nr_plates', 'number_of_genes', 'avg_genes_per_well',
                'avg_cells_per_well', 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']
    
    if clean:
        df = remove_columns_with_single_value(df)
        features = [feature for feature in features if feature in df.columns]

    X = df[features]
    y = df[target]

    # Initialize a model (you could pass it as an argument if you'd like to use a different one)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    shap.summary_plot(shap_values, X)
    save_shap_plot(plt.gcf(), src='figures', variable='shap', i=1)
    #save_shap_plot(fig, src, variable, i)
    return plt.gcf()


def remove_constant_columns(df):
    """Return ``df`` limited to columns that contain more than one unique value.

    :param df: Source DataFrame.
    :returns: Copy of ``df`` with constant columns dropped.
    """
    return df.loc[:, df.nunique() > 1]


# to justify using beta for sim classifier

# Fit a Beta distribution to these outputs
#a, b, loc, scale = beta.fit(predicted_probs, floc=0, fscale=1)  # Fix location and scale to match the support of the sigmoid

# Sample from this fitted Beta distribution
#simulated_probs = beta.rvs(a, b, size=1000)

# Plot the empirical vs simulated distribution
#plt.hist(predicted_probs, bins=30, alpha=0.5, label='Empirical')
#plt.hist(simulated_probs, bins=30, alpha=0.5, label='Simulated from Beta')
#plt.legend()
#plt.show()