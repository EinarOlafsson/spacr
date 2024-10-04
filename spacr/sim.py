
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
    """
    Generates a list of randomly selected genes.

    Args:
        number_of_genes (int): The number of genes to be selected.
        number_of_all_genes (int): The total number of genes available.

    Returns:
        list: A list of randomly selected genes.
    """
    genes_ls = list(range(number_of_all_genes))
    random.shuffle(genes_ls)
    gene_list = genes_ls[:number_of_genes]
    return gene_list

# plate_map is a table with a row for each well, containing well metadata: plate_id, row_id, and column_id
def generate_plate_map(nr_plates):
    #print('nr_plates',nr_plates)
    """
    Generate a plate map based on the number of plates.

    Parameters:
    nr_plates (int): The number of plates to generate the map for.

    Returns:
    pandas.DataFrame: The generated plate map dataframe.
    """
    plate_row_column = [f"{i+1}_{ir+1}_{ic+1}" for i in range(nr_plates) for ir in range(16) for ic in range(24)]
    df= pd.DataFrame({'plate_row_column': plate_row_column})
    df["plate_id"], df["row_id"], df["column_id"] = zip(*[r.split("_") for r in df['plate_row_column']])
    return df

def gini_coefficient(x):
    """
    Compute Gini coefficient of array of values.

    Parameters:
    x (array-like): Array of values.

    Returns:
    float: Gini coefficient.

    """
    diffsum = np.sum(np.abs(np.subtract.outer(x, x)))
    return diffsum / (2 * len(x) ** 2 * np.mean(x))

def gini(x):
    """
    Calculate the Gini coefficient for a given array of values.

    Parameters:
    x (array-like): Input array of values.

    Returns:
    float: The Gini coefficient.

    Notes:
    This implementation has a time and memory complexity of O(n**2), where n is the length of x.
    Avoid passing in large samples to prevent performance issues.
    """
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def gini_gene_well(x):
    """
    Calculate the Gini coefficient for a given income distribution.

    The Gini coefficient measures income inequality in a population.
    A value of 0 represents perfect income equality (everyone has the same income),
    while a value of 1 represents perfect income inequality (one individual has all the income).

    Parameters:
    x (array-like): An array-like object representing the income distribution.

    Returns:
    float: The Gini coefficient for the given income distribution.
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def gini(x):
    """
    Calculate the Gini coefficient for a given array of values.

    Parameters:
    x (array-like): The input array of values.

    Returns:
    float: The Gini coefficient.

    References:
    - Based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    - From: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    - All values are treated equally, arrays must be 1d.
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x))  # ranks of x
    return 1 - (2 * (r * x).sum() + s) / (n * s)

def dist_gen(mean, sd, df):
    """
    Generate a Poisson distribution based on a gamma distribution.

    Parameters:
    mean (float): Mean of the gamma distribution.
    sd (float): Standard deviation of the gamma distribution.
    df (pandas.DataFrame): Input data.

    Returns:
    tuple: A tuple containing the generated Poisson distribution and the length of the input data.
    """
    length = len(df)
    shape = (mean / sd) ** 2  # Calculate shape parameter
    scale = (sd ** 2) / mean  # Calculate scale parameter
    rate = np.random.gamma(shape, scale, size=length)  # Generate random rate from gamma distribution
    data = np.random.poisson(rate)  # Use the random rate for a Poisson distribution
    return data, length

def generate_gene_weights(positive_mean, positive_variance, df):
    """
    Generate gene weights using a beta distribution.

    Parameters:
    - positive_mean (float): The mean value for the positive distribution.
    - positive_variance (float): The variance value for the positive distribution.
    - df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    - weights (numpy.ndarray): An array of gene weights generated using a beta distribution.
    """
    # alpha and beta for positive distribution
    a1 = positive_mean*(positive_mean*(1-positive_mean)/positive_variance - 1)
    b1 = a1*(1-positive_mean)/positive_mean
    weights = np.random.beta(a1, b1, len(df))
    return weights

def normalize_array(arr):
    """
    Normalize an array by scaling its values between 0 and 1.

    Parameters:
    arr (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array.

    """
    min_value = np.min(arr)
    max_value = np.max(arr)
    normalized_arr = (arr - min_value) / (max_value - min_value)
    return normalized_arr

def generate_power_law_distribution(num_elements, coeff):
    """
    Generate a power law distribution.

    Parameters:
    - num_elements (int): The number of elements in the distribution.
    - coeff (float): The coefficient of the power law.

    Returns:
    - normalized_distribution (ndarray): The normalized power law distribution.
    """
    base_distribution = np.arange(1, num_elements + 1)
    powered_distribution = base_distribution ** -coeff
    normalized_distribution = powered_distribution / np.sum(powered_distribution)
    return normalized_distribution

# distribution generator function
def power_law_dist_gen(df, avg, well_ineq_coeff):
    """
    Generate a power-law distribution for wells.

    Parameters:
    - df: DataFrame: The input DataFrame containing the wells.
    - avg: float: The average value for the distribution.
    - well_ineq_coeff: float: The inequality coefficient for the power-law distribution.

    Returns:
    - dist: ndarray: The generated power-law distribution for the wells.
    """
    # Generate a power-law distribution for wells
    distribution = generate_power_law_distribution(len(df), well_ineq_coeff)
    dist = np.random.choice(distribution, len(df)) * avg
    return dist

# plates is a table with for each cell in the experiment with columns [plate_id, row_id, column_id, gene_id, is_active]
def run_experiment(plate_map, number_of_genes, active_gene_list, avg_genes_per_well, sd_genes_per_well, avg_cells_per_well, sd_cells_per_well, well_ineq_coeff, gene_ineq_coeff):
    """
    Run a simulation experiment.

    Args:
        plate_map (DataFrame): The plate map containing information about the wells.
        number_of_genes (int): The total number of genes.
        active_gene_list (list): The list of active genes.
        avg_genes_per_well (float): The average number of genes per well.
        sd_genes_per_well (float): The standard deviation of genes per well.
        avg_cells_per_well (float): The average number of cells per well.
        sd_cells_per_well (float): The standard deviation of cells per well.
        well_ineq_coeff (float): The coefficient for well inequality.
        gene_ineq_coeff (float): The coefficient for gene inequality.

    Returns:
        tuple: A tuple containing the following:
            - cell_df (DataFrame): The DataFrame containing information about the cells.
            - genes_per_well_df (DataFrame): The DataFrame containing gene counts per well.
            - wells_per_gene_df (DataFrame): The DataFrame containing well counts per gene.
            - df_ls (list): A list containing gene counts per well, well counts per gene, Gini coefficients for wells,
              Gini coefficients for genes, gene weights array, and well weights.
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
    """
    Classifies the data in the DataFrame based on the given parameters and a classifier error rate.

    Args:
        positive_mean (float): The mean of the positive distribution.
        positive_variance (float): The variance of the positive distribution.
        negative_mean (float): The mean of the negative distribution.
        negative_variance (float): The variance of the negative distribution.
        classifier_accuracy (float): The likelihood (0 to 1) that a gene is correctly classified according to its true label.
        df (pandas.DataFrame): The DataFrame containing the data to be classified.

    Returns:
        pandas.DataFrame: The DataFrame with an additional 'score' column containing the classification scores.
    """
    def calc_alpha_beta(mean, variance):
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
        if np.random.rand() < classifier_accuracy:  # With classifier_accuracy probability, choose the correct distribution
            return np.random.beta(a1, b1) if is_active else np.random.beta(a2, b2)
        else:  # With 1-classifier_accuracy probability, choose the incorrect distribution
            return np.random.beta(a2, b2) if is_active else np.random.beta(a1, b1)

    # Calculate alpha and beta for both distributions
    a1, b1 = calc_alpha_beta(positive_mean, positive_variance)
    a2, b2 = calc_alpha_beta(negative_mean, negative_variance)
    df['score'] = df['is_active'].apply(get_score)

    return df

def classifier_v2(positive_mean, positive_variance, negative_mean, negative_variance, df):
    """
    Classifies the data in the DataFrame based on the given parameters.

    Args:
        positive_mean (float): The mean of the positive distribution.
        positive_variance (float): The variance of the positive distribution.
        negative_mean (float): The mean of the negative distribution.
        negative_variance (float): The variance of the negative distribution.
        df (pandas.DataFrame): The DataFrame containing the data to be classified.

    Returns:
        pandas.DataFrame: The DataFrame with an additional 'score' column containing the classification scores.
    """
    def calc_alpha_beta(mean, variance):
        if mean <= 0 or mean >= 1:
            raise ValueError("Mean must be between 0 and 1 exclusively.")
        max_variance = mean * (1 - mean)
        if variance <= 0 or variance >= max_variance:
            raise ValueError(f"Variance must be positive and less than {max_variance}.")
        
        alpha = mean * (mean * (1 - mean) / variance - 1)
        beta = alpha * (1 - mean) / mean
        return alpha, beta

    # Calculate alpha and beta for both distributions
    a1, b1 = calc_alpha_beta(positive_mean, positive_variance)
    a2, b2 = calc_alpha_beta(negative_mean, negative_variance)

    # Apply the beta distribution based on 'is_active' status
    df['score'] = df['is_active'].apply(lambda is_active: np.random.beta(a1, b1) if is_active else np.random.beta(a2, b2))
    return df

def compute_roc_auc(cell_scores):
    """
    Compute the Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) for cell scores.

    Parameters:
    - cell_scores (DataFrame): DataFrame containing cell scores with columns 'is_active' and 'score'.

    Returns:
    - cell_roc_dict (dict): Dictionary containing the ROC curve information, including the threshold, true positive rate (TPR), false positive rate (FPR), and ROC AUC.

    """
    fpr, tpr, thresh = roc_curve(cell_scores['is_active'], cell_scores['score'], pos_label=1)
    roc_auc = auc(fpr, tpr)
    cell_roc_dict = {'threshold':thresh,'tpr': tpr,'fpr': fpr, 'roc_auc':roc_auc}
    return cell_roc_dict

def compute_precision_recall(cell_scores):
    """
    Compute precision, recall, F1 score, and PR AUC for a given set of cell scores.

    Parameters:
    - cell_scores (DataFrame): A DataFrame containing the cell scores with columns 'is_active' and 'score'.

    Returns:
    - cell_pr_dict (dict): A dictionary containing the computed precision, recall, F1 score, PR AUC, and threshold values.
    """
    pr, re, th = precision_recall_curve(cell_scores['is_active'], cell_scores['score'])
    th = np.insert(th, 0, 0)
    f1_score = 2 * (pr * re) / (pr + re)
    pr_auc = auc(re, pr)
    cell_pr_dict = {'threshold':th,'precision': pr,'recall': re, 'f1_score':f1_score, 'pr_auc': pr_auc}
    return cell_pr_dict

def get_optimum_threshold(cell_pr_dict):
    """
    Calculates the optimum threshold based on the f1_score in the given cell_pr_dict.

    Parameters:
    cell_pr_dict (dict): A dictionary containing precision, recall, and f1_score values for different thresholds.

    Returns:
    float: The optimum threshold value.
    """
    cell_pr_dict_df = pd.DataFrame(cell_pr_dict)
    max_x = cell_pr_dict_df.loc[cell_pr_dict_df['f1_score'].idxmax()]
    optimum = float(max_x['threshold'])
    return optimum

def update_scores_and_get_cm(cell_scores, optimum):
    """
    Update the cell scores based on the given optimum value and calculate the confusion matrix.

    Args:
        cell_scores (DataFrame): The DataFrame containing the cell scores.
        optimum (float): The optimum value used for updating the scores.

    Returns:
        tuple: A tuple containing the updated cell scores DataFrame and the confusion matrix.
    """
    cell_scores[optimum] = cell_scores.score.map(lambda x: 1 if x >= optimum else 0)
    cell_cm = metrics.confusion_matrix(cell_scores.is_active, cell_scores[optimum])
    return cell_scores, cell_cm

def cell_level_roc_auc(cell_scores):
    """
    Compute the ROC AUC and precision-recall metrics at the cell level.

    Args:
        cell_scores (list): List of scores for each cell.

    Returns:
        cell_roc_dict_df (DataFrame): DataFrame containing the ROC AUC metrics for each cell.
        cell_pr_dict_df (DataFrame): DataFrame containing the precision-recall metrics for each cell.
        cell_scores (list): Updated list of scores after applying the optimum threshold.
        cell_cm (array): Confusion matrix for the cell-level classification.
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
    """
    Generate well scores based on cell scores.

    Args:
        cell_scores (DataFrame): DataFrame containing cell scores.

    Returns:
        DataFrame: DataFrame containing well scores with average active score, gene list, and score.

    """
    # Compute mean and list of unique gene_ids
    well_score = cell_scores.groupby(['plate_row_column']).agg(
        average_active_score=('is_active', 'mean'),
        gene_list=('gene_id', lambda x: np.unique(x).tolist()))
    well_score['score'] = np.log10(well_score['average_active_score'] + 1)
    return well_score

def sequence_plates(well_score, number_of_genes, avg_reads_per_gene, sd_reads_per_gene, sequencing_error=0.01):

    """
    Simulates the sequencing of plates and calculates gene fractions and metadata.

    Parameters:
    well_score (pd.DataFrame): DataFrame containing well scores and gene lists.
    number_of_genes (int): Number of genes.
    avg_reads_per_gene (float): Average number of reads per gene.
    sd_reads_per_gene (float): Standard deviation of reads per gene.
    sequencing_error (float, optional): Probability of introducing sequencing error. Defaults to 0.01.

    Returns:
    gene_fraction_map (pd.DataFrame): DataFrame containing gene fractions for each well.
    metadata (pd.DataFrame): DataFrame containing metadata for each well.
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
    """
    Calculate regression ROC AUC and other statistics.

    Parameters:
    results_df (DataFrame): DataFrame containing the results of regression analysis.
    active_gene_list (list): List of active gene IDs.
    control_gene_list (list): List of control gene IDs.
    alpha (float, optional): Significance level for determining hits. Default is 0.05.
    optimal (bool, optional): Whether to use the optimal threshold for classification. Default is False.

    Returns:
    tuple: A tuple containing the following:
    - results_df (DataFrame): Updated DataFrame with additional columns.
    - reg_roc_dict_df (DataFrame): DataFrame containing regression ROC curve data.
    - reg_pr_dict_df (DataFrame): DataFrame containing precision-recall curve data.
    - reg_cm (ndarray): Confusion matrix.
    - sim_stats (DataFrame): DataFrame containing simulation statistics.
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
    """
    Plots a histogram of the given data.

    Parameters:
    - data: The data to be plotted.
    - x_label: The label for the x-axis.
    - ax: The matplotlib axis object to plot on.
    - color: The color of the histogram bars.
    - title: The title of the plot.
    - binwidth: The width of each histogram bin.
    - log: Whether to use a logarithmic scale for the y-axis.

    Returns:
    None
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
    """
    Plot the ROC (Receiver Operating Characteristic) and PR (Precision-Recall) curves.

    Parameters:
    - data: DataFrame containing the data to be plotted.
    - ax: The matplotlib axes object to plot on.
    - title: The title of the plot.
    - x_label: The label for the x-axis.
    - y_label: The label for the y-axis.
    """
    ax.plot(data[x_label], data[y_label], color='black', lw=0.5)
    ax.plot([0, 1], [0, 1], color='black', lw=0.5, linestyle="--", label='random classifier')
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(loc="lower right")

def plot_confusion_matrix(data, ax, title):
    """
    Plots a confusion matrix using a heatmap.

    Parameters:
    data (numpy.ndarray): The confusion matrix data.
    ax (matplotlib.axes.Axes): The axes object to plot the heatmap on.
    title (str): The title of the plot.

    Returns:
    None
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
    """
    Run the simulation based on the given settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        tuple: A tuple containing the simulation results and distances.
        - cell_scores (DataFrame): Scores for each cell.
        - cell_roc_dict_df (DataFrame): ROC AUC scores for each cell.
        - cell_pr_dict_df (DataFrame): Precision-Recall AUC scores for each cell.
        - cell_cm (DataFrame): Confusion matrix for each cell.
        - well_score (DataFrame): Scores for each well.
        - gene_fraction_map (DataFrame): Fraction of genes for each well.
        - metadata (DataFrame): Metadata for each well.
        - results_df (DataFrame): Results of the regression analysis.
        - reg_roc_dict_df (DataFrame): ROC AUC scores for each gene.
        - reg_pr_dict_df (DataFrame): Precision-Recall AUC scores for each gene.
        - reg_cm (DataFrame): Confusion matrix for each gene.
        - sim_stats (dict): Additional simulation statistics.
        - genes_per_well_df (DataFrame): Number of genes per well.
        - wells_per_gene_df (DataFrame): Number of wells per gene.
        dists (list): List of distances.
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
    """
    Visualizes the distributions of given distances.

    Args:
        dists (list): List of distance arrays.
        src (str): Source directory for saving the plot.
        v (int): Number of vertices.
        i (int): Index of the plot.

    Returns:
        None
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
    """
    Visualizes various plots based on the given output data.

    Args:
        output (list): A list containing the following elements:
            - cell_scores (DataFrame): DataFrame containing cell scores.
            - cell_roc_dict_df (DataFrame): DataFrame containing ROC curve data for cell classification.
            - cell_pr_dict_df (DataFrame): DataFrame containing precision-recall curve data for cell classification.
            - cell_cm (array-like): Confusion matrix for cell classification.
            - well_score (DataFrame): DataFrame containing well scores.
            - gene_fraction_map (dict): Dictionary mapping genes to fractions.
            - metadata (dict): Dictionary containing metadata.
            - results_df (DataFrame): DataFrame containing results.
            - reg_roc_dict_df (DataFrame): DataFrame containing ROC curve data for gene regression.
            - reg_pr_dict_df (DataFrame): DataFrame containing precision-recall curve data for gene regression.
            - reg_cm (array-like): Confusion matrix for gene regression.
            - sim_stats (dict): Dictionary containing simulation statistics.
            - genes_per_well_df (DataFrame): DataFrame containing genes per well data.
            - wells_per_gene_df (DataFrame): DataFrame containing wells per gene data.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
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
    """
    Creates a SQLite database at the specified path.

    Args:
        db_path (str): The path where the database should be created.

    Returns:
        None
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
    """
    Append a pandas DataFrame to an SQLite database table.

    Parameters:
    src (str): The source directory where the database file is located.
    table (pandas.DataFrame): The DataFrame to be appended to the database table.
    table_name (str): The name of the database table.

    Returns:
    None
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
    """
    Save simulation data to specified location.

    Args:
        src (str): The directory path where the data will be saved.
        output (list): A list of dataframes containing simulation output.
        settings (dict): A dictionary containing simulation settings.
        save_all (bool, optional): Flag indicating whether to save all tables or only a subset. Defaults to False.
        i (int, optional): The simulation number. Defaults to 0.
        variable (str, optional): The variable name. Defaults to 'all'.

    Returns:
        None
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
    """
    Save a matplotlib figure as a PDF file.

    Parameters:
    - fig: The matplotlib figure to be saved.
    - src: The directory where the file will be saved.
    - variable: The name of the variable being plotted.
    - i: The index of the figure.

    Returns:
    None
    """
    os.makedirs(f'{src}/{variable}', exist_ok=True)
    filename_fig = f'{src}/{variable}/{str(i)}_figure.pdf'
    fig.savefig(filename_fig, dpi=600, format='pdf', bbox_inches='tight')
    return
    
def run_and_save(i, settings, time_ls, total_sims):
    
    """
    Run the simulation and save the results.

    Args:
        i (int): The simulation index.
        settings (dict): The simulation settings.
        time_ls (list): The list to store simulation times.
        total_sims (int): The total number of simulations.

    Returns:
        tuple: A tuple containing the simulation index, simulation time, and None.
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
    """
    Validates and adjusts Beta distribution parameters in simulation settings to ensure they are possible.
    
    Args:
    sim_params (list of dict): List of dictionaries, each containing the simulation parameters.
    
    Returns:
    list of dict: The adjusted list of simulation parameter sets.
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

    """
    Generate a list of parameter sets for simulation based on the given settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        list: A list of parameter sets for simulation.
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

    """
    Run multiple simulations in parallel using the provided settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        None
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
    return list(range(start, stop + 1, step))

def generate_floats(start, stop, step):
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
    """
    Removes columns from the DataFrame that have the same value in all rows.

    Args:
    df (pandas.DataFrame): The original DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame with the columns removed that contained only one unique value.
    """
    to_drop = [column for column in df.columns if df[column].nunique() == 1]
    return df.drop(to_drop, axis=1)

def read_simulations_table(db_path):
    """
    Reads the 'simulations' table from an SQLite database into a pandas DataFrame.
    
    Args:
    db_path (str): The file path to the SQLite database.
    
    Returns:
    pandas.DataFrame: DataFrame containing the 'simulations' table data.
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
    
    """
    Creates separate line plots for 'prauc' against a specified 'variable', 
    for each unique combination of conditions defined by 'grouping_vars', displayed on a grid.

    Args:
    df (pandas.DataFrame): DataFrame containing the necessary columns.
    variable (str): Name of the column to use as the x-axis for grouping and plotting.
    x_rotation (int, optional): Degrees to rotate the x-axis labels.
    legend (bool, optional): Whether to display a legend.
    grid (bool, optional): Whether to display grid lines.
    verbose (bool, optional): Whether to print the filter conditions.

    Returns:
    None
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
    """
    Plots a correlation matrix for the specified variables and the target variable.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    variables (list): List of column names to include in the correlation matrix.
    target_variable (str): The target variable column name.

    Returns:
    None
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
    """
    Trains a RandomForestRegressor to determine the importance of each feature in predicting the target.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    target (str): The target variable column name.
    exclude (list or str, optional): Column names to exclude from features.

    Returns:
    matplotlib.figure.Figure: The figure object containing the feature importance plot.
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
    """
    Calculates permutation importance for the given features in the dataframe.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    features (list): List of column names to include as features.
    target (str): The name of the target variable column.

    Returns:
    dict: Dictionary containing the importances and standard deviations.
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
    
    """
    Creates partial dependence plots for the specified features, with improved layout to avoid text overlap.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    target (str): The target variable.

    Returns:
    None
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
    import os
    os.makedirs(f'{src}/{variable}', exist_ok=True)
    filename_fig = f'{src}/{variable}/{str(i)}_figure.pdf'
    fig.savefig(filename_fig, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved figure as {filename_fig}")

def generate_shap_summary_plot(df,target='prauc', clean=True):
    """
    Generates a SHAP summary plot for the given features in the dataframe.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    features (list): List of column names to include as features.
    target (str): The name of the target variable column.

    Returns:
    None
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
    """
    Removes columns in the DataFrame where all entries have the same value.

    Parameters:
    df (pd.DataFrame): The input DataFrame from which to remove constant columns.

    Returns:
    pd.DataFrame: A DataFrame with the constant columns removed.
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