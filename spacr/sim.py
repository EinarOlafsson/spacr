
import os, gc, random, warnings, traceback, itertools, matplotlib, sqlite3
import time as tm
from time import time, sleep
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve
import statsmodels.api as sm
from multiprocessing import cpu_count, Value, Array, Lock, Pool, Manager

from .logger import log_function_call

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
    print('nr_plates',nr_plates)
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
    - df: DataFrame
        The input DataFrame containing the wells.
    - avg: float
        The average value for the distribution.
    - well_ineq_coeff: float
        The inequality coefficient for the power-law distribution.

    Returns:
    - dist: ndarray
        The generated power-law distribution for the wells.
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
    
    for gene in genes:
        gene_to_well_mapping[gene] = np.random.choice(plate_map['plate_row_column'], size=int(gpw[gene-1]), p=well_weights) # Generate a number of wells for each gene according to well_weights
    
    gene_to_well_mapping = {gene: wells for gene, wells in gene_to_well_mapping.items() if len(wells) >= 2}
    
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

# classifier is a function that takes a cell state (active=1/inactive=0) and produces a score in [0, 1]
# For the input cell, it checks if it is active or inactive, and then samples from an appropriate beta distribution to give a score
def classifier(positive_mean, positive_variance, negative_mean, negative_variance, df):
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
    # alpha and beta for positive distribution
    a1 = positive_mean*(positive_mean*(1-positive_mean)/positive_variance - 1)
    b1 = a1*(1-positive_mean)/positive_mean
    # alpha and beta for negative distribution
    a2 = negative_mean*(negative_mean*(1-negative_mean)/negative_variance - 1)
    b2 = a2*(1-negative_mean)/negative_mean
    df['score'] = df['is_active'].apply(lambda is_active: np.random.beta(a1, b1) if is_active else np.random.beta(a2, b2))
    return df

def compute_roc_auc(cell_scores):
    """
    Compute the Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) for cell scores.

    Parameters:
    - cell_scores (DataFrame): DataFrame containing cell scores with columns 'is_active' and 'score'.

    Returns:
    - cell_roc_dict (dict): Dictionary containing the ROC curve information, including the threshold, true positive rate (TPR),
                            false positive rate (FPR), and ROC AUC.

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
    cell_scores = classifier(settings['positive_mean'], settings['positive_variance'], settings['negative_mean'], settings['negative_variance'], df=cell_level)
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
    return [cell_scores, cell_roc_dict_df, cell_pr_dict_df, cell_cm, well_score, gene_fraction_map, metadata, results_df, reg_roc_dict_df, reg_pr_dict_df, reg_cm, sim_stats, genes_per_well_df, wells_per_gene_df], dists

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
        #plot_histogram(temp, f'{names[index]}', ax[n], 'slategray', f'{names[index]}', binwidth=None, log=False)
        n+=1
    save_plot(fig2, src, 'dists', i)
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
    plot_histogram(active_distribution, "score", ax[n], 'slategray', 'Cell scores', binwidth=0.01, log=False)
    plot_histogram(inactive_distribution, "score", ax[n], 'teal', 'Cell scores', binwidth=0.01, log=False)
    ax[n].set_xlim([0, 1])
    n+=1
    
    #plot classifier cell predictions by inactive and active well average
    ##inactive_distribution_well['score'] = pd.to_numeric(inactive_distribution['score'], errors='coerce')
    ##inactive_distribution_well = inactive_distribution_well.groupby('plate_row_column')['score'].mean()
    
    ##active_distribution_well['score'] = pd.to_numeric(active_distribution['score'], errors='coerce')
    ##active_distribution_well = active_distribution_well.groupby('plate_row_column')['score'].mean()
    
    #inactive_distribution_well = inactive_distribution.groupby(['plate_row_column']).mean()
    #active_distribution_well = active_distribution.groupby(['plate_row_column']).mean()
    
    plot_histogram(active_distribution, "score", ax[n], 'slategray', 'Well scores', binwidth=0.01, log=False)
    plot_histogram(inactive_distribution, "score", ax[n], 'teal', 'Well scores', binwidth=0.01, log=False)
    ax[n].set_xlim([0, 1])
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
    plot_histogram(well_score, "score", ax[n], 'teal', 'Well score', binwidth=0.005, log=False)
    ax[n].set_xlim([0, 1])
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

        if save_all:
            for i, df in enumerate(output):
                df = output[i]
                if table_names[i] == 'well_score':
                    df['gene_list'] = df['gene_list'].astype(str)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                append_database(src, df, table_names[i])
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
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
    
    if settings['random_seed']:
        random.seed(42) # sims will be too similar with random seed
    src = settings['src']
    plot = settings['plot']
    v = settings['variable']
    start_time = time()  # Start time of the simulation
    now = datetime.now() # get current date
    date_string = now.strftime("%y%m%d") # format as a string in 'ddmmyy' format        
    #try:
    output, dists = run_simulation(settings)
    sim_time = time() - start_time  # Elapsed time for the simulation
    settings['sim_time'] = sim_time
    src = os.path.join(f'{src}/{date_string}',settings['name'])
    save_data(src, output, settings, save_all=False, i=i, variable=v)
    if vis_dists:
        vis_dists(dists,src, v, i)
    if plot:
        fig = visualize_all(output)
        save_plot(fig, src, v, i)
        plt.close(fig)
        plt.figure().clear() 
        plt.cla() 
        plt.clf()
        del fig
    del output, dists
    #except Exception as e:
    #    print(e, end='\r', flush=True)
    #    sim_time = time() - start_time
        #print(traceback.format_exc(), end='\r', flush=True)
    time_ls.append(sim_time)
    return i, sim_time, None
    
def generate_paramiters(settings):
    """
    Generate a list of parameter sets for simulation based on the given settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        list: A list of parameter sets for simulation.
    """
    sim_ls = []
    for avg_genes_per_well in settings['avg_genes_per_well']:
        replicates = settings['replicates']
        sett = settings.copy()
        sett['avg_genes_per_well'] = avg_genes_per_well
        sett['sd_genes_per_well'] = int(avg_genes_per_well / 2)
        for avg_cells_per_well in settings['avg_cells_per_well']:
            sett['avg_cells_per_well'] = avg_cells_per_well
            sett['sd_cells_per_well'] = int(avg_cells_per_well / 2)
            for positive_mean in settings['positive_mean']:
                sett['positive_mean'] = positive_mean
                sett['negative_mean'] = 1-positive_mean
                sett['positive_variance'] = (1-positive_mean)/2
                sett['negative_variance'] = (1-positive_mean)/2
                for avg_reads_per_gene in settings['avg_reads_per_gene']:
                    sett['avg_reads_per_gene'] = int(avg_reads_per_gene)
                    sett['sd_reads_per_gene'] = int(avg_reads_per_gene/2)
                    for sequencing_error in settings['sequencing_error']:
                        sett['sequencing_error'] = sequencing_error
                        for well_ineq_coeff in settings['well_ineq_coeff']:
                            sett['well_ineq_coeff'] = well_ineq_coeff
                            for gene_ineq_coeff in settings['gene_ineq_coeff']:
                                sett['gene_ineq_coeff'] = gene_ineq_coeff
                                for nr_plates in settings['nr_plates']:
                                    sett['nr_plates'] = nr_plates
                                    for number_of_genes in settings['number_of_genes']:
                                        sett['number_of_genes'] = number_of_genes
                                        for number_of_active_genes in settings['number_of_active_genes']:
                                            sett['number_of_active_genes'] = number_of_active_genes
                                            for i in [*range(1,replicates+1)]:
                                                sim_ls.append(sett)
                                                #print(sett)
    #print('Number of simulations:',len(sim_ls))
    return sim_ls

#altered for one set of settings see negative_mean and variance
def generate_paramiters_single(settings):
    """
    Generate a list of parameter sets for single simulations based on the given settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        list: A list of parameter sets for single simulations.
    """
    sim_ls = []
    for avg_genes_per_well in settings['avg_genes_per_well']:
        replicates = settings['replicates']
        sett = settings.copy()
        sett['avg_genes_per_well'] = avg_genes_per_well
        sett['sd_genes_per_well'] = int(avg_genes_per_well / 2)
        for avg_cells_per_well in settings['avg_cells_per_well']:
            sett['avg_cells_per_well'] = avg_cells_per_well
            sett['sd_cells_per_well'] = int(avg_cells_per_well / 2)
            for positive_mean in settings['positive_mean']:
                sett['positive_mean'] = positive_mean
                sett['negative_mean'] = 0.2
                sett['positive_variance'] = 0.13
                sett['negative_variance'] = 0.13
                for avg_reads_per_gene in settings['avg_reads_per_gene']:
                    sett['avg_reads_per_gene'] = int(avg_reads_per_gene)
                    sett['sd_reads_per_gene'] = int(avg_reads_per_gene/2)
                    for sequencing_error in settings['sequencing_error']:
                        sett['sequencing_error'] = sequencing_error
                        for well_ineq_coeff in settings['well_ineq_coeff']:
                            sett['well_ineq_coeff'] = well_ineq_coeff
                            for gene_ineq_coeff in settings['gene_ineq_coeff']:
                                sett['gene_ineq_coeff'] = gene_ineq_coeff
                                for nr_plates in settings['nr_plates']:
                                    sett['nr_plates'] = nr_plates
                                    for number_of_genes in settings['number_of_genes']:
                                        sett['number_of_genes'] = number_of_genes
                                        for number_of_active_genes in settings['number_of_active_genes']:
                                            sett['number_of_active_genes'] = number_of_active_genes
                                            for i in [*range(1,replicates+1)]:
                                                sim_ls.append(sett)
                                                #print(sett)
    #print('Number of simulations:',len(sim_ls))
    return sim_ls

def run_multiple_simulations(settings):
    """
    Run multiple simulations in parallel using the provided settings.

    Args:
        settings (dict): A dictionary containing the simulation settings.

    Returns:
        None
    """

    sim_ls = generate_paramiters(settings)
    print(f'Running {len(sim_ls)} simulations. Standard deviations for each variable are variable / 2')

    max_workers = settings['max_workers'] or cpu_count() - 4
    with Manager() as manager:
        time_ls = manager.list()
        total_sims = len(sim_ls)
        with Pool(max_workers) as pool:
            result = pool.starmap_async(run_and_save, [(index, settings, time_ls, total_sims) for index, settings in enumerate(sim_ls)])
            while not result.ready():
                sleep(0.01)
                sims_processed = len(time_ls)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                time_left = (((total_sims - sims_processed) * average_time) / max_workers) / 60
                print(f'Progress: {sims_processed}/{total_sims} Time/simulation {average_time:.3f}sec Time Remaining {time_left:.3f} min.', end='\r', flush=True)
            result.get()
            
