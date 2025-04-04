import os, shap, re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro
from math import pi

from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from statsmodels.othermod.betareg import BetaModel
from scipy.optimize import minimize
from scipy.special import gammaln, psi, expit
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import FunctionTransformer
from patsy import dmatrices

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine, euclidean, mahalanobis, cityblock, minkowski, chebyshev, hamming, jaccard, braycurtis
from xgboost import XGBClassifier

import numpy as np
from scipy.stats import kstest, normaltest
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")


class QuasiBinomial(Binomial):
    """Custom Quasi-Binomial family with adjustable variance."""
    def __init__(self, link=logit(), dispersion=1.0):
        super().__init__(link=link)
        self.dispersion = dispersion

    def variance(self, mu):
        """Adjust the variance with the dispersion parameter."""
        return self.dispersion * super().variance(mu)
    
def calculate_p_values(X, y, model):
    # Predict y values
    y_pred = model.predict(X)
    # Calculate residuals
    residuals = y - y_pred
    # Calculate the standard error of the residuals
    dof = X.shape[0] - X.shape[1] - 1
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)
    # Calculate the standard error of the coefficients
    X_design = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
    # Use pseudoinverse instead of inverse to handle singular matrices
    coef_var_covar = residual_std_error ** 2 * np.linalg.pinv(X_design.T @ X_design)
    coef_standard_errors = np.sqrt(np.diag(coef_var_covar))
    # Calculate t-statistics
    t_stats = model.coef_ / coef_standard_errors[1:]  # Skip intercept error
    # Calculate p-values
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), dof)) for t in t_stats]
    return np.array(p_values)  # Ensure p_values is a 1-dimensional array

def perform_mixed_model(y, X, groups, alpha=1.0):
    # Ensure groups are defined correctly and check for multicollinearity
    if groups is None:
        raise ValueError("Groups must be defined for mixed model regression")

    # Check for multicollinearity by calculating the VIF for each feature
    X_np = X.values
    vif = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
    print(f"VIF: {vif}")
    if any(v > 10 for v in vif):
        print(f"Multicollinearity detected with VIF: {vif}. Applying Ridge regression to the fixed effects.")
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        X_ridge = ridge.coef_ * X  # Adjust X with Ridge coefficients
        model = MixedLM(y, X_ridge, groups=groups)
    else:
        model = MixedLM(y, X, groups=groups)

    result = model.fit()
    return result

def create_volcano_filename(csv_path, regression_type, alpha, dst):
    """Create and return the volcano plot filename based on regression type and alpha."""
    volcano_filename = os.path.splitext(os.path.basename(csv_path))[0] + '_volcano_plot.pdf'
    volcano_filename = f"{regression_type}_{volcano_filename}" if regression_type != 'quantile' else f"{alpha}_{volcano_filename}"

    if dst:
        return os.path.join(dst, volcano_filename)
    return os.path.join(os.path.dirname(csv_path), volcano_filename)

def scale_variables(X, y):
    """Scale independent (X) and dependent (y) variables using MinMaxScaler."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled

def process_model_coefficients(model, regression_type, X, y, nc, pc, controls):
    """Return DataFrame of model coefficients and p-values."""
    if regression_type in ['ols', 'gls', 'wls', 'rlm', 'glm', 'mixed', 'quantile', 'logit', 'probit', 'poisson']:
        coefs = model.params
        p_values = model.pvalues

        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'p_value': p_values.values
        })

    elif regression_type in ['ridge', 'lasso']:
        coefs = model.coef_.flatten()
        p_values = calculate_p_values(X, y, model)

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefs,
            'p_value': p_values
        })
        
    else:
        coefs = model.coef_
        intercept = model.intercept_
        feature_names = X.design_info.column_names

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs
        })
        coef_df.loc[0, 'coefficient'] += intercept
        coef_df['p_value'] = np.nan  # Placeholder since sklearn doesn't provide p-values

    coef_df['-log10(p_value)'] = -np.log10(coef_df['p_value'])
    coef_df['grna'] = coef_df['feature'].str.extract(r'\[(.*?)\]')[0]
    coef_df['condition'] = coef_df.apply(lambda row: 'nc' if nc in row['feature'] else 'pc' if pc in row['feature'] else ('control' if row['grna'] in controls else 'other'),axis=1)
    return coef_df[~coef_df['feature'].str.contains('row|column')]

def check_distribution(y):
    """Check the type of distribution to recommend a model."""
    if np.all((y == 0) | (y == 1)):
        print("Detected binary data.")
        return 'logit'
    elif (y > 0).all() and (y < 1).all():
        print("Detected continuous data between 0 and 1 (excluding 0 and 1).")
        return 'beta'
    elif (y >= 0).all() and (y <= 1).all():
        print("Detected continuous data between 0 and 1 (including 0 or 1).")
        # Consider quasi-binomial regression
        return 'quasi_binomial'
    else:
        print("Using OLS as a fallback.")
        return 'ols'

def select_glm_family(y):
    """Select the appropriate GLM family based on the data."""
    if np.all((y == 0) | (y == 1)):
        print("Using Binomial family (for binary data).")
        return sm.families.Binomial()
    elif (y >= 0).all() and (y <= 1).all():
        print("Using Quasi-Binomial family (for proportion data including 0 and 1).")
        return QuasiBinomial()
    elif np.all(y.astype(int) == y) and (y >= 0).all():
        print("Using Poisson family (for count data).")
        return sm.families.Poisson()
    else:
        print("Using Gaussian family (for continuous data).")
        return sm.families.Gaussian()

def prepare_formula(dependent_variable, random_row_column_effects=False):
    """Return the regression formula using random effects for plate, row, and column."""
    if random_row_column_effects:
        # Random effects for row and column + gene weighted by gene_fraction + grna weighted by fraction
        return f'{dependent_variable} ~ fraction:grna + gene_fraction:gene'
    return f'{dependent_variable} ~ fraction:grna + gene_fraction:gene + rowID + columnID'

def fit_mixed_model(df, formula, dst):
    from .plot import plot_histogram

    """Fit the mixed model with plate, row_name, and columnID as random effects and return results."""
    # Specify random effects for plate, row, and column
    model = smf.mixedlm(formula, 
                        data=df, 
                        groups=df['plateID'], 
                        re_formula="1 + rowID + columnID", 
                        vc_formula={"rowID": "0 + rowID", "columnID": "0 + columnID"})
    
    mixed_model = model.fit()

    # Plot residuals
    df['residuals'] = mixed_model.resid
    plot_histogram(df, 'residuals', dst=dst)

    # Return coefficients and p-values
    coefs = mixed_model.params
    p_values = mixed_model.pvalues

    coef_df = pd.DataFrame({
        'feature': coefs.index,
        'coefficient': coefs.values,
        'p_value': p_values.values
    })
    
    return mixed_model, coef_df

def check_and_clean_data(df, dependent_variable):
    """Check for collinearity, missing values, or invalid types in relevant columns. Clean data accordingly."""
    
    def handle_missing_values(df, columns):
        """Handle missing values in specified columns."""
        missing_summary = df[columns].isnull().sum()
        print("Missing values summary:")
        print(missing_summary)
        
        # Drop rows with missing values in these fields
        df_cleaned = df.dropna(subset=columns)
        if df_cleaned.shape[0] < df.shape[0]:
            print(f"Dropped {df.shape[0] - df_cleaned.shape[0]} rows with missing values in {columns}.")
        return df_cleaned
    
    def ensure_valid_types(df, columns):
        """Ensure that specified columns are categorical."""
        for col in columns:
            if not pd.api.types.is_categorical_dtype(df[col]):
                df[col] = pd.Categorical(df[col])
                print(f"Converted {col} to categorical type.")
        return df

    def check_collinearity(df, columns):
        """Check for collinearity using VIF (Variance Inflation Factor)."""
        print("Checking for collinearity...")
        
        # Only include fraction and the dependent variable for collinearity check
        df_encoded = df[columns]
        
        # Ensure all data in df_encoded is numeric
        df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
        
        # Check for perfect multicollinearity (i.e., rank deficiency)
        if np.linalg.matrix_rank(df_encoded.values) < df_encoded.shape[1]:
            print("Warning: Perfect multicollinearity detected! Dropping correlated columns.")
            df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_encoded.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]
        except np.linalg.LinAlgError:
            print("LinAlgError: Unable to compute VIF due to matrix singularity.")
            return df_encoded

        print("Variance Inflation Factor (VIF) for each feature:")
        print(vif_data)
        
        # Drop columns with VIF > 10 (a common threshold to identify multicollinearity)
        high_vif_columns = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
        if high_vif_columns:
            print(f"Dropping columns with high VIF: {high_vif_columns}")
            df_encoded.drop(columns=high_vif_columns, inplace=True)
        
        return df_encoded
    
    # Step 1: Handle missing values in relevant fields
    df = handle_missing_values(df, ['fraction', dependent_variable])
    
    # Step 2: Ensure grna, gene, plate, row, column, and prc are categorical types
    df = ensure_valid_types(df, ['grna', 'gene', 'plateID', 'rowID', 'columnID', 'prc'])
    
    # Step 3: Check for multicollinearity in fraction and the dependent variable
    df_cleaned = check_collinearity(df, ['fraction', dependent_variable])
    
    # Ensure that the prc, plate, row, and column columns are still included for random effects
    df_cleaned['gene'] = df['gene']
    df_cleaned['grna'] = df['grna']
    df_cleaned['prc'] = df['prc']
    df_cleaned['plateID'] = df['plateID']
    df_cleaned['rowID'] = df['rowID']
    df_cleaned['columnID'] = df['columnID']

    # Create a new column 'gene_fraction' that sums the fractions by gene within the same well
    df_cleaned['gene_fraction'] = df_cleaned.groupby(['prc', 'gene'])['fraction'].transform('sum')

    print("Data is ready for model fitting.")
    return df_cleaned

def check_normality(y, variable_name):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    from scipy.stats import shapiro

    stat, p = shapiro(y)
    alpha = 0.05
    if p > alpha:
        print(f"{variable_name} is normally distributed (fail to reject H0)")
        return True
    else:
        print(f"{variable_name} is not normally distributed (reject H0)")
        return False

def minimum_cell_simulation(settings, num_repeats=10, sample_size=100, tolerance=0.02, smoothing=10, increment=10):
    """
    Plot the mean absolute difference with standard deviation as shaded area vs. sample size.
    Detect and mark the elbow point (inflection) with smoothing and tolerance control.
    """

    from spacr.utils import correct_metadata_column_names

    # Load and process data
    if isinstance(settings['score_data'], str):
        settings['score_data'] = [settings['score_data']]

    dfs = []
    for i, score_data in enumerate(settings['score_data']):
        df = pd.read_csv(score_data)
        df = correct_metadata_column_names(df)
        df['plateID'] = f'plate{i + 1}'
        
        if 'prc' not in df.columns:
            df['prc'] = df['plateID'] + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
            
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Compute the number of cells per well and select the top 100 wells by cell count
    cell_counts = df.groupby('prc').size().reset_index(name='cell_count')
    top_wells = cell_counts.nlargest(sample_size, 'cell_count')['prc']

    # Filter the data to include only the top 100 wells
    df = df[df['prc'].isin(top_wells)]

    # Initialize storage for absolute difference data
    diff_data = []

    # Group by wells and iterate over them
    for i, (prc, group) in enumerate(df.groupby('prc')):
        original_mean = group[settings['score_column']].mean()  # Original full-well mean
        max_cells = len(group)
        sample_sizes = np.arange(2, max_cells + 1, increment)  # Sample sizes from 2 to max cells

        # Iterate over sample sizes and compute absolute difference
        for sample_size in sample_sizes:
            abs_diffs = []

            # Perform multiple random samples to reduce noise
            for _ in range(num_repeats):
                sample = group.sample(n=sample_size, replace=False)
                sampled_mean = sample[settings['score_column']].mean()
                abs_diff = abs(sampled_mean - original_mean)  # Absolute difference
                abs_diffs.append(abs_diff)

            # Compute the average absolute difference across all repeats
            avg_abs_diff = np.mean(abs_diffs)

            # Store the result for plotting
            diff_data.append((sample_size, avg_abs_diff))

    # Convert absolute difference data to DataFrame for plotting
    diff_df = pd.DataFrame(diff_data, columns=['sample_size', 'avg_abs_diff'])

    # Group by sample size to calculate mean and standard deviation
    summary_df = diff_df.groupby('sample_size').agg(
        mean_abs_diff=('avg_abs_diff', 'mean'),
        std_abs_diff=('avg_abs_diff', 'std')
    ).reset_index()

    # Apply smoothing using a rolling window
    summary_df['smoothed_mean_abs_diff'] = summary_df['mean_abs_diff'].rolling(window=smoothing, min_periods=1).mean()

    # Convert percentage to fraction
    if isinstance(settings['tolerance'], int):
        tolerance_fraction = settings['tolerance'] / 100  # Convert 2% to 0.02
    elif isinstance(settings['tolerance'], float):
        tolerance_fraction = settings['tolerance']
    else:
        raise ValueError("Tolerance must be an integer 0 - 100 or float 0.0 - 1.0.")

    # Compute the relative threshold for each well
    relative_thresholds = {
        prc: tolerance_fraction * group[settings['score_column']].mean()  # Compute % of original mean
        for prc, group in df.groupby('prc')
    }

    # Detect the elbow point when mean absolute difference is below the relative threshold
    summary_df['relative_threshold'] = summary_df['sample_size'].map(
        lambda size: np.mean([relative_thresholds[prc] for prc in top_wells])  # Average across selected wells
    )

    elbow_df = summary_df[summary_df['smoothed_mean_abs_diff'] <= summary_df['relative_threshold']]

    # Select the first occurrence if it exists; otherwise, use the last point
    if not elbow_df.empty:
        elbow_point = elbow_df.iloc[0]  # First point where condition is met
    else:
        elbow_point = summary_df.iloc[-1]  # Fallback to last point

    # Plot the mean absolute difference with standard deviation as shaded area
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        summary_df['sample_size'], summary_df['smoothed_mean_abs_diff'], color='teal', label='Smoothed Mean Absolute Difference'
    )
    ax.fill_between(
        summary_df['sample_size'],
        summary_df['smoothed_mean_abs_diff'] - summary_df['std_abs_diff'],
        summary_df['smoothed_mean_abs_diff'] + summary_df['std_abs_diff'],
        color='teal', alpha=0.3, label='±1 Std. Dev.'
    )

    if settings['min_cell_count'] is None:
        # Mark the elbow point (inflection) on the plot
        ax.axvline(elbow_point['sample_size'], color='black', linestyle='--', label='Elbow Point')
    else:
        ax.axvline(settings['min_cell_count'], color='black', linestyle='--', label='Elbow Point')

    # Formatting the plot
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Mean Absolute Difference vs. Sample Size with Standard Deviation')
    ax.legend().remove()

    # Save the plot if a destination is provided
    dst = os.path.dirname(settings['count_data'][0])
    if dst is not None:
        fig_path = os.path.join(dst, 'results')
        os.makedirs(fig_path, exist_ok=True)
        fig_file_path = os.path.join(fig_path, 'cell_min_threshold.pdf')
        fig.savefig(fig_file_path, format='pdf', dpi=600, bbox_inches='tight')
        print(f"Saved {fig_file_path}")

    plt.show()
    return elbow_point['sample_size']

def process_model_coefficients(model, regression_type, X, y, nc, pc, controls):
    """Return DataFrame of model coefficients, standard errors, and p-values."""
    
    if regression_type == 'beta':
        # Extract coefficients and standard errors
        coefs = model.params
        std_err = model.bse
        
        # Compute Wald test (coefficient / standard error)
        wald_stats = coefs / std_err
        
        # Calculate two-tailed p-values
        p_values = 2 * (1 - st.norm.cdf(np.abs(wald_stats)))
        
        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'std_err': std_err.values,
            'wald_stat': wald_stats.values,
            'p_value': p_values
        })

    elif regression_type in ['ols', 'glm', 'logit', 'probit', 'quasi_binomial']:
        coefs = model.params
        p_values = model.pvalues

        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'p_value': p_values.values
        })

    elif regression_type in ['ridge', 'lasso']:
        coefs = model.coef_.flatten()
        p_values = calculate_p_values(X, y, model)

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefs,
            'p_value': p_values
        })

    else:
        raise ValueError(f"Unsupported regression type: {regression_type}")

    # Additional formatting
    coef_df['-log10(p_value)'] = -np.log10(coef_df['p_value'])
    coef_df['grna'] = coef_df['feature'].str.extract(r'\[(.*?)\]')[0]
    coef_df['condition'] = coef_df.apply(
        lambda row: 'nc' if nc in row['feature'] else 
                    'pc' if pc in row['feature'] else 
                    ('control' if row['grna'] in controls else 'other'), 
        axis=1
    )

    return coef_df[~coef_df['feature'].str.contains('row|column')]

def check_distribution(y, epsilon=1e-6):
    """Check the distribution of y and recommend an appropriate model."""
    
    # Check if the dependent variable is binary (only 0 and 1)
    if np.all((y == 0) | (y == 1)):
        print("Detected binary data.")
        return 'logit'
    
    # Continuous data between 0 and 1 (excluding exact 0 and 1)
    elif (y > 0).all() and (y < 1).all():
        # Check if the data is close to 0 or 1 (boundary issues)
        if np.any((y < epsilon) | (y > 1 - epsilon)):
            print("Detected continuous data near 0 or 1. Using quasi-binomial.")
            return 'quasi_binomial'
        else:
            print("Detected continuous data between 0 and 1 (no boundary issues). Using beta regression.")
            return 'beta'
    
    # Continuous data between 0 and 1 (including exact 0 or 1)
    elif (y >= 0).all() and (y <= 1).all():
        print("Detected continuous data with boundary values (0 or 1). Using quasi-binomial.")
        return 'quasi_binomial'
    
    # Check if the data is normally distributed for OLS suitability
    stat, p_value = stats.normaltest(y)  # D’Agostino and Pearson’s test for normality
    print(f"Normality test p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("Detected normally distributed data. Using OLS.")
        return 'ols'
    
    # Check if the data fits a Beta distribution
    if stats.kstest(y, 'beta', args=(2, 2)).pvalue > 0.05:
        # Check if the data is close to 0 or 1 (boundary issues)
        if np.any((y < epsilon) | (y > 1 - epsilon)):
            print("Detected continuous data near 0 or 1. Using quasi-binomial.")
            return 'quasi_binomial'
        else:
            print("Detected continuous data between 0 and 1 (no boundary issues). Using beta regression.")
            return 'beta'
    
    print("Detected non-normally distributed data. Using GLM.")
    return 'glm'

def pick_glm_family_and_link(y):
    """Select the appropriate GLM family and link function based on data."""
    if np.all((y == 0) | (y == 1)):
        print("Binary data detected. Using Binomial family with Logit link.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    elif (y > 0).all() and (y < 1).all():
        print("Data strictly between 0 and 1. Beta regression recommended.")
        raise ValueError("Use BetaModel for this data; GLM is not applicable.")

    elif (y >= 0).all() and (y <= 1).all():
        print("Data between 0 and 1 (including boundaries). Using Quasi-Binomial.")
        return sm.families.Binomial(link=sm.families.links.Logit())

    stat, p_value = normaltest(y)
    print(f"Normality test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Normally distributed data detected. Using Gaussian with Identity link.")
        return sm.families.Gaussian(link=sm.families.links.Identity())

    if (y >= 0).all() and np.all(y.astype(int) == y):
        print("Count data detected. Using Poisson with Log link.")
        return sm.families.Poisson(link=sm.families.links.Log())

    if (y > 0).all() and kstest(y, 'invgauss', args=(1,)).pvalue > 0.05:
        print("Inverse Gaussian distribution detected. Using InverseGaussian with Log link.")
        return sm.families.InverseGaussian(link=sm.families.links.Log())

    if (y >= 0).all():
        print("Overdispersed count data detected. Using Negative Binomial with Log link.")
        return sm.families.NegativeBinomial(link=sm.families.links.Log())

    print("Using default Gaussian family with Identity link.")
    return sm.families.Gaussian(link=sm.families.links.Identity())

def regression_model(X, y, regression_type='ols', groups=None, alpha=1.0, cov_type=None):
    def plot_regression_line(X, y, model):
        """Helper to plot regression line for lasso and ridge models."""
        y_pred = model.predict(X)
        plt.scatter(X.iloc[:, 1], y, color='blue', label='Data')
        plt.plot(X.iloc[:, 1], y_pred, color='red', label='Regression line')
        plt.xlabel('Features')
        plt.ylabel('Dependent Variable')
        plt.legend()
        plt.show()
        
    def find_best_alpha(model_cls):
        """Find optimal alpha using cross-validation."""
        alphas = np.logspace(-5, 5, 100)  # Search over a range of alphas
        if model_cls == 'lasso':
            model_cv = LassoCV(alphas=alphas, cv=5).fit(X, y)
        elif model_cls == 'ridge':
            model_cv = RidgeCV(alphas=alphas, cv=5).fit(X, y)
        print(f"Optimal alpha for {model_cls}: {model_cv.alpha_}")
        return model_cv

    # Dictionary of models
    model_map = {
        'ols': lambda: sm.OLS(y, X).fit(cov_type=cov_type) if cov_type else sm.OLS(y, X).fit(),
        'glm': lambda: sm.GLM(y, X, family=pick_glm_family_and_link(y)).fit(),
        'beta': lambda: BetaModel(endog=y, exog=X).fit(),
        'logit': lambda: sm.Logit(y, X).fit(),
        'probit': lambda: sm.Probit(y, X).fit(),
        'lasso': lambda: find_best_alpha('lasso') if alpha in [0, None] else Lasso(alpha=alpha).fit(X, y),
        'ridge': lambda: find_best_alpha('ridge') if alpha in [0, None] else Ridge(alpha=alpha).fit(X, y)
    }

    # Select the model based on regression_type
    if regression_type in model_map:
        model = model_map[regression_type]()
    elif regression_type == 'mixed':
        model = perform_mixed_model(y, X, groups, alpha=alpha)
    else:
        raise ValueError(f"Unsupported regression type {regression_type}")

    # Plot regression line for Lasso and Ridge
    if regression_type in ['lasso', 'ridge']:
        plot_regression_line(X, y, model)

    # Handle GLM-specific statistics
    if regression_type == 'glm':
        llf_model = model.llf  # Log-likelihood of the fitted model
        llf_null = model.null_deviance / -2  # Log-likelihood of the null model
        mcfadden_r2 = 1 - (llf_model / llf_null)
        print(f"McFadden's R²: {mcfadden_r2:.4f}")
        print(model.summary())

    if regression_type in ['lasso', 'ridge']:
        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y, model.predict(X))
        print(f"{regression_type.capitalize()} Regression MSE: {mse:.4f}")

        # Display coefficients
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        })
        print(coef_df)

    return model

def regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0,
               random_row_column_effects=False, nc='233460', pc='220950', controls=[''],
               dst=None, cov_type=None, plot=False):
    
    from .plot import volcano_plot, plot_histogram
    #from .ml import create_volcano_filename, check_and_clean_data, prepare_formula, scale_variables
        
    # Generate the volcano filename
    volcano_path = create_volcano_filename(csv_path, regression_type, alpha, dst)

    # Determine regression type if not specified
    if regression_type is None:
        regression_type = check_distribution(df[dependent_variable])

    print(f"Using regression type: {regression_type}")

    df = check_and_clean_data(df, dependent_variable)

    # Handle mixed effects if row/column effect is treated as random
    if random_row_column_effects:
        regression_type = 'mixed'
        formula = prepare_formula(dependent_variable, random_row_column_effects=True)
        mixed_model, coef_df = fit_mixed_model(df, formula, dst)
        model = mixed_model
    else:
        # Prepare the formula
        formula = prepare_formula(dependent_variable, random_row_column_effects=False)
        y, X = dmatrices(formula, data=df, return_type='dataframe')
        
        # Plot histogram of the dependent variable
        plot_histogram(y, dependent_variable, dst=dst)
        plot_histogram(df, 'fraction', dst=dst)

        # Scale the independent variables and dependent variable
        if regression_type in ['beta', 'quasi_binomial', 'logit']:
            print('Data will not be scaled')
        else:
            X, y = scale_variables(X, y)

        # Perform the regression
        groups = df['prc'] if regression_type == 'mixed' else None
        print(f'Performing {regression_type} regression')
        
        model = regression_model(X, y, regression_type=regression_type, groups=groups, alpha=alpha, cov_type=cov_type)

        # Process the model coefficients
        coef_df = process_model_coefficients(model, regression_type, X, y, nc, pc, controls)
        display(coef_df)
    if plot:
        volcano_plot(coef_df, volcano_path)
        
    return model, coef_df, regression_type

def save_summary_to_file(model, file_path='summary.csv'):
    """
    Save the model's summary output to a CSV or text file.
    """
    # Get the summary as a string
    summary_str = model.summary().as_text()

    # Save it as a plain text file or CSV
    with open(file_path, 'w') as f:
        f.write(summary_str)

def perform_regression(settings):
    
    from .plot import plot_plates, plot_data_from_csv
    from .utils import merge_regression_res_with_metadata, save_settings, calculate_shortest_distance, correct_metadata
    from .settings import get_perform_regression_default_settings
    from .toxo import go_term_enrichment_by_column, custom_volcano_plot, plot_gene_phenotypes, plot_gene_heatmaps
    from .sequencing import graph_sequencing_stats

    def _perform_regression_read_data(settings):
        
        if not isinstance(settings['score_data'], list):
            settings['score_data'] = [settings['score_data']]
        if not isinstance(settings['count_data'], list):
            settings['count_data'] = [settings['count_data']]
            
        score_data_df = pd.DataFrame()
        for i, score_data in enumerate(settings['score_data']):
            df = pd.read_csv(score_data)
            df = correct_metadata(df)
            if not 'plateID' in df.columns:
                df['plateID'] = f'plate{i+1}'

            score_data_df = pd.concat([score_data_df, df])
            print('Score data:', len(score_data_df))
            
        count_data_df = pd.DataFrame()
        for i, count_data in enumerate(settings['count_data']):
            df = pd.read_csv(count_data)
            df = correct_metadata(df)
            if not 'plateID' in df.columns:
                df['plateID'] = f'plate{i+1}'
                
            count_data_df = pd.concat([count_data_df, df])
            print('Count data:', len(count_data_df))

        print(f"Dependent variable: {len(score_data_df)}")
        print(f"Independent variable: {len(count_data_df)}")

        if settings['dependent_variable'] not in score_data_df.columns:
            print(f'Columns in DataFrame:')
            for col in score_data_df.columns:
                print(col)
            if not settings['dependent_variable'] == 'pathogen_nucleus_shortest_distance':
                raise ValueError(f"Dependent variable {settings['dependent_variable']} not found in the DataFrame")
        
        reg_types = ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge', None]
        if settings['regression_type'] not in reg_types:
            print(f'Possible regression types: {reg_types}')
            raise ValueError(f"Unsupported regression type {settings['regression_type']}")

        return count_data_df, score_data_df
    
    def _perform_regression_set_paths(settings):

        if isinstance(settings['score_data'], list):
            score_data = settings['score_data'][0]
        else:
            score_data = settings['score_data']
        
        score_source = os.path.splitext(os.path.basename(score_data))[0]
        
        if isinstance(settings['count_data'], list):
            src = os.path.dirname(settings['count_data'][0])
            csv_path = settings['count_data'][0]
        else:
            src = os.path.dirname(settings['count_data'])
            csv_path = settings['count_data']

        settings['src'] = src
    
        if settings['regression_type'] is None:
            res_folder = os.path.join(src, 'results', score_source, 'auto')
        else:
            res_folder = os.path.join(src, 'results', score_source, settings['regression_type'])
        
        if isinstance(settings['count_data'], list):
            res_folder = os.path.join(res_folder, 'list')

        os.makedirs(res_folder, exist_ok=True)
        results_filename = 'results.csv'
        results_filename_gene = 'results_gene.csv'
        results_filename_grna = 'results_grna.csv'
        hits_filename = 'results_significant.csv'
        results_path=os.path.join(res_folder, results_filename)
        results_path_gene=os.path.join(res_folder, results_filename_gene)
        results_path_grna=os.path.join(res_folder, results_filename_grna)
        hits_path=os.path.join(res_folder, hits_filename)

        return results_path, results_path_gene, results_path_grna, hits_path, res_folder, csv_path
    
    def _count_variable_instances(df, column_1, column_2):
        if column_1 is not None:
            n_grna = df[column_1].value_counts().reset_index()
            n_grna.columns = [column_1, f'n_{column_1}']
        if column_2 is not None:
            n_gene = df[column_2].value_counts().reset_index()
            n_gene.columns = [column_2, f'n_{column_2}']
        if column_1 is not None and column_2 is not None:
            return df, n_grna, n_gene
        elif column_1 is not None:
            return df, n_grna
        elif column_2 is not None:
            return df, n_gene
        else:
            return df
        
    def grna_metricks(df):
        df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)

        # --- 2) Compute GRNA-level Well Counts ---
        # For each (grna, plate), count the number of unique prc (wells)
        grna_well_counts = (df.groupby(['grna', 'plateID'])['prc'].nunique().reset_index(name='grna_well_count'))

        # --- 3) Compute Gene-level Well Counts ---
        # For each (gene, plate), count the number of unique prc
        gene_well_counts = (df.groupby(['gene', 'plateID'])['prc'].nunique().reset_index(name='gene_well_count'))

        # --- 4) Merge These Counts into a Single DataFrame ---
        # Because each grna is typically associated with one gene, we bring them together.
        # First, create a unique (grna, gene, plate) reference from the original df
        unique_triplets = df[['grna', 'gene', 'plateID']].drop_duplicates()

        # Merge the grna_well_count
        merged_df = pd.merge(unique_triplets, grna_well_counts, on=['grna', 'plateID'], how='left')

        # Merge the gene_well_count
        merged_df = pd.merge(merged_df, gene_well_counts, on=['gene', 'plateID'], how='left')

        # Keep only the columns needed (if you want to keep 'gene', remove the drop below)
        final_grna_df = merged_df[['grna', 'plateID', 'grna_well_count', 'gene_well_count']]

        # --- 5) Compute gene_count per prc ---
        # For each prc (well), how many distinct genes are there?
        prc_gene_count_df = (df.groupby('prc')['gene'].nunique().reset_index(name='gene_count'))
        prc_gene_count_df[['plateID', 'rowID', 'columnID']] = prc_gene_count_df['prc'].str.split('_', expand=True)
        
        return final_grna_df, prc_gene_count_df
    
    def get_outlier_reference_values(df, outlier_col, return_col):
        """
        Detect outliers in 'outlier_col' of 'df' using the 1.5 × IQR rule,
        and return values from 'return_col' that correspond to those outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.
        outlier_col : str
            Column in which to check for outliers.
        return_col : str
            Column whose values to return for rows that are outliers in 'outlier_col'.
        
        Returns:
        --------
        pd.Series
            A Series containing values from 'return_col' for the outlier rows.
        """
        # Calculate Q1, Q3, and IQR for the outlier_col
        Q1 = df[outlier_col].quantile(0.05)
        Q3 = df[outlier_col].quantile(0.95)
        IQR = Q3 - Q1
        
        # Determine the outlier cutoffs
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create a mask for outliers
        outlier_mask = (df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)
        
        outliers = df.loc[outlier_mask, return_col]
        
        outliers_ls = outliers.unique().tolist()
        
        return outliers_ls

    settings = get_perform_regression_default_settings(settings)
    count_data_df, score_data_df = _perform_regression_read_data(settings)
    
    if "rowID" in count_data_df.columns:
        num_parts = len(count_data_df['rowID'].iloc[0].split('_'))
        if num_parts == 2:
            split = count_data_df['rowID'].str.split('_', expand=True)
            count_data_df['rowID'] = split[1]
    
    if "prc" in score_data_df.columns:
        num_parts = len(score_data_df['prc'].iloc[0].split('_'))
        if num_parts == 3:
            split = score_data_df['prc'].str.split('_', expand=True)
            score_data_df['plateID'] = settings['plateID']
            score_data_df['prc'] = score_data_df['plateID'] + '_' + split[1] + '_' + split[2]
        
    results_path, results_path_gene, results_path_grna, hits_path, res_folder, csv_path = _perform_regression_set_paths(settings)
    save_settings(settings, name='regression', show=True)

    count_source = os.path.dirname(settings['count_data'][0])
    volcano_path = os.path.join(count_source, 'volcano_plot.pdf')

    if isinstance(settings['filter_value'], list):
        filter_value = settings['filter_value']
    else:
        filter_value = []
    if isinstance(settings['filter_column'], str):
        filter_column = settings['filter_column']
    
    score_data_df = clean_controls(score_data_df, settings['filter_value'], settings['filter_column'])
    
    if settings['verbose']:
        print(f"Dependent variable after clean_controls: {len(score_data_df)}")

    sim_min_count = minimum_cell_simulation(settings, tolerance=settings['tolerance'])
    
    if settings['min_cell_count'] is None:
        settings['min_cell_count'] = sim_min_count
        
    if settings['verbose']:
        print(f"Minimum cell count: {settings['min_cell_count']}")
        print(f"Dependent variable after minimum cell count filter: {len(score_data_df)}")
        display(score_data_df)

    orig_dv = settings['dependent_variable']

    dependent_df, dependent_variable = process_scores(score_data_df, settings['dependent_variable'], settings['plateID'], settings['min_cell_count'], settings['agg_type'], settings['transform'])
    
    if settings['verbose']:
        print(f"Dependent variable after process_scores: {len(dependent_df)}")
        display(dependent_df)
    
    if settings['fraction_threshold'] is None:
        settings['fraction_threshold'] = graph_sequencing_stats(settings)

    independent_df = process_reads(count_data_df, settings['fraction_threshold'], settings['plateID'], filter_column=filter_column, filter_value=filter_value)
        
    independent_df, n_grna, n_gene = _count_variable_instances(independent_df, column_1='grna', column_2='gene')
    
    if settings['verbose']:
        print(f"Independent variable after process_reads: {len(independent_df)}")
    
    merged_df = pd.merge(independent_df, dependent_df, on='prc')
    
    if settings['verbose']:
        display(independent_df)
        display(dependent_df)
        display(merged_df)
    
    
    merged_df[['plateID', 'rowID', 'columnID']] = merged_df['prc'].str.split('_', expand=True)
        
    try:
        os.makedirs(res_folder, exist_ok=True)
        data_path = os.path.join(res_folder, 'regression_data.csv')
        merged_df.to_csv(data_path, index=False)
        print(f"Saved regression data to {data_path}")
        
        cell_settings = {'src':data_path,
                        'graph_name':'cell_count',
                        'data_column':['cell_count'],
                        'grouping_column':'plateID',
                        'graph_type':'jitter_bar',
                        'theme':'bright',
                        'save':True,
                        'y_lim':[None,None],
                        'log_y':False,
                        'log_x':False,
                        'representation':'well',
                        'verbose':False}
        
        _, _ = plot_data_from_csv(settings=cell_settings)
        
        final_grna_df, prc_gene_count_df = grna_metricks(merged_df)
        
        if settings['outlier_detection']:
            outliers_grna = get_outlier_reference_values(final_grna_df,outlier_col='grna_well_count',return_col='grna')
            if len (outliers_grna) > 0:
                merged_df = merged_df[~merged_df['grna'].isin(outliers_grna)]
                final_grna_df, prc_gene_count_df = grna_metricks(merged_df)
                merged_df.to_csv(data_path, index=False)
                print(f"Saved regression data to {data_path}")

        grna_data_path = os.path.join(res_folder, 'grna_well.csv')
        final_grna_df.to_csv(grna_data_path, index=False)
        print(f"Saved grna per well data to {grna_data_path}")
        
        wells_per_gene_settings = {'src':grna_data_path,
                    'graph_name':'wells_per_gene',
                    'data_column':['grna_well_count'],
                    'grouping_column':'plateID',
                    'graph_type':'jitter_bar',
                    'theme':'bright',
                    'save':True,
                    'y_lim':[None,None],
                    'log_y':False,
                    'log_x':False,
                    'representation':'object',
                    'verbose':True}
        
        _, _ = plot_data_from_csv(settings=wells_per_gene_settings)
        
        grna_well_data_path = os.path.join(res_folder, 'well_grna.csv')
        prc_gene_count_df.to_csv(grna_well_data_path, index=False)
        print(f"Saved well per grna data to {grna_well_data_path}")
        
        grna_per_well_settings = {'src':grna_well_data_path,
                                'graph_name':'gene_per_well',
                                'data_column':['gene_count'],
                                'grouping_column':'plateID',
                                'graph_type':'jitter_bar',
                                'theme':'bright',
                                'save':True,
                                'y_lim':[None,None],
                                'log_y':False,
                                'log_x':False,
                                'representation':'well',
                                'verbose':False}
        
        _, _ = plot_data_from_csv(settings=grna_per_well_settings)
        
    except Exception as e:
        print(e)
        
    _ = plot_plates(merged_df, variable=orig_dv, grouping='mean', min_max='allq', cmap='viridis', min_count=None, dst=res_folder)                

    model, coef_df, regression_type = regression(merged_df, csv_path, dependent_variable, settings['regression_type'], settings['alpha'], settings['random_row_column_effects'], nc=settings['negative_control'], pc=settings['positive_control'], controls=settings['controls'], dst=res_folder, cov_type=settings['cov_type'])
    
    coef_df['grna'] = coef_df['feature'].apply(lambda x: re.search(r'grna\[(.*?)\]', x).group(1) if 'grna' in x else None)
    coef_df['gene'] = coef_df['feature'].apply(lambda x: re.search(r'gene\[(.*?)\]', x).group(1) if 'gene' in x else None)
    
    coef_df = coef_df.merge(n_grna, how='left', on='grna')
    coef_df = coef_df.merge(n_gene, how='left', on='gene')

    gene_coef_df = coef_df[coef_df['n_gene'] != None]
    grna_coef_df = coef_df[coef_df['n_grna'] != None]
    gene_coef_df = gene_coef_df.dropna(subset=['n_gene'])
    grna_coef_df = grna_coef_df.dropna(subset=['n_grna'])
    
    if settings['controls'] is not None:

        control_coef_df = grna_coef_df[grna_coef_df['grna'].isin(settings['controls'])]
        mean_coef = control_coef_df['coefficient'].mean()
        significant_c = control_coef_df[control_coef_df['p_value']<= 0.05]
        mean_coef_c = significant_c['coefficient'].mean()
        
        if settings['verbose']:
            print(mean_coef, mean_coef_c)
        
        if settings['threshold_method'] in ['var','variance']:
            coef_mes = control_coef_df['coefficient'].var()
        elif settings['threshold_method'] in ['std', 'standard_deveation']:
            coef_mes = control_coef_df['coefficient'].std()
        else:
            raise ValueError(f"Unsupported threshold method {settings['threshold_method']}. Supported methods: ['var','variance','std','standard_deveation']")
        
        reg_threshold = mean_coef + (settings['threshold_multiplier'] * coef_mes)
    
    coef_df.to_csv(results_path, index=False)
    gene_coef_df.to_csv(results_path_gene, index=False)
    grna_coef_df.to_csv(results_path_grna, index=False)
    
    if regression_type == 'lasso':
        significant = coef_df[coef_df['coefficient'] > 0]
        
    else:
        significant = coef_df[coef_df['p_value']<= 0.05]
        if settings['controls'] is not None:
            significant_high = significant[significant['coefficient'] >= reg_threshold]
            significant_low = significant[significant['coefficient'] <= reg_threshold]
            significant = pd.concat([significant_high, significant_low])
        significant.sort_values(by='coefficient', ascending=False, inplace=True)
        significant = significant[~significant['feature'].str.contains('row|column')]
        
    if regression_type in ['ols', 'beta']:
        if settings['verbose']:
            print(model.summary())
            save_summary_to_file(model, file_path=f'{res_folder}/mode_summary.csv')
    
    significant.to_csv(hits_path, index=False)
    significant_grna_filtered = significant[significant['n_grna'] > settings['min_n']]
    significant_gene_filtered = significant[significant['n_gene'] > settings['min_n']]
    significant_filtered = pd.concat([significant_grna_filtered, significant_gene_filtered])
    filtered_hit_path = os.path.join(os.path.dirname(hits_path), 'results_significant_filtered.csv')
    significant_filtered.to_csv(filtered_hit_path, index=False)

    if isinstance(settings['metadata_files'], str):
        settings['metadata_files'] = [settings['metadata_files']]

    for metadata_file in settings['metadata_files']:
        file = os.path.basename(metadata_file)
        filename, _ = os.path.splitext(file)
        _ = merge_regression_res_with_metadata(hits_path, metadata_file, name=filename)
        merged_df = merge_regression_res_with_metadata(results_path, metadata_file, name=filename)
        gene_merged_df = merge_regression_res_with_metadata(results_path_gene, metadata_file, name=filename)
        grna_merged_df = merge_regression_res_with_metadata(results_path_grna, metadata_file, name=filename)

    if settings['toxo']:
        data_path = merged_df
        data_path_gene = gene_merged_df
        data_path_grna = grna_merged_df
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, 'resources', 'data', 'lopit.csv')
        
        if settings['volcano'] == 'all':
            print('all')
            gene_list = custom_volcano_plot(data_path, metadata_path, metadata_column='tagm_location', point_size=600, figsize=20, threshold=reg_threshold, save_path=volcano_path, x_lim=settings['x_lim'],y_lims=settings['y_lims'])
            display(gene_list)
        elif settings['volcano'] == 'gene':
            print('gene')
            gene_list = custom_volcano_plot(data_path_gene, metadata_path, metadata_column='tagm_location', point_size=600, figsize=20, threshold=reg_threshold, save_path=volcano_path, x_lim=settings['x_lim'],y_lims=settings['y_lims'])
            display(gene_list)
        elif settings['volcano'] == 'grna':
            print('grna')
            gene_list = custom_volcano_plot(data_path_grna, metadata_path, metadata_column='tagm_location', point_size=600, figsize=20, threshold=reg_threshold, save_path=volcano_path, x_lim=settings['x_lim'],y_lims=settings['y_lims'])
            display(gene_list)
        
        phenotype_plot = os.path.join(res_folder,'phenotype_plot.pdf')
        transcription_heatmap = os.path.join(res_folder,'transcription_heatmap.pdf')
        data_GT1 = pd.read_csv(settings['metadata_files'][1], low_memory=False)
        data_ME49 = pd.read_csv(settings['metadata_files'][0], low_memory=False)
        
        columns = ['sense - Tachyzoites', 'sense - Tissue cysts', 'sense - EES1', 'sense - EES2', 'sense - EES3', 'sense - EES4', 'sense - EES5']
        
        print('Plotting gene phenotypes and heatmaps')
        print(gene_list)

        plot_gene_phenotypes(data=data_GT1, gene_list=gene_list, save_path=phenotype_plot)
        plot_gene_heatmaps(data=data_ME49, gene_list=gene_list, columns=columns, x_column='Gene ID', normalize=True, save_path=transcription_heatmap)
        
        #if len(significant) > 2:
        #    metadata_path = os.path.join(base_dir, 'resources', 'data', 'toxoplasma_metadata.csv')
        #    go_term_enrichment_by_column(significant, metadata_path)
    
    print('Significant Genes')
    grnas = significant['grna'].unique().tolist()
    genes = significant['gene'].unique().tolist()
    print(f"Found p<0.05 coedfficients for {len(grnas)} gRNAs and {len(genes)} genes")
    display(significant)

    output = {'results':coef_df,
              'significant':significant}

    return output

def process_reads(csv_path, fraction_threshold, plate, filter_column=None, filter_value=None):
    
    from .utils import correct_metadata

    if isinstance(csv_path, pd.DataFrame):
        csv_df = csv_path
    else:
        # Read the CSV file into a DataFrame
        csv_df = pd.read_csv(csv_path)

    csv_df = correct_metadata(csv_df)    
    
    if 'grna_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'grna_name': 'grna'})
    if 'plate_row' in csv_df.columns:
        csv_df[['plateID', 'rowID']] = csv_df['plate_row'].str.split('_', expand=True)

    if not 'plateID' in csv_df.columns:
        if not plate is None:
            csv_df['plateID'] = plate
        else:
            csv_df['plateID'] = 'plate1'
            
    if 'prcfo' in csv_df.columns:
        #csv_df = csv_df.loc[:, ~csv_df.columns.duplicated()].copy()
        csv_df[['plateID', 'rowID', 'columnID', 'fieldID', 'objectID']] = csv_df['prcfo'].str.split('_', expand=True)
        csv_df['prc'] = csv_df['plateID'].astype(str) + '_' + csv_df['rowID'].astype(str) + '_' + csv_df['columnID'].astype(str)

    if isinstance(filter_column, str):
        filter_column = [filter_column]

    if isinstance(filter_value, str):
        filter_value = [filter_value]
            
    if isinstance(filter_column, list):            
        for filter_col in filter_column:
            for value in filter_value:
                csv_df = csv_df[csv_df[filter_col] != value]

    # Ensure the necessary columns are present
    if not all(col in csv_df.columns for col in ['rowID','columnID','grna','count']):
        raise ValueError("The CSV file must contain 'grna', 'count', 'rowID', and 'columnID' columns.")

    # Create the prc column
    csv_df['prc'] = csv_df['plateID'] + '_' + csv_df['rowID'] + '_' + csv_df['columnID']

    # Group by prc and calculate the sum of counts
    grouped_df = csv_df.groupby('prc')['count'].sum().reset_index()
    grouped_df = grouped_df.rename(columns={'count': 'total_counts'})
    merged_df = pd.merge(csv_df, grouped_df, on='prc')
    merged_df['fraction'] = merged_df['count'] / merged_df['total_counts']

    # Filter rows with fraction under the threshold
    if fraction_threshold is not None:
        observations_before = len(merged_df)
        merged_df = merged_df[merged_df['fraction'] >= fraction_threshold]
        observations_after = len(merged_df)
        removed = observations_before - observations_after
        print(f'Removed {removed} observation below fraction threshold: {fraction_threshold}')

    merged_df = merged_df[['prc', 'grna', 'fraction']]

    if not all(col in merged_df.columns for col in ['grna', 'gene']):
        try:
            merged_df[['org', 'gene', 'grna']] = merged_df['grna'].str.split('_', expand=True)
            merged_df = merged_df.drop(columns=['org'])
            merged_df['grna'] = merged_df['gene'] + '_' + merged_df['grna']
        except:
            print('Error splitting grna into org, gene, grna.')

    return merged_df

def apply_transformation(X, transform):
    if transform == 'log':
        transformer = FunctionTransformer(np.log1p, validate=True)
    elif transform == 'sqrt':
        transformer = FunctionTransformer(np.sqrt, validate=True)
    elif transform == 'square':
        transformer = FunctionTransformer(np.square, validate=True)
    else:
        transformer = None
    return transformer

def check_normality(data, variable_name, verbose=False):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    stat, p_value = shapiro(data)
    if verbose:
        print(f"Shapiro-Wilk Test for {variable_name}:\nStatistic: {stat}, P-value: {p_value}")
    if p_value > 0.05:
        if verbose:
            print(f"Normal distribution: The data for {variable_name} is normally distributed.")
        return True
    else:
        if verbose:
            print(f"Normal distribution: The data for {variable_name} is not normally distributed.")
        return False

def clean_controls(df,values, column):
    if column in df.columns:
        if isinstance(values, list):
            for value in values:
                df = df[~df[column].isin([value])]
                print(f'Removed data from {value}')
    return df

def process_scores(df, dependent_variable, plate, min_cell_count=25, agg_type='mean', transform=None, regression_type='ols'):
    from .utils import calculate_shortest_distance, correct_metadata
    df = df.reset_index(drop=True)
    if 'prcfo' in df.columns:
        df = df.loc[:, ~df.columns.duplicated()].copy()
        if not all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df[['plateID', 'rowID', 'columnID', 'fieldID', 'objectID']] = df['prcfo'].str.split('_', expand=True)
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
    else:
        
         
        df = correct_metadata(df)
        
            
        if not plate is None:
            df['plateID'] = plate
    
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
        else:
            raise ValueError("The DataFrame must contain 'plateID', 'rowID', and 'columnID' columns.")

    df = df[['prc', dependent_variable]]
    # Group by prc and calculate the mean and count of the dependent_variable
    grouped = df.groupby('prc')[dependent_variable]
        
    if regression_type != 'poisson':
    
        print(f'Using agg_type: {agg_type}')

        if agg_type == 'median':
            dependent_df = grouped.median().reset_index()
        elif agg_type == 'mean':
            dependent_df = grouped.mean().reset_index()
        elif agg_type == 'quantile':
            dependent_df = grouped.quantile(0.75).reset_index()
        elif agg_type == None:
            dependent_df = df.reset_index()
            if 'prcfo' in dependent_df.columns:
                dependent_df = dependent_df.drop(columns=['prcfo'])
        else:
            raise ValueError(f"Unsupported aggregation type {agg_type}")
            
    if regression_type == 'poisson':
        agg_type = 'count'
        print(f'Using agg_type: {agg_type} for poisson regression')
        dependent_df = grouped.sum().reset_index()        
        
    # Calculate cell_count for all cases
    cell_count = grouped.size().reset_index(name='cell_count')

    if agg_type is None:
        dependent_df = pd.merge(dependent_df, cell_count, on='prc')
    else:
        dependent_df['cell_count'] = cell_count['cell_count']

    dependent_df = dependent_df[dependent_df['cell_count'] >= min_cell_count]

    is_normal = check_normality(dependent_df[dependent_variable], dependent_variable)

    if not transform is None:
        transformer = apply_transformation(dependent_df[dependent_variable], transform=transform)
        transformed_var = f'{transform}_{dependent_variable}'
        dependent_df[transformed_var] = transformer.fit_transform(dependent_df[[dependent_variable]])
        dependent_variable = transformed_var
        is_normal = check_normality(dependent_df[transformed_var], transformed_var)

    if not is_normal:
        print(f'{dependent_variable} is not normally distributed')
    else:
        print(f'{dependent_variable} is normally distributed')

    return dependent_df, dependent_variable

def generate_ml_scores(settings):
    
    from .io import _read_and_merge_data, _read_db
    from .plot import plot_plates
    from .utils import get_ml_results_paths, add_column_to_database, calculate_shortest_distance
    from .settings import set_default_analyze_screen

    settings = set_default_analyze_screen(settings)

    srcs = settings['src']

    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    display(settings_df)
    
    if isinstance(srcs, str):
        srcs = [srcs]
    
    df = pd.DataFrame()
    for idx, src in enumerate(srcs):
        
        if idx == 0:
            src1 = src

        db_loc = [src+'/measurements/measurements.db']
        tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
        
        dft, _ = _read_and_merge_data(db_loc, 
                                    tables,
                                    settings['verbose'],
                                    nuclei_limit=settings['nuclei_limit'],
                                    pathogen_limit=settings['pathogen_limit'])
        df = pd.concat([df, dft])
    
    try:
        df = calculate_shortest_distance(df, 'pathogen', 'nucleus')
    except Exception as e:
        print(e)
    
    if settings['annotation_column'] is not None:

        settings['location_column'] = settings['annotation_column']
        
        png_list_df = _read_db(db_loc[0], tables=['png_list'])[0]
        if not {'prcfo', settings['annotation_column']}.issubset(png_list_df.columns):
            raise ValueError("The 'png_list_df' DataFrame must contain 'prcfo' and 'test' columns.")
        annotated_df = png_list_df[['prcfo', settings['annotation_column']]].set_index('prcfo')
        df = annotated_df.merge(df, left_index=True, right_index=True)
        unique_values = df[settings['annotation_column']].dropna().unique()
        
        if len(unique_values) == 1:
            unannotated_rows = df[df[settings['annotation_column']].isna()].index
            existing_value = unique_values[0]
            next_value = existing_value + 1

            settings['positive_control'] = str(existing_value)
            settings['negative_control'] = str(next_value)

            existing_count = df[df[settings['annotation_column']] == existing_value].shape[0]
            num_to_select = min(existing_count, len(unannotated_rows))
            selected_rows = np.random.choice(unannotated_rows, size=num_to_select, replace=False)
            df.loc[selected_rows, settings['annotation_column']] = next_value

            # Print the counts for existing_value and next_value
            existing_count_final = df[df[settings['annotation_column']] == existing_value].shape[0]
            next_count_final = df[df[settings['annotation_column']] == next_value].shape[0]

            print(f"Number of rows with value {existing_value}: {existing_count_final}")
            print(f"Number of rows with value {next_value}: {next_count_final}")
            df[settings['annotation_column']] = df[settings['annotation_column']].apply(str)
    
    if settings['channel_of_interest'] in [0,1,2,3]:
        if f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity" and f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity" in df.columns:
            df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"]/df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
    
    output, figs = ml_analysis(df,
                               settings['channel_of_interest'],
                               settings['location_column'],
                               settings['positive_control'],
                               settings['negative_control'],
                               settings['exclude'],
                               settings['n_repeats'],
                               settings['top_features'],
                               settings['reg_alpha'],
                               settings['reg_lambda'],
                               settings['learning_rate'],                               
                               settings['n_estimators'],
                               settings['test_size'],
                               settings['model_type_ml'],
                               settings['n_jobs'],
                               settings['remove_low_variance_features'],
                               settings['remove_highly_correlated_features'],
                               settings['prune_features'],
                               settings['cross_validation'],
                               settings['verbose'])
    
    shap_fig = shap_analysis(output[3], output[4], output[5])

    features = output[0].select_dtypes(include=[np.number]).columns.tolist()
    train_features_df = pd.DataFrame(output[9], columns=['feature'])
    
    if not settings['heatmap_feature'] in features:
        raise ValueError(f"Variable {settings['heatmap_feature']} not found in the dataframe. Please choose one of the following: {features}")
    
    plate_heatmap = plot_plates(df=output[0],
                                variable=settings['heatmap_feature'],
                                grouping=settings['grouping'],
                                min_max=settings['min_max'],
                                cmap=settings['cmap'],
                                min_count=settings['minimum_cell_count'],
                                verbose=settings['verbose'])

    data_path, permutation_path, feature_importance_path, model_metricks_path, permutation_fig_path, feature_importance_fig_path, shap_fig_path, plate_heatmap_path, settings_csv, ml_features = get_ml_results_paths(src1, settings['model_type_ml'], settings['channel_of_interest'])
    df, permutation_df, feature_importance_df, _, _, _, _, _, metrics_df, _ = output

    settings_df.to_csv(settings_csv, index=False)
    df.to_csv(data_path, mode='w', encoding='utf-8')
    permutation_df.to_csv(permutation_path, mode='w', encoding='utf-8')
    feature_importance_df.to_csv(feature_importance_path, mode='w', encoding='utf-8')
    train_features_df.to_csv(ml_features, mode='w', encoding='utf-8')
    metrics_df.to_csv(model_metricks_path, mode='w', encoding='utf-8')

    plate_heatmap.savefig(plate_heatmap_path, format='pdf')
    figs[0].savefig(permutation_fig_path, format='pdf')
    figs[1].savefig(feature_importance_fig_path, format='pdf')
    shap_fig.savefig(shap_fig_path, format='pdf')

    if settings['save_to_db']:
        settings['csv_path'] = data_path
        settings['db_path'] = os.path.join(src1, 'measurements', 'measurements.db')
        settings['table_name'] = 'png_list'
        settings['update_column'] = 'predictions'
        settings['match_column'] = 'prcfo'
        add_column_to_database(settings)

    return [output, plate_heatmap]

def ml_analysis(df, channel_of_interest=3, location_column='columnID', positive_control='c2', negative_control='c1', exclude=None, n_repeats=10, top_features=30, reg_alpha=0.1, reg_lambda=1.0, learning_rate=0.00001, n_estimators=1000, test_size=0.2, model_type='xgboost', n_jobs=-1, remove_low_variance_features=True, remove_highly_correlated_features=True, prune_features=False, cross_validation=False, verbose=False):
    
    """
    Calculates permutation importance for numerical features in the dataframe,
    comparing groups based on specified column values and uses the model to predict 
    the class for all other rows in the dataframe.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    feature_string (str): String to filter features that contain this substring.
    location_column (str): Column name to use for comparing groups.
    positive_control, negative_control (str): Values in location_column to create subsets for comparison.
    exclude (list or str, optional): Columns to exclude from features.
    n_repeats (int): Number of repeats for permutation importance.
    top_features (int): Number of top features to plot based on permutation importance.
    n_estimators (int): Number of trees in the random forest, gradient boosting, or XGBoost model.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    model_type (str): Type of model to use ('random_forest', 'logistic_regression', 'gradient_boosting', 'xgboost').
    n_jobs (int): Number of jobs to run in parallel for applicable models.

    Returns:
    pandas.DataFrame: The original dataframe with added prediction and data usage columns.
    pandas.DataFrame: DataFrame containing the importances and standard deviations.
    """

    from .utils import filter_dataframe_features
    from .plot import plot_permutation, plot_feature_importance

    random_state = 42
    
    if 'cells_per_well' in df.columns:
        df = df.drop(columns=['cells_per_well'])

    df_metadata = df[[location_column]].copy()

    df, features = filter_dataframe_features(df, channel_of_interest, exclude, remove_low_variance_features, remove_highly_correlated_features, verbose)
    print('After filtration:', len(df))
    
    if verbose:
        print(f'Found {len(features)} numerical features in the dataframe')
        print(f'Features used in training: {features}')
        print(f'Features: {features}')
        
    df = pd.concat([df, df_metadata[location_column]], axis=1)

    # Subset the dataframe based on specified column values
    if isinstance(negative_control, str):
        df1 = df[df[location_column] == negative_control].copy()
    elif isinstance(negative_control, list):
        df1 = df[df[location_column].isin(negative_control)].copy()
    
    if isinstance(positive_control, str):
        df2 = df[df[location_column] == positive_control].copy()
    elif isinstance(positive_control, list):
        df2 = df[df[location_column].isin(positive_control)].copy()
        
    # Create target variable
    df1['target'] = 0 # Negative control
    df2['target'] = 1 # Positive control

    # Combine the subsets for analysis
    combined_df = pd.concat([df1, df2])
    combined_df = combined_df.drop(columns=[location_column])
    
    if verbose:
        print(f'Found {len(df1)} samples for {negative_control} and {len(df2)} samples for {positive_control}. Total: {len(combined_df)}')
    
    X = combined_df[features]
    y = combined_df['target']
    
    if prune_features:
        before_pruning = len(X.columns)
        selector = SelectKBest(score_func=f_classif, k=top_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get the selected feature names
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        features = selected_features.tolist()
        
        after_pruning = len(X.columns)
        print(f"Removed {before_pruning - after_pruning} features using SelectKBest")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Add data usage labels
    combined_df['data_usage'] = 'train'
    combined_df.loc[X_test.index, 'data_usage'] = 'test'
    df['data_usage'] = 'not_used'
    df.loc[combined_df.index, 'data_usage'] = combined_df['data_usage']
    
    # Initialize the model based on model_type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'gradient_boosting':
        model = HistGradientBoostingClassifier(max_iter=n_estimators, random_state=random_state)  # Supports n_jobs internally
    elif model_type == 'xgboost':
        model = XGBClassifier(reg_alpha=reg_alpha, reg_lambda=reg_lambda, learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state, nthread=n_jobs, use_label_encoder=False, eval_metric='logloss')
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Perform k-fold cross-validation
    if cross_validation:
        
        # Cross-validation setup
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        fold_metrics = []

        for fold_idx, (train_index, test_index) in enumerate(kfold.split(X, y), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Predict for the current test set
            predictions_test = model.predict(X_test)
            combined_df.loc[X_test.index, 'predictions'] = predictions_test

            # Get prediction probabilities for the test set
            prediction_probabilities_test = model.predict_proba(X_test)

            # Find the optimal threshold
            optimal_threshold = find_optimal_threshold(y_test, prediction_probabilities_test[:, 1])
            if verbose:
                print(f'Fold {fold_idx} - Optimal threshold: {optimal_threshold}')

            # Assign predictions and probabilities to the test set in the DataFrame
            df.loc[X_test.index, 'predictions'] = predictions_test
            for i in range(prediction_probabilities_test.shape[1]):
                df.loc[X_test.index, f'prediction_probability_class_{i}'] = prediction_probabilities_test[:, i]

            # Evaluate performance for the current fold
            fold_report = classification_report(y_test, predictions_test, output_dict=True)
            fold_metrics.append(pd.DataFrame(fold_report).transpose())

            if verbose:
                print(f"Fold {fold_idx} Classification Report:")
                print(classification_report(y_test, predictions_test))

        # Aggregate metrics across all folds
        metrics_df = pd.concat(fold_metrics).groupby(level=0).mean()

        # Re-train on full data (X, y) and then apply to entire df
        model.fit(X, y)  
        all_predictions = model.predict(df[features])  # Predict on entire df
        df['predictions'] = all_predictions

        # Get prediction probabilities for all rows in df
        prediction_probabilities = model.predict_proba(df[features])
        for i in range(prediction_probabilities.shape[1]):
            df[f'prediction_probability_class_{i}'] = prediction_probabilities[:, i]

        #if verbose:
        #    print("\nFinal Classification Report on Full Dataset:")
        #    print(classification_report(y, all_predictions))

        # Generate metrics DataFrame
        #final_report_dict = classification_report(y, all_predictions, output_dict=True)
        #metrics_df = pd.DataFrame(final_report_dict).transpose()
    
    else:
        model.fit(X_train, y_train)
        # Predicting the target variable for the test set
        predictions_test = model.predict(X_test)
        combined_df.loc[X_test.index, 'predictions'] = predictions_test

        # Get prediction probabilities for the test set
        prediction_probabilities_test = model.predict_proba(X_test)

        # Find the optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, prediction_probabilities_test[:, 1])
        if verbose:
            print(f'Optimal threshold: {optimal_threshold}')

        # Predicting the target variable for all other rows in the dataframe
        X_all = df[features]
        all_predictions = model.predict(X_all)
        df['predictions'] = all_predictions

        # Get prediction probabilities for all rows in the dataframe
        prediction_probabilities = model.predict_proba(X_all)
        for i in range(prediction_probabilities.shape[1]):
            df[f'prediction_probability_class_{i}'] = prediction_probabilities[:, i]
            
        if verbose:
            print("\nClassification Report:")
            print(classification_report(y_test, predictions_test))
            
        report_dict = classification_report(y_test, predictions_test, output_dict=True)
        metrics_df = pd.DataFrame(report_dict).transpose()
        
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)

    # Create a DataFrame for permutation importances
    permutation_df = pd.DataFrame({
        'feature': [features[i] for i in perm_importance.importances_mean.argsort()],
        'importance_mean': perm_importance.importances_mean[perm_importance.importances_mean.argsort()],
        'importance_std': perm_importance.importances_std[perm_importance.importances_mean.argsort()]
    }).tail(top_features)

    permutation_fig = plot_permutation(permutation_df)
    if verbose:
        permutation_fig.show()

    # Feature importance for models that support it
    if model_type in ['random_forest', 'xgboost', 'gradient_boosting']:
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False).head(top_features)
        
        feature_importance_fig = plot_feature_importance(feature_importance_df)
        if verbose:
            feature_importance_fig.show()

    else:
        feature_importance_df = pd.DataFrame()

    df = _calculate_similarity(df, features, location_column, positive_control, negative_control)

    df['prcfo'] = df.index.astype(str)
    df[['plateID', 'rowID', 'columnID', 'fieldID', 'object']] = df['prcfo'].str.split('_', expand=True)
    df['prc'] = df['plateID'] + '_' + df['rowID'] + '_' + df['columnID']
    
    return [df, permutation_df, feature_importance_df, model, X_train, X_test, y_train, y_test, metrics_df, features], [permutation_fig, feature_importance_fig]

def shap_analysis(model, X_train, X_test):
    
    """
    Performs SHAP analysis on the given model and data.

    Args:
    model: The trained model.
    X_train (pandas.DataFrame): Training feature set.
    X_test (pandas.DataFrame): Testing feature set.
    Returns:
    fig: Matplotlib figure object containing the SHAP summary plot.
    """
    
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    # Create a new figure
    fig, ax = plt.subplots()
    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    # Save the current figure (the one that SHAP just created)
    fig = plt.gcf()
    plt.close(fig)  # Close the figure to prevent it from displaying immediately
    return fig

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the optimal threshold for binary classification based on the F1-score.

    Args:
    y_true (array-like): True binary labels.
    y_pred_proba (array-like): Predicted probabilities for the positive class.

    Returns:
    float: The optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def _calculate_similarity(df, features, col_to_compare, val1, val2):
    """
    Calculate similarity scores of each well to the positive and negative controls using various metrics.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    features (list): List of feature columns to use for similarity calculation.
    col_to_compare (str): Column name to use for comparing groups.
    val1, val2 (str): Values in col_to_compare to create subsets for comparison.

    Returns:
    pandas.DataFrame: DataFrame with similarity scores.
    """
    # Separate positive and negative control wells
    if isinstance(val1, str):
        pos_control = df[df[col_to_compare] == val1][features].mean()
    elif isinstance(val1, list):
        pos_control = df[df[col_to_compare].isin(val1)][features].mean()
    if isinstance(val2, str):
        neg_control = df[df[col_to_compare] == val2][features].mean()
    elif isinstance(val2, list):
        neg_control = df[df[col_to_compare].isin(val2)][features].mean()
    
    # Standardize features for Mahalanobis distance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Regularize the covariance matrix to avoid singularity
    cov_matrix = np.cov(scaled_features, rowvar=False)
    inv_cov_matrix = None
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add a small value to the diagonal elements for regularization
        epsilon = 1e-5
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon)
        
    # Calculate similarity scores
    def safe_similarity(func, row, control, *args, **kwargs):
        try:
            return func(row, control, *args, **kwargs)
        except Exception:
            return np.nan
        
    # Calculate similarity scores
    try:
        df['similarity_to_pos_euclidean'] = df[features].apply(lambda row: safe_similarity(euclidean, row, pos_control), axis=1)
        df['similarity_to_neg_euclidean'] = df[features].apply(lambda row: safe_similarity(euclidean, row, neg_control), axis=1)
        df['similarity_to_pos_cosine'] = df[features].apply(lambda row: safe_similarity(cosine, row, pos_control), axis=1)
        df['similarity_to_neg_cosine'] = df[features].apply(lambda row: safe_similarity(cosine, row, neg_control), axis=1)
        df['similarity_to_pos_mahalanobis'] = df[features].apply(lambda row: safe_similarity(mahalanobis, row, pos_control, inv_cov_matrix), axis=1)
        df['similarity_to_neg_mahalanobis'] = df[features].apply(lambda row: safe_similarity(mahalanobis, row, neg_control, inv_cov_matrix), axis=1)
        df['similarity_to_pos_manhattan'] = df[features].apply(lambda row: safe_similarity(cityblock, row, pos_control), axis=1)
        df['similarity_to_neg_manhattan'] = df[features].apply(lambda row: safe_similarity(cityblock, row, neg_control), axis=1)
        df['similarity_to_pos_minkowski'] = df[features].apply(lambda row: safe_similarity(minkowski, row, pos_control, p=3), axis=1)
        df['similarity_to_neg_minkowski'] = df[features].apply(lambda row: safe_similarity(minkowski, row, neg_control, p=3), axis=1)
        df['similarity_to_pos_chebyshev'] = df[features].apply(lambda row: safe_similarity(chebyshev, row, pos_control), axis=1)
        df['similarity_to_neg_chebyshev'] = df[features].apply(lambda row: safe_similarity(chebyshev, row, neg_control), axis=1)
        df['similarity_to_pos_braycurtis'] = df[features].apply(lambda row: safe_similarity(braycurtis, row, pos_control), axis=1)
        df['similarity_to_neg_braycurtis'] = df[features].apply(lambda row: safe_similarity(braycurtis, row, neg_control), axis=1)
    except Exception as e:
        print(f"Error calculating similarity scores: {e}")    
    return df

def interperate_vision_model(settings={}):
    
    from .io import _read_and_merge_data, _results_to_csv
    from .settings import set_interperate_vision_model_defaults
    from .utils import save_settings
    
    settings = set_interperate_vision_model_defaults(settings)
    save_settings(settings, name='interperate_vision_model', show=True)
    
    # Function to create radar plot for individual and combined values
    def create_extended_radar_plot(values, labels, title):
        values = list(values) + [values[0]]  # Close the loop for radar chart
        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.show()

    def extract_compartment_channel(feature_name):
        # Identify compartment as the first part before an underscore
        compartment = feature_name.split('_')[0]
        
        if compartment == 'cells':
            compartment = 'cell'

        # Identify channels based on substring presence
        channels = []
        if 'channel_0' in feature_name:
            channels.append('channel_0')
        if 'channel_1' in feature_name:
            channels.append('channel_1')
        if 'channel_2' in feature_name:
            channels.append('channel_2')
        if 'channel_3' in feature_name:
            channels.append('channel_3')

        # If multiple channels are found, join them with a '+'
        if channels:
            channel = ' + '.join(channels)
        else:
            channel = 'morphology'  # Use 'morphology' if no channel identifier is found

        return (compartment, channel)

    def read_and_preprocess_data(settings):

        df, _ = _read_and_merge_data(
            locs=[settings['src']+'/measurements/measurements.db'], 
            tables=settings['tables'], 
            verbose=True, 
            nuclei_limit=settings['nuclei_limit'], 
            pathogen_limit=settings['pathogen_limit']
        )

        scores_df = pd.read_csv(settings['scores'])

        # Clean and align columns for merging
        df['object_label'] = df['object_label'].str.replace('o', '')

        if 'rowID' not in scores_df.columns:
            if 'row' in scores_df.columns:
                scores_df['rowID'] = scores_df['row']
            if 'row_name' in scores_df.columns:
                scores_df['rowID'] = scores_df['row_name']

        if 'columnID' not in scores_df.columns:
            if 'col' in scores_df.columns:
                scores_df['columnID'] = scores_df['col']
            if 'column' in scores_df.columns:
                scores_df['columnID'] = scores_df['column']

        if 'object_label' not in scores_df.columns:
            scores_df['object_label'] = scores_df['object']

        # Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type
        df['object_label'] = df['object_label'].str.replace('o', '').astype(str)

        # Ensure 'object_label' in scores_df is also a string
        scores_df['object_label'] = scores_df['object'].astype(str)

        # Ensure all join columns have the same data type in both DataFrames
        df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']] = df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']].astype(str)
        scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']] = scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']].astype(str)

        # Select only the necessary columns from scores_df for merging
        scores_df = scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label', settings['score_column']]]

        # Now merge DataFrames
        merged_df = pd.merge(df, scores_df, on=['plateID', 'rowID', 'columnID', 'fieldID', 'object_label'], how='inner')

        # Separate numerical features and the score column
        X = merged_df.select_dtypes(include='number').drop(columns=[settings['score_column']])
        y = merged_df[settings['score_column']]

        return X, y, merged_df
    
    X, y, merged_df = read_and_preprocess_data(settings)
    
    # Step 1: Feature Importance using Random Forest
    if settings['feature_importance'] or settings['feature_importance']:
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X, y)
        
        if settings['feature_importance']:
            print(f"Feature Importance ...")
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            top_feature_importance_df = feature_importance_df.head(settings['top_features'])

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(top_feature_importance_df['feature'], top_feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f"Top {settings['top_features']} Features - Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()
        
        if settings['save']:
            _results_to_csv(feature_importance_df, filename='feature_importance.csv')
    
    # Step 2: Permutation Importance
    if settings['permutation_importance']:
        print(f"Permutation Importance ...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=settings['n_jobs'])
        perm_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='importance', ascending=False)
        top_perm_importance_df = perm_importance_df.head(settings['top_features'])

        # Plot Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(top_perm_importance_df['feature'], top_perm_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f"Top {settings['top_features']} Features - Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()
        
        if settings['save']:
            _results_to_csv(perm_importance_df, filename='permutation_importance.csv')
    
    # Step 3: SHAP Analysis
    if settings['shap']:
        print(f"SHAP Analysis ...")

        # Select top N features based on Random Forest importance and fit the model on these features only
        top_features = feature_importance_df.head(settings['top_features'])['feature']
        X_top = X[top_features]

        # Refit the model on this subset of features
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X_top, y)

        # Sample a smaller subset of rows to speed up SHAP
        if settings['shap_sample']:
            sample = int(len(X_top) / 100)
            X_sample = X_top.sample(min(sample, len(X_top)), random_state=42)
        else:
            X_sample = X_top

        # Initialize SHAP explainer with the same subset of features
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample, max_evals=1500)

        # Plot SHAP summary for the selected sample and top features
        shap.summary_plot(shap_values, X_sample, max_display=settings['top_features'])

        # Convert SHAP values to a DataFrame for easier manipulation
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        
        # Apply the function to create MultiIndex columns with compartment and channel
        shap_df.columns = pd.MultiIndex.from_tuples(
            [extract_compartment_channel(feat) for feat in shap_df.columns], 
            names=['compartment', 'channel']
        )
        
        # Aggregate SHAP values by compartment and channel
        compartment_mean = shap_df.abs().groupby(level='compartment', axis=1).mean().mean(axis=0)
        channel_mean = shap_df.abs().groupby(level='channel', axis=1).mean().mean(axis=0)

        # Calculate combined importance for each pair of compartments and channels
        combined_compartment = {}
        for i, comp1 in enumerate(compartment_mean.index):
            for comp2 in compartment_mean.index[i+1:]:
                combined_compartment[f"{comp1} + {comp2}"] = shap_df.loc[:, (comp1, slice(None))].abs().mean().mean() + \
                                                              shap_df.loc[:, (comp2, slice(None))].abs().mean().mean()
        
        combined_channel = {}
        for i, chan1 in enumerate(channel_mean.index):
            for chan2 in channel_mean.index[i+1:]:
                combined_channel[f"{chan1} + {chan2}"] = shap_df.loc[:, (slice(None), chan1)].abs().mean().mean() + \
                                                          shap_df.loc[:, (slice(None), chan2)].abs().mean().mean()

        # Prepare values and labels for radar charts
        all_compartment_importance = list(compartment_mean.values) + list(combined_compartment.values())
        all_compartment_labels = list(compartment_mean.index) + list(combined_compartment.keys())

        all_channel_importance = list(channel_mean.values) + list(combined_channel.values())
        all_channel_labels = list(channel_mean.index) + list(combined_channel.keys())

        # Create radar plots for compartments and channels
        create_extended_radar_plot(all_compartment_importance, all_compartment_labels, "SHAP Importance by Compartment (Individual and Combined)")
        create_extended_radar_plot(all_channel_importance, all_channel_labels, "SHAP Importance by Channel (Individual and Combined)")
    
    return merged_df