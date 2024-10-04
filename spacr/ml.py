import os, shap
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro

import matplotlib.pyplot as plt
from IPython.display import display

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import FunctionTransformer
from patsy import dmatrices

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cosine, euclidean, mahalanobis, cityblock, minkowski, chebyshev, hamming, jaccard, braycurtis

from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

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

    # Define the dictionary with callables (lambdas) to delay evaluation
    model_map = {
        'ols': lambda: sm.OLS(y, X).fit(cov_type=cov_type) if cov_type else sm.OLS(y, X).fit(),
        'gls': lambda: sm.GLS(y, X).fit(),
        'wls': lambda: sm.WLS(y, X, weights=1 / np.sqrt(X.iloc[:, 1])).fit(),
        'rlm': lambda: sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(),
        'glm': lambda: sm.GLM(y, X, family=sm.families.Gaussian()).fit(),
        'quantile': lambda: sm.QuantReg(y, X).fit(q=alpha),
        'logit': lambda: sm.Logit(y, X).fit(),
        'probit': lambda: sm.Probit(y, X).fit(),
        'poisson': lambda: sm.Poisson(y, X).fit(),
        'lasso': lambda: Lasso(alpha=alpha).fit(X, y),
        'ridge': lambda: Ridge(alpha=alpha).fit(X, y)
    }

    # Call the appropriate model only when needed
    if regression_type in model_map:
        model = model_map[regression_type]() 
    elif regression_type == 'mixed':
        model = perform_mixed_model(y, X, groups, alpha=alpha)
    else:
        raise ValueError(f"Unsupported regression type {regression_type}")

    if regression_type in ['lasso', 'ridge']:
        plot_regression_line(X, y, model)

    return model

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

def process_model_coefficients(model, regression_type, X, y, highlight):
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
    coef_df['highlight'] = coef_df['feature'].apply(lambda x: highlight in x)

    return coef_df[~coef_df['feature'].str.contains('row|column')]

def prepare_formula(dependent_variable, random_row_column_effects=False):
    """Return the regression formula using random effects for plate, row, and column."""
    if random_row_column_effects:
        # Random effects for row and column + gene weighted by gene_fraction + grna weighted by fraction
        return f'{dependent_variable} ~ fraction:grna + gene_fraction:gene'
    return f'{dependent_variable} ~ fraction:grna + gene_fraction:gene + row + column'

def fit_mixed_model(df, formula, dst):
    from .plot import plot_histogram

    """Fit the mixed model with plate, row, and column as random effects and return results."""
    # Specify random effects for plate, row, and column
    model = smf.mixedlm(formula, 
                        data=df, 
                        groups=df['plate'], 
                        re_formula="1 + row + column", 
                        vc_formula={"row": "0 + row", "column": "0 + column"})
    
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
    df = ensure_valid_types(df, ['grna', 'gene', 'plate', 'row', 'column', 'prc'])
    
    # Step 3: Check for multicollinearity in fraction and the dependent variable
    df_cleaned = check_collinearity(df, ['fraction', dependent_variable])
    
    # Ensure that the prc, plate, row, and column columns are still included for random effects
    df_cleaned['gene'] = df['gene']
    df_cleaned['grna'] = df['grna']
    df_cleaned['prc'] = df['prc']
    df_cleaned['plate'] = df['plate']
    df_cleaned['row'] = df['row']
    df_cleaned['column'] = df['column']

    #display(df_cleaned)

    # Create a new column 'gene_fraction' that sums the fractions by gene within the same well
    df_cleaned['gene_fraction'] = df_cleaned.groupby(['prc', 'gene'])['fraction'].transform('sum')

    print("Data is ready for model fitting.")
    return df_cleaned

def regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0, random_row_column_effects=False, highlight='220950', dst=None, cov_type=None):
    from .plot import volcano_plot, plot_histogram

    # Generate the volcano filename
    volcano_path = create_volcano_filename(csv_path, regression_type, alpha, dst)

    # Check if the data is normally distributed
    is_normal = check_normality(df[dependent_variable], dependent_variable)

    if is_normal:
        print(f"To avoid violating assumptions, it is recommended to use a regression model that assumes normality.")
        print(f"Recommended regression type: ols (Ordinary Least Squares)")
    else:
        print(f"To avoid violating assumptions, it is recommended to use a regression model that does not assume normality.")
        print(f"Recommended regression type: glm (Generalized Linear Model)")

    # Determine regression type if not specified
    if regression_type is None:
        regression_type = 'ols' if is_normal else 'glm'

    #display('before check_and_clean_data:',df)
    df = check_and_clean_data(df, dependent_variable)
    #display('after check_and_clean_data:',df)

    # Handle mixed effects if row/column effect is treated as random
    if random_row_column_effects:
        regression_type = 'mixed'
        formula = prepare_formula(dependent_variable, random_row_column_effects=True)
        mixed_model, coef_df = fit_mixed_model(df, formula, dst)
        model = mixed_model
    else:
        # Regular regression models
        formula = prepare_formula(dependent_variable, random_row_column_effects=False)
        y, X = dmatrices(formula, data=df, return_type='dataframe')

        # Plot histogram of the dependent variable
        plot_histogram(y, dependent_variable, dst=dst)

        # Scale the independent variables and dependent variable
        X, y = scale_variables(X, y)

        # Perform the regression
        groups = df['prc'] if regression_type == 'mixed' else None
        print(f'performing {regression_type} regression')

        model = regression_model(X, y, regression_type=regression_type, groups=groups, alpha=alpha, cov_type=cov_type)

        # Process the model coefficients
        coef_df = process_model_coefficients(model, regression_type, X, y, highlight)

    # Plot the volcano plot
    volcano_plot(coef_df, volcano_path)

    return model, coef_df

def perform_regression(settings):

    from .plot import plot_plates
    from .utils import merge_regression_res_with_metadata, save_settings
    from .settings import get_perform_regression_default_settings
    from .toxo import go_term_enrichment_by_column, custom_volcano_plot

    if isinstance(settings['score_data'], list) and isinstance(settings['count_data'], list):
        settings['plate'] = None
        if len(settings['score_data']) == 1:
            settings['score_data'] = settings['score_data'][0]
        if len(settings['count_data']) == 1:
            settings['count_data'] = settings['count_data'][0]
        else:
            count_data_df = pd.DataFrame()
            for i, count_data in enumerate(settings['count_data']):
                df = pd.read_csv(count_data)
                df['plate_name'] = f'plate{i+1}'
                count_data_df = pd.concat([count_data_df, df])
                print('Count data:', len(count_data_df))

            score_data_df = pd.DataFrame()
            for i, score_data in enumerate(settings['score_data']):
                df = pd.read_csv(score_data)
                df['plate_name'] = f'plate{i+1}'
                score_data_df = pd.concat([score_data_df, df])
                print('Score data:', len(score_data_df))
    else:
        count_data_df = pd.read_csv(settings['count_data'])
        score_data_df = pd.read_csv(settings['score_data'])

    reg_types = ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge']
    if settings['regression_type'] not in reg_types:
        print(f'Possible regression types: {reg_types}')
        raise ValueError(f"Unsupported regression type {settings['regression_type']}")
    
    if settings['dependent_variable'] not in score_data_df.columns:
        print(f'Columns in DataFrame:')
        for col in score_data_df.columns:
            print(col)
        raise ValueError(f"Dependent variable {settings['dependent_variable']} not found in the DataFrame")

    if isinstance(settings['count_data'], list):
        src = os.path.dirname(settings['count_data'][0])
        csv_path = settings['count_data'][0]
    else:
        src = os.path.dirname(settings['count_data'])
        csv_path = settings['count_data']

    settings['src'] = src
    fldr = 'results_' + settings['regression_type']
    if isinstance(settings['count_data'], list):
        fldr = fldr + '_list'

    if settings['regression_type'] == 'quantile':
        fldr = fldr + '_' + str(settings['alpha'])

    res_folder = os.path.join(src, fldr)
    os.makedirs(res_folder, exist_ok=True)
    results_filename = 'results.csv'
    hits_filename = 'results_significant.csv'
    results_path=os.path.join(res_folder, results_filename)
    hits_path=os.path.join(res_folder, hits_filename)
    
    settings = get_perform_regression_default_settings(settings)
    save_settings(settings, name='regression', show=True)
    
    score_data_df = clean_controls(score_data_df, settings['pc'], settings['nc'], settings['other'])

    if 'prediction_probability_class_1' in score_data_df.columns:
        if not settings['class_1_threshold'] is None:
            score_data_df['predictions'] = (score_data_df['prediction_probability_class_1'] >= settings['class_1_threshold']).astype(int)

    dependent_df, dependent_variable = process_scores(score_data_df, settings['dependent_variable'], settings['plate'], settings['min_cell_count'], settings['agg_type'], settings['transform'])
        
    independent_df = process_reads(count_data_df, settings['fraction_threshold'], settings['plate'])
    
    merged_df = pd.merge(independent_df, dependent_df, on='prc')

    data_path = os.path.join(res_folder, 'regression_data.csv')
    merged_df.to_csv(data_path, index=False)

    merged_df[['plate', 'row', 'column']] = merged_df['prc'].str.split('_', expand=True)
    
    if settings['transform'] is None:
        _ = plot_plates(score_data_df, variable=dependent_variable, grouping='mean', min_max='allq', cmap='viridis', min_count=settings['min_cell_count'], dst = res_folder)                

    model, coef_df = regression(merged_df, csv_path, dependent_variable, settings['regression_type'], settings['alpha'], settings['random_row_column_effects'], highlight=settings['highlight'], dst=res_folder, cov_type=settings['cov_type'])
    
    coef_df.to_csv(results_path, index=False)
    
    if settings['regression_type'] == 'lasso':
        significant = coef_df[coef_df['coefficient'] > 0]
        
    else:
        significant = coef_df[coef_df['p_value']<= 0.05]
        #significant = significant[significant['coefficient'] > 0.1]
        significant.sort_values(by='coefficient', ascending=False, inplace=True)
        significant = significant[~significant['feature'].str.contains('row|column')]
        
    if settings['regression_type'] == 'ols':
        print(model.summary())
    
    significant.to_csv(hits_path, index=False)

    if isinstance(settings['metadata_files'], str):
        settings['metadata_files'] = [settings['metadata_files']]

    for metadata_file in settings['metadata_files']:
        file = os.path.basename(metadata_file)
        filename, _ = os.path.splitext(file)
        _ = merge_regression_res_with_metadata(hits_path, metadata_file, name=filename)
        merged_df = merge_regression_res_with_metadata(results_path, metadata_file, name=filename)

    if settings['toxo']:
        
        data_path = merged_df
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, 'resources', 'data', 'lopit.csv')

        custom_volcano_plot(data_path, metadata_path, metadata_column='tagm_location', string_list=[settings['highlight']], point_size=50, figsize=20)

        metadata_path = os.path.join(base_dir, 'resources', 'data', 'toxoplasma_metadata.csv')

        go_term_enrichment_by_column(significant, metadata_path)
    
    print('Significant Genes')
    display(significant)

    output = {'results':coef_df,
              'significant':significant}

    return output

def process_reads(csv_path, fraction_threshold, plate):

    if isinstance(csv_path, pd.DataFrame):
        csv_df = csv_path
    else:
        # Read the CSV file into a DataFrame
        csv_df = pd.read_csv(csv_path)
    
    if 'plate_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'plate_name': 'plate'})
    if 'column_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'column_name': 'column'})
    if 'row_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'row_name': 'row'})
    if 'grna_name' in csv_df.columns:
        csv_df = csv_df.rename(columns={'grna_name': 'grna'})
    if 'plate_row' in csv_df.columns:
        csv_df[['plate', 'row']] = csv_df['plate_row'].str.split('_', expand=True)
    if not 'plate' in csv_df.columns:
        if not plate is None:
            csv_df['plate'] = plate
        else:
            csv_df['plate'] = 'plate1'
    
    # Ensure the necessary columns are present
    if not all(col in csv_df.columns for col in ['row','column','grna','count']):
        raise ValueError("The CSV file must contain 'grna', 'count', 'row', and 'column' columns.")

    # Create the prc column
    csv_df['prc'] = csv_df['plate'] + '_' + csv_df['row'] + '_' + csv_df['column']

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

def clean_controls(df,pc,nc,other):
    if 'col' in df.columns:
        df['column'] = df['col']
    if nc != None:
        df = df[~df['column'].isin([nc])]
    if pc != None:
        df = df[~df['column'].isin([pc])]
    if other != None:
        df = df[~df['column'].isin([other])]
        print(f'Removed data from {nc, pc, other}')
    return df

def process_scores(df, dependent_variable, plate, min_cell_count=25, agg_type='mean', transform=None, regression_type='ols'):

    if 'plate_name' in df.columns:
        df.drop(columns=['plate'], inplace=True)
        df = df.rename(columns={'plate_name': 'plate'})
    
    if plate is not None:
        df['plate'] = plate

    if 'col' not in df.columns:
        df['col'] = df['column']

    df['prc'] = df['plate'] + '_' + df['row'] + '_' + df['col']
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
    
    from .io import _read_and_merge_data
    from .plot import plot_plates
    from .utils import get_ml_results_paths
    from .settings import set_default_analyze_screen

    settings = set_default_analyze_screen(settings)

    src = settings['src']

    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    display(settings_df)

    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']

    nuclei_limit, pathogen_limit, uninfected = settings['nuclei_limit'], settings['pathogen_limit'], settings['uninfected']
    
    df, _ = _read_and_merge_data(db_loc, 
                                 tables,
                                 settings['verbose'],
                                 nuclei_limit,
                                 pathogen_limit,
                                 uninfected)
    
    if settings['channel_of_interest'] in [0,1,2,3]:

        df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"]/df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
    
    output, figs = ml_analysis(df,
                               settings['channel_of_interest'],
                               settings['location_column'],
                               settings['positive_control'],
                               settings['negative_control'],
                               settings['exclude'],
                               settings['n_repeats'],
                               settings['top_features'],
                               settings['n_estimators'],
                               settings['test_size'],
                               settings['model_type_ml'],
                               settings['n_jobs'],
                               settings['remove_low_variance_features'],
                               settings['remove_highly_correlated_features'],
                               settings['verbose'])
    
    shap_fig = shap_analysis(output[3], output[4], output[5])

    features = output[0].select_dtypes(include=[np.number]).columns.tolist()

    if not settings['heatmap_feature'] in features:
        raise ValueError(f"Variable {settings['heatmap_feature']} not found in the dataframe. Please choose one of the following: {features}")
    
    plate_heatmap = plot_plates(df=output[0],
                                variable=settings['heatmap_feature'],
                                grouping=settings['grouping'],
                                min_max=settings['min_max'],
                                cmap=settings['cmap'],
                                min_count=settings['minimum_cell_count'],
                                verbose=settings['verbose'])

    data_path, permutation_path, feature_importance_path, model_metricks_path, permutation_fig_path, feature_importance_fig_path, shap_fig_path, plate_heatmap_path, settings_csv = get_ml_results_paths(src, settings['model_type_ml'], settings['channel_of_interest'])
    df, permutation_df, feature_importance_df, _, _, _, _, _, metrics_df = output

    settings_df.to_csv(settings_csv, index=False)
    df.to_csv(data_path, mode='w', encoding='utf-8')
    permutation_df.to_csv(permutation_path, mode='w', encoding='utf-8')
    feature_importance_df.to_csv(feature_importance_path, mode='w', encoding='utf-8')
    metrics_df.to_csv(model_metricks_path, mode='w', encoding='utf-8')
    
    plate_heatmap.savefig(plate_heatmap_path, format='pdf')
    figs[0].savefig(permutation_fig_path, format='pdf')
    figs[1].savefig(feature_importance_fig_path, format='pdf')
    shap_fig.savefig(shap_fig_path, format='pdf')

    return [output, plate_heatmap]

def ml_analysis(df, channel_of_interest=3, location_column='col', positive_control='c2', negative_control='c1', exclude=None, n_repeats=10, top_features=30, n_estimators=100, test_size=0.2, model_type='xgboost', n_jobs=-1, remove_low_variance_features=True, remove_highly_correlated_features=True, verbose=False):
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
    df = pd.concat([df, df_metadata[location_column]], axis=1)

    # Subset the dataframe based on specified column values
    df1 = df[df[location_column] == negative_control].copy()
    df2 = df[df[location_column] == positive_control].copy()

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

    print(X)
    print(y)

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
        model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, nthread=n_jobs, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)

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

    df = _calculate_similarity(df, features, location_column, positive_control, negative_control)

    df['prcfo'] = df.index.astype(str)
    df[['plate', 'row', 'col', 'field', 'object']] = df['prcfo'].str.split('_', expand=True)
    df['prc'] = df['plate'] + '_' + df['row'] + '_' + df['col']
    
    return [df, permutation_df, feature_importance_df, model, X_train, X_test, y_train, y_test, metrics_df], [permutation_fig, feature_importance_fig]

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
    pos_control = df[df[col_to_compare] == val1][features].mean()
    neg_control = df[df[col_to_compare] == val2][features].mean()
    
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
    df['similarity_to_pos_euclidean'] = df[features].apply(lambda row: euclidean(row, pos_control), axis=1)
    df['similarity_to_neg_euclidean'] = df[features].apply(lambda row: euclidean(row, neg_control), axis=1)
    df['similarity_to_pos_cosine'] = df[features].apply(lambda row: cosine(row, pos_control), axis=1)
    df['similarity_to_neg_cosine'] = df[features].apply(lambda row: cosine(row, neg_control), axis=1)
    df['similarity_to_pos_mahalanobis'] = df[features].apply(lambda row: mahalanobis(row, pos_control, inv_cov_matrix), axis=1)
    df['similarity_to_neg_mahalanobis'] = df[features].apply(lambda row: mahalanobis(row, neg_control, inv_cov_matrix), axis=1)
    df['similarity_to_pos_manhattan'] = df[features].apply(lambda row: cityblock(row, pos_control), axis=1)
    df['similarity_to_neg_manhattan'] = df[features].apply(lambda row: cityblock(row, neg_control), axis=1)
    df['similarity_to_pos_minkowski'] = df[features].apply(lambda row: minkowski(row, pos_control, p=3), axis=1)
    df['similarity_to_neg_minkowski'] = df[features].apply(lambda row: minkowski(row, neg_control, p=3), axis=1)
    df['similarity_to_pos_chebyshev'] = df[features].apply(lambda row: chebyshev(row, pos_control), axis=1)
    df['similarity_to_neg_chebyshev'] = df[features].apply(lambda row: chebyshev(row, neg_control), axis=1)
    df['similarity_to_pos_hamming'] = df[features].apply(lambda row: hamming(row, pos_control), axis=1)
    df['similarity_to_neg_hamming'] = df[features].apply(lambda row: hamming(row, neg_control), axis=1)
    df['similarity_to_pos_jaccard'] = df[features].apply(lambda row: jaccard(row, pos_control), axis=1)
    df['similarity_to_neg_jaccard'] = df[features].apply(lambda row: jaccard(row, neg_control), axis=1)
    df['similarity_to_pos_braycurtis'] = df[features].apply(lambda row: braycurtis(row, pos_control), axis=1)
    df['similarity_to_neg_braycurtis'] = df[features].apply(lambda row: braycurtis(row, neg_control), axis=1)
    
    return df

