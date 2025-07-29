spacr.ml
========

.. py:module:: spacr.ml






Module Contents
---------------

.. py:class:: QuasiBinomial(link=logit(), dispersion=1.0)

   Bases: :py:obj:`statsmodels.genmod.families.Binomial`


   Quasi-Binomial family for generalized linear models with adjustable dispersion.

   Extends the standard Binomial family from `statsmodels` to allow for modeling overdispersion
   in binomial data, where the observed variance exceeds the nominal binomial variance.

   This is often used in biological and ecological count data where extra-binomial variation is common.

   Initialize the Quasi-Binomial family with a specified link function and dispersion factor.

   :param link: The link function to use. Defaults to logit().
   :type link: statsmodels.genmod.families.links
   :param dispersion: A positive float to scale the variance.
                      Use values >1 to model overdispersion. Defaults to 1.0.
   :type dispersion: float


   .. py:attribute:: dispersion
      :value: 1.0



   .. py:method:: variance(mu)

      Compute the variance function for the quasi-binomial model.

      :param mu: The mean response values.
      :type mu: array-like

      :returns: Variance scaled by the dispersion parameter.
      :rtype: array-like



.. py:function:: calculate_p_values(X, y, model)

   Calculate p-values for model coefficients using residuals and t-statistics.

   :param X: The input feature matrix used for fitting the model.
   :type X: np.ndarray
   :param y: The target variable.
   :type y: np.ndarray
   :param model: A fitted sklearn linear model (must have `.predict` and `.coef_` attributes).
   :type model: object

   :returns: An array of p-values corresponding to each coefficient in the model.
   :rtype: np.ndarray


.. py:function:: perform_mixed_model(y, X, groups, alpha=1.0)

   Perform mixed effects regression and return the fitted model.

   :param y: Target variable.
   :type y: pd.Series or np.ndarray
   :param X: Feature matrix.
   :type X: pd.DataFrame
   :param groups: Grouping variable for mixed model.
   :type groups: pd.Series or np.ndarray
   :param alpha: Regularization strength for Ridge regression when multicollinearity is detected.
   :type alpha: float, optional

   :returns: Fitted mixed model results.
   :rtype: statsmodels.regression.mixed_linear_model.MixedLMResults


.. py:function:: create_volcano_filename(csv_path, regression_type, alpha, dst)

   Construct the file name for saving the volcano plot.

   :param csv_path: Path to the CSV file containing model results.
   :type csv_path: str
   :param regression_type: Type of regression performed.
   :type regression_type: str
   :param alpha: Alpha value (used for quantile regression naming).
   :type alpha: float
   :param dst: Destination directory to save the file.
   :type dst: str or None

   :returns: Full path for the output volcano plot PDF file.
   :rtype: str


.. py:function:: scale_variables(X, y)

   Min-max scale both independent and dependent variables.

   :param X: Feature matrix.
   :type X: pd.DataFrame
   :param y: Target variable.
   :type y: np.ndarray

   :returns: * **X_scaled** (*pd.DataFrame*) -- Scaled feature matrix.
             * **y_scaled** (*np.ndarray*) -- Scaled target variable.


.. py:function:: select_glm_family(y)

   Automatically select a GLM family based on the nature of the dependent variable.

   :param y: Target variable.
   :type y: np.ndarray

   :returns: GLM family appropriate for the data.
   :rtype: sm.families.Family


.. py:function:: prepare_formula(dependent_variable, random_row_column_effects=False)

   Prepare the formula for regression modeling based on model design.

   :param dependent_variable: Name of the dependent variable.
   :type dependent_variable: str
   :param random_row_column_effects: Whether to include row and column IDs as random effects.
   :type random_row_column_effects: bool, optional

   :returns: Regression formula string.
   :rtype: str


.. py:function:: fit_mixed_model(df, formula, dst)

   Fit a mixed linear model with random effects for plate, row, and column.

   :param df: Input data.
   :type df: pd.DataFrame
   :param formula: Regression formula.
   :type formula: str
   :param dst: Output folder to save diagnostic plots.
   :type dst: str

   :returns: * **mixed_model** (*statsmodels.regression.mixed_linear_model.MixedLMResults*) -- Fitted model.
             * **coef_df** (*pd.DataFrame*) -- DataFrame of coefficients and p-values.


.. py:function:: check_and_clean_data(df, dependent_variable)

   Validate and preprocess a DataFrame before model fitting.

   :param df: Input data.
   :type df: pd.DataFrame
   :param dependent_variable: Name of the dependent variable column.
   :type dependent_variable: str

   :returns: Cleaned and validated DataFrame, ready for model fitting.
   :rtype: pd.DataFrame


.. py:function:: minimum_cell_simulation(settings, num_repeats=10, sample_size=100, tolerance=0.02, smoothing=10, increment=10)

   Estimate the minimum number of cells required per well to stabilize phenotype measurements.

   This function simulates phenotype score stability by repeatedly subsampling increasing numbers
   of cells per well, calculating the mean absolute difference from the full-well mean. It identifies
   the minimal sample size (elbow point) at which the average difference drops below a user-defined
   tolerance, smoothed across wells.

   :param settings: A dictionary with keys:
                    - 'score_data' (list or str): CSV path(s) with cell-level data.
                    - 'score_column' (str): Column name of the phenotype score.
                    - 'tolerance' (float or int): Allowed deviation from true mean (e.g. 0.02 for 2%).
                    - 'min_cell_count' (int or None): Optional fixed value for annotation.
                    - 'count_data' (list): Used to define the output folder for figure saving.
   :type settings: dict
   :param num_repeats: Number of times to resample for each sample size. Default is 10.
   :type num_repeats: int, optional
   :param sample_size: Number of top wells to simulate (by cell count). Default is 100.
   :type sample_size: int, optional
   :param tolerance: Tolerance threshold to define the elbow point. Default is 0.02.
   :type tolerance: float or int, optional
   :param smoothing: Window size for smoothing the mean absolute difference curve. Default is 10.
   :type smoothing: int, optional
   :param increment: Step size between tested sample sizes. Default is 10.
   :type increment: int, optional

   :returns: Estimated minimal required number of cells (elbow point).
   :rtype: int


.. py:function:: process_model_coefficients(model, regression_type, X, y, nc, pc, controls)

   Extract model coefficients, standard errors, and p-values into a DataFrame with annotations.

   Supports various regression types including beta, OLS, GLM, ridge, lasso, and quasi-binomial.
   Adds classification labels (nc, pc, control, other) and computes -log10(p-values).

   :param model: Fitted regression model object.
   :param regression_type: Type of regression (e.g., 'beta', 'ols', 'glm', 'ridge', 'lasso').
   :type regression_type: str
   :param X: Feature matrix used for fitting the model.
   :type X: pd.DataFrame
   :param y: Target values used in model fitting.
   :type y: np.ndarray
   :param nc: Identifier for negative control features.
   :type nc: str
   :param pc: Identifier for positive control features.
   :type pc: str
   :param controls: List of gRNAs used as general controls.
   :type controls: list

   :returns: Table of coefficients, p-values, and annotations.
   :rtype: pd.DataFrame


.. py:function:: check_distribution(y, epsilon=1e-06)

   Analyze distribution of the target variable and recommend appropriate regression model.

   Checks for:
   - Binary data (logit)
   - Continuous [0, 1) data (beta)
   - Continuous [0, 1] data (quasi-binomial)
   - Normal distribution (OLS)
   - Beta distribution fit
   - Default to GLM otherwise

   :param y: Dependent variable values.
   :type y: np.ndarray
   :param epsilon: Threshold for boundary proximity detection. Default is 1e-6.
   :type epsilon: float, optional

   :returns: Suggested regression model name (e.g., 'logit', 'beta', 'ols', 'glm').
   :rtype: str


.. py:function:: pick_glm_family_and_link(y)

   Select the appropriate GLM family and link function based on distribution of y.

   Inspects binary, count, proportion, normal, and overdispersed data and maps to:
   - Binomial with Logit link
   - Gaussian with Identity link
   - Poisson with Log link
   - Inverse Gaussian with Log link
   - Negative Binomial with Log link

   :param y: Response variable to inspect.
   :type y: np.ndarray

   :returns: GLM family object with appropriate link function.
   :rtype: sm.families.Family

   :raises ValueError: If data is suitable for Beta regression, which GLM cannot handle.


.. py:function:: regression_model(X, y, regression_type='ols', groups=None, alpha=1.0, cov_type=None)

   Fit a regression model of the specified type to the data.

   Supports multiple regression types: OLS, GLM, beta, logit, probit, ridge, lasso, and mixed models.
   Automatically performs hyperparameter tuning (alpha) for lasso and ridge if alpha is 0 or None.
   For GLMs, prints McFadden's pseudo R². For regularized models, prints MSE and coefficient table.

   :param X: Design matrix of independent variables.
   :type X: pd.DataFrame
   :param y: Dependent variable.
   :type y: pd.Series or np.ndarray
   :param regression_type: Type of regression ('ols', 'glm', 'beta', 'logit', 'probit', 'lasso', 'ridge', 'mixed').
   :type regression_type: str
   :param groups: Grouping variable for mixed models.
   :type groups: array-like, optional
   :param alpha: Regularization strength for lasso and ridge.
   :type alpha: float
   :param cov_type: Covariance estimator type for OLS (e.g., 'HC3').
   :type cov_type: str, optional

   :returns: Fitted model object (statsmodels or sklearn estimator).
   :rtype: model


.. py:function:: regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0, random_row_column_effects=False, nc='233460', pc='220950', controls=[''], dst=None, cov_type=None, plot=False)

   Perform regression analysis on a DataFrame with optional plotting and mixed effects support.

   Handles data cleaning, formula preparation, scaling, model fitting, and coefficient extraction.
   Automatically determines appropriate regression type if not specified.

   :param df: Input data.
   :type df: pd.DataFrame
   :param csv_path: Path to input CSV used for labeling plots/files.
   :type csv_path: str
   :param dependent_variable: Name of column to regress against.
   :type dependent_variable: str
   :param regression_type: Type of regression (auto-detected if None).
   :type regression_type: str, optional
   :param alpha: Regularization parameter for lasso/ridge.
   :type alpha: float
   :param random_row_column_effects: Whether to model row/column as random effects.
   :type random_row_column_effects: bool
   :param nc: Identifier for negative controls.
   :type nc: str
   :param pc: Identifier for positive controls.
   :type pc: str
   :param controls: List of general control gRNAs.
   :type controls: list
   :param dst: Destination folder for output files.
   :type dst: str, optional
   :param cov_type: Covariance estimator for OLS.
   :type cov_type: str, optional
   :param plot: Whether to generate volcano plot.
   :type plot: bool

   :returns: (fitted model, coefficients DataFrame, regression type used)
   :rtype: tuple


.. py:function:: save_summary_to_file(model, file_path='summary.csv')

   Save the model's summary output to a text or CSV file.

   :param model: Fitted statsmodels model with a `.summary()` method.
   :param file_path: Output path to save the summary (e.g., 'summary.txt').
   :type file_path: str

   :returns: None


.. py:function:: perform_regression(settings)

   Perform full regression analysis pipeline for pooled CRISPR-Cas9 screens.

   This function integrates data loading, filtering, transformation, regression modeling,
   statistical significance assessment, visualization, and optional downstream Toxoplasma-specific analysis.
   It handles both grna-level and gene-level aggregation and outputs results with optional metadata integration.

   :param settings: Dictionary containing regression settings. Recommended to start with
                    `get_perform_regression_default_settings()` to initialize.
   :type settings: dict

   Required Keys in `settings`:
       - score_data (str or list): Path(s) to CSV(s) containing phenotypic scores.
       - count_data (str or list): Path(s) to CSV(s) containing barcode abundance counts.
       - dependent_variable (str): Column name in `score_data` to use as dependent variable.
       - regression_type (str or None): Type of regression model to use ('ols', 'glm', 'lasso', etc.).
       - filter_value (list): Values to exclude from `filter_column`.
       - filter_column (str): Column in data to apply exclusion filter.
       - plateID (str): Identifier to use when reconstructing `prc` labels.
       - alpha (float): Regularization strength for lasso/ridge regressions.
       - random_row_column_effects (bool): If True, treat row/column as random effects.
       - negative_control (str): Label for negative controls.
       - positive_control (str): Label for positive controls.
       - controls (list): List of control gRNAs for estimating coefficient threshold.
       - volcano (str): Type of volcano plot to generate ('all', 'gene', 'grna').
       - transform (bool): Whether to transform dependent variable.
       - agg_type (str): Aggregation method for per-well phenotypes ('mean', 'median', etc.).
       - metadata_files (str or list): Metadata CSV(s) to merge with regression results.
       - toxo (bool): If True, generate Toxoplasma-specific downstream plots.
       - threshold_method (str): Method to calculate effect size threshold ('std' or 'var').
       - threshold_multiplier (float): Multiplier for threshold method.
       - outlier_detection (bool): Whether to remove grnas with outlier well counts.
       - cov_type (str or None): Covariance type for OLS (e.g., 'HC3').
       - verbose (bool): Whether to print verbose output.
       - x_lim, y_lims: Optional axes limits for volcano plots.
       - tolerance (float): Tolerance for determining minimal cell count via simulation.
       - min_cell_count (int or None): Minimum number of cells per well for inclusion.
       - min_n (int): Minimum number of replicates for inclusion in filtered results.

   :returns:     'results': Full coefficient table (pandas DataFrame).
                 'significant': Filtered table of statistically significant hits (pandas DataFrame).
   :rtype: dict

   Side Effects:
       - Saves regression results, metadata-merged results, volcano plots, and phenotype plots to disk.
       - Optionally generates publication-ready plots for Toxoplasma gene expression and phenotype scores.
       - Writes cell count, well-grna, and well-gene plots to disk.

   :raises ValueError: If required keys are missing or regression type is invalid.


.. py:function:: process_reads(csv_path, fraction_threshold, plate, filter_column=None, filter_value=None)

   Process barcode count data and compute fractional abundance of each gRNA per well.

   This function loads a count table, standardizes metadata columns, computes the
   relative abundance of each gRNA in a well (`fraction`), and filters based on a
   minimum threshold. It also parses `grna` strings into gene-level information if
   applicable.

   :param csv_path: Path to CSV file or preloaded DataFrame with count data.
   :type csv_path: str or pd.DataFrame
   :param fraction_threshold: Minimum fraction of reads per well for inclusion.
   :type fraction_threshold: float
   :param plate: Plate identifier to use if not present in metadata.
   :type plate: str
   :param filter_column: Columns to filter values from.
   :type filter_column: str or list, optional
   :param filter_value: Values to exclude from filter_column(s).
   :type filter_value: str or list, optional

   :returns: Filtered DataFrame with columns ['prc', 'grna', 'fraction'].
   :rtype: pd.DataFrame

   :raises ValueError: If required metadata columns are missing.


.. py:function:: apply_transformation(X, transform)

   Apply a mathematical transformation to a variable.

   Supported transformations include logarithm, square root, and square.

   :param X: Input variable or DataFrame column.
   :type X: array-like
   :param transform: Type of transformation ('log', 'sqrt', or 'square').
   :type transform: str

   :returns: Transformer object, or None if transform is unrecognized.
   :rtype: FunctionTransformer or None


.. py:function:: check_normality(data, variable_name, verbose=False)

   Check if data follows a normal distribution using the Shapiro-Wilk test.

   :param data: Data to test for normality.
   :type data: array-like
   :param variable_name: Name of the variable (used in print statements).
   :type variable_name: str
   :param verbose: Whether to print test results and interpretation.
   :type verbose: bool

   :returns: True if data is normally distributed (p > 0.05), False otherwise.
   :rtype: bool


.. py:function:: clean_controls(df, values, column)

   Remove rows from a DataFrame based on specified control values in a given column.

   :param df: Input DataFrame.
   :type df: pd.DataFrame
   :param values: List of control values to remove.
   :type values: list
   :param column: Column from which to remove control values.
   :type column: str

   :returns: Filtered DataFrame with control rows removed.
   :rtype: pd.DataFrame


.. py:function:: process_scores(df, dependent_variable, plate, min_cell_count=25, agg_type='mean', transform=None, regression_type='ols')

   Aggregate and transform single-cell phenotype scores for regression input.

   Groups phenotypic measurements by well (`prc`) and applies optional aggregation
   (mean, median, quantile). Filters out wells with low cell counts. Optionally applies
   mathematical transformation and checks for normality of the dependent variable.

   :param df: Input DataFrame with single-cell phenotype scores.
   :type df: pd.DataFrame
   :param dependent_variable: Name of the phenotype column to model.
   :type dependent_variable: str
   :param plate: Plate ID to assign if not present in metadata.
   :type plate: str
   :param min_cell_count: Minimum number of cells per well to retain.
   :type min_cell_count: int
   :param agg_type: Aggregation method ('mean', 'median', 'quantile', None).
   :type agg_type: str
   :param transform: Optional transformation ('log', 'sqrt', or 'square').
   :type transform: str or None
   :param regression_type: Type of regression (affects aggregation logic for 'poisson').
   :type regression_type: str

   :returns:

             (aggregated_df, dependent_variable_name)
                 - aggregated_df: Aggregated and filtered phenotype DataFrame.
                 - dependent_variable_name: Possibly transformed column name used for modeling.
   :rtype: tuple

   :raises ValueError: If required metadata columns are missing or aggregation type is invalid.


.. py:function:: generate_ml_scores(settings)

   Perform machine learning analysis on single-cell measurement data from one or more screen sources.

   This function loads and merges measurement databases, computes additional features (e.g., recruitment
   scores, shortest distance between compartments), handles annotations (for supervised classification),
   trains a machine learning model (e.g., XGBoost), computes SHAP values, and generates output files
   including predictions, importance scores, SHAP plots, and well-level heatmaps.

   :param settings: Dictionary of analysis parameters and options. Required keys include:
                    - 'src' (str or list): Path(s) to screen folders with measurement databases.
                    - 'channel_of_interest' (int): Channel index used for computing recruitment scores.
                    - 'location_column' (str): Column used to group wells for training (e.g., 'columnID').
                    - 'positive_control' (str): Value of `location_column` representing positive control wells.
                    - 'negative_control' (str): Value of `location_column` representing negative control wells.
                    - 'exclude' (list): List of feature names to exclude.
                    - 'n_repeats' (int): Number of repetitions for permutation importance.
                    - 'top_features' (int): Number of top features to keep and plot.
                    - 'reg_alpha', 'reg_lambda' (float): Regularization strength for XGBoost.
                    - 'learning_rate' (float): Learning rate for XGBoost.
                    - 'n_estimators' (int): Number of boosting rounds for the model.
                    - 'test_size' (float): Fraction of labeled data to use for testing.
                    - 'model_type_ml' (str): Type of model to use ('xgboost', 'random_forest', etc.).
                    - 'n_jobs' (int): Number of parallel jobs.
                    - 'remove_low_variance_features' (bool): Whether to drop near-constant features.
                    - 'remove_highly_correlated_features' (bool): Whether to drop redundant features.
                    - 'prune_features' (bool): Whether to perform SelectKBest-based pruning.
                    - 'cross_validation' (bool): Whether to use 5-fold cross-validation.
                    - 'heatmap_feature' (str): Feature to plot as a plate heatmap.
                    - 'grouping' (str): Column used to group wells in the heatmap (e.g., 'prc').
                    - 'min_max' (tuple): Min/max range for heatmap normalization.
                    - 'minimum_cell_count' (int): Minimum number of cells per well for heatmap inclusion.
                    - 'cmap' (str): Colormap for the heatmap.
                    - 'save_to_db' (bool): If True, update the database with prediction column.
                    - 'verbose' (bool): If True, print detailed logs.
                    - 'annotation_column' (str or None): If provided, annotate data from the `png_list` table.
   :type settings: dict

   :returns:     - output (list): Includes DataFrames and model objects from the `ml_analysis` function.
                 - plate_heatmap (matplotlib.Figure): Plate-level heatmap of the selected feature.
   :rtype: list

   :raises ValueError: If required columns are missing or specified features are not found.


.. py:function:: ml_analysis(df, channel_of_interest=3, location_column='columnID', positive_control='c2', negative_control='c1', exclude=None, n_repeats=10, top_features=30, reg_alpha=0.1, reg_lambda=1.0, learning_rate=1e-05, n_estimators=1000, test_size=0.2, model_type='xgboost', n_jobs=-1, remove_low_variance_features=True, remove_highly_correlated_features=True, prune_features=False, cross_validation=False, verbose=False)

   Train a machine learning classifier to distinguish between positive and negative control wells,
   compute feature importance via permutation and model-based methods, and assign predictions to all rows.

   This function supports several classifier types (XGBoost, random forest, logistic regression, etc.),
   with options for feature selection, cross-validation, and SHAP-ready output. It returns predictions,
   probability estimates, and feature importances, and computes classification metrics.

   :param df: Input DataFrame with features and metadata.
   :type df: pd.DataFrame
   :param channel_of_interest: Channel index used for filtering relevant features.
   :type channel_of_interest: int
   :param location_column: Column that defines grouping (e.g., 'columnID') to select control wells.
   :type location_column: str
   :param positive_control: Identifier(s) in `location_column` for positive control group.
   :type positive_control: str or list
   :param negative_control: Identifier(s) in `location_column` for negative control group.
   :type negative_control: str or list
   :param exclude: Feature names or substrings to exclude from analysis.
   :type exclude: list or str, optional
   :param n_repeats: Number of repetitions for permutation importance calculation.
   :type n_repeats: int
   :param top_features: Number of top features to retain and visualize.
   :type top_features: int
   :param reg_alpha: L1 regularization term (only for XGBoost).
   :type reg_alpha: float
   :param reg_lambda: L2 regularization term (only for XGBoost).
   :type reg_lambda: float
   :param learning_rate: Learning rate for gradient-based models (e.g., XGBoost).
   :type learning_rate: float
   :param n_estimators: Number of trees or boosting iterations.
   :type n_estimators: int
   :param test_size: Fraction of labeled data used for testing.
   :type test_size: float
   :param model_type: Model type to train: 'xgboost', 'random_forest', 'logistic_regression', or 'gradient_boosting'.
   :type model_type: str
   :param n_jobs: Number of CPU cores to use (where supported).
   :type n_jobs: int
   :param remove_low_variance_features: If True, drop near-constant features.
   :type remove_low_variance_features: bool
   :param remove_highly_correlated_features: If True, drop features with high pairwise correlation.
   :type remove_highly_correlated_features: bool
   :param prune_features: If True, apply SelectKBest to reduce features before training.
   :type prune_features: bool
   :param cross_validation: If True, use 5-fold stratified cross-validation instead of single split.
   :type cross_validation: bool
   :param verbose: If True, print detailed logs and show diagnostic plots.
   :type verbose: bool

   :returns:     - [0] df (pd.DataFrame): Original DataFrame with added predictions, probabilities, and metadata.
                 - [1] permutation_df (pd.DataFrame): Permutation importance for top features.
                 - [2] feature_importance_df (pd.DataFrame): Model-based feature importance scores (if supported).
                 - [3] model (sklearn-compatible): Trained model object.
                 - [4] X_train (pd.DataFrame): Training feature matrix.
                 - [5] X_test (pd.DataFrame): Test feature matrix.
                 - [6] y_train (pd.Series): Training target labels.
                 - [7] y_test (pd.Series): Test target labels.
                 - [8] metrics_df (pd.DataFrame): Summary classification metrics (precision, recall, f1).
                 - [9] features (list): Final list of features used in training.

             list:
                 - [0] permutation_fig (matplotlib.Figure): Bar plot of permutation importances.
                 - [1] feature_importance_fig (matplotlib.Figure): Bar plot of model importances.
   :rtype: list


.. py:function:: shap_analysis(model, X_train, X_test)

   Compute and return a SHAP summary plot figure for a trained model.

   :param model: A trained machine learning model compatible with SHAP.
   :param X_train: Training features used to initialize the SHAP explainer.
   :type X_train: pd.DataFrame
   :param X_test: Test features for which SHAP values will be computed.
   :type X_test: pd.DataFrame

   :returns: SHAP summary plot figure.
   :rtype: matplotlib.figure.Figure


.. py:function:: find_optimal_threshold(y_true, y_pred_proba)

   Determine the optimal decision threshold that maximizes the F1-score
   based on predicted probabilities.

   :param y_true: Ground-truth binary labels.
   :type y_true: array-like
   :param y_pred_proba: Predicted class probabilities (positive class).
   :type y_pred_proba: array-like

   :returns: Threshold that yields the highest F1-score.
   :rtype: float


.. py:function:: interperate_vision_model(settings={})

   Perform feature interpretation on vision model predictions using feature importance,
   permutation importance, and SHAP analysis.

   Steps:
       1. Loads and merges measurement and score data.
       2. Computes Random Forest feature importances.
       3. Computes permutation-based feature importances.
       4. Runs SHAP analysis on selected features.
       5. Aggregates SHAP values by compartment and channel and visualizes them using radar plots.

   :param settings: Dictionary containing configuration options, including:
                    - src (str): Path to measurement database.
                    - tables (list): List of tables to include.
                    - score_column (str): Column with predicted or measured scores.
                    - top_features (int): Number of features to retain and plot.
                    - shap (bool): Whether to perform SHAP analysis.
                    - shap_sample (bool): Whether to subsample data before SHAP.
                    - feature_importance (bool): Whether to compute Random Forest importance.
                    - permutation_importance (bool): Whether to compute permutation importance.
                    - n_jobs (int): Number of parallel jobs for model training and permutation.
                    - save (bool): Whether to save result CSVs.
   :type settings: dict

   :returns: Merged and scored dataset used for interpretation.
   :rtype: pd.DataFrame


