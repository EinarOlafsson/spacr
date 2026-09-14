# Notes from `spacr/sim.py`

Prose lifted out of `spacr/sim.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [generate_plate_map](#generate_plate_map) (1 entry)
- [gini](#gini) (1 entry)
- [dist_gen](#dist_gen) (3 entries)
- [generate_gene_weights](#generate_gene_weights) (1 entry)
- [power_law_dist_gen](#power_law_dist_gen) (1 entry)
- [run_experiment](#run_experiment) (9 entries)
- [classifier.get_score](#classifierget_score) (3 entries)
- [compute_precision_recall](#compute_precision_recall) (1 entry)
- [generate_well_score](#generate_well_score) (1 entry)
- [sequence_plates](#sequence_plates) (3 entries)
- [regression_roc_auc](#regression_roc_auc) (9 entries)
- [plot_roc_pr](#plot_roc_pr) (1 entry)
- [plot_confusion_matrix](#plot_confusion_matrix) (1 entry)
- [run_simulation](#run_simulation) (4 entries)
- [vis_dists](#vis_dists) (2 entries)
- [visualize_all](#visualize_all) (18 entries)
- [append_database](#append_database) (1 entry)
- [save_data](#save_data) (1 entry)
- [run_and_save](#run_and_save) (6 entries)
- [validate_and_adjust_beta_params](#validate_and_adjust_beta_params) (2 entries)
- [generate_parameters](#generate_parameters) (1 entry)
- [run_multiple_simulations](#run_multiple_simulations) (2 entries)
- [generate_floats](#generate_floats) (3 entries)
- [read_simulations_table](#read_simulations_table) (3 entries)
- [plot_simulations](#plot_simulations) (12 entries)
- [plot_correlation_matrix](#plot_correlation_matrix) (4 entries)
- [plot_feature_importance](#plot_feature_importance) (6 entries)
- [calculate_permutation_importance](#calculate_permutation_importance) (4 entries)
- [plot_partial_dependences](#plot_partial_dependences) (7 entries)
- [generate_shap_summary_plot](#generate_shap_summary_plot) (3 entries)

## Module level

### line 22, trailing  _(unsure)_

```python
from .plot import save_figure
```

every kept figure goes through the format/DPI preference

### line 1674  _(unsure)_

to justify using beta for sim classifier

### lines 1676-1677

Fit a Beta distribution to these outputs a, b, loc, scale = beta.fit(predicted_probs, floc=0, fscale=1)  # Fix location and scale to match the support of the sigmoid

### lines 1679-1680

Sample from this fitted Beta distribution simulated_probs = beta.rvs(a, b, size=1000)

### lines 1682-1686

Plot the empirical vs simulated distribution plt.hist(predicted_probs, bins=30, alpha=0.5, label='Empirical') plt.hist(simulated_probs, bins=30, alpha=0.5, label='Simulated from Beta') plt.legend() plt.show()

## generate_plate_map

### line 64  _(unsure)_

```python
def generate_plate_map(nr_plates):
```

plate_map is a table with a row for each well, containing well metadata: plate_id, row_id, and column_id

## gini

### line 110, trailing  _(unsure)_

```python
r = np.argsort(np.argsort(-x))
```

ranks of x

## dist_gen

### line 123, trailing  _(unsure)_

```python
shape = (mean / sd) ** 2
```

Calculate shape parameter

### line 124, trailing  _(unsure)_

```python
scale = (sd ** 2) / mean
```

Calculate scale parameter

### line 125, trailing  _(unsure)_

```python
rate = np.random.gamma(shape, scale, size=length)
```

Generate random rate from gamma distribution

## generate_gene_weights

### line 137  _(unsure)_

```python
a1 = positive_mean*(positive_mean*(1-positive_mean)/positive_variance - 1)
```

alpha and beta for positive distribution

## power_law_dist_gen

### line 168  _(unsure)_

```python
def power_law_dist_gen(df, avg, well_ineq_coeff):
```

distribution generator function

## run_experiment

### lines 203-205

```python
plate_map = plate_map.reset_index(drop=True)
```

The per-well loops below address wells positionally (``plate_map.loc[i]`` for i in range(len(plate_map))), so a caller-filtered plate map with a non-contiguous index would raise KeyError. Renumber defensively.

### line 215, trailing  _(unsure)_

```python
gene_weights_array = np.array(list(gene_weights.values()))
```

Convert the values to an array

### line 220, trailing  _(unsure)_

```python
for gene in range(1, number_of_genes + 1):
```

ensures gene-1 is within bounds

### line 222, trailing  _(unsure)_

```python
max_index = len(plate_map['plate_row_column'])
```

this should be the number of choices available from plate_map

### line 225, trailing  _(unsure)_

```python
num_samples = max_index - 1
```

adjust to maximum possible index

### line 228, trailing  _(unsure)_

```python
break
```

break the loop if gene-1 is out of bounds for gpw

### line 233, trailing  _(unsure)_

```python
present_genes = [gene for gene, wells in gene_to_well_mapping.items() if plate_map.loc[i, 'plate_...
```

Select genes present in the current well

### line 234, trailing  _(unsure)_

```python
present_gene_weights = [gene_weights[gene] for gene in present_genes]
```

For sampling, filter gene_weights according to present_genes

### line 259  _(unsure)_

```python
genes_per_well_df = pd.DataFrame(gene_counts_per_well, columns=['genes_per_well'])
```

Create DataFrames

## classifier.get_score

### line 337  _(unsure)_

```python
def get_score(is_active):
```

Apply the beta distribution based on 'is_active' status with consideration for classifier error

### line 340, trailing  _(unsure)_

```python
if np.random.rand() < classifier_accuracy:
```

With classifier_accuracy probability, choose the correct distribution

### line 342, trailing

```python
else:
```

With 1-classifier_accuracy probability, choose the incorrect distribution

## compute_precision_recall

### lines 371-374

```python
th = np.append(th, 1.0)
```

sklearn returns one more precision/recall point than thresholds, and pr[i]/re[i] belong to th[i]. The trailing point (precision 1, recall 0) is "predict nothing positive", so pad at the END — padding at the front shifts every row onto the threshold below it.

## generate_well_score

### line 433  _(unsure)_

```python
well_score = cell_scores.groupby(['plate_row_column']).agg(
```

Compute mean and list of unique gene_ids

## sequence_plates

### line 468  _(unsure)_

```python
error = np.random.binomial(1, sequencing_error)
```

Decide whether to introduce error or not

### line 471  _(unsure)_

```python
wrong_well = np.random.choice(all_wells)
```

Randomly select a different well

### lines 483-484  _(unsure)_

```python
metadata['sum_reads'] = gene_counts_map.sum(axis=1)
```

Totalled after the loop: a mis-assigned read can land in a well that has already been visited, so a running per-iteration total under-reports it.

## regression_roc_auc

### line 508  _(unsure)_

```python
actives_list = ['gene_' + str(i) for i in active_gene_list]
```

asign active genes a value of 1 and inactive genes a value of 0

### line 517  _(unsure)_

```python
results_df['size'] = results_df['active']
```

generate a size column and handdf.replace([np.inf, -np.inf], np.nan, inplace=True)le infinate and NaN values create a new column for -log(p)

### line 522  _(unsure)_

```python
control_df = results_df[results_df['color'] == 'control']
```

calculate cutoff for hits based on randomly chosen 'control' genes

### line 529  _(unsure)_

```python
active_df = results_df[results_df['color'] == 'active']
```

calculate discriptive statistics for active genes

### line 535  _(unsure)_

```python
inactive_df = results_df[results_df['color'] == 'inactive']
```

calculate discriptive statistics for active genes

### line 541  _(unsure)_

```python
results_df['score'] = np.where(((results_df['coef'] >= cutoff) | (results_df['coef'] <= -cutoff))...
```

generate score column for hits and non hitts

### line 544

```python
fpr, tpr, thresh = roc_curve(results_df['active'], results_df['score'])
```

calculate regression roc based on controll cutoff

### line 550, trailing

```python
th = np.append(th, 1.0)
```

pr[i]/re[i] pair with th[i]; the extra point is the "predict nothing" end

### lines 561-562

```python
optimal_row = reg_pr_dict_df['f1_score'].idxmax()
```

``idxmax`` gives the row of the best F1, not the threshold itself — take the threshold recorded on that row so the value is a usable score cutoff.

## plot_roc_pr

### lines 629-630

```python
ax.plot(data[x_label], data[y_label], color=SIM_ACTIVE, lw=1.2)
```

The curve is the result and the diagonal is the null, so they must not be the same weight in the same colour: they were both 0.5 pt black.

## plot_confusion_matrix

### line 651  _(unsure)_

```python
sns.heatmap(data, cmap=Palette.SEQUENTIAL, ax=ax)
```

'Blues' is already the style's single-hue ramp for a count.

## run_simulation

### line 681  _(unsure)_

```python
active_gene_list = generate_gene_list(settings['number_of_active_genes'], settings['number_of_gen...
```

try:

### lines 686-687

```python
control_columns = ['1', '2', '3', '23', '24']
```

generate_plate_map writes bare column numbers ('1' ... '24'), so the outer control columns have to be matched without a 'c' prefix.

### line 689, trailing

```python
plate_map = plate_map[~plate_map['column_id'].isin(control_columns)].reset_index(drop=True)
```

Drop rows where 'column_id' is in [1,2,3,23,24]

### lines 710-711

```python
output = [cell_scores, cell_roc_dict_df, cell_pr_dict_df, cell_cm, well_score, gene_fraction_map,...
```

except Exception as e:

print(f"An error occurred while saving data: {e}")

## vis_dists

### lines 736-737  _(unsure)_

```python
sns.histplot(data=temp, x=f'{names[index]}', kde=False, binwidth=None, stat='count', element="ste...
```

One series per panel, so no panel has a claim to make: the style's histogram fill, not a saturated teal.

### lines 742-745

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## visualize_all

### line 791  _(unsure)_

```python
n+=1
```

ax[n].set_xscale('log')

### lines 797-805

```python
plot_histogram(active_distribution, "score", ax[n], SIM_ACTIVE, 'Cell scores', log=False)#, binwi...
```

THE ACTIVE GUIDES ARE THE CLAIM. The two series were slategray and teal, both full strength, so the panel gave the background distribution the same weight as the thing the simulation is about.

THE LEGEND WAS ALSO INVERTED, which is a wrong figure and not a style complaint: `active_distribution` is plotted first, and the first legend patch said 'Inactive'. Every colour on this panel therefore named the opposite series. Fixed here rather than filed -- the plotting order is unchanged, only the labels now follow it.

### line 806, trailing

```python
plot_histogram(active_distribution, "score", ax[n], SIM_ACTIVE, 'Cell scores', log=False)
```

, binwidth=0.01, log=False)

### line 807, trailing

```python
plot_histogram(inactive_distribution, "score", ax[n], SIM_INACTIVE, 'Cell scores', log=False)
```

, binwidth=0.01, log=False)

### lines 809-810

```python
legend_elements = [Patch(facecolor=SIM_ACTIVE, edgecolor=SIM_ACTIVE, label='Active'),
```

The legend now names the series it is actually drawn from: the first call plots `active_distribution` and used to be labelled 'Inactive'.

### line 820  _(unsure)_

```python
inactive_distribution_well = inactive_distribution.groupby(['plate_id', 'row_id', 'column_id'])['...
```

plot classifier cell predictions by inactive and active well average

### line 825, trailing

```python
plot_histogram(inactive_distribution_well, "score", ax[n], SIM_INACTIVE, 'Well scores', log=False)
```

, binwidth=0.01, log=False)

### line 826, trailing

```python
plot_histogram(active_distribution_well, "score", ax[n], SIM_ACTIVE, 'Well scores', log=False)
```

, binwidth=0.01, log=False)

### line 827, trailing

```python
plot_histogram(mixed_distribution_well, "score", ax[n], SIM_MIXED, 'Well scores', log=False)
```

, binwidth=0.01, log=False)

### line 836  _(unsure)_

```python
n+=1
```

ax[n].legend()

### line 843  _(unsure)_

```python
plot_roc_pr(cell_pr_dict_df, ax[n], 'Precision recall (Cell)', 'recall', 'precision')
```

plot Presision recall (cell classification)

### line 849  _(unsure)_

```python
plot_confusion_matrix(cell_cm, ax[n], 'Confusion Matrix Cell')
```

Confusion matrix at optimal threshold

### line 855  _(unsure)_

```python
n+=1
```

ax[n].set_xlim([0, 1])

### lines 863-865

```python
categories = ['inactive', 'control', 'active']
```

The simulator's volcano, and the same rule as the screen's: the genes made active are the claim, the controls are the reference, everything else is grey.

### line 884  _(unsure)_

```python
df = results_df[['gene', 'coef', 'std err', 'p']].copy()
```

error plot

### lines 898-899

```python
ax[n].plot(df['rank'], df['coef'], '-', color=Palette.GREY_DARK, lw=1.2)
```

One series with its own error band: the band takes the line's hue at the 0.25 the published figures use, not a second colour at 0.4.

### line 912

```python
plot_roc_pr(reg_pr_dict_df, ax[n], 'Precision recall (gene)', 'recall', 'precision')
```

plot Presision recall (regression classification)

### line 917  _(unsure)_

```python
plot_confusion_matrix(reg_cm, ax[n], 'Confusion Matrix Reg')
```

Confusion matrix at optimal threshold

## append_database

### line 962  _(unsure)_

```python
if conn is not None:
```

connect() itself can fail (unwritable directory), leaving conn unbound.

## save_data

### line 996, trailing  _(unsure)_

```python
indices_to_keep= [0,12]
```

Specify the indices to remove

## run_and_save

### line 1051, trailing  _(unsure)_

```python
random.seed(42)
```

sims will be too similar with a fixed seed — opt-in only

### line 1055, trailing  _(unsure)_

```python
start_time = time()
```

Start time of the simulation

### line 1059  _(unsure)_

```python
output, dists = run_simulation(settings)
```

try:

### line 1061, trailing  _(unsure)_

```python
sim_time = time() - start_time
```

Elapsed time for the simulation

### lines 1070-1073

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1081-1083

```python
time_ls.append(sim_time)
```

except Exception as e:

print(e, end='\r', flush=True) sim_time = time() - start_time

## validate_and_adjust_beta_params

### line 1104, trailing

```python
params['positive_variance'] = max_pos_variance * 0.99
```

Adjust to 99% of the maximum allowed variance

### line 1109, trailing

```python
params['negative_variance'] = max_neg_variance * 0.99
```

Adjust to 99% of the maximum allowed variance

## generate_parameters

### lines 1123-1124  _(unsure)_

```python
if not isinstance(settings.get('positive_mean'), (list, tuple)):
```

positive_mean is sweepable like every other key, but callers historically pass a bare float (or omit it) — only supply the default in that case.

## run_multiple_simulations

### line 1185, trailing  _(unsure)_

```python
now = datetime.now()
```

get current date

### line 1186, trailing  _(unsure)_

```python
start_time = now.strftime("%y%m%d")
```

format as a string in 'ddmmyy' format

## generate_floats

### line 1231  _(unsure)_

```python
num_decimals = str(step)[::-1].find('.')
```

Determine the number of decimal places in 'step'

### line 1235, trailing  _(unsure)_

```python
num_decimals = 0
```

integral step (e.g. 1) — str(step) has no '.' at all

### lines 1237-1239

```python
n_steps = int(math.floor(round((stop - start) / step, 9))) + 1
```

Repeated `current += step` accumulates float error (0.1 three times is 0.30000000000000004), which silently drops the inclusive upper bound. Step off the index instead so `stop` is always reached.

## read_simulations_table

### line 1260  _(unsure)_

```python
from .database_concurrency import connect as _connect_database
```

Create a connection object using the connect function

### line 1265  _(unsure)_

```python
try:
```

Read the 'simulations' table into a pandas DataFrame

### line 1272  _(unsure)_

```python
conn.close()
```

Close the connection to SQLite database

## plot_simulations

### line 1297  _(unsure)_

```python
required_columns = {variable, 'prauc'} | set(grouping_vars)
```

Check if the necessary columns are present in the DataFrame

### lines 1304-1305

```python
grouping_vars = [col for col in grouping_vars if df[col].nunique() > 1]
```

Grouping on a column that never varies just adds a constant to every subplot title; drop those so the panel grid reflects real conditions.

### line 1308  _(unsure)_

```python
if grouping_vars:
```

if not dependent is None:

### line 1314  _(unsure)_

```python
unique_combinations = df.iloc[[0]][[]]
```

Nothing to split on — a single panel over the whole DataFrame.

### line 1318  _(unsure)_

```python
num_rows = math.ceil(np.sqrt(num_combinations))
```

Determine the layout of the subplots

### line 1332  _(unsure)_

```python
condition = {var: row[var] for var in grouping_vars}
```

Filter the DataFrame for the current combination of variables

### line 1338, trailing  _(unsure)_

```python
grouped = grouped.sort_index()
```

Sort by the variable for orderly plots

### lines 1340-1342

```python
ax.plot(grouped.index, grouped['mean'], marker='o', linestyle='-', color=SIM_ACTIVE, label='Mean ...
```

Plotting the mean of 'prauc' with std deviation as shaded area One series with its own spread: the band is the line's hue at the 0.25 the published figures use, not a second grey at 0.5.

### line 1346  _(unsure)_

```python
ax.set_xlabel(variable)
```

Setting plot labels and title

### line 1355  _(unsure)_

```python
ax.set_xticks(grouped.index)
```

Set x-ticks and rotate them as specified

### lines 1361-1362  _(unsure)_

```python
ax.text(0.95, 0.05, verbose_text, transform=ax.transAxes,
```

No box: the style draws none, and a white round panel is a white rectangle on the dark theme.

### line 1368  _(unsure)_

```python
for ax in axes[idx+1:]:
```

Hide any unused axes if there are any

## plot_correlation_matrix

### line 1416

```python
if clean:
```

'inactive_mean', 'inactive_std', 'inactive_var', 'active_mean', 'active_std', 'inactive_var', 'cutoff', 'TP', 'FP', 'TN', 'FN',

### line 1422  _(unsure)_

```python
relevant_data = df[grouping_vars]
```

Subsetting the DataFrame to include only the relevant variables

### line 1428  _(unsure)_

```python
corr_matrix = relevant_data.corr()
```

Calculating the correlation matrix

### lines 1432-1435

```python
with figure_style(theme_target(), frame='box'):
```

Plotting the correlation matrix

THE MAP HAS TO BE CENTRED. It was diverging but unbounded, so seaborn scaled it to the data: an all-positive block ran to the hot end and read as a finding when the weakest correlation in it might be 0.05.

## plot_feature_importance

### line 1459  _(unsure)_

```python
features = ['number_of_active_genes', 'number_of_control_genes', 'avg_reads_per_gene',
```

Define the features for the model

### line 1468  _(unsure)_

```python
if isinstance(exclude, list):
```

Remove excluded features if specified

### line 1474  _(unsure)_

```python
model = RandomForestRegressor(n_estimators=1000, random_state=42)
```

Train the model

### lines 1482-1483  _(unsure)_

```python
with figure_style(theme_target()):
```

Plot horizontal bar chart

One series ranked by length: grey, opaque. It was a translucent teal.

### line 1488  _(unsure)_

```python
plt.yticks(range(len(indices)), [features[i] for i in indices])
```

Bar k carries importances[indices][k], so its label must be features[indices[k]].

### line 1490, trailing  _(unsure)_

```python
plt.gca().invert_yaxis()
```

Invert the axis to have the highest importance at the top

## calculate_permutation_importance

### line 1525  _(unsure)_

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
```

Initialize a model (you could pass it as an argument if you'd like to use a different one)

### line 1531  _(unsure)_

```python
sorted_idx = perm_importance.importances_mean.argsort()
```

Plotting

### line 1534  _(unsure)_

```python
with figure_style(theme_target()):
```

Create a figure and a set of subplots

### lines 1540-1541  _(unsure)_

```python
ax.set_yticklabels([features[i] for i in sorted_idx])
```

sorted_idx indexes the feature list that was fitted, not df.columns — those only coincide when df happens to start with exactly these columns.

## plot_partial_dependences

### lines 1566-1569

```python
X = df[features].astype(float)
```

scikit-learn 1.7 rejects integer columns for partial dependence because its evaluation grid is continuous and assigning grid values back into an integer array would round them. A float view preserves every input value while making the interpolation contract explicit.

### line 1573  _(unsure)_

```python
model = GradientBoostingRegressor()
```

Train a model

### line 1578, trailing  _(unsure)_

```python
n_cols = 4
```

Number of columns in subplot grid

### line 1579, trailing  _(unsure)_

```python
n_rows = (len(features) + n_cols - 1) // n_cols
```

Calculate rows needed

### lines 1588-1589

```python
axs = np.atleast_1d(axs).flatten()
```

Flatten the array of axes (subplots always returns an array here, since ncols > 1 — a bare `[axs]` would hand the whole row to a single feature).

### line 1597, trailing  _(unsure)_

```python
ax.set_title(feature)
```

Set title to the name of the feature

### line 1599  _(unsure)_

```python
for ax in axs[len(features):]:
```

Hide unused axes if any

## generate_shap_summary_plot

### line 1646  _(unsure)_

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
```

Initialize a model (you could pass it as an argument if you'd like to use a different one)

### line 1650  _(unsure)_

```python
explainer = shap.TreeExplainer(model)
```

Calculate SHAP values

### lines 1654-1657

```python
with figure_style(theme_target()):
```

Summary plot. shap builds the figure itself, so the style has to be open around the CALL -- there is no earlier point to reach the axes it creates. Its own colour map is left alone: a SHAP summary encodes the feature value on that ramp, so it is the data and not decoration.
