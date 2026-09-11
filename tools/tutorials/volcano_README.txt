VOLCANO EXPLORER: REAL REGRESSION TUTORIAL OUTPUTS, NOT A NEW ANALYSIS

Extract the archive and keep its originals. Open Regression, then its nested
Volcano Explorer / Publication figure tool. Use Open results to choose the
included results folder. It contains unchanged outputs of the completed
downloaded-data Regression lesson. It is not a new experiment or a complete
archive of the original run. No regression or GPU job is required to plot it.

IMPORTANT CURRENT UI WORKAROUND: after loading results, BEFORE editing any
setting, Save style to a new JSON file and Load style from that same file.
This synchronizes controls with the actual renderer. In the recorded build,
the initial controls showed alpha=0.000001, multiplier=0 and log transform off
while the actual figure used alpha=0.05, multiplier=3 and log transform on.
Editing before synchronizing could pull many wrong defaults into the style.
The included baseline_style.json is the same recorded baseline; volcano_style.json
is the final demonstrated export style. These are appearance settings, not a
record that a different statistical model was executed.

Each point is one guide from ONE correction family: 434 guides, outcome log_pred,
minimum_wells_threshold=2, plus-one permutation P values and BH adjustment.
The minimum raw P is 0.005, but the minimum adjusted P is 0.39454545454545453;
none is significant at the original alpha 0.05. These are nonparametric marginal
effects, not coefficients from a simultaneously fitted multivariable model.
Do not mix support thresholds or outcomes into one unlabelled correction family.

X is standardized_marginal_effect. Default Y is -log10(adjusted_p_value).
Switching to raw permutation_p_value changes the plotted quantity, not the
statistical analysis. Changing alpha moves a reference line; source significant
flags remain the original ones. At alpha 0.5, nineteen adjusted values fall
below the new line's P cutoff, but zero points acquire a new significance flag.
Restore 0.05 and adjusted_p_value before exporting. Do not call this new testing.

With an explicit effect cutoff 0.1 and multiplier 3, vertical cut lines are at
-0.3 and +0.3. This is not a log2-fold-change axis, and an effect threshold is
not a substitute for statistical correction. Restore Auto after the exercise.
Colour by wells_with_guide displays actual support; it is not an accuracy score.

Save style / Load style preserves the whole recorded style. Export PDF and
Export PNG re-render rather than capture the screen. The demonstration sets
nominal figure size 7 by 5 inches, DPI 150. Tight bounding-box cropping changes
final dimensions: PDF 492.536 x 348.485 points, PNG 1026 x 726 pixels. The PDF
contains embedded fonts and vector marks, not raster images. Inspect labels,
contrast and warnings; the app warned about white marker outlines on light paper.

Some native control labels and detail names remain clipped in this build; this
lesson does not certify universal GUI text fitting. The exported PDF is readable.
Optional annotation joins, localization plots, broken axes and control-MAD rules
are outside this verified example. No new hits, biological validation or AI
provider request is claimed. API: spacr.volcano_style.VolcanoStyle and
render_volcano(results, style, save_path='volcano.pdf').
