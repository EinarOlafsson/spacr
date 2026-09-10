# Tutorial-only fixtures

`control_chart_campaign.csv` is entirely synthetic: 30 fictitious plates,
three negative-control, three positive-control and three sample wells per plate
(270 rows). Dates are illustrative sequence labels, not acquisition records.
The negative-control mean was programmed to rise after plate 20.

The generator is `../control_chart_evidence.py`, adapted from the application's
`tests/qt/test_control_chart_screen.py:campaign`, with NumPy seed 11. It never
loads images, runs a model, or claims that a biological assay was validated.
The tutorial uses the normal CSV file picker and threaded chart computation.
The source CSV is hashed before and after; exported point identities, means,
sample standard deviations, limits, baseline membership, z values and rule-one
flags are independently checked using Python's standard library.

This is not a replacement for a core module's downloadable microscopy data.
The Control Charts specialist panel currently has no Load test data button.
