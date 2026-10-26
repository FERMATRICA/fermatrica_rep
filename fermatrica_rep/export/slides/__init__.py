"""
Generate specific slides for PPTX export:

1. fit and predict
2. decomposition (dynamic)
3. waterfall (decomposition for certain period)
4. options summary
5. efficiency curves
6. adstock (decay, carry-over)
"""


from fermatrica_rep.export.slides import fit_predict
from fermatrica_rep.export.slides import decomposition
from fermatrica_rep.export.slides import incr_roi
from fermatrica_rep.export.slides import waterfall
from fermatrica_rep.export.slides import options
from fermatrica_rep.export.slides import adstocks

# Backward-compatible callable aliases (deprecated — use .create_legacy() or .create() directly)
fit_predict_legacy = fit_predict.create_legacy
decomposition_legacy = decomposition.create_legacy
incr_roi_legacy = incr_roi.create_legacy
waterfall_legacy = waterfall.create_legacy
options_legacy = options.create_legacy
adstocks_legacy = adstocks.create_legacy
