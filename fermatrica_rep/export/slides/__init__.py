"""
Generate specific slides for PPTX export:

1. fit and predict
2. decomposition (dynamic)
3. waterfall (decomposition for certain period)
4. options summary
5. efficiency curves
6. adstock (decay, carry-over)
"""


from fermatrica_rep.export.slides.fit_predict import create as fit_predict
from fermatrica_rep.export.slides.decomposition import create as decomposition
from fermatrica_rep.export.slides.incr_roi import create as incr_roi
from fermatrica_rep.export.slides.waterfall import create as waterfall
from fermatrica_rep.export.slides.options import create as options
from fermatrica_rep.export.slides.adstocks import create as adstocks
