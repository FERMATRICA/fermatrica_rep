"""
Exports data to XLSX (in some way, to be finished later) ot to PPTX. As for PPTX,
generates PowerPoint presentation with standard slides describing model. Slides / components
available to export now are:

- Retro analysis
    - Fit and predict by superbrand
    - Fit and predict by SKU / arbitrary set of variables
    - Dynamic decomposition by superbrand
    - Dynamic decomposition by SKU / arbitrary set of variables
    - Waterfall decomposition for arbitrary historical period
- Marketing tools efficiency
    - Short-term curves
        - Incremental KPI volume
        - Incremental KPI value
        - ROI
    - Long-term curves
        - Incremental KPI volume
        - Incremental KPI value
        - ROI
- Summary table for set of predefined budget options for future periods
- Detailed report for every predefined budget option for future periods
    - Fit and predict by superbrand
    - Fit and predict by SKU / arbitrary set of variables
    - Dynamic decomposition by superbrand
    - Dynamic decomposition by SKU / arbitrary set of variables
    - Waterfall decomposition for defined future period

"""


import fermatrica_rep.export.basics
import fermatrica_rep.export.slides
