"""
FERMATRICA_REP reporting system for FERMATRICA econometrics framework.
"""


import fermatrica_rep.analysis
import fermatrica_rep.basics
import fermatrica_rep.options
import fermatrica_rep.stats
import fermatrica_rep.fit
import fermatrica_rep.curves
import fermatrica_rep.decomposition
import fermatrica_rep.waterfall
import fermatrica_rep.meta_model.model_rep
import fermatrica_rep.transformation
import fermatrica_rep.category
import fermatrica_rep.elasticity_price

from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.stats import metrics_group_table, metrics_table, predictors_table, vif_table
from fermatrica_rep.fit import fit_main_plot_vol, fit_main_plot_val, fit_mult_plot_vol, fit_mult_plot_val
from fermatrica_rep.category import category_plot
from fermatrica_rep.decomposition import extract_effect, decompose_main_plot, decompose_sub_plot, decompose_basic
from fermatrica_rep.waterfall import waterfall_plot, waterfall_data
from fermatrica_rep.curves import curves_simple_plot, curves_simple_data
from fermatrica_rep.curves_full import curves_full_plot_short, curves_full_plot_long
from fermatrica_rep.adstocks import adstocks_data
from fermatrica_rep.transformation import transformation_plot
from fermatrica_rep.options.calc import option_report, option_report_multi_var
from fermatrica_rep.options.calc_multi import option_report_multi_post
from fermatrica_rep.options.optim import optimize_budget_local_cobyla, optimize_target_local_cobyla
from fermatrica_rep.options.define import OptionSettings, trans_dict_create, budget_dict_create, media_ssn_apply
from fermatrica_rep import options
from fermatrica_rep.export import export_pptx
from fermatrica_rep.meta_model.meta_model import MetaModel
