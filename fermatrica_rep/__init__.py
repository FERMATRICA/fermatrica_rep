"""
FERMATRICA_REP reporting system for FERMATRICA econometrics framework.
"""


import fermatrica_rep.analysis
import fermatrica_rep.basics
import fermatrica_rep.options
import fermatrica_rep.analysis.stats
import fermatrica_rep.analysis.fit
import fermatrica_rep.analysis.curves
import fermatrica_rep.analysis.decomposition
import fermatrica_rep.analysis.waterfall
import fermatrica_rep.meta_model.model_rep
import fermatrica_rep.analysis.transformation
import fermatrica_rep.analysis.category
import fermatrica_rep.analysis.elasticity_price

from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.diagnostics import MetaModelDiagnostics, DiagnosticsReport, CheckResult
from fermatrica_rep.options.scope import OptionScope, find_scope, filter_scopes
from fermatrica_rep.options.option import Option
from fermatrica_rep.analysis.stats import metrics_group_table, metrics_table, predictors_table, vif_table, tests_table_ols
from fermatrica_rep.analysis.fit import fit_main_plot_vol, fit_main_plot_val, fit_mult_plot_vol, fit_mult_plot_val
from fermatrica_rep.analysis.category import category_plot
from fermatrica_rep.analysis.decomposition import extract_effect, decompose_main_plot, decompose_sub_plot, decompose_basic
from fermatrica_rep.analysis.waterfall import waterfall_plot, waterfall_data
from fermatrica_rep.analysis.curves import curves_simple_plot, curves_simple_data
from fermatrica_rep.analysis.curves_full import curves_full_plot_short, curves_full_plot_long
from fermatrica_rep.analysis.adstocks import adstocks_data
from fermatrica_rep.analysis.elasticity_price import elasticity_data
from fermatrica_rep.analysis.transformation import transformation_plot
from fermatrica_rep.options.calc import option_report, option_report_multi_var
from fermatrica_rep.options.calc_multi import option_report_multi_post
from fermatrica_rep.options.parallel import compute_multi, compute_grid
from fermatrica_rep.options.optim import (optimize_budget_local_cobyla, optimize_target_local_cobyla
                                          , optimize_budget_local_ncma, optimize_target_local_ncma
                                          , optimize_budget_local_cobyla_multi, optimize_target_local_cobyla_multi)
from fermatrica_rep.options.define import (OptionSettings, option_settings_list_create, trans_dict_create
                                           , budget_dict_create, media_ssn_apply, media_ssn_create, budget_ssn_create
                                           , scopes_read, media_prices_read, seasonality_read)
from fermatrica_rep.options.context import load_context, ComputeContext, campaigns_to_options
from fermatrica_rep import options
from fermatrica_rep.export import export_pptx, export_results
from fermatrica_rep.export import options_table, options_detail, curves, adstocks
from fermatrica_rep.export import summary_data, export_summaries
from fermatrica_rep.meta_model.model_exp import ModelConfExp, export_model
from fermatrica_rep.pipeline import generate_pipeline
