"""
Decompose model into effects (impacts).

This module describes dynamic decomposition, for waterfall decomposition see `fermatrica_rep.waterfall`
"""


import copy
import gc
import itertools
import logging
import re

from line_profiler import profile

import numpy as np
import patsy
import statsmodels.regression.linear_model

import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

from fermatrica_utils import groupby_eff, pandas_force_materialize, pandas_force_materialize_series

from fermatrica import params_to_dict, fermatrica_error, Model, ModelConf
from fermatrica.model.lhs_fun import *  # to run LHS
from fermatrica_rep.basics import _filter_df
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode, _pack_results
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel
from fermatrica_rep.meta_model.meta_model import MetaModel

pio.templates.default = 'ggplot2'


"""
Helpers
"""

def _transform_lhs_generate_part(fn_src: str
                                 , fn_name: str):
    """
    Creates part of LHS transformation function from source

    :param fn_src: pandas DataFrame containing order, type and source code of LHS transformations
    :param fn_name: function name to generate
    :return:
    """

    fn_src = \
'''
def ''' + fn_name + '''(ds: pd.DataFrame
                        , coef: "DotDict"
                        , y=None):
    rtrn = ''' + fn_src + \
'''
    return rtrn
'''

    d = {}

    if not isinstance(fn_src, str):
        fn_src = ''.join(fn_src)

    exec(fn_src, globals(), d)

    return d[fn_name]


def _debug_print_sum(data, var_name, ds, color='31', threshold=None, use_abs=False):
    """
    Helper function for debug printing DataFrame/Series sums filtered by listed==2.
    
    :param data: DataFrame or Series to check
    :param var_name: Variable name to display
    :param ds: Original dataset with 'listed' column
    :param color: ANSI color code (31=red, 33=yellow, 36=cyan, 32=green)
    :param threshold: Optional threshold for breakpoint (e.g., 1.0)
    :param use_abs: If True, use abs() for threshold comparison
    """
    mask_deb = ds.reset_index()['listed']
    if isinstance(data, pd.Series):
        sum_val = data[mask_deb == 2].sum()
    else:  # DataFrame
        sum_val = data[mask_deb == 2].sum().sum()
    
    print(f'\033[{color}m_ {var_name} {sum_val}\033[0m')
    
    if threshold is not None:
        check_val = abs(sum_val) if use_abs else sum_val
        if check_val < threshold:
            breakpoint()


"""
Extract RHS effects from specific model types
"""

@profile
def decompose_basic(model_cur: statsmodels.regression.linear_model.OLS | statsmodels.regression.linear_model.OLSResults | list
                    , ds: pd.DataFrame | list):
    """
    Wrapper
    Simplistic decomposition from OLS statsmodels object. If possible, use `extract_effects` instead.

    :param model_cur: OLS statsmodels object or list of OLS statsmodels object
    :param ds: dataset or list of datasets
    :return: table of effects (impacts) or list of tables of effects
    """

    if isinstance(model_cur, list):
        ret = [None]*len(model_cur)
        for i in range(len(model_cur)):
            ret[i] = _decompose_basic_worker(model_cur[i], ds[i])
    else: ret = _decompose_basic_worker(model_cur, ds)

    return ret


@profile
def _decompose_basic_worker(model_cur: statsmodels.regression.linear_model.OLS | statsmodels.regression.linear_model.OLSResults
                            , ds: pd.DataFrame):
    """
    Worker
    Simplistic decomposition from OLS statsmodels object. If possible, use `extract_effects` instead.

    :param model_cur: OLS statsmodels object
    :param ds: dataset
    :return: table of effects (impacts)
    """

    y, X = patsy.dmatrices(model_cur.formula, ds, return_type='dataframe')
    coefs = pd.Series(model_cur.fit().params)

    return X.dot(pd.DataFrame(np.diag(coefs), index=coefs.index, columns=coefs.index))


@profile
def _rhs_effects_lme(model_conf: "ModelConf"
                     , model_cur
                     , ds: pd.DataFrame
                     , id_vars: list | tuple
                     , if_detail
                     , if_debug: bool = False) -> pd.DataFrame:
    """
    The function extract effects (impacts) from RHS.

    If if_detail == True, all brackets in the formula are expanded,
     and effects are calculated for each component of the formula,
     including unions inside "I()"

    For example, if if_detail == True and formula ==
     I(a * par1 + (b + c) * par2) * par3,
     next effects will be extracted:
     a * par1 * par3
     b * par2 * par3
     c * par2 * par3

    The formulae must be logical, so the next examples will NOT work:
     I(a * par1 * par2 + b) -> use I(a * par12 + b) instead
     I(par2 * a * par1 + b) -> use I(a * par12 + b) or I(par12 * a + b) instead
     I(1 - m) * var -> use var - var * m instead (summands are split on '+' only)

    :param model_conf: ModelConf
    :param model_cur: object of statsmodels LME model. Separately from model_conf because different LME models could be used
    :param ds: dataset
    :param id_vars: index (dimension) variables. Mostly ['date'], not very important
    :param if_detail: expand tokens
    :param if_debug: enable debug mode
    :return: DataFrame of effects

    """

    fixed = model_cur.fe_params
    fixed.rename(index={'Intercept': 'intercept'}, inplace=True)

    random = model_cur.random_effects

    ds = ds.copy(deep=True)
    
    ds['intercept'] = 1

    split_m = ds[id_vars]
    split_m = pandas_force_materialize(split_m)

    for i, j in fixed.reset_index(name='value').iterrows():

        cols = j['index'].replace(':', '*').replace('I(', '(')
        if if_detail:
            # not universal (!!!) unbracketing, so check if while loop doesn't run forever

            itr = 0
            while ")" in cols:
                stack = []
                for i, c in enumerate(cols):

                    if c == '(':
                        stack.append(i)
                    elif c == ')' and stack:
                        start = stack.pop()
                        tmp = cols[start + 1: i]

                        stop_symbols = r"[a-zA-Z0-9_]+"

                        tmp_regex = tmp.replace('+', r'\+').replace('*', r'\*')

                        str_w_multiplier = re.search(stop_symbols + r" {0,1}\* {0,1}\(" + tmp_regex + r"\)"
                                                     + r"|"
                                                     + r"\(" + tmp_regex + r"\) {0,1}\* {0,1}" + stop_symbols, cols)

                        if str_w_multiplier:
                            str_w_multiplier = str_w_multiplier.group()
                            multiplier = re.sub(r" {0,1}\* {0,1}\(" + tmp_regex + r"\)"
                                                + r"|"
                                                + r"\(" + tmp_regex + r"\) {0,1}\* {0,1}", "", str_w_multiplier)

                            uncovered_str = [i + " * " + multiplier for i in tmp.split("+")]
                            uncovered_str = ' +'.join(uncovered_str)
                            cols = cols.replace(str_w_multiplier, uncovered_str)
                        else:
                            cols = cols.replace("(", "").replace(")", "")

                        break
                itr = itr + 1  # += not working as expected
                if itr > 10:
                    break

            # check if unbracketing was successfull
            if ")" not in cols:
                cols = cols.split("+")
            else:
                cols = [cols]

            for cl in cols:
                if re.search(r'[*( \-+:]', cl):
                    eval_result = ds.eval(cl, engine='python')
                    eval_result = pandas_force_materialize_series(pd.Series(eval_result) if not isinstance(eval_result, pd.Series) else eval_result)
                    split_m[cl] = (eval_result.values * j['value']).copy()
                    split_m[cl] = pandas_force_materialize_series(split_m[cl])
                else:
                    ds_col = ds[cl].copy(deep=True)
                    ds_col = pandas_force_materialize_series(ds_col)
                    split_m[cl] = (ds_col.values * j['value']).copy()
                    split_m[cl] = pandas_force_materialize_series(split_m[cl])
            # Materialize after each parameter
            split_m = split_m.copy(deep=True)
        else:

            if re.search(r'[*( \-+:]', cols):
                cols_all = ds.columns.to_list()

                cols_splt = re.split(r' *[*() \-+:] *', cols)
                cols_splt = [x for x in cols_splt if x in cols_all]

                # CRITICAL: Extract only needed columns, materialize minimally
                ds_cur = ds[cols_splt].copy(deep=True)
                
                # Now eval should work on clean DataFrame
                eval_result = ds_cur.eval(cols, engine='python').values.copy()
                eval_result = (eval_result * j['value']).copy()

                # eval_result = eval_result.copy(deep=True)
                split_m[j['index']] = eval_result

            else:
                ds_col = ds[cols].copy(deep=True)
                ds_col = pandas_force_materialize_series(ds_col)
                split_m[j['index']] = (ds_col.values * j['value']).copy()
                
            # Materialize after each parameter
            split_m[j['index']] = pandas_force_materialize_series(split_m[j['index']])
            split_m = split_m.copy(deep=True)
            
    # Materialize after loop
    split_m = pandas_force_materialize(split_m)

    ft = model_conf.fixed_effect_var

    j = pd.DataFrame(random).T.reset_index()
    j.columns = [ft, ft + '_' + 'intercept']
    j.iloc[0, 0] = j.iloc[0, 0]  # materialize
    j = j.copy(deep=True)
    
    split_m[ft] = ds[ft].values.copy()
    split_m[ft] = pandas_force_materialize_series(split_m[ft])

    split_m = split_m.merge(j, how='left', on=ft)
    split_m = split_m.reset_index(drop=True)  # break index references
    split_m = pandas_force_materialize(split_m)
    
    split_m[ft + '_' + 'intercept'] = split_m[ft + '_' + 'intercept'].values.copy()  # * split_m['Intercept']
    split_m[ft + '_' + 'intercept'] = pandas_force_materialize_series(split_m[ft + '_' + 'intercept'])

    del split_m[ft]

    split_m.drop(id_vars, axis=1, inplace=True)
    split_m = pandas_force_materialize(split_m)

    return split_m


@profile
def _rhs_effects_ols(model_cur: statsmodels.regression.linear_model.OLS | statsmodels.regression.linear_model.OLSResults
                     , ds: pd.DataFrame
                     , id_vars
                     , if_detail) -> pd.DataFrame:
    """
    The function extract effects of RHS.

    If if_detail == True, all brackets in the formula are expanded,
     and effects are calculated for each component of the formula,
     including unions inside "I()"

    For example, if if_detail == True and formula ==
     I(a * par1 + (b + c) * par2) * par3,
     next effects will be extracted:
     a * par1 * par3
     b * par2 * par3
     c * par2 * par3

    The formulas must be logical, so, the next examples will NOT work:
     I(a * par1 * par2 + b) -> use I(a * par12 + b) instead
     I(par2 * a * par1 + b) -> use I(a * par12 + b) or I(par12 * a + b) instead
     I(1 - m) * var -> use var - var * m instead (summands are split on '+' only)

    :param model_cur: object of statsmodels OLS model. Separately from model_conf because different LME models could be used
    :param ds: dataset
    :param id_vars: index (dimension) variables. Mostly ['date'], not very important
    :param if_detail: expand tokens
    :return:

    """

    params = model_cur.params
    params.rename(index={'Intercept': 'intercept'}, inplace=True)

    ds = ds.copy(deep=True)
    ds['intercept'] = 1

    split_m = ds[id_vars].copy(deep=True)

    for i, j in params.reset_index(name='value').iterrows():

        cols = j['index'].replace(':', '*').replace('I(', '(')
        if if_detail:
            while ")" in cols:
                stack = []
                for i, c in enumerate(cols):

                    if c == '(':
                        stack.append(i)
                    elif c == ')' and stack:
                        start = stack.pop()
                        tmp = cols[start + 1: i]

                        stop_symbols = r"[a-zA-Z0-9_]+"

                        tmp_regex = tmp.replace('+', r'\+').replace('*', r'\*')

                        str_w_multiplier = re.search(stop_symbols + r" {0,1}\* {0,1}\(" + tmp_regex + r"\)"
                                      + r"|"
                                      + r"\(" + tmp_regex + r"\) {0,1}\* {0,1}" + stop_symbols, cols)

                        if str_w_multiplier:
                            str_w_multiplier = str_w_multiplier.group()
                            multiplier = re.sub(r" {0,1}\* {0,1}\(" + tmp_regex + r"\)"
                                                + r"|"
                                                + r"\(" + tmp_regex + r"\) {0,1}\* {0,1}", "", str_w_multiplier)

                            uncovered_str = [i + " * " + multiplier for i in tmp.split("+")]
                            uncovered_str = ' +'.join(uncovered_str)
                            cols = cols.replace(str_w_multiplier, uncovered_str)
                        else:
                            cols = cols.replace("(", "").replace(")", "")

                        break
            cols = cols.split("+")
            for i in cols:
                split_m[i] = (ds.eval(i, engine='python') * j['value']).values.copy()
                split_m[i] = pandas_force_materialize_series(split_m[i])
            # Materialize after each parameter
            split_m = split_m.copy(deep=True)
        else:
            if "np.log1p" in cols:
                cols = cols.replace("np.log1p", "")
                split_m[j['index']] = np.log1p(ds.eval(cols, engine='python')).values * j['value']
                split_m[j['index']] = pandas_force_materialize_series(split_m[j['index']])
            else:
                split_m[j['index']] = (ds.eval(cols, engine='python') * j['value']).values.copy()
                split_m[j['index']] = pandas_force_materialize_series(split_m[j['index']])
            # Materialize after each parameter
            split_m = split_m.copy(deep=True)

    split_m.drop(id_vars, axis=1, inplace=True)
    split_m = pandas_force_materialize(split_m)

    return split_m


@profile
def rhs_effects(model_conf: "ModelConf"
                , model_cur
                , ds: pd.DataFrame
                , id_vars
                , if_detail
                , if_debug: bool = False) -> pd.DataFrame:
    """
    Extract RHS effects from different type statsmodels models (at least LME ans OLS are supported).

    :param model_conf: ModelConf
    :param model_cur: object of statsmodels model. Separately from model_conf because different models could be used
    :param ds: dataset
    :param id_vars: index (dimension) variables. Mostly ['date'], not very important
    :param if_detail: expand tokens
    :return:
    """

    if model_conf.model_type == 'OLS':
        rtrn = _rhs_effects_ols(model_cur, ds, id_vars, if_detail)
    elif model_conf.model_type == 'LME':
        rtrn = _rhs_effects_lme(model_conf=model_conf
                                , model_cur=model_cur
                                , ds=ds
                                , id_vars=id_vars
                                , if_detail=if_detail
                                , if_debug=if_debug)
    elif model_conf.model_type == 'LMEA':
        rtrn = _rhs_effects_lme(model_conf=model_conf
                                , model_cur=model_cur
                                , ds=ds
                                , id_vars=id_vars
                                , if_detail=if_detail
                                , if_debug=if_debug)
    else:
        rtrn = None
        fermatrica_error('Unknown model type in model decomposition: ' + model_conf.model_type)

    return rtrn


"""
Extract all effects
"""


@profile
def extract_effect(model: "Model | ModelNode | MetaModel | list"
                   , ds: pd.DataFrame | list | None = None
                   , model_rep: "ModelRep | list | None" = None
                   , if_detail: bool = False
                   , listed_filter: tuple | list = (2, 3, 4)
                   , if_debug: bool = False
                   , *, brands: list | None = None
                   , brands_var: str = 'superbrand'
                   , filters: dict | None = None
                   , whole_graph: bool | str = False) -> pd.DataFrame | list | dict:
    """
        Wrapper
        Decompose complex model / extract effects (impacts) of all types.

        :param model: Model, ModelNode, MetaModel, or list of Model objects (deprecated)
        :param ds: dataset or list of datasets (optional if ModelNode passed)
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects (optional if ModelNode passed)
        :param if_detail: expand tokens
        :param listed_filter: filter listed values to culaculate splits
        :param if_debug: enable debug mode
        :param brands: simple filter values (e.g. umbrella brands), applied to `brands_var`
        :param brands_var: column the simple filter values apply to
        :param filters: complex filter ``{column: [values]}``
        :param whole_graph: expand link factors through the DAG. ``False`` (default) —
            current behavior, each node decomposed independently. ``True`` — return
            ``{node_name: split_m_m}`` with every node's frame expanded transitively
            (a predecessor's whole output is split into the predecessor's own factor
            impacts, proportionally weighted by the successor's link contribution).
            A node name (str) returns a single ``split_m_m`` for that node, expanded
            the same way. The MetaModel must be computed first (``mm.compute()`` —
            every node's ``dt_pred`` must carry ``predicted``).
        :return: table with column per every extracted effect / impact, list, or dict keyed by node name
        """

    mm = _convert_to_metamodel(model, ds=ds, model_rep=model_rep)

    if whole_graph is not False:
        return _extract_effect_whole_graph(mm, whole_graph
                                           , if_detail=if_detail
                                           , listed_filter=listed_filter
                                           , if_debug=if_debug
                                           , brands=brands
                                           , brands_var=brands_var
                                           , filters=filters)

    results_dict = {}
    for name, node in mm.nodes.items():
        results_dict[name] = _extract_effect_worker(model=node.model
                                       , ds=node.ds
                                       , model_rep=node.model_rep
                                       , if_detail=if_detail
                                       , listed_filter=listed_filter
                                       , if_debug=if_debug
                                       , brands=brands
                                       , brands_var=brands_var
                                       , filters=filters)

    if isinstance(model, MetaModel):
        names = list(mm.nodes.keys())
    elif isinstance(model, list):
        names = True
    else:
        names = None
    return _pack_results(list(results_dict.values()), names)


@profile
def _extract_effect_worker(model: "Model"
                           , ds: pd.DataFrame
                           , model_rep: "ModelRep"
                           , if_detail: bool = False
                           , listed_filter: tuple | list = (2, 3, 4)
                           , if_debug: bool = False
                           , brands: list | None = None
                           , brands_var: str = 'superbrand'
                           , filters: dict | None = None) -> pd.DataFrame:
    """
    Worker
    Decompose complex model / extract effects (impacts) of all types.

    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param if_detail: expand tokens
    :param listed_filter: filter listed values to culaculate splits
    :param brands: simple filter values (e.g. umbrella brands), applied to `brands_var`
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return: table with column per every extracted effect / impact
    """

    wide, _ = _effects_wide(model, ds, model_rep, if_detail=if_detail, if_debug=if_debug)

    return _wide_to_split(wide, model.conf, model_rep, listed_filter
                          , brands=brands, brands_var=brands_var, filters=filters
                          , if_debug=if_debug)


@profile
def _effects_wide(model: "Model"
                  , ds: pd.DataFrame
                  , model_rep: "ModelRep"
                  , if_detail: bool = False
                  , if_debug: bool = False) -> tuple:
    """
    Compute the pre-melt wide effects frame (id columns + one column per factor).

    Rows align positionally with ``ds`` rows. Returns ``(wide, cln_id)`` where
    ``cln_id`` is the list of id columns carried alongside the effect columns.

    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param if_detail: expand tokens
    :param if_debug: enable debug mode
    :return: (wide, cln_id)
    """

    ds = pandas_force_materialize(ds, if_hard=False)

    model_conf = model.conf

    split_m = rhs_effects(model.conf, model.obj.models['main'], ds, ['date'], if_detail)

    split_m = pandas_force_materialize(split_m)

    lhs = model_conf.model_lhs

    if lhs is not None:
        to_original = True
        
        # Fetch coef once for all LHS transformations
        coef = model_conf.params[((model_conf.params['variable'] == '') | (model_conf.params['variable'].isna())) &
                                (model_conf.params['fun'] == 'coef')][['arg', 'value']]
        if not coef.empty:
            coef = params_to_dict(coef)
        
        # free_before type LHS transformations
        mask = '^free_before$'

        if ((lhs['if_active'] == 1) & (lhs['type'].str.match(mask))).sum() > 0:

            lhs_free_before = lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask))]

            lhs_free_before_funs = []
            for token in lhs_free_before.token:
                fun_cur = _transform_lhs_generate_part(token, 'fn_cur')
                lhs_free_before_funs.append(fun_cur)

            y = split_m.sum(axis=1)
            y = pandas_force_materialize_series(y)

            for fn_cur, col_name in zip(lhs_free_before_funs, lhs_free_before.name):
                y_new = fn_cur(ds=ds, coef=coef)

                split_m[col_name] = copy.deepcopy(y_new) - y
                split_m[col_name] = pandas_force_materialize_series(split_m[col_name])
                y = y_new
            
            split_m = pandas_force_materialize(split_m)

        # multiplicators

        mask = '^multiplicative$'

        if ((lhs['if_active'] == 1) & (lhs['type'].str.match(mask))).sum() > 0:
            
            lhs_mult = lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask))]

            lhs_funs = []
            for i in lhs_mult.token:
                fun_cur = _transform_lhs_generate_part(i, 'fn_cur')
                lhs_funs.append(fun_cur)

            rowsums = split_m.sum(axis=1)
            rowsums = pandas_force_materialize_series(rowsums)

            # y = [rowsums / eval(compile(i, '<string>', 'eval'), globals(), locals()) - rowsums for i in lhs_mult.token]
            y_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = rowsums / res_cur.values - rowsums
                y_list.append(y_tmp)
            y = pd.concat(y_list, ignore_index=True, axis=1, copy=True)
            y = pandas_force_materialize(y)

            # y = y.T
            y_pos_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values - 1
                y_pos_list.append(pd.DataFrame(y_tmp))
            y_pos = pd.concat(y_pos_list, ignore_index=True, axis=1, copy=True)
            y_pos = pandas_force_materialize(y_pos)

            if if_debug:
                _debug_print_sum(y_pos, 'y_pos', ds, color='31')

            # y_pos = y_pos.T
            # Use numpy directly to avoid fillna() view corruption
            y_pos_index = y_pos.index.copy()
            y_pos_columns = y_pos.columns.copy()
            y_pos_values = y_pos.values.copy()  # Extract to numpy ONCE
            
            # Replace NaN with 0 using numpy (avoid pandas fillna)
            y_pos_values = np.nan_to_num(y_pos_values, nan=0.0, copy=False)
            
            # Apply mask to numpy array
            mask_values = y_pos_values > 0.0
            y_pos_values[mask_values] = 1.0
            y_pos_values[~mask_values] = 0.0
            
            # Rebuild from numpy - breaks ALL references
            y_pos = pandas_force_materialize(pd.DataFrame(y_pos_values, index=y_pos_index, columns=y_pos_columns))
            
            if if_debug:
                _debug_print_sum(y_pos, 'y_pos', ds, color='31', threshold=1.0)

            y_dnm_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values - 1
                y_dnm_list.append(pd.DataFrame(y_tmp))
            y_dnm = pd.concat(y_dnm_list, ignore_index=True, axis=1, copy=True)
            y_dnm = pandas_force_materialize(y_dnm)

            # y_dnm = y_dnm.T
            # Single-step: multiply → sum → materialize final Series
            # Use pure numpy to avoid pandas multiplication issues
            y_dnm_np = y_dnm.values.copy()
            y_pos_np = y_pos.values.copy()
            y_dnm_mult_np = np.multiply(y_dnm_np, y_pos_np, dtype=np.float64)
            y_dnm_mult = pd.DataFrame(y_dnm_mult_np, index=y_dnm.index.copy(), columns=y_dnm.columns.copy())
            y_dnm_mult = pandas_force_materialize(y_dnm_mult)
            y_dnm = pandas_force_materialize_series(y_dnm_mult.sum(axis=1))

            if if_debug:
                _debug_print_sum(y_dnm, 'y_dnm', ds, color='31', threshold=1.0)

            y_num_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values
                y_num_list.append(pd.DataFrame(y_tmp))
            y_num = pd.concat(y_num_list, ignore_index=True, axis=1, copy=True)
            y_num = pandas_force_materialize(y_num)

            if if_debug:
                _debug_print_sum(y_num, 'y_num', ds, color='31', threshold=1.0)

            # Single-step: multiply → replace → prod → subtract → materialize final Series
            # Use pure numpy to avoid pandas multiplication issues
            y_num_np = y_num.values.copy()
            y_pos_np = y_pos.values.copy()
            y_num_mult_np = np.multiply(y_num_np, y_pos_np, dtype=np.float64)
            y_num_mult = pd.DataFrame(y_num_mult_np, index=y_num.index.copy(), columns=y_num.columns.copy())
            y_num_mult = pandas_force_materialize(y_num_mult, if_hard=True, if_soft=False)
            y_num = pandas_force_materialize_series(y_num_mult.replace(0, np.nan).prod(skipna=True, axis=1) - 1)

            if if_debug:
                _debug_print_sum(y_num, 'y_num', ds, color='31', threshold=1.0)

            # Single-step: ratio → mul → multiply → fillna → materialize
            ratio = pandas_force_materialize_series(y_num / y_dnm)
            # Use numpy broadcasting to avoid .mul() view issues
            y_pos_np = y_pos.values.copy()
            ratio_np = ratio.values.reshape(-1, 1)  # Reshape for broadcasting
            y_pos_adjusted_np = np.multiply(y_pos_np, ratio_np, dtype=np.float64)
            y_pos_adjusted = pd.DataFrame(y_pos_adjusted_np, index=y_pos.index.copy(), columns=y_pos.columns.copy())
            y_pos_adjusted = pandas_force_materialize(y_pos_adjusted)
                
            # Use pure numpy to avoid pandas multiplication issues
            y_np = y.values.astype(np.float64, copy=True)
            y_pos_adjusted_np = y_pos_adjusted.values.astype(np.float64, copy=True)
            y_pos_mult_np = np.multiply(y_np, y_pos_adjusted_np, dtype=np.float64)
            # Replace NaN with 0 using numpy (avoid pandas fillna)
            y_pos_mult_np = np.nan_to_num(y_pos_mult_np, nan=0.0, copy=False)
            y_pos = pd.DataFrame(y_pos_mult_np, index=y.index.copy(), columns=y.columns.copy())
            y_pos = pandas_force_materialize(y_pos)

            if if_debug:
                _debug_print_sum(y_pos, 'y_pos', ds, color='31', threshold=1.0, use_abs=True)

            # neg
            y_neg_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values - 1
                y_neg_list.append(pd.DataFrame(y_tmp))
            y_neg = pd.concat(y_neg_list, ignore_index=True, axis=1, copy=True)
            y_neg = pandas_force_materialize(y_neg)

            # y_neg = y_neg.T
            # Use numpy directly to avoid fillna() view corruption
            y_neg_index = y_neg.index.copy()
            y_neg_columns = y_neg.columns.copy()
            y_neg_values = y_neg.values.copy()  # Extract to numpy ONCE
            
            # Replace NaN with 0 using numpy (avoid pandas fillna)
            y_neg_values = np.nan_to_num(y_neg_values, nan=0.0, copy=False)

            # Apply mask to numpy array
            mask_neg_values = y_neg_values < 0.0
            y_neg_values[mask_neg_values] = 1.0
            y_neg_values[~mask_neg_values] = 0.0
            
            # Rebuild from numpy - breaks ALL references
            y_neg = pandas_force_materialize(pd.DataFrame(y_neg_values, index=y_neg_index, columns=y_neg_columns))

            if if_debug:
                _debug_print_sum(y_neg, 'y_neg', ds, color='33', threshold=1.0, use_abs=True)

            y_dnm_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values - 1
                y_dnm_list.append(pd.DataFrame(y_tmp))
            y_dnm = pd.concat(y_dnm_list, ignore_index=True, axis=1, copy=True)
            y_dnm = pandas_force_materialize(y_dnm)

            # y_dnm = y_dnm.T
            # Single-step: multiply → sum → materialize final Series
            # Use pure numpy to avoid pandas multiplication issues
            y_dnm_np = y_dnm.values.copy()
            y_neg_np = y_neg.values.copy()
            y_dnm_mult_np = np.multiply(y_dnm_np, y_neg_np, dtype=np.float64)
            y_dnm_mult = pd.DataFrame(y_dnm_mult_np, index=y_dnm.index.copy(), columns=y_dnm.columns.copy())
            y_dnm_mult = pandas_force_materialize(y_dnm_mult)
            y_dnm = pandas_force_materialize_series(y_dnm_mult.sum(axis=1))

            if if_debug:
                _debug_print_sum(y_dnm, 'y_dnm', ds, color='33', threshold=1.0, use_abs=True)

            y_num_list = []
            for i, fn_cur in zip(lhs_mult.token, lhs_funs):
                res_cur = fn_cur(ds=ds, coef=coef)
                y_tmp = 1 / res_cur.values
                y_num_list.append(pd.DataFrame(y_tmp))
            y_num = pd.concat(y_num_list, ignore_index=True, axis=1, copy=True)
            y_num = pandas_force_materialize(y_num)

            # Single-step: multiply → replace → prod → subtract → materialize final Series
            # Use pure numpy to avoid pandas multiplication issues
            y_num_np = y_num.values.copy()
            y_neg_np = y_neg.values.copy()
            y_num_mult_np = np.multiply(y_num_np, y_neg_np, dtype=np.float64)
            y_num_mult = pd.DataFrame(y_num_mult_np, index=y_num.index.copy(), columns=y_num.columns.copy())
            y_num_mult = pandas_force_materialize(y_num_mult)
            y_num = pandas_force_materialize_series(y_num_mult.replace(0, np.nan).prod(skipna=True, axis=1) - 1)

            if if_debug:
                _debug_print_sum(y_num, 'y_num', ds, color='33', threshold=1.0, use_abs=True)

            # Single-step: ratio → mul → multiply → fillna → materialize
            ratio = pandas_force_materialize_series(y_num / y_dnm)
            # Use numpy broadcasting to avoid .mul() view issues
            y_neg_np = y_neg.values.copy()
            ratio_np = ratio.values.reshape(-1, 1)  # Reshape for broadcasting
            y_neg_adjusted_np = np.multiply(y_neg_np, ratio_np, dtype=np.float64)
            y_neg_adjusted = pd.DataFrame(y_neg_adjusted_np, index=y_neg.index.copy(), columns=y_neg.columns.copy())
            y_neg_adjusted = pandas_force_materialize(y_neg_adjusted)

            if if_debug:
                _debug_print_sum(y_neg_adjusted, 'y_neg_adjusted', ds, color='33', threshold=1.0, use_abs=True)

            # Use pure numpy to avoid pandas multiplication issues
            y_np = y.values.astype(np.float64, copy=True)
            y_neg_adjusted_np = y_neg_adjusted.values.astype(np.float64, copy=True)
            y_neg_mult_np = np.multiply(y_np, y_neg_adjusted_np, dtype=np.float64)

            # Replace NaN with 0 using numpy (avoid pandas fillna)
            y_neg_mult_np = np.nan_to_num(y_neg_mult_np, nan=0.0, copy=False)
            y_neg = pd.DataFrame(y_neg_mult_np, index=y.index.copy(), columns=y.columns.copy())
            y_neg = pandas_force_materialize(y_neg)

            if if_debug:
                _debug_print_sum(y_neg, 'y_neg', ds, color='36')
                _debug_print_sum(y_pos, 'y_pos', ds, color='36')
                _debug_print_sum(rowsums, 'rowsums', ds, color='36')

            # Use numpy to avoid chained .mul().div() view issues
            # Explicit dtype conversion to avoid object dtype issues
            y_neg_np = y_neg.values.astype(np.float64, copy=True)
            y_pos_np = y_pos.values.astype(np.float64, copy=True)
            rowsums_np = rowsums.values.astype(np.float64, copy=True).reshape(-1, 1)  # Reshape for broadcasting
            y_pos_sum_np = y_pos_np.sum(axis=1, keepdims=True)
            
            numerator_np = np.multiply(y_neg_np, (rowsums_np + y_pos_sum_np), dtype=np.float64)
            y_final_np = np.divide(numerator_np, rowsums_np, dtype=np.float64) + y_pos_np
            y = pd.DataFrame(y_final_np, index=y_neg.index.copy(), columns=y_neg.columns.copy())
            y = pandas_force_materialize(y)

            y.columns = lhs_mult.name

            split_m = pd.concat([split_m, y], axis=1)
            split_m = pandas_force_materialize(split_m)

            if if_debug:
                _debug_print_sum(split_m, 'split_m_m', ds, color='32')

        # free_after type LHS transformations
        mask = '^free_after'

        if ((lhs['if_active'] == 1) & (lhs['type'].str.match(mask))).sum() > 0:

            lhs_free_after = lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask))]

            lhs_free_after_funs = []
            for token in lhs_free_after.token:
                fun_cur = _transform_lhs_generate_part(token, 'fn_cur')
                lhs_free_after_funs.append(fun_cur)

            y = split_m.sum(axis=1).array
            y = y.copy()

            for fn_cur, col_name in zip(lhs_free_after_funs, lhs_free_after.name):
                y_new = fn_cur(ds=ds, coef=coef, y=y).array

                split_m[col_name] = y_new - y
                split_m[col_name] = pandas_force_materialize_series(split_m[col_name])
                y = y_new
            
            split_m = pandas_force_materialize(split_m)

    cln_id = ["date", "pack_key", "bs_key", "listed", model_conf.price_var] + model_conf.bs_key
    cln_id = [x for x in cln_id if x in ds.columns.copy()]

    # Materialize select_eff result immediately
    sel_eff_df = select_eff(ds, cln_id).reset_index(names=['tmp1____'])
    sel_eff_df = sel_eff_df.copy(deep=True)
    
    split_m_reset = split_m.reset_index(names=['tmp2____'])
    split_m_reset = split_m_reset.copy(deep=True)
    
    split_m = pd.concat([sel_eff_df, split_m_reset], axis=1)
    del split_m['tmp1____']
    del split_m['tmp2____']
    split_m = pandas_force_materialize(split_m)

    return split_m, cln_id


@profile
def _wide_to_split(wide: pd.DataFrame
                   , model_conf: "ModelConf"
                   , model_rep: "ModelRep"
                   , listed_filter: tuple | list = (2, 3, 4)
                   , brands: list | None = None
                   , brands_var: str = 'superbrand'
                   , filters: dict | None = None
                   , id_vars_extra: list | None = None
                   , rename_map: dict | None = None
                   , if_debug: bool = False) -> pd.DataFrame:
    """
    Convert a wide effects frame into the long ``split_m_m`` report format.

    Melts the wide frame, sums per ``(id columns, variable)``, computes
    ``value_rub``, applies the ``listed_filter`` groupby, renames factors to
    display names, sorts, applies user filters and refreshes palettes.

    ``rename_map`` (raw factor name -> display name, recursive for dotted
    whole-DAG names) replaces the default ``model_conf`` / ``model_rep``
    lookup when provided; ``id_vars_extra`` extends the id columns beyond
    ``cln_id`` (join keys added by the whole-DAG expansion).

    :param wide: pre-melt wide effects frame (see ``_effects_wide``)
    :param model_conf: ModelConf
    :param model_rep: ModelRep object (export settings)
    :param listed_filter: filter listed values to culaculate splits
    :param brands: simple filter values (e.g. umbrella brands), applied to `brands_var`
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param id_vars_extra: extra id columns to carry through the melt
    :param rename_map: pre-built raw factor -> display name map
    :param if_debug: enable debug mode
    :return: table with column per every extracted effect / impact
    """

    cln_id = ["date", "pack_key", "bs_key", "listed", model_conf.price_var] + model_conf.bs_key
    cln_id = [x for x in cln_id if x in wide.columns]
    if id_vars_extra:
        cln_id = cln_id + [x for x in id_vars_extra if x not in cln_id]

    split_m = wide

    cln_sort = [x for x in cln_id if x in ['superbrand', 'bs_key', 'master', 'date']]

    split_m_m = split_m.melt(id_vars=cln_id)
    split_m_m = split_m_m[~split_m_m['value'].isna()]
    split_m_m = split_m_m.reset_index(drop=True)  # break index references
    split_m_m = split_m_m.copy(deep=True)

    split_m_m = split_m_m.groupby(cln_id + ['variable'])['value'].\
        sum().\
        reset_index()
    split_m_m = pandas_force_materialize(split_m_m)

    split_m_m['value_rub'] = split_m_m['value'].values * split_m_m[model_conf.price_var].values
    split_m_m['value_rub'] = pandas_force_materialize_series(split_m_m['value_rub'])

    pattern = re.compile(r'^[0-9]+_')

    if not model_rep.vis_dict.empty:

        if rename_map is None:
            rename_map = _build_rename_dict(model_conf, model_rep, split_m_m.variable.unique())

        split_m_m['variable'] = split_m_m['variable'].replace(rename_map)
        split_m_m['variable'] = pandas_force_materialize_series(split_m_m['variable'])
        split_m_m = split_m_m.copy(deep=True)

        mask = split_m_m['listed'].isin(listed_filter)

        split_m_m = groupby_eff(split_m_m, cln_id + ['variable'], ['value', 'value_rub'], mask) \
            .agg({'value': 'sum', 'value_rub': 'sum'}) \
            .reset_index()
        split_m_m = pandas_force_materialize(split_m_m)

        split_m_m['variable'] = split_m_m['variable'].astype('category')
        split_m_m['variable'] = pandas_force_materialize_series(split_m_m['variable'])
        split_m_m = split_m_m.copy(deep=True)

    split_m_m = split_m_m.sort_values(['variable'] + cln_sort, axis=0, ascending=True)
    split_m_m = split_m_m.reset_index(drop=True)  # break index references after sort
    split_m_m['variable'] = split_m_m['variable'].str.replace(pattern, '', regex=True)
    split_m_m['variable'] = pandas_force_materialize_series(split_m_m['variable'])
    split_m_m = split_m_m.copy(deep=True)

    # prevent inconsistence due to lazy loading in later functions

    split_m_m = pandas_force_materialize(split_m_m)

    # enforce data types to avoid issues in later functions

    split_m_m['value'] = split_m_m['value'].astype(float)
    split_m_m['value_rub'] = split_m_m['value_rub'].astype(float)

    # apply user filters to the output split (post-computation)

    split_m_m = _filter_df(split_m_m, brands=brands, brands_var=brands_var, filters=filters)

    # update palettes

    lst = split_m_m.loc[split_m_m['value'] > 0, 'variable'].unique().tolist()
    lst.sort()

    model_rep.fill_colours_tools(lst)

    return split_m_m


def _build_rename_dict(model_conf: "ModelConf"
                       , model_rep: "ModelRep"
                       , names: list | pd.Index) -> dict:
    """
    Build the raw factor name -> display name map used by the decomposition
    rename step (intercept/base, LHS, and RHS tokens, with ``vis_dict``
    translations and ``NN_`` sort prefixes).
    """
    language = model_rep.language
    pattern = re.compile(r'^[0-9]+_')

    rename_dict = {}

    for name in names:

        if 'ntercept' in name:
            disp_name_base = getattr(model_conf, 'display_var_base', 'base')

            if disp_name_base in model_conf.model_rhs.display_var.values:
                disp_name = disp_name_base

                if pattern.match(disp_name) is None:
                    ptrn = ''
                else:
                    ptrn = pattern.match(disp_name)[0]

                if pattern.sub('', disp_name) in model_rep.vis_dict.loc[model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']), 'variable'].array:
                    rename_dict[name] = ptrn + \
                        model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == pattern.sub('', disp_name)) &
                        (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type'])), language].iloc[0]
                else:
                    rename_dict[name] = ptrn + pattern.sub('', disp_name)
            else:
                if disp_name_base in model_rep.vis_dict.loc[model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']), 'variable'].array:
                    rename_dict[name] = '01_' +\
                        model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == disp_name_base) &
                                     (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type'])), language].iloc[0]
                else:
                    rename_dict[name] = '01_' + disp_name_base

        elif model_conf.model_lhs is not None and name in model_conf.model_lhs.name.values:

            disp_name = model_conf.model_lhs.loc[model_conf.model_lhs['name'] == name, 'display_var'].iloc[0]

            if pattern.match(disp_name) is None:
                ptrn = ''
            else:
                ptrn = pattern.match(disp_name)[0]

            if pattern.sub('', disp_name) in model_rep.vis_dict.loc[model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']), 'variable'].array:

                rename_dict[name] = ptrn + \
                    model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == pattern.sub('', disp_name)) &
                    (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type'])), language].iloc[0]

            else:
                rename_dict[name] = ptrn + pattern.sub('', disp_name)

        elif name.replace(' ', '') in model_conf.model_rhs.token.str.replace(' ', '').values:

            disp_name = model_conf.model_rhs.loc[model_conf.model_rhs.token.str.replace(' ', '') == name.replace(' ', ''), 'display_var'].iloc[0]

            if pattern.match(disp_name) is None:
                ptrn = ''
            else:
                ptrn = pattern.match(disp_name)[0]

            if pattern.sub('', disp_name) in model_rep.vis_dict.loc[model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']), 'variable'].array:

                rename_dict[name] = ptrn + \
                    model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == pattern.sub('', disp_name)) &
                    (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type'])), language].iloc[0]

            else:
                rename_dict[name] = ptrn + pattern.sub('', disp_name)

    return rename_dict


def _validate_dt_pred(mm: "MetaModel"):
    """
    Check that every node's ``dt_pred`` is computed with a ``predicted`` column.

    The whole-DAG expansion needs the predecessor predictions to identify link
    columns and broadcast factor shares, so the caller must run ``mm.compute()``
    first (no-arg refresh or option path). Raises ``FermatricaError`` otherwise.
    """
    missing = [name for name, node in mm.nodes.items()
               if node.dt_pred is None
               or not isinstance(node.dt_pred, pd.DataFrame)
               or 'predicted' not in node.dt_pred.columns]
    if missing:
        fermatrica_error(
            "extract_effect(whole_graph=...) requires every node's dt_pred to be "
            "computed with a 'predicted' column — run mm.compute() (no-arg refresh "
            "or option path) first. Missing for node(s): " + ", ".join(missing)
        )


def _edge_merge_link_columns(mm: "MetaModel", node_name: str, pred_name: str) -> list:
    """
    Identify the link columns an edge's merge function writes into ``node.ds``.

    Re-runs the merge (exactly like ``MetaModel.compute``) on a copy of the
    node's dataset with the predecessor's ``predicted`` perturbed by ``+1e6``;
    columns whose values changed or appeared vs the untouched dataset are
    candidate link columns. The perturbation preserves ``notna()`` masks, so
    only the actual broadcast columns differ.
    """
    merge_fn = mm._edge_merge_fn(pred_name, node_name)
    if merge_fn is None:
        return []
    node = mm.nodes[node_name]
    pred = mm.nodes[pred_name]
    mm._load_adhoc_from_node(node)
    ds_probe = node.ds.copy(deep=True)
    dt_pred_perturbed = pred.dt_pred.copy(deep=True)
    dt_pred_perturbed['predicted'] = dt_pred_perturbed['predicted'] + 1e6
    ds_merged = mm._apply_merge_fn(merge_fn, ds_cur=ds_probe, dt_pred_prev=dt_pred_perturbed, allow_past=True)

    candidates = []
    for col in ds_merged.columns:
        if col not in node.ds.columns:
            candidates.append(col)
        elif not ds_merged[col].equals(node.ds[col]):
            candidates.append(col)
    return candidates


def _search_join_keys(dt_pred_P: pd.DataFrame, ds_cur: pd.DataFrame, link_col: str) -> list | None:
    """
    Find the smallest key set ``K`` that validates ``link_col`` as a prediction
    broadcast.

    ``K`` is a subset (size 1-3) of the common ``{date, superbrand, bs_key,
    master, form, market, region, pack_key}`` columns plus the derived
    ``{year, month}`` pair (from ``date``). ``K`` is valid iff broadcasting
    ``P.dt_pred['predicted']`` on ``K`` equals ``ds_cur[link_col]`` on all its
    not-NaN rows (rtol/atol tolerance) and ``P.dt_pred`` has unique ``(K)``
    rows. Returns ``None`` when no key set validates the column.
    """
    pool = ['date', 'superbrand', 'bs_key', 'master', 'form', 'market', 'region', 'pack_key']
    pool = [c for c in pool if c in dt_pred_P.columns and c in ds_cur.columns]
    if 'date' in pool:
        units = pool + ['year_month']
    else:
        units = pool

    link = ds_cur[link_col]
    notna_mask = link.notna().to_numpy()
    if not notna_mask.any():
        return None
    link_vals = link.to_numpy()

    def _eff(keys):
        out = []
        for k in keys:
            if k == 'year_month':
                out += ['year', 'month']
            else:
                out.append(k)
        return out

    def _derive(df, keys_eff):
        if not any(k in ('year', 'month') for k in keys_eff):
            return df
        df = df.copy()
        for k in ('year', 'month'):
            if k in keys_eff and k not in df.columns and 'date' in df.columns:
                df[k] = getattr(df['date'].dt, k)
        return df

    for size in (1, 2, 3):
        for combo in itertools.combinations(units, size):
            if 'date' in combo and 'year_month' in combo:
                continue
            keys_eff = _eff(combo)
            if not all(k in dt_pred_P.columns or k in ('year', 'month') for k in keys_eff):
                continue
            tmp = _derive(dt_pred_P, keys_eff)[keys_eff + ['predicted']].dropna(subset=['predicted'])
            if tmp.empty or tmp.duplicated(subset=keys_eff).any():
                continue
            check = _derive(ds_cur, keys_eff)[keys_eff].merge(tmp, on=keys_eff, how='left')['predicted'].to_numpy()
            if not np.isclose(check[notna_mask], link_vals[notna_mask], rtol=1e-5, atol=1e-8).all():
                continue
            return keys_eff
    return None


def _link_factor_columns(wide: pd.DataFrame, link_cols: list, id_cols: list) -> list:
    """
    Wide factor columns that represent a predecessor link.

    A column is a link factor when one of its name tokens (split on
    non-alphanumeric chars, so ``category_predicted_softmax`` matches the raw
    link ``category_predicted``) starts with a raw link column name. Only
    additive columns qualify — operator-bearing (interaction) terms that
    reference a link stay single factors with a warning. Dotted (already
    expanded) columns are skipped.
    """
    link_factors = []
    for col in wide.columns:
        if col in id_cols or '.' in col:
            continue
        tokens = re.split(r'[^\w]+', col)
        if not any(t.startswith(lc) for t in tokens for lc in link_cols):
            continue
        if re.search(r'[*:+\-()]', col):
            logging.warning(
                f"extract_effect(whole_graph): interaction term '{col}' references a link "
                f"column ({', '.join(link_cols)}) and is left unexpanded"
            )
            continue
        link_factors.append(col)
    return link_factors


def _expand_links(wide_N: pd.DataFrame, link_factors: list, wide_P: pd.DataFrame, keys: list, id_cols_P: list) -> pd.DataFrame:
    """
    Replace each additive link factor of ``wide_N`` with proportionally split
    leaves of the (already expanded) predecessor wide frame.

    Per predecessor row ``denom = sum of wide_P factor columns``; leaf value =
    ``wide_N[link] * wide_P[f] / denom`` (``0`` where ``|denom| < 1e-12``),
    broadcast onto ``wide_N`` rows on ``keys``. By construction the expanded
    columns sum to the original link factor per row, preserving the
    "decomposition sums equal prediction" invariant.
    """
    factor_cols = [c for c in wide_P.columns if c not in id_cols_P]
    if not factor_cols:
        return wide_N

    denom = wide_P[factor_cols].sum(axis=1, skipna=True).to_numpy()
    safe = np.abs(denom) < 1e-12

    wide_N = wide_N.copy(deep=True)

    share_df = {}
    for k in keys:
        if k in wide_P.columns:
            share_df[k] = wide_P[k].to_numpy()
        elif k in ('year', 'month') and 'date' in wide_P.columns:
            share_df[k] = getattr(wide_P['date'].dt, k).to_numpy()
        else:
            return wide_N

    for L in link_factors:
        share_cols = dict(share_df)
        for f in factor_cols:
            share_cols[f"{L}.{f}"] = np.where(safe, 0.0, wide_P[f].to_numpy() / denom)
        merged = wide_N[keys].merge(pd.DataFrame(share_cols), on=keys, how='left')
        for f in factor_cols:
            wide_N[f"{L}.{f}"] = wide_N[L].to_numpy() * merged[f"{L}.{f}"].to_numpy()
        wide_N = wide_N.drop(columns=[L])

    return wide_N


def _rename_expanded(name: str, node_map: dict, pred_map: dict) -> str:
    """
    Resolve a dotted whole-DAG factor name through the node's own rename map
    and the predecessor's (recursive) rename map, stripping ``NN_`` sort
    prefixes from both segments.
    """
    head, _, rest = name.partition('.')
    head_disp = node_map.get(head, head)
    rest_disp = pred_map.get(rest, rest)
    return re.sub(r'^[0-9]+_', '', str(head_disp)) + '.' + re.sub(r'^[0-9]+_', '', str(rest_disp))


def _extract_effect_whole_graph(mm: "MetaModel"
                                , whole_graph: bool | str
                                , if_detail: bool = False
                                , listed_filter: tuple | list = (2, 3, 4)
                                , if_debug: bool = False
                                , brands: list | None = None
                                , brands_var: str = 'superbrand'
                                , filters: dict | None = None) -> pd.DataFrame | dict:
    """
    Decompose a MetaModel as a whole: expand each node's link factors into the
    predecessor's own factor impacts (transitively), proportionally split by
    the successor's link contribution.

    ``whole_graph=True`` returns ``{node_name: split_m_m}`` for every node;
    a node name returns a single ``split_m_m`` for that node. An unknown node
    name raises ``FermatricaError``. The MetaModel must be computed first
    (every node's ``dt_pred`` carries ``predicted``).
    """
    order = mm._topological_order()
    _validate_dt_pred(mm)

    if isinstance(whole_graph, str):
        if whole_graph not in mm.nodes:
            fermatrica_error(
                "extract_effect(whole_graph='" + whole_graph + "'): unknown node name. "
                "Known nodes: " + ", ".join(sorted(mm.nodes))
            )
        targets = [whole_graph]
    else:
        targets = list(mm.nodes.keys())

    links = {}
    for node_name in order:
        node_links = {}
        for pred_name in mm._predecessors(node_name):
            merge_fn = mm._edge_merge_fn(pred_name, node_name)
            if merge_fn is None:
                continue
            for link_col in _edge_merge_link_columns(mm, node_name, pred_name):
                keys = _search_join_keys(mm.nodes[pred_name].dt_pred, mm.nodes[node_name].ds, link_col)
                if keys is not None:
                    node_links[link_col] = (pred_name, keys)
                else:
                    logging.warning(
                        f"extract_effect(whole_graph): no join keys found for link column "
                        f"'{link_col}' on edge '{pred_name}' -> '{node_name}' — factor left unexpanded"
                    )
        links[node_name] = node_links

    wide = {}
    id_cols = {}
    rename_maps = {}
    for node_name in order:
        node = mm.nodes[node_name]
        w, ids = _effects_wide(node.model, node.ds, node.model_rep, if_detail, if_debug)
        ids = list(ids)

        for link_col, (pred_name, keys) in links.get(node_name, {}).items():
            for k in keys:
                if k in w.columns:
                    continue
                if k in node.ds.columns:
                    w[k] = node.ds[k].values
                elif k in ('year', 'month') and 'date' in w.columns:
                    w[k] = getattr(w['date'].dt, k)
                else:
                    continue
                if k not in ids:
                    ids.append(k)

        if node.model_rep.vis_dict.empty:
            own_map = {}
        else:
            own_map = _build_rename_dict(node.model.conf, node.model_rep, [c for c in w.columns if c not in ids])

        for pred_name in mm._predecessors(node_name):
            pred_node = mm.nodes[pred_name]
            wp = wide[pred_name]
            ids_p = list(id_cols[pred_name])
            for link_col, (pred, keys) in links.get(node_name, {}).items():
                if pred != pred_name:
                    continue
                for k in keys:
                    if k not in wp.columns:
                        if k in pred_node.ds.columns:
                            wp[k] = pred_node.ds[k].values
                        elif k in ('year', 'month') and 'date' in wp.columns:
                            wp[k] = getattr(wp['date'].dt, k)
                    if k not in ids_p:
                        ids_p.append(k)
                id_cols[pred_name] = ids_p

                link_factors = _link_factor_columns(w, [link_col], ids)
                if not link_factors:
                    continue
                w = _expand_links(w, link_factors, wp, keys, id_cols[pred_name])
                pred_map = rename_maps[pred_name]
                for L in link_factors:
                    for f in [c for c in wp.columns if c not in id_cols[pred_name]]:
                        dotted = f"{L}.{f}"
                        if L in own_map and f in pred_map:
                            own_map[dotted] = _rename_expanded(dotted, own_map, pred_map)
            wide[pred_name] = wp

        wide[node_name] = w
        id_cols[node_name] = ids
        rename_maps[node_name] = own_map

    results = {}
    for node_name in targets:
        node = mm.nodes[node_name]
        w = wide[node_name]
        derived = [x for x in ["date", "pack_key", "bs_key", "listed", node.model.conf.price_var] + node.model.conf.bs_key
                   if x in w.columns]
        extra = [k for k in id_cols[node_name] if k not in derived]
        results[node_name] = _wide_to_split(w, node.model.conf, node.model_rep, listed_filter
                                            , brands=brands, brands_var=brands_var, filters=filters
                                            , id_vars_extra=extra, rename_map=rename_maps[node_name]
                                            , if_debug=if_debug)

    if isinstance(whole_graph, str):
        return results[whole_graph]
    return results


@profile
def decompose_main_plot(split_m_m: pd.DataFrame | list | dict
                        , brands: list | None = None
                        , model_rep: "ModelRep | ModelNode | MetaModel | list | None" = None
                        , *, brands_var: str = 'superbrand'
                        , filters: dict | None = None
                        , period: str | list = 'day'
                        , show_future: bool | str = True
                        , show_pretrain: bool = True
                        , contour_line: bool = True) -> go.Figure | list | dict:
    """
        Wrapper
        Plot main dynamic decomposition (without faceting)

        :param split_m_m: prepared dataset (see `extract_effect()`) or list/dict of prepared datasets
        :param brands: list of umbrella brands to preserve, None to keep all
        :param model_rep: ModelRep, ModelNode, MetaModel, or list of ModelRep objects (deprecated)
        :param brands_var: column the simple filter values apply to
        :param filters: complex filter ``{column: [values]}``
        :param period: time period / interval to group by or list of time periods / intervals to group by
        :param show_future: show future periods or not
        :param show_pretrain: show pretrain periods or not
        :param contour_line: add contours or not
        :return:
        """

    mm = _convert_to_metamodel(model_rep)

    results = []
    for i, (name, node) in enumerate(mm.nodes.items()):
        splits_i = split_m_m[name] if isinstance(split_m_m, dict) else (split_m_m[i] if isinstance(split_m_m, list) else split_m_m)
        period_i = period[name] if isinstance(period, dict) else (period[i] if isinstance(period, list) else period)
        results.append(_decompose_main_plot_worker(split_m_m=splits_i, brands=brands, model_rep=node.model_rep,
                                                    period=period_i, show_future=show_future,
                                                    show_pretrain=show_pretrain, contour_line=contour_line,
                                                    brands_var=brands_var, filters=filters))

    if isinstance(model_rep, MetaModel):
        return dict(zip(list(mm.nodes.keys()), results))
    if isinstance(model_rep, list):
        return results
    return results[0]


@profile
def _decompose_main_plot_worker(split_m_m: pd.DataFrame
                                , brands: list | None = None
                                , model_rep: "ModelRep" = None
                                , period: str = 'day'
                                , show_future: bool | str = True
                                , show_pretrain: bool | str = False
                                , contour_line: bool = True
                                , brands_var: str = 'superbrand'
                                , filters: dict | None = None) -> go.Figure:
    """
    Worker
    Plot main dynamic decomposition (without faceting)

    :param split_m_m: prepared dataset (see `extract_effect()`)
    :param brands: list of umbrella brands to preserve, None to keep all
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by
    :param show_future: show future periods or not
    :param show_pretrain: show pretrain periods or not
    :param contour_line: add contours or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    language = model_rep.language

    if show_future == 'True':
        show_future = True
    elif show_future == 'False':
        show_future = False

    split = split_m_m.copy()

    split = _filter_df(split, brands=brands, brands_var=brands_var, filters=filters)

    if not brands:
        if brands_var in split.columns:
            brands = split[brands_var].unique()
        elif 'superbrand' in split.columns:
            brands = split['superbrand'].unique()
        else:
            brands = []

    if period in ['day', 'd', 'D']:
        split['date'] = split['date'].dt.floor(freq='D')
    elif period in ['week', 'w', 'W']:
        split['date'] = split['date'].dt.to_period('W').dt.start_time
    elif period in ['month', 'm', 'M']:
        split['date'] = split['date'].dt.to_period('M').dt.start_time
    elif period in ['quarter', 'q', 'Q']:
        split['date'] = split['date'].dt.to_period('Q').dt.start_time
    elif period in ['year', 'y', 'Y']:
        split['date'] = split['date'].dt.to_period('Y').dt.start_time

    mask = split['listed'].isin([1, 2, 3, 4])

    if not show_future:
        mask = mask & ~split['listed'].isin([4])
    if not show_pretrain:
        mask = mask & ~split['listed'].isin([1])

    tmp = groupby_eff(split, ['date', 'variable'], ['value'], mask, sort=False)
    tmp = tmp.value.sum().reset_index()

    fig = px.bar(tmp
                 , x="date"
                 , y="value"
                 , color="variable"
                 , color_discrete_map=model_rep.palette_tools
                 , title=', '.join(str(x) for x in brands) + ": " +
                         model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                                (model_rep.vis_dict['variable'] == "factors_title"), language].iloc[0]
                 , labels=
                 {"value": model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "shared") &
                                                  (model_rep.vis_dict['variable'] == "packs"), language].iloc[0]
                     , "variable": model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                                          (model_rep.vis_dict['variable'] == "factor"), language].iloc[0]}
                 , text_auto='.2s'
                 , height=800
                 )

    for i in fig.data:
        if len(i.name) > 40:
            i.name = i.name[0:40] + '...'

    rangeslider_visible = True
    rangeselector = dict(
        buttons=list([
            dict(count=6,
                 label="6m",
                 step="month",
                 stepmode="backward"),
            dict(count=1,
                 label="1y",
                 step="year",
                 stepmode="backward"),
            dict(count=3,
                 label="3y",
                 step="year",
                 stepmode="backward"),
            dict(step="all")]))
    tickformat = "%b\n%Y"
    dtick = "M3"
    tickmode = 'linear'

    y_min = tmp.loc[tmp['value'] < 0, :].groupby('date')['value'].sum().min() * 1.05
    y_max = tmp.loc[tmp['value'] > 0, :].groupby('date')['value'].sum().max() * 1.05
    if pd.isna(y_min) or y_min > .0:
        y_min = 0

    for i in tmp['date'].dt.year.unique():
        fig.add_trace(go.Scatter(
            x=[str(i) + '-01'] * 2
            , y=[y_min, y_max]
            , line_color='grey'
            , mode='lines'
            , line_width=1.3
            , name=''
            , showlegend=False
            , hoverinfo='skip'
        ))

    if show_future and split.loc[split['listed'].isin([4]), 'date'].shape[0] > 0:

        date_end = split.loc[split['listed'].isin([4]), 'date'].sort_values().iloc[0]
        fig.add_trace(go.Scatter(
            x=[date_end] * 2
            , y=[y_min, y_max]
            , line_color='red'
            , mode='lines'
            , line_dash='dash'
            , opacity=.7
            , line_width=2
            , name=''
            , showlegend=False
            , hoverinfo='skip'
        ))

    fig.update_xaxes(tick0="2000-01-01"
                     , dtick=dtick
                     , tickformat=tickformat
                     , tickmode=tickmode
                     # , ticklabelstep=3
                     , showticklabels=True
                     # , minor=dict(ticks="inside", showgrid=True)
                     , rangeslider_visible=rangeslider_visible
                     , rangeselector=rangeselector
                     , tickfont_size=10
                     )

    fig.update_yaxes(showticklabels=True
                     , matches=None
                     )
    fig.update_layout(
        bargap=0  # gap between bars of adjacent location coordinates
        , bargroupgap=0  # gap between bars of the same location coordinate
        , yaxis_range=[y_min, y_max]
    )

    if contour_line:
        fig.update_traces(marker_line_color='gray',
                          marker_line_width=1, opacity=1)

    return fig


@profile
def decompose_sub_plot(split_m_m: pd.DataFrame | list | dict
                       , brands: list | None = None
                       , model_rep: "ModelRep | ModelNode | MetaModel | list | None" = None
                       , sku_var: list | tuple = ('superbrand', 'market')
                       , *, brands_var: str = 'superbrand'
                       , filters: dict | None = None
                       , period: str | list = 'day'
                       , show_future: bool | str = True
                       , show_pretrain: bool | str = True
                       , contour_line: bool = True) -> go.Figure | list | dict:
    """
        Wrapper
        Plot dynamic decomposition with faceting by `sku_var`

        :param split_m_m: prepared dataset (see `extract_effect()`) or list/dict of prepared datasets
        :param brands: list of umbrella brands to preserve, None to keep all
        :param model_rep: ModelRep, ModelNode, MetaModel, or list of ModelRep objects (deprecated)
        :param sku_var: list or tuple of variables to group by
        :param brands_var: column the simple filter values apply to
        :param filters: complex filter ``{column: [values]}``
        :param period: time period / interval to group by or list of time periods / intervals to group by
        :param show_future: show future periods or not
        :param show_pretrain: show pretrain periods or not
        :param contour_line: add contours or not
        :return:
        """

    mm = _convert_to_metamodel(model_rep)

    results = []
    for i, (name, node) in enumerate(mm.nodes.items()):
        splits_i = split_m_m[name] if isinstance(split_m_m, dict) else (split_m_m[i] if isinstance(split_m_m, list) else split_m_m)
        period_i = period[name] if isinstance(period, dict) else (period[i] if isinstance(period, list) else period)
        results.append(_decompose_sub_plot_worker(split_m_m=splits_i, brands=brands, model_rep=node.model_rep,
                                                   sku_var=sku_var, period=period_i, show_future=show_future,
                                                   show_pretrain=show_pretrain, contour_line=contour_line,
                                                   brands_var=brands_var, filters=filters))

    if isinstance(model_rep, MetaModel):
        return dict(zip(list(mm.nodes.keys()), results))
    if isinstance(model_rep, list):
        return results
    return results[0]


@profile
def _decompose_sub_plot_worker(split_m_m: pd.DataFrame
                               , brands: list | None = None
                               , model_rep: "ModelRep" = None
                               , sku_var: list | tuple = ('superbrand', 'market')
                               , period: str = 'day'
                               , show_future: bool | str = True
                               , show_pretrain: bool | str = True
                               , contour_line: bool = True
                               , brands_var: str = 'superbrand'
                               , filters: dict | None = None) -> go.Figure:
    """
    Worker
    Plot dynamic decomposition with faceting by `sku_var`

    :param split_m_m: prepared dataset (see `extract_effect()`)
    :param brands: list of umbrella brands to preserve, None to keep all
    :param model_rep: ModelRep object (export settings)
    :param sku_var: list or tuple of variables to group by
    :param period: time period / interval to group by
    :param show_future: show future periods or not
    :param contour_line: add contours or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    language = model_rep.language

    if show_future == 'True':
        show_future = True
    elif show_future == 'False':
        show_future = False

    split = split_m_m.copy()

    split = _filter_df(split, brands=brands, brands_var=brands_var, filters=filters)

    if not brands:
        if brands_var in split.columns:
            brands = split[brands_var].unique()
        elif 'superbrand' in split.columns:
            brands = split['superbrand'].unique()
        else:
            brands = []

    if period in ['day', 'd', 'D']:
        split['date'] = split['date'].dt.floor(freq='D')
    elif period in ['week', 'w', 'W']:
        split['date'] = split['date'].dt.to_period('W').dt.start_time
    elif period in ['month', 'm', 'M']:
        split['date'] = split['date'].dt.to_period('M').dt.start_time
    elif period in ['quarter', 'q', 'Q']:
        split['date'] = split['date'].dt.to_period('Q').dt.start_time
    elif period in ['year', 'y', 'Y']:
        split['date'] = split['date'].dt.to_period('Y').dt.start_time

    mask = split['listed'].isin([1, 2, 3, 4])

    if not show_future:
        mask = mask & ~split['listed'].isin([4])
    if not show_pretrain:
        mask = mask & ~split['listed'].isin([1])

    # cleanse sku_var from non-existing variables
    sku_var = [x for x in sku_var if x in split.columns]

    tmp = groupby_eff(split, list(set(['date', 'superbrand'] + sku_var + ['variable'])), ['value'], mask, sort=False)
    tmp = tmp.value.sum().reset_index()
    tmp['sku'] = ''

    for i in sku_var:
        tmp['sku'] = tmp['sku'] + ' ' + tmp[i].astype('str').str.title()

    fig = px.bar(tmp
                 , x="date"
                 , y="value"
                 , color="variable"
                 , color_discrete_map=model_rep.palette_tools
                 , title=', '.join(str(x).title() for x in brands) + ": " +
                         model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                                (model_rep.vis_dict['variable'] == "factors_title"), language].iloc[0]
                 , labels=
                 {"value": model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "shared") &
                                                  (model_rep.vis_dict['variable'] == "packs"), language].iloc[0]
                     , "variable": model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                                          (model_rep.vis_dict['variable'] == "factor"), language].iloc[0]}
                 , facet_col=tmp['sku']
                 , facet_col_wrap=2
                 , facet_col_spacing=0.06
                 , facet_row_spacing=1 / (len(tmp['sku'].unique()) * 2)
                 # , text_auto='.2s'
                 , height=np.ceil(len(tmp['sku'].unique()) / 2) * 400
                 )

    for i in fig.data:
        if len(i.name) > 40:
            i.name = i.name[0:40] + '...'

    rangeslider_visible = False
    rangeselector = None
    tickformat = None
    dtick = None
    tickmode = None

    for i in tmp['date'].dt.year.unique():
        fig.add_vline(x=str(i) + '-01', line_width=1.3, line_color="black")

    if show_future and split.loc[split['listed'].isin([4]), 'date'].shape[0] > 0:
        date_end = split.loc[split['listed'].isin([4]), 'date'].sort_values().iloc[0]
        fig.add_vline(x=date_end, line_width=2, line_color="red", line_dash="dash", opacity=0.7, )

    fig.update_xaxes(tick0="2000-01-01"
                     , dtick=dtick
                     , tickformat=tickformat
                     , tickmode=tickmode
                     , showticklabels=True
                     # , minor=dict(ticks="inside", showgrid=True)
                     , rangeslider_visible=rangeslider_visible
                     , rangeselector=rangeselector
                     , tickfont_size=10
                     )

    fig.update_yaxes(showticklabels=True
                     , matches=None
                     )

    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("sku=", "")))

    fig.update_layout(
        bargap=0  # gap between bars of adjacent location coordinates.
        , bargroupgap=0  # gap between bars of the same location coordinate.
    )

    if contour_line:
        fig.update_traces(marker_line_color='gray',
                          marker_line_width=1, opacity=0.9)

    return fig
