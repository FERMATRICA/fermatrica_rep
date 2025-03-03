"""
Decompose model into effects (impacts).

This module describes dynamic decomposition, for waterfall decomposition see `fermatrica_rep.waterfall`
"""


import re

from line_profiler_pycharm import profile

import patsy
import statsmodels.regression.linear_model

import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

from fermatrica_utils import groupby_eff

from fermatrica import params_to_dict, fermatrica_error, Model, ModelConf
from fermatrica.model.lhs_fun import *  # to run LHS
from fermatrica_rep.meta_model.model_rep import ModelRep

pio.templates.default = 'ggplot2'


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
                     , if_detail) -> pd.DataFrame:
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

    :param model_conf: ModelConf
    :param model_cur: object of statsmodels LME model. Separately from model_conf because different LME models could be used
    :param ds: dataset
    :param id_vars: index (dimension) variables. Mostly ['date'], not very important
    :param if_detail: expand tokens
    :return:

    """

    fixed = model_cur.fe_params
    fixed.rename(index={'Intercept': 'intercept'}, inplace=True)

    random = model_cur.random_effects

    ds = ds.copy()
    ds['intercept'] = 1

    split_m = select_eff(ds, id_vars)

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
                            multiplier = re.sub(r"\(" + tmp_regex + r"\) {0,1}\* {0,1}", "", str_w_multiplier)

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
                    split_m[cl] = ds.eval(cl) * j['value']
                else:
                    split_m[cl] = ds[cl] * j['value']
        else:

            if re.search(r'[*( \-+:]', cols):
                cols_all = ds.columns.to_list()

                cols_splt = re.split(r' *[*() \-+:] *', cols)
                cols_splt = [x for x in cols_splt if x in cols_all]

                split_m[j['index']] = select_eff(ds, cols_splt).eval(cols) * j['value']

            else:
                split_m[j['index']] = ds[cols] * j['value']

    ft = model_conf.fixed_effect_var

    j = pd.DataFrame(random).T.reset_index()
    j.columns = [ft, ft + '_' + 'intercept']
    split_m[ft] = ds[ft]

    split_m = split_m.merge(j, how='left', on=ft)
    split_m[ft + '_' + 'intercept'] = split_m[ft + '_' + 'intercept']  # * split_m['Intercept']

    del split_m[ft]

    split_m.drop(id_vars, axis=1, inplace=True)

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

    :param model_cur: object of statsmodels OLS model. Separately from model_conf because different LME models could be used
    :param ds: dataset
    :param id_vars: index (dimension) variables. Mostly ['date'], not very important
    :param if_detail: expand tokens
    :return:

    """

    params = model_cur.params
    params.rename(index={'Intercept': 'intercept'}, inplace=True)

    ds = ds.copy()
    ds['intercept'] = 1

    split_m = select_eff(ds, id_vars)

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
                            multiplier = re.sub(r"\(" + tmp_regex + r"\) {0,1}\* {0,1}", "", str_w_multiplier)

                            uncovered_str = [i + " * " + multiplier for i in tmp.split("+")]
                            uncovered_str = ' +'.join(uncovered_str)
                            cols = cols.replace(str_w_multiplier, uncovered_str)
                        else:
                            cols = cols.replace("(", "").replace(")", "")

                        break
            cols = cols.split("+")
            for i in cols:
                split_m[i] = ds.eval(i) * j['value']
        else:
            if "np.log1p" in cols:
                cols = cols.replace("np.log1p", "")
                split_m[j['index']] = np.log1p(ds.eval(cols)) * j['value']
            else:
                split_m[j['index']] = ds.eval(cols) * j['value']

    split_m.drop(id_vars, axis=1, inplace=True)

    return split_m


@profile
def rhs_effects(model_conf: "ModelConf"
                , model_cur
                , ds: pd.DataFrame
                , id_vars
                , if_detail) -> pd.DataFrame:
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
        rtrn = _rhs_effects_lme(model_conf, model_cur, ds, id_vars, if_detail)
    elif model_conf.model_type == 'LMEA':
        rtrn = _rhs_effects_lme(model_conf, model_cur, ds, id_vars, if_detail)
    else:
        rtrn = None
        fermatrica_error('Unknown model type in model decomposition: ' + model_conf.model_type)

    return rtrn


"""
Extract all effects
"""


@profile
def extract_effect(model: "Model | list"
                   , ds: pd.DataFrame | list
                   , model_rep: "ModelRep | list"
                   , if_detail: bool = False) -> pd.DataFrame | list:
    """
        Wrapper
        Decompose complex model / extract effects (impacts) of all types.

        :param model: Model object or list of Model objects
        :param ds: dataset or list of datasets
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects
        :param if_detail: expand tokens
        :return: table with column per every extracted effect / impact or list of tables with columns per every extracted effect
        """

    if isinstance(model, list):
        split_m_m = [None] * len(model)
        for i in range(len(model)):
            split_m_m[i] = _extract_effect_worker(model[i], ds[i], model_rep[i], if_detail)
    else:
        split_m_m = _extract_effect_worker(model, ds, model_rep, if_detail)

    return split_m_m


@profile
def _extract_effect_worker(model: "Model"
                           , ds: pd.DataFrame
                           , model_rep: "ModelRep"
                           , if_detail: bool = False) -> pd.DataFrame:
    """
    Worker
    Decompose complex model / extract effects (impacts) of all types.

    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param if_detail: expand tokens
    :return: table with column per every extracted effect / impact
    """

    language = model_rep.language
    model_conf = model.conf

    split_m = rhs_effects(model.conf, model.obj.models['main'], ds, ['date'], if_detail)

    lhs = model_conf.model_lhs

    if lhs is not None:
        to_original = True
        # free_before type LHS transformations
        mask = '^free_before$'

        if ((lhs['if_active'] == 1) & (lhs['type'].str.match(mask))).sum() > 0:

            y = split_m.sum(axis=1)

            for i in range(sum((lhs['if_active'] == 1) & (lhs['type'].str.match(mask)))):
                y_new = eval(lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask)), 'token'].iloc[i])

                split_m[lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask)), 'name'].iloc[i]] = y_new - y
                y = y_new

        # multiplicators
        lhs_mult = lhs

        if lhs['type'].str.match('^multiplicative$').sum() > 0:
            lhs_mult = lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match('^multiplicative$'))]

            rowsums = split_m.sum(axis=1)

            coef = model_conf.params[((model_conf.params['variable'] == '') | (model_conf.params['variable'].isna())) &
                                    (model_conf.params['fun'] == 'coef')][['arg', 'value']]

            coef = params_to_dict(coef)

            # y = [rowsums / eval(compile(i, '<string>', 'eval'), globals(), locals()) - rowsums for i in lhs_mult.token]
            y = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = rowsums / eval(compile(i, '<string>', 'eval'), globals(), locals()).array - rowsums
                y = pd.concat([y, y_tmp], ignore_index=True, axis=1)

            # y = y.T
            y_pos = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array - 1
                y_pos = pd.concat([y_pos, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            # y_pos = y_pos.T
            y_pos = y_pos.fillna(0)
            y_pos[y_pos <= 0] = 0
            y_pos[y_pos > 0] = 1

            y_dnm = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array - 1
                y_dnm = pd.concat([y_dnm, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            # y_dnm = y_dnm.T
            y_dnm *= y_pos
            y_dnm = y_dnm.sum(axis=1)

            y_num = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array
                y_num = pd.concat([y_num, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            y_num *= y_pos

            y_num.replace(0, np.nan, inplace=True)
            y_num = y_num.prod(skipna=True, axis=1)
            y_num -= 1

            y_pos = y * y_pos.mul((y_num / y_dnm), axis=0)
            y_pos.fillna(0, inplace=True)

            # neg
            y_neg = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array - 1
                y_neg = pd.concat([y_neg, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            # y_pos = y_pos.T
            y_neg = y_neg.fillna(0)
            y_neg[y_neg >= 0] = 0
            y_neg[y_neg < 0] = 1

            y_dnm = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array - 1
                y_dnm = pd.concat([y_dnm, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            # y_dnm = y_dnm.T
            y_dnm *= y_neg
            y_dnm = y_dnm.sum(axis=1)

            y_num = pd.DataFrame()
            for i in lhs_mult.token:
                y_tmp = 1 / eval(compile(i, '<string>', 'eval'), globals(), locals()).array
                y_num = pd.concat([y_num, pd.DataFrame(y_tmp)], ignore_index=True, axis=1)

            y_num *= y_neg

            y_num = y_num.replace(0, np.nan)
            y_num = y_num.prod(skipna=True, axis=1)
            y_num -= 1

            y_neg = y * y_neg.mul((y_num / y_dnm), axis=0)
            y_neg.fillna(0, inplace=True)

            y = y_neg.mul((rowsums + y_pos.sum(axis=1)), axis=0).div(rowsums, axis=0) + y_pos

            y.columns = lhs_mult.name

            split_m = pd.concat([split_m, y], axis=1)

        # free_after type LHS transformations
        mask = '^free_after'

        if ((lhs['if_active'] == 1) & (lhs['type'].str.match(mask))).sum() > 0:

            y = split_m.sum(axis=1).array

            for i in range(sum((lhs['if_active'] == 1) & (lhs['type'].str.match(mask)))):
                y_new = eval(lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask)), 'token'].iloc[i]).array

                split_m[lhs.loc[(lhs['if_active'] == 1) & (lhs['type'].str.match(mask)), 'name'].iloc[i]] = y_new - y
                y = y_new

    cln_id = ["date", "pack_key", "bs_key", "listed", model_conf.price_var] + model_conf.bs_key
    cln_id = [x for x in cln_id if x in ds.columns.tolist()]

    split_m = pd.concat([select_eff(ds, cln_id).reset_index(names=['tmp1____']), split_m.reset_index(names=['tmp2____'])], axis=1)
    del split_m['tmp1____']
    del split_m['tmp2____']

    cln_sort = [x for x in cln_id if x in ['superbrand', 'bs_key', 'master', 'date']]

    split_m_m = split_m.melt(id_vars=cln_id)
    split_m_m = split_m_m[~split_m_m['value'].isna()]

    split_m_m = split_m_m.groupby(cln_id + ['variable'])['value'].\
        sum().\
        reset_index()

    split_m_m['value_rub'] = split_m_m['value'] * split_m_m[model_conf.price_var]

    pattern = re.compile(r'^[0-9]+_')

    if not model_rep.vis_dict.empty:

        rename_dict = {}

        for name in split_m_m.variable.unique():

            if 'ntercept' in name:
                rename_dict[name] = '01_' +\
                    model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == 'base') &
                                 (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type'])), language].iloc[0]

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

        split_m_m['variable'].replace(rename_dict, inplace=True)

        mask = split_m_m['listed'].isin([2, 3, 4])

        split_m_m = groupby_eff(split_m_m, cln_id + ['variable'], ['value', 'value_rub'], mask) \
            .agg({'value': 'sum', 'value_rub': 'sum'}) \
            .reset_index()

        split_m_m['variable'] = split_m_m['variable'].astype('category')

    split_m_m = split_m_m.sort_values(['variable'] + cln_sort, axis=0, ascending=True)
    split_m_m['variable'] = split_m_m['variable'].str.replace(pattern, '', regex=True)

    # update palettes

    lst = split_m_m.loc[split_m_m['value'] > 0, 'variable'].unique().tolist()
    lst.sort()

    model_rep.fill_colours_tools(lst)

    return split_m_m


@profile
def decompose_main_plot(split_m_m: pd.DataFrame | list
                        , brands: list | None
                        , model_rep: "ModelRep | list"
                        , period: str | list = 'day'
                        , show_future: bool | str = True
                        , contour_line: bool = True) -> go.Figure:
    """
        Wrapper
        Plot main dynamic decomposition (without faceting)

        :param split_m_m: prepared dataset (see `extract_effect()`) or list of prepared datasets
        :param brands: list of umbrella brands to preserve
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects
        :param period: time period / interval to group by or list of time periods / intervals to group by
        :param show_future: show future periods or not
        :param contour_line: add contours or not
        :return:
        """

    if isinstance(split_m_m, list):
        fig = [None] * len(split_m_m)
        if isinstance(period, list):
            for i in range(len(split_m_m)):
                fig[i] = _decompose_main_plot_worker(split_m_m[i], brands, model_rep[i], period[i], show_future, contour_line)
        else:
            for i in range(len(split_m_m)):
                fig[i] = _decompose_main_plot_worker(split_m_m[i], brands, model_rep[i], period, show_future, contour_line)
    else:
        fig = _decompose_main_plot_worker(split_m_m, brands, model_rep, period, show_future, contour_line)

    return fig


@profile
def _decompose_main_plot_worker(split_m_m: pd.DataFrame
                                , brands: list | None
                                , model_rep: "ModelRep"
                                , period: str = 'day'
                                , show_future: bool | str = True
                                , contour_line: bool = True) -> go.Figure:
    """
    Worker
    Plot main dynamic decomposition (without faceting)

    :param split_m_m: prepared dataset (see `extract_effect()`)
    :param brands: list of umbrella brands to preserve
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by
    :param show_future: show future periods or not
    :param contour_line: add contours or not
    :return:
    """

    language = model_rep.language

    if not brands:
        brands = split_m_m['superbrand'].unique()

    if show_future == 'True':
        show_future = True
    elif show_future == 'False':
        show_future = False

    split = split_m_m.copy()

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

    if show_future:
        mask = split['superbrand'].isin(brands) & split['listed'].isin([2, 3, 4])
    else:
        mask = split['superbrand'].isin(brands) & split['listed'].isin([2, 3])

    tmp = groupby_eff(split, ['date', 'variable'], ['value'], mask, sort=False)
    tmp = tmp.value.sum().reset_index()

    fig = px.bar(tmp
                 , x="date"
                 , y="value"
                 , color="variable"
                 , color_discrete_map=model_rep.palette_tools
                 , title=', '.join(brands) + ": " +
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
def decompose_sub_plot(split_m_m: pd.DataFrame
                       , brands: list | None
                       , model_rep: "ModelRep"
                       , sku_var: list | tuple = ('superbrand', 'market')
                       , period: str | list = 'day'
                       , show_future: bool | str = True
                       , contour_line: bool = True) -> go.Figure:
    """
        Wrapper
        Plot dynamic decomposition with faceting by `sku_var`

        :param split_m_m: prepared dataset (see `extract_effect()`) or list of prepared datasets
        :param brands: list of umbrella brands to preserve, None to keep all
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects
        :param sku_var: list or tuple of variables to group by
        :param period: time period / interval to group by or list of time periods / intervals to group by
        :param show_future: show future periods or not
        :param contour_line: add contours or not
        :return:
        """
    if isinstance(split_m_m, list):
        fig = [None] * len(split_m_m)
        if isinstance(period, list):
            for i in range(len(split_m_m)):
                fig[i] = _decompose_sub_plot_worker(split_m_m[i], brands, model_rep[i], sku_var, period[i], show_future,
                                                    contour_line)
        else:
            for i in range(len(split_m_m)):
                fig[i] = _decompose_sub_plot_worker(split_m_m[i], brands, model_rep[i], sku_var, period, show_future,
                                                    contour_line)
    else:
        fig = _decompose_sub_plot_worker(split_m_m, brands, model_rep, sku_var, period, show_future, contour_line)

    return fig


@profile
def _decompose_sub_plot_worker(split_m_m: pd.DataFrame
                               , brands: list | None
                               , model_rep: "ModelRep"
                               , sku_var: list | tuple = ('superbrand', 'market')
                               , period: str = 'day'
                               , show_future: bool | str = True
                               , contour_line: bool = True) -> go.Figure:
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
    :return:
    """

    language = model_rep.language

    if not brands:
        brands = split_m_m['superbrand'].unique()

    if show_future == 'True':
        show_future = True
    elif show_future == 'False':
        show_future = False

    split = split_m_m.copy()

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

    if show_future:
        mask = split['superbrand'].isin(brands) & split['listed'].isin([2, 3, 4])
    else:
        mask = split['superbrand'].isin(brands) & split['listed'].isin([2, 3])

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
                 , title=', '.join([x.title() for x in brands]) + ": " +
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
