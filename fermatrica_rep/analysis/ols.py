"""
OLS low-level statistics analysis and helpers
"""


import numpy as np
import pandas as pd

import statsmodels.api as sm


def get_model_ds(model: sm.regression.linear_model.RegressionResultsWrapper
                 , ds: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose linear combinations from model regressors.

    :param model: statsmodels OLS model object
    :param ds: dataset
    :return:
    """

    ds_test = pd.DataFrame()

    for token in model.params.index:
        if token == 'Intercept':
            ds_test[token] = model.params[token]
        elif token in ds.columns:
            ds_test[token] = ds[token]
        elif '*' in token:
            strip = token.strip("I()").split('*', 1)[0].replace(")", "").replace(" ", "").split('+')
            coeff = token.strip("I()").split('*', 1)[1].replace(" ", "")
            ds_test[token] = ds[strip].sum(axis=1) * ds[coeff]
        else:
            strip = token.strip("I()").replace(" ", "").split('+')
            ds_test[token] = ds[strip].sum(axis=1)

    ds_test['Intercept'] = 1
    ds_test['date'] = ds['date']
    ds_test['listed'] = ds['listed']
    ds_test = ds_test[ds_test['listed'] != 1]

    return ds_test


def trans(ds: pd.DataFrame
          , params: pd.DataFrame) -> pd.DataFrame:
    """


    :param ds:
    :param params:
    :return:
    """

    var_fun_uniques = params[['variable', 'fun']].drop_duplicates()

    for ind, var_fun in var_fun_uniques.iterrows():
        params_subset = params[
            (params['variable'] == var_fun['variable']) & (params['fun'] == var_fun['fun'])]
        var = var_fun['variable']
        fun_name = var_fun['fun']
        var_new = var + '_' + fun_name
        ds[var_new] = eval(fun_name)(ds, var, params_subset, index_vars=params_subset['index_vars'].iloc[0])

    return ds


def rhs_comparison(model_init: sm.regression.linear_model.RegressionResultsWrapper
                   , model_upd: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """
    Compare signs and regression coefficients between older (initial) and new (updated) models

    :param model_init: older model (statsmodels OLS model object)
    :param model_upd: new model (statsmodels OLS model object)
    :return: table with comparison between models
    """

    ds_upd = pd.read_html(model_upd.summary().tables[1].as_html(), header=0, index_col=0)[0]
    ds_upd.reset_index(inplace=True)
    ds_upd.rename(columns={"coef": "coeff_upd"}, inplace=True)

    ds_init = pd.read_html(model_init.summary().tables[1].as_html(), header=0, index_col=0)[0]
    ds_init.reset_index(inplace=True)
    ds_init.rename(columns={"coef": "coeff_init"}, inplace=True)

    comparison = ds_init[['index', 'coeff_init']].merge(ds_upd[['index', 'coeff_upd']], on='index', how='left')
    comparison['coeff_diff_prc'] = round(
        (comparison['coeff_upd'] - comparison['coeff_init']) / comparison['coeff_init'] * 100, 3)
    comparison['sign_change'] = np.where(comparison['coeff_upd'] * comparison['coeff_init'] > 0, 'OK',
                                         'Sign_changed!')

    return comparison


def impacts_comparison(model_init: sm.regression.linear_model.RegressionResultsWrapper
                       , model_upd: sm.regression.linear_model.RegressionResultsWrapper
                       , ds_init: pd.DataFrame
                       , ds_upd: pd.DataFrame) -> pd.DataFrame:
    """
    Compare impacts between older (initial) and new (updated) models

    :param model_init: older model (statsmodels OLS model object)
    :param model_upd: new model (statsmodels OLS model object)
    :param ds_init: older dataset
    :param ds_upd: new dataset
    :return:
    """

    ds_init = ds_init[ds_init['listed'] == 2]
    ds_upd = ds_upd[ds_upd['listed'] == 2]
    dec_init = ds_init[ds_init.columns[~ds_init.columns.isin(['date', 'listed'])]].dot(np.diag(model_init.params.values))
    dec_upd = ds_upd[ds_upd.columns[~ds_upd.columns.isin(['date', 'listed'])]].dot(np.diag(model_upd.params.values))

    comparison_imp = pd.DataFrame({'imp_init': list(dec_init.sum()/dec_init.sum(axis=1).sum()*100)
                                    , 'imp_upd': list(dec_upd.sum()/dec_upd.sum(axis=1).sum()*100)}
                                    , columns=['imp_init', 'imp_upd'], index=model_init.params.index)
    comparison_imp['diff_prc'] = round(
        (comparison_imp['imp_upd'] - comparison_imp['imp_init']) / comparison_imp['imp_init'] * 100, 3)

    return comparison_imp




