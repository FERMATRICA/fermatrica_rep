"""
Calculate metrics and tests.
"""


# basic packages
import copy
import pandas as pd

import statsmodels.api as sm
import statsmodels.regression.linear_model
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import shapiro, spearmanr
import statsmodels.stats as smstats
from statsmodels.stats.weightstats import ztest

# framework
from fermatrica_utils import groupby_eff, rm_1_item_groups
import fermatrica.evaluation.metrics as mtr
from fermatrica.model.model import Model


def predictors_table(model: "Model") -> pd.DataFrame:
    """
    Linear regression coefficients estimations, standard errors and p-values.

    :param model: Model object
    :return: table with predictors data
    """

    model_cur = model.obj.models['main']

    # gather predictors together

    tbl = model_cur.params.to_frame()
    tbl.rename(columns={tbl.columns[-1]: 'Estimate'}, inplace=True)
    tbl['Estimate'] = tbl['Estimate'].map('{:,.3E}'.format)

    if hasattr(model_cur, 'bse'):
        tbl = pd.concat([tbl, model_cur.bse], axis=1)
        tbl.rename(columns={tbl.columns[-1]: 'Std.Error'}, inplace=True)
        tbl['Std.Error'] = tbl['Std.Error'].map('{:,.3E}'.format)

    if hasattr(model_cur, 'tvalues'):
        tbl = pd.concat([tbl, model_cur.tvalues], axis=1)
        tbl.rename(columns={tbl.columns[-1]: 't-value'}, inplace=True)
        tbl['t-value'] = tbl['t-value'].map('{:,.3E}'.format)

    if hasattr(model_cur, 'pvalues'):
        tbl = pd.concat([tbl, model_cur.pvalues], axis=1)
        tbl.rename(columns={tbl.columns[-1]: 'p-value'}, inplace=True)
        tbl['p-value'] = tbl['p-value'].map('{:,.8f}'.format)

    tbl = tbl.reset_index().rename(columns={'index': 'Predictor'})

    return tbl


def basic_metrics_table(dt_pred: pd.DataFrame
                        , mape_threshold: float | None = None) -> pd.DataFrame:
    """
    Basic (model-blind) metric: RMSE, MAPE, R^2.

    :param dt_pred: prediction data
    :param mape_threshold: threshold for mape_adj
    :return: table
    """

    dct = {}

    # R^2, RMSE

    dct['R^2 train'] = '{:,.4f}'.format(mtr.r_squared(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted']))
    dct['RMSE train'] = '{:,.4f}'.format(mtr.rmse(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted']))

    if dt_pred[dt_pred['listed'] == 3].shape[0] > 0:
        dct['RMSE test'] = '{:,.4f}'.format(mtr.rmse(dt_pred[dt_pred['listed'] == 3]['observed'], dt_pred[dt_pred['listed'] == 3]['predicted']))

    # MAPE

    dct['MAPE total'] = '{:,.2f}'.format(mtr.mapef(dt_pred[dt_pred['listed'].isin([2, 3])]['observed'], dt_pred[dt_pred['listed'].isin([2, 3])]['predicted']))
    dct['MAPE train'] = '{:,.2f}'.format(mtr.mapef(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted']))

    if mape_threshold is not None:
        dct['MAPE adj train'] = '{:,.2f}'.format(
            mtr.mape_adj(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted'], mape_threshold))

    if dt_pred[dt_pred['listed'] == 3].shape[0] > 0:
        dct['MAPE test'] = '{: .2f}'.format(mtr.mapef(dt_pred[dt_pred['listed'] == 3]['observed'], dt_pred[dt_pred['listed'] == 3]['predicted']))

        if mape_threshold is not None:
            dct['MAPE adj test'] = '{:,.2f}'.format(
                mtr.mape_adj(dt_pred[dt_pred['listed'] == 3]['observed'], dt_pred[dt_pred['listed'] == 3]['predicted'], mape_threshold))
        
            mape_train = mtr.mapef(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted'])
            mape_adj_train = mtr.mape_adj(dt_pred[dt_pred['listed'] == 2]['observed'], dt_pred[dt_pred['listed'] == 2]['predicted'], mape_threshold)
            mape_test = mtr.mapef(dt_pred[dt_pred['listed'] == 3]['observed'], dt_pred[dt_pred['listed'] == 3]['predicted'])
            mape_adj_test = mtr.mape_adj(dt_pred[dt_pred['listed'] == 3]['observed'], dt_pred[dt_pred['listed'] == 3]['predicted'], mape_threshold)

            dct['MAPE test/train - 1'] = '{:,.2f}'.format(mape_test / mape_train - 1)
            dct['MAPE adj test/train - 1'] = '{:,.2f}'.format(mape_adj_test / mape_adj_train - 1)

    # gather

    tbl = pd.DataFrame.from_dict(dct, orient='index', columns=['Statictics'])
    tbl = tbl.reset_index().rename({'index': 'Metrics'}, axis=1)

    return tbl


def tests_table_ols(model: "Model | statsmodels.regression.linear_model.OLS | statsmodels.regression.linear_model.OLSResults"
                    , dt_pred: pd.DataFrame):
    """
    Tests specific for time series OLS model (not panel / LME).

    :param model: Model object
    :param dt_pred: prediction data
    :return:
    """

    if isinstance(model, Model):
        mdl = model.obj.models['main'].model
    else:
        mdl = model

    dct = {}

    # Ljung-Box autocorrelation test
    lb_autocorr_test = sm.stats.acorr_ljungbox(mdl.fit().resid, lags=[1], return_df=True)
    dct['Ljung-Box autocorrelation test'] = ['{:,.4f}'.format(lb_autocorr_test['lb_stat'].to_numpy()[0]), '{:,.4f}'.format(lb_autocorr_test['lb_pvalue'].to_numpy()[0])]

    # Augmented Dickey-Fuller test for stationary
    adf_test = adfuller(mdl.fit().resid, autolag="t-stat", regression="ct")
    dct['Augmented Dickey-Fuller test for stationary'] = ['{:,.4f}'.format(adf_test[0]), '{:,.4f}'.format(adf_test[1])]

    # KPSS test for stationary
    kpss_test = kpss(mdl.fit().resid, regression="ct")
    dct['KPSS test for stationary'] = ['{:,.4f}'.format(kpss_test[0]), '{:,.4f}'.format(kpss_test[1])]

    # Shapiro-Wilk normality test
    sw_norm_test = shapiro(mdl.fit().resid)
    dct['Shapiro-Wilk normality test'] = ['{:,.4f}'.format(sw_norm_test[0]), '{:,.4f}'.format(sw_norm_test[1])]

    # Ramsey’s RESET test for neglected nonlinearity / omitted variables
    if pd.__version__ <= '1.5.3':
        reset_test = smstats.diagnostic.linear_reset(mdl.fit(), power=2, test_type='fitted')
        dct['Ramsey’s RESET test, fitted y, powers = 2'] = ['{:,.4f}'.format(reset_test.statistic), '{:,.4f}'.format(reset_test.pvalue)]
    else:
        res_np = sm.OLS(mdl.endog, mdl.exog).fit()
        reset_test = smstats.diagnostic.linear_reset(res_np, power=2, test_type='fitted')
        dct['Ramsey’s RESET test, fitted y, powers = 2'] = ['{:,.4f}'.format(reset_test.statistic),
                                                            '{:,.4f}'.format(reset_test.pvalue)]

    # z-test for a systematic bias in prediction on test period
    z_test = ztest(dt_pred[dt_pred['listed'] == 3]['observed'] - dt_pred[dt_pred['listed'] == 3]['predicted'], x2=None, value=0, alternative='two-sided')
    dct['z-test for a systematic bias in forecast'] = ['{:,.4f}'.format(z_test[0]), '{:,.4f}'.format(z_test[1])]

    # Spearman test
    if dt_pred[dt_pred['listed'] == 3].shape[0] < 40:
        spearman = spearmanr(dt_pred[dt_pred['listed'].isin([2, 3])]['observed']
                            , dt_pred[dt_pred['listed'].isin([2, 3])]['predicted'],
                            axis=0, nan_policy='propagate', alternative='two-sided')
    else:
        spearman = spearmanr(dt_pred[dt_pred['listed'] == 3]['observed']
                            , dt_pred[dt_pred['listed'] == 3]['predicted'],
                            axis=0, nan_policy='propagate', alternative='two-sided')
    dct['Spearman correlation'] = ['{:,.4f}'.format(spearman.statistic), '{:,.4f}'.format(spearman.pvalue)]

    # gather
    tbl = pd.DataFrame.from_dict(dct, orient='index', columns=['Statictics', 'P-Value'])
    tbl = tbl.reset_index().rename({'index': 'Metrics'}, axis=1)

    return tbl


def metrics_table(model: "Model"
                  , dt_pred: pd.DataFrame
                  , mape_threshold: float | None = None) -> pd.DataFrame:
    """
    All model-level metrics and stats available: basic metrics and model-specific metrics.

    :param model: Model object
    :param dt_pred: prediction data
    :param mape_threshold: threshold for mape_adj
    :return:
    """

    # basic statistics

    tbl = basic_metrics_table(dt_pred, mape_threshold)

    # linear models

    if model.conf.model_type == 'OLS':
        tbl = pd.concat([tbl, tests_table_ols(model, dt_pred)], axis=0)

    # panel models

    elif model.conf.model_type == 'LME':
        pass

    elif model.conf.model_type == 'LMEA':
        pass

    elif model.conf.model_type == 'FE':
        pass

    return tbl


def metrics_group_table(dt_pred: pd.DataFrame
                        , group_vars: list | tuple = ('market', 'superbrand')
                        ) -> pd.DataFrame:
    """
    Calculate metrics by group, e.g. by superbrand (umbrella brand), market, region etc.

    :param dt_pred: prediction data
    :param group_vars: calculate metrics by these variables (columns)
    :return: table
    """

    if type(group_vars) == tuple:
        group_vars = list(group_vars)

    group_vars_ext = copy.deepcopy(group_vars)
    group_vars_ext.append('date')

    # get metrics

    tbl = groupby_eff(dt_pred[dt_pred['listed'].isin([2, 3])], group_vars_ext, ['observed', 'predicted']).sum().reset_index()
    tbl = rm_1_item_groups(tbl, group_vars)

    def worker(x: pd.DataFrame):
        rtrn = pd.DataFrame({'r_sq': [mtr.r_squared(x['observed'], x['predicted'])]
                             , 'rmse': [mtr.rmse(x['observed'], x['predicted'])]
                             , 'smape': [mtr.smape(x['observed'], x['predicted'])]})
        return rtrn

    tbl = groupby_eff(tbl, group_vars, ['observed', 'predicted']).apply(worker)

    # pseudographics

    tbl['pc'] = tbl['r_sq'] * 10
    tbl.loc[tbl['pc'] < 0, 'pc'] = 0

    tbl.loc[:, 'pc'] = tbl['pc'].apply(lambda x: ''.join(['='*int(x)]))

    mask = tbl['pc'].str.len() < 10
    tbl.loc[mask, 'pc'] = tbl.loc[mask, 'pc'].apply(lambda x: x + ''.join(['.' * int(10 - len(x))]))

    #

    return tbl


def stats_compare(model_init: "Model"
                  , model_update: "Model"
                  , dt_pred_init: pd.DataFrame
                  , dt_pred_upd: pd.DataFrame
                  , mape_threshold: float | None = None
                  , if_return_ds: bool = False):
    """
    Compare two versions of the model (older and newer).

    :param model_init: older Model
    :param model_update: new Model
    :param dt_pred_init: older dataset
    :param dt_pred_upd: new dataset
    :param mape_threshold: threshold for mape_adj (not calculated if None is provided)
    :param if_return_ds: return or print
    :return: table with statistics and tests or void
    """

    stats_tbl = metrics_table(model_init, dt_pred_init, mape_threshold).merge(
        metrics_table(model_update, dt_pred_upd, mape_threshold), on='Metrics', how='left')

    stats_tbl.rename(columns={
        "Statictics_x": "Statictics"
        , "P-Value_x": "P-Value"
        , "Statictics_y": "Statictics_upd"
        , "P-Value_y": "P-Value_upd"
    }, inplace=True)

    if if_return_ds:
        return stats_tbl

    else:
        print("\n==== Compare Models =====\n")
        print(stats_tbl)

        pass


def vif_table(model_cur: statsmodels.regression.linear_model.OLS | statsmodels.regression.linear_model.OLSResults
              ) -> pd.DataFrame:
    """
    Calculate VIFs for OLS model.

    :param model_cur: statsmodels OLS object
    :return: VIF table
    """

    ds = mtr.vif(model_cur)
    ds = pd.DataFrame({'Variable': ds.index, 'VIF': ds.values}, index=range(0, len(ds)))

    ds['Variable'] = ds['Variable'].map('{:.45}'.format)
    ds['VIF'] = ds['VIF'].map('{:.3f}'.format)

    return ds
