"""
Export model as extended ModelConf (ModelConfExt) XLSX workbook.

Effectively adds new sheets to standard ModelConf with model output data:
fit-predict, decomposition, waterfall etc.
"""


import copy
import io
import numpy as np
import pandas as pd
import re

from fermatrica_utils import select_eff
from fermatrica.model.model_conf import ModelConf


class ModelConfExp(ModelConf):
    """
    Export model from dashboard: basic ModelConf + calculated metrics, curves, decomposition etc.
    """

    state: pd.DataFrame | None
    current_options: pd.DataFrame | None
    decomposition: pd.DataFrame | None
    curves: pd.DataFrame | None
    fit: pd.DataFrame | None

    incremental_volume: pd.DataFrame | None
    incremental_value: pd.DataFrame | None
    profit: pd.DataFrame | None
    roi: pd.DataFrame | None

    incremental_volume_short: pd.DataFrame | None
    incremental_value_short: pd.DataFrame | None
    profit_short: pd.DataFrame | None
    roi_short: pd.DataFrame | None

    incremental_volume_long: pd.DataFrame | None
    incremental_value_long: pd.DataFrame | None
    profit_long: pd.DataFrame | None
    roi_long: pd.DataFrame | None

    statistics: pd.DataFrame | None
    vifs: pd.DataFrame | None
    predictors: pd.DataFrame | None

    def __init__(self
                 , model_conf: "ModelConf"
                 , board_name_list: list | tuple =('statistics', 'fitness', 'decomposition', 'waterfall'
                                                   , 'curves_simple', 'curves_full', 'optim', 'transformations', 'export')
                 , ds: pd.DataFrame | None = None
                 , dt_pred: pd.DataFrame | None = None
                 , split_m_m: pd.DataFrame | None = None
                 , curves_simple_data: pd.DataFrame | None = None
                 , curves_full_data: pd.DataFrame | None = None
                 , price: float | None = 1.
                 , conv: float | None = 1.
                 , period_fit: str = 'M'
                 , period_decomp: str = 'M'
                 , metrics: pd.DataFrame | None = None
                 , predictors: pd.DataFrame | None = None
                 , vifs: pd.DataFrame | None = None
                 , current_options: pd.DataFrame | None = None
                 , current_opt_budget_summary: pd.DataFrame | None = None
                 , current_opt_budget_budget_fr: pd.DataFrame | None = None
                 , current_opt_target_summary: pd.DataFrame | None = None
                 , current_opt_target_budget_fr: pd.DataFrame | None = None
                 ):
        """
        Initialize ModelConfExp instance expanding existing ModelConf object
        with data to be exported

        :param model_conf: ModelConf object to extend
        :param board_name_list: which boards to export (list of string board names)
        :param ds: main dataset
        :param dt_pred: prediction data
        :param split_m_m: decomposed prediction data
        :param curves_simple_data: simple curves (simple, fast, limited) data
        :param curves_full_data: full curves (complex, flexible, slow) data
        :param price: product price (numeric)
        :param conv: single conversion rate (numeric)
        :param period_fit: time period to plot fit ('day', 'week', 'month', 'quarter', 'year')
        :param period_decomp: time period to plot decomposition ('day', 'week', 'month', 'quarter', 'year')
        :param metrics: calculated metrics as dataframe
        :param predictors: regression coefficients with estimations and statistics as dataframe
        :param vifs: VIF table
        :param current_options: options dictionary of dictionaries (for every year)
        :param current_opt_budget_summary: optimized budget option summary as dictionary
        :param current_opt_budget_budget_fr: optimized budget option summary as dataframe
        :param current_opt_target_summary: optimized target option summary as dictionary
        :param current_opt_target_budget_fr: optimized target option summary as dataframe
        """

        # "inherit" from ModelConf instance (w/o calling the constructor)
        model_conf_dict = copy.deepcopy(vars(model_conf))
        model_conf_dict = {x: model_conf_dict[x] for x in model_conf_dict if x not in ['_StableClass__if_stable']}

        vars(self).update(model_conf_dict)

        self.__if_stable = False

        self._state(current_options
                    , metrics
                    , predictors
                    , vifs
                    , dt_pred
                    , split_m_m
                    , curves_simple_data
                    , curves_full_data
                    )

        self._decompose(split_m_m, period_decomp, board_name_list)

        self._fit(dt_pred, ds, period_fit, board_name_list)

        self._curves_simple(curves_simple_data, price, conv, board_name_list)

        self._curves_full(curves_full_data, board_name_list)

        self._current_options(current_options)

        if "statistics" in board_name_list:
            self.metrics = metrics
            self.predictors = predictors
            self.vifs = vifs

        if "optim" in board_name_list:
            if current_opt_budget_summary is not None:
                self.opt_budget_summary = current_opt_budget_summary
                self.opt_budget_budget_fr = current_opt_budget_budget_fr
            if current_opt_target_summary is not None:
                self.opt_target_summary = current_opt_target_summary
                self.opt_target_budget_fr = current_opt_target_budget_fr

        self._init_finish()

    def _decompose(self
                   , split_m_m: pd.DataFrame | None
                   , period_decomp: str
                   , board_name_list: list | tuple):
        """
        Prepare decomposition data.

        :param split_m_m: decomposed prediction data
        :param period_decomp: time period to plot decomposition ('day', 'week', 'month', 'quarter', 'year')
        :param board_name_list: which boards to export (list of string board names)
        :return:
        """

        if "decomposition" in board_name_list:

            if period_decomp in ['day', 'd', 'D']:
                split_m_m['date'] = split_m_m['date'].dt.floor(freq='D')
            elif period_decomp in ['week', 'w', 'W']:
                split_m_m['date'] = split_m_m['date'].dt.to_period('W').dt.start_time
            elif period_decomp in ['month', 'm', 'M']:
                split_m_m['date'] = split_m_m['date'].dt.to_period('M').dt.start_time
            elif period_decomp in ['quarter', 'q', 'Q']:
                split_m_m['date'] = split_m_m['date'].dt.to_period('Q').dt.start_time
            elif period_decomp in ['year', 'y', 'Y']:
                split_m_m['date'] = split_m_m['date'].dt.to_period('Y').dt.start_time

            cln_pivot = split_m_m.columns.drop(['value', 'value_rub']).tolist()
            tmp = split_m_m.groupby(cln_pivot, as_index=False).sum()
            cln_pivot.remove('variable')
            split_pivot = tmp.pivot(index=cln_pivot, values='value', columns='variable')
            split_pivot = split_pivot.reset_index()

            self.decomposition = split_pivot

        pass

    def _fit(self
             , dt_pred: pd.DataFrame | None
             , ds: pd.DataFrame | None
             , period_fit: str
             , board_name_list: list | tuple):
        """
        Prepare fit-predict data.

        :param dt_pred: prediction data
        :param ds: main dataset
        :param period_fit: time period to plot fit ('day', 'week', 'month', 'quarter', 'year')
        :param board_name_list: which boards to export (list of string board names)
        :return:
        """

        if "fitness" in board_name_list:

            cln_id = ["date", "bs_key", "listed"] + self.bs_key
            if pd.notna(self.price_var):
                cln_id = cln_id + [self.price_var]

            cln_var = [x for x in dt_pred.columns if re.match(r'(observed|predicted)', x)]

            tmp = select_eff(dt_pred.loc[dt_pred['listed'].isin([2, 3, 4]), :], cln_id + cln_var)

            if period_fit in ['day', 'd', 'D']:
                tmp['date'] = tmp['date'].dt.floor(freq='D')
            elif period_fit in ['week', 'w', 'W']:
                tmp['date'] = tmp['date'].dt.to_period('W').dt.start_time
            elif period_fit in ['month', 'm', 'M']:
                tmp['date'] = tmp['date'].dt.to_period('M').dt.start_time
            elif period_fit in ['quarter', 'q', 'Q']:
                tmp['date'] = tmp['date'].dt.to_period('Q').dt.start_time
            elif period_fit in ['year', 'y', 'Y']:
                tmp['date'] = tmp['date'].dt.to_period('Y').dt.start_time

            if pd.notna(self.price_var):

                tmp['observed_value'] = tmp['observed'] * ds[self.price_var]
                tmp['predicted_value'] = tmp['predicted'] * ds[self.price_var]

                tmp = tmp.drop(['listed', self.price_var], axis=1) \
                    .groupby([x for x in cln_id if x not in {'listed', self.price_var}], as_index=False) \
                    .sum()

            else:
                tmp['observed_value'] = tmp['observed']
                tmp['predicted_value'] = tmp['predicted']

                tmp = tmp.drop(['listed'], axis=1) \
                    .groupby([x for x in cln_id if x not in {'listed'}], as_index=False) \
                    .sum()

            self.fit = tmp

        pass

    def _curves_simple(self
                       , curves_simple_data: pd.DataFrame | None
                       , price: float | None
                       , conv: float | None
                       , board_name_list: list | tuple):
        """
        Prepare simple curves data.

        :param curves_simple_data: simple curves (simple, fast, limited) data
        :param price: product price (numeric)
        :param conv: single conversion rate (numeric)
        :param board_name_list: which boards to export (list of string board names)
        :return:
        """

        if "curves_simple" in board_name_list:

            price_array = np.full(curves_simple_data.shape[0], np.nan)
            price_array[0] = price
            price_series = pd.Series(price_array, name='price', index=curves_simple_data.index)

            conv_array = np.full(curves_simple_data.shape[0], np.nan)
            conv_array[0] = conv
            conv_series = pd.Series(conv_array, name='conv', index=curves_simple_data.index)

            bdg = pd.Series(curves_simple_data.index * 1_000_000, name='budget', index=curves_simple_data.index)

            self.incremental_volume = pd.concat([price_series
                                                    , conv_series
                                                    , bdg
                                                    , curves_simple_data]
                                                , axis=1)
            curves_simple_value_data = curves_simple_data * price * conv

            self.incremental_value = pd.concat([price_series
                                                   , conv_series
                                                   , bdg
                                                   , curves_simple_value_data]
                                               , axis=1)

            self.profit = pd.concat([price_series
                                        , conv_series
                                        , pd.Series(curves_simple_value_data.index * 1_000_000, name='budget')
                                        , bdg]
                                    , axis=1)

            self.roi = pd.concat([price_series
                                     , conv_series
                                     , pd.Series(curves_simple_value_data.index * 1_000_000, name='budget')
                                     , bdg]
                                 , axis=1)

        pass

    def _curves_full(self
                     , curves_full_data: pd.DataFrame | None
                     , board_name_list: list | tuple):
        """
        Prepare full curves data.

        :param curves_full_data: full curves (complex, flexible, slow) data
        :param board_name_list: which boards to export (list of string board names)
        :return:
        """

        if ("curves_full" in board_name_list) and (curves_full_data is not None):
            if hasattr(self, "incremental_volume"):
                del self.incremental_volume, self.incremental_value, self.profit, self.roi

            zero = curves_full_data.loc[:, ["pred_exact_vol", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_exact_vol').zero.iloc[0]

            self.incremental_volume_short = curves_full_data.loc[:, ["pred_exact_vol", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_exact_vol') - zero

            zero = curves_full_data.loc[:, ["pred_exact_val", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_exact_val').zero.iloc[0]

            self.incremental_value_short = curves_full_data.loc[:, ["pred_exact_val", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_exact_val') - zero

            self.profit_short = self.incremental_value_short.sub(self.incremental_value_short.index * 1_000_000, axis=0)
            self.roi_short = self.incremental_value_short.div(self.incremental_value_short.index * 1_000_000, axis=0)

            zero = curves_full_data.loc[:, ["pred_long_vol", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_long_vol').zero.iloc[0]

            self.incremental_volume_long = curves_full_data.loc[:, ["pred_long_vol", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_long_vol') - zero

            zero = curves_full_data.loc[:, ["pred_long_val", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_long_val').zero.iloc[0]

            self.incremental_value_long = curves_full_data.loc[:, ["pred_long_val", "option", "bdg"]].pivot(
                index='bdg', columns='option', values='pred_long_val') - zero

            self.profit_long = self.incremental_value_long.sub(self.incremental_value_long.index * 1_000_000, axis=0)
            self.roi_long = self.incremental_value_long.div(self.incremental_value_long.index * 1_000_000, axis=0)

            self.incremental_volume_short = self.incremental_volume_short.reset_index()
            self.incremental_value_short = self.incremental_value_short.reset_index()
            self.profit_short = self.profit_short.reset_index()
            self.roi_short = self.roi_short.reset_index()
            self.incremental_volume_long = self.incremental_volume_long.reset_index()
            self.incremental_value_long = self.incremental_value_long.reset_index()
            self.profit_long = self.profit_long.reset_index()
            self.roi_long = self.roi_long.reset_index()

        pass

    def _current_options(self
                         , current_options: dict):
        """
        Prepare current options as pandas DataFrame.

        :param current_options: options dictionary of dictionaries (for every year)
        :return:
        """

        if current_options:
            self.current_options = pd.DataFrame(current_options).reset_index()

        pass

    def _state(self
               , current_options: dict
               , metrics: pd.DataFrame | None
               , predictors: pd.DataFrame | None
               , vifs: pd.DataFrame | None
               , dt_pred: pd.DataFrame | None
               , split_m_m: pd.DataFrame | None
               , curves_simple_data: pd.DataFrame | None
               , curves_full_data: pd.DataFrame | None
               ):
        """
        Check state of data available and prepare it to be exported as 'info'

        :param current_options: options dictionary of dictionaries (for every year)
        :param metrics: calculated metrics as dataframe
        :param predictors: regression coefficients with estimations and statistics as dataframe
        :param vifs: VIF table
        :param dt_pred: prediction data
        :param split_m_m: decomposed prediction data
        :param curves_simple_data: simple curves (simple, fast, limited) data
        :param curves_full_data: full curves (complex, flexible, slow) data
        :return:
        """

        dict_info = {}

        if not current_options:
            dict_info["options_info"] = "Warning : Current option is not defined / contain zeros"
        if (metrics is None) or metrics.empty:
            dict_info["metrics_info"] = "Warning : Metrics table is not defined / contain zeros"
        if (predictors is None) or predictors.empty:
            dict_info["predictors_info"] = "Warning : Predictors table is not defined / contain zeros"
        if (vifs is None) or vifs.empty:
            dict_info["vifs_info"] = "Warning : Vifs table is not defined / contain zeros"

        if (dt_pred is None) or dt_pred.empty:
            dict_info["ds_pred_info"] = "Warning : ds_pred is not defined / contain zeros"
        if (split_m_m is None) or split_m_m.empty:
            dict_info["split_m_m_info"] = "Warning : split_m_m is not defined / contain zeros"
        if (curves_simple_data is None) or curves_simple_data.empty:
            dict_info["curves_simple_data_info"] = "Warning : curves_simple_data is not defined / contain zeros"
        else:
            dict_info["curves_info"] = "Simple curves data has been exported"

        if (curves_full_data is None) or curves_full_data.empty:
            dict_info["curves_full_data_info"] = "Warning : curves_full_data is not defined / contain zeros"
        else:
            dict_info["curves_info"] = "Full curves data has been exported"

        self.state = pd.DataFrame(dict_info, index=["Export general information: "]).T

        pass


def export_model(model_exp: "ModelConfExp"
                 , mode: str = "client"  # "full", "client", "analysis"
                 ):
    """
    Export model results and info prepared as ModelConfExp

    :param model_exp: ModelConfExp object (extended ModelConf)
    :param mode: export mode: "full", "client", "analysis"
    :return:
    """

    model_conf_list = {}

    # list of data frames to save

    model_conf_list['info'] = model_exp.state
    if hasattr(model_exp, 'current_options') and model_exp.current_options is not None:
        model_conf_list['current_options'] = model_exp.current_options
    if hasattr(model_exp, 'params') and model_exp.params is not None and mode != "client":
        model_conf_list['params'] = model_exp.params
    if hasattr(model_exp, 'model_rhs') and model_exp.model_rhs is not None and mode != "client":
        model_conf_list['RHS'] = model_exp.model_rhs
    if hasattr(model_exp, 'model_lhs') and model_exp.model_lhs is not None and mode != "client":
        model_conf_list['LHS'] = model_exp.model_lhs
    if hasattr(model_exp, 'trans_path_df') and model_exp.trans_path_df is not None and mode != "client":
        model_conf_list['trans_path_df'] = model_exp.trans_path_df
    if hasattr(model_exp, 'metrics') and model_exp.metrics is not None:
        model_conf_list['metrics'] = model_exp.metrics
    if hasattr(model_exp, 'predictors') and model_exp.predictors is not None:
        model_conf_list['predictors'] = model_exp.predictors
    if hasattr(model_exp, 'vifs') and model_exp.vifs is not None:
        model_conf_list['vifs'] = model_exp.vifs
    if hasattr(model_exp, 'fit') and model_exp.fit is not None:
        model_conf_list['fit'] = model_exp.fit
    if hasattr(model_exp, 'decomposition') and model_exp.decomposition is not None:
        model_conf_list['decomposition'] = model_exp.decomposition

    if hasattr(model_exp, 'incremental_volume') and model_exp.incremental_volume is not None:
        model_conf_list['incremental_volume'] = model_exp.incremental_volume
    if hasattr(model_exp, 'incremental_value') and model_exp.incremental_value is not None:
        model_conf_list['incremental_value'] = model_exp.incremental_value
    if hasattr(model_exp, 'profit') and model_exp.profit is not None:
        model_conf_list['profit'] = model_exp.profit
    if hasattr(model_exp, 'roi') and model_exp.roi is not None:
        model_conf_list['roi'] = model_exp.roi

    if hasattr(model_exp, 'incremental_volume_short') and model_exp.incremental_volume_short is not None:
        model_conf_list['incremental_volume_short'] = model_exp.incremental_volume_short
    if hasattr(model_exp, 'incremental_value_short') and model_exp.incremental_value_short is not None:
        model_conf_list['incremental_value_short'] = model_exp.incremental_value_short
    if hasattr(model_exp, 'profit_short') and model_exp.profit_short is not None:
        model_conf_list['profit_short'] = model_exp.profit_short
    if hasattr(model_exp, 'roi_short') and model_exp.roi_short is not None:
        model_conf_list['roi_short'] = model_exp.roi_short

    if hasattr(model_exp, 'incremental_volume_long') and model_exp.incremental_volume_long is not None:
        model_conf_list['incremental_volume_long'] = model_exp.incremental_volume_long
    if hasattr(model_exp, 'incremental_value_long') and model_exp.incremental_value_long is not None:
        model_conf_list['incremental_value_long'] = model_exp.incremental_value_long
    if hasattr(model_exp, 'profit_long') and model_exp.profit_long is not None:
        model_conf_list['profit_long'] = model_exp.profit_long
    if hasattr(model_exp, 'roi_long') and model_exp.roi_long is not None:
        model_conf_list['roi_long'] = model_exp.roi_long

    if hasattr(model_exp, 'opt_budget_budget_fr') and model_exp.opt_budget_budget_fr is not None:
        model_conf_list['Optimized split budget'] = model_exp.opt_budget_budget_fr
    if hasattr(model_exp, 'opt_budget_summary') and model_exp.opt_budget_summary is not None:
        model_conf_list['Optimized summary budget'] = model_exp.opt_budget_summary
    if hasattr(model_exp, 'opt_target_budget_fr') and model_exp.opt_target_budget_fr is not None:
        model_conf_list['Optimized split target'] = model_exp.opt_target_budget_fr
    if hasattr(model_exp, 'opt_target_summary') and model_exp.opt_target_summary is not None:
        model_conf_list['Optimized summary target'] = model_exp.opt_target_summary

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df_name, ds in model_conf_list.items():
            print(df_name)
            ds.to_excel(writer, sheet_name=df_name, index=False, freeze_panes=(1, 0))
        writer.close()

    data = output.getvalue()
    output.seek(0)

    return data
