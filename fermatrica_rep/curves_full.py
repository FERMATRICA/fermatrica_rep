"""
Calculate efficiency curves per marketing tool.

The version defined here is general-purpose and could be used with models of any design.
It calculates curves with calculating row of budget options. This approach is model-blind
and very convenient when model design is complex, LHS effects are great etc.

However, it implies large amount of calculations, so it is rather slow and resource demanding
(multiprocessing is used). If your model contains RHS only, you may consider using
`fermatrica_rep.curves` module at least for WIP-analysis.

Beware! This module contains only plot functions. Use outputs from `options.calc_multi.option_report_multi_post`
as input data.
"""


import pandas as pd
import plotly.graph_objects as go

from fermatrica_rep.model_rep import ModelRep


def _curves_full_plot_worker(curves_full_df: pd.DataFrame
                             , model_rep: "ModelRep"
                             , title: str
                             , var: str):
    """
    Plot single type of the curve: incremental, return or ROI

    :param curves_full_df: prepared dataset
    :param model_rep: ModelRep object (export settings)
    :param title: figure title
    :param var: variable to calculate (type of curve)
    :return: plot of the single curve type
    """

    layout = {'title': {'text': title}}

    fig = go.Figure(layout=layout)

    tmp = curves_full_df.groupby('option').agg({var: 'nunique'})
    opts = tmp.loc[tmp[var] > 1].reset_index().option.tolist()

    for opt in opts:

        tmp_obs = curves_full_df.loc[(curves_full_df["option"] == opt)]
        if 'max_obs_budg' in curves_full_df.columns.to_list():
            tmp_obs = tmp_obs.loc[(tmp_obs['max_obs_budg'] >= tmp_obs['bdg'])]

        fig.add_trace(go.Scatter(
            x=tmp_obs["bdg"],
            y=tmp_obs["value"].values,
            line_color=model_rep.palette_tools[opt],
            mode='lines',
            name=opt.replace('bdg_', '')
        ))

        tmp_extra = curves_full_df.loc[(curves_full_df["option"] == opt)]
        if 'max_obs_budg' in curves_full_df.columns.to_list():
            tmp_extra = tmp_extra.loc[(tmp_extra['max_obs_budg'] <= tmp_extra['bdg'])]

        fig.add_trace(go.Scatter(
            x=tmp_extra["bdg"],
            y=tmp_extra["value"].values,
            line_color=model_rep.palette_tools[opt],
            mode='lines',
            name=opt.replace('bdg_', ''),
            line=dict(dash='dash'),
            showlegend=True
        ))

    return fig


def curves_full_plot_short(curves_full_df: pd.DataFrame
                           , model_rep: "ModelRep") -> dict:
    """
    Plot efficiency curves per marketing tool. "Short" means only short-term effect is measured:
    period of "promoting" is equal to period of effect calculating. (Not the same as "on-air",
    because in "promoting" period some silence dates could be included.)

    This version could be used with models of any design. It calculates curves with calculating
    row of budget options. This approach is model-blind and convenient when model design is complex,
    LHS effects are great etc.

    :param curves_full_df: prepared dataset
    :param model_rep: ModelRep object (export settings)
    :return: dict of plotly figures
    """

    tmp = curves_full_df.groupby('option').agg({'pred_exact_val': 'nunique'})
    lst = tmp.loc[tmp["pred_exact_val"] > 1].reset_index().option.tolist()
    lst.sort()

    model_rep.fill_colours_tools(lst)

    for i in lst:
        model_rep.palette_tools[i + '_extrapolated'] = model_rep.palette_tools[i]

    zero = curves_full_df.loc[curves_full_df["option"] == "zero", "pred_exact_vol"].values
    curves_full_df["value"] = curves_full_df["pred_exact_vol"] - zero
    fig_vol = _curves_full_plot_worker(curves_full_df, model_rep, 'Incremental Volume', "pred_exact_vol")

    zero = curves_full_df.loc[curves_full_df["option"] == "zero", "pred_exact_val"].values
    curves_full_df["value"] = curves_full_df["pred_exact_val"] - zero
    fig_val = _curves_full_plot_worker(curves_full_df, model_rep, 'Incremental Value', "pred_exact_val")

    curves_full_df["value"] = curves_full_df["value"] - curves_full_df["bdg"] * 1e6
    fig_incr = _curves_full_plot_worker(curves_full_df, model_rep, 'Return / Profit', "pred_exact_val")

    curves_full_df["value"] = (curves_full_df["pred_exact_val"] - zero) / (curves_full_df["bdg"] * 1e6)
    fig_roi = _curves_full_plot_worker(curves_full_df, model_rep, 'ROI', "pred_exact_val")

    return {'Incremental Volume': fig_vol,
            'Incremental Value': fig_val,
            'Return / Profit': fig_incr,
            'ROI': fig_roi}


def curves_full_plot_long(curves_full_df: pd.DataFrame
                          , model_rep: "ModelRep"):
    """
    Plot efficiency curves per marketing tool. "Long" means short-term and long-term effects
    are measured both: period of "promoting" is much shorter than period of effect calculating.

    This version could be used with models of any design. It calculates curves with calculating
    row of budget options. This approach is model-blind and convenient when model design is complex,
    LHS effects are great etc.

    :param curves_full_df: prepared dataset
    :param model_rep: ModelRep object (export settings)
    :return: dict of plotly figures
    """

    tmp = curves_full_df.groupby('option').agg({'pred_long_val': 'nunique'})
    lst = tmp.loc[tmp["pred_long_val"] > 1].reset_index().option.tolist()
    lst.sort()

    model_rep.fill_colours_tools(lst)

    for i in lst:
        model_rep.palette_tools[i + '_extrapolated'] = model_rep.palette_tools[i]

    zero = curves_full_df.loc[curves_full_df["option"] == "zero", "pred_long_vol"].values
    curves_full_df["value"] = curves_full_df["pred_long_vol"] - zero
    fig_vol = _curves_full_plot_worker(curves_full_df, model_rep, 'Incremental Volume', "pred_long_vol")

    zero = curves_full_df.loc[curves_full_df["option"] == "zero", "pred_long_val"].values
    curves_full_df["value"] = curves_full_df["pred_long_val"] - zero
    fig_val = _curves_full_plot_worker(curves_full_df, model_rep, 'Incremental Value', "pred_long_val")

    curves_full_df["value"] = curves_full_df["value"] - curves_full_df["bdg"] * 1e6
    fig_incr = _curves_full_plot_worker(curves_full_df, model_rep, 'Return / Profit', "pred_long_val")

    curves_full_df["value"] = (curves_full_df["pred_long_val"] - zero) / (curves_full_df["bdg"] * 1e6)
    fig_roi = _curves_full_plot_worker(curves_full_df, model_rep, 'ROI', "pred_long_val")

    return {'Incremental Volume long': fig_vol,
            'Incremental Value long': fig_val,
            'Return / Profit long': fig_incr,
            'ROI long': fig_roi}
