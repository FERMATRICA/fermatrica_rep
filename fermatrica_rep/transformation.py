"""
Report transformation chains: from raw variable through all transformations one after another
to the final variable (feature).

Only standard transformations could be reported. Custom transformations don't follow transformation
API and therefore are not reportable.
"""


import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from fermatrica_utils import groupby_eff

pio.templates.default = 'ggplot2'


def transformation_main_plot(ds: pd.DataFrame
                             , init_var: str
                             , fin_var: str
                             , height: int = 300
                             , if_main=True):
    """
    Plot two bound variables one vs another.
    
    :param ds: main dataset
    :param init_var: initial variable name
    :param fin_var: final variable name
    :param height: plot height
    :param if_main: different treatment of combined initial-final plot and one step plot
    :return: 
    """

    tmp = groupby_eff(ds, ['date'], [init_var, fin_var], ds['listed'].isin([2, 3]), as_index=False, sort=True)
    tmp = tmp.sum()

    # Create figure with secondary y-axis
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    tmp['color_init'] = init_var
    tmp['color_fin'] = fin_var
    fig1 = px.line(x=tmp['date'], y=tmp[init_var], height=height, color=tmp['color_init'])
    fig2 = px.line(x=tmp['date'], y=tmp[fin_var], height=height, color=tmp['color_fin'])

    fig2.update_traces(yaxis="y2")

    subfig.add_traces(fig1.data + fig2.data)

    title = ' '.join(fin_var.replace(init_var, '').split('_'))
    subfig.update_layout(title=title
                         , yaxis=dict(title=init_var)  # , rangemode='tozero'
                         , yaxis2=dict(title=fin_var, showgrid=False)  # , rangemode='tozero'
                         , xaxis=dict(title='date'
                                      , mirror=True
                                      , ticks='outside'
                                      , showline=True))
    if if_main:
        subfig.update_xaxes(showline=True,
                            linewidth=1,
                            linecolor='black',
                            mirror=True)

        subfig.update_yaxes(showline=True,
                            linewidth=1,
                            linecolor='black',
                            mirror=True)
    subfig.update_traces(line=dict(color='LightCoral'), selector=dict(yaxis='y'), showlegend=True)
    subfig.update_traces(line=dict(color='LightSeaGreen'), selector=dict(yaxis='y2'))

    subfig.layout['height'] = height

    return subfig


def transformation_plot(ds: pd.DataFrame
                        , init_var: str
                        , fin_var: str
                        , sub: bool = True):
    """
    Plot transformations of the variable. Depending on `sub` argument:
    initial (raw) variable vs final variable (feature) or step by step in the transformation chain.

    :param ds:
    :param init_var:
    :param fin_var:
    :param sub:
    :return:
    """

    if not sub:
        fig = transformation_main_plot(ds=ds, init_var=init_var, fin_var=fin_var, height=650, if_main=True)
        return {'main': fig}
    else:

        transforms = fin_var.replace(init_var, '').split('_')[1:]

        chain = [init_var]

        for i in transforms:
            chain.append(chain[len(chain) - 1] + '_' + i)

        fig = {}
        for i in range(len(chain)-1):
            fig[i] = transformation_main_plot(ds=ds, init_var=chain[i], fin_var=chain[i+1], height=400, if_main=False)

        return fig










