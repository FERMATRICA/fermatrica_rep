"""
Basic / common utilities required by FERMATRICA_REP.
"""


import copy
import pandas as pd
import re
import colorcet as cc

"""
Data utilities
"""


def coef_var_align(params: pd.DataFrame
                   , coefs: pd.Series):
    """
    Align trans_path_df with regression coefficients. Used in rather simple models.

    :param params: trans_path_df
    :param coefs: regression coefficients (from statsmodels model object)
    :return: table (dataframe) with merged data
    """

    params_subset = params.copy()

    params_subset['coef_name'] = ''
    params_subset['coef_value'] = .0

    for ind, param_row in params_subset.iterrows():
        rtrn = _coef_var_align_one(param_row['variable_fin'], coefs)
        if rtrn is not None:
            params_subset.loc[ind, 'coef_name'] = rtrn.index.values[0]
            params_subset.loc[ind, 'coef_value'] = rtrn.values[0]

    params_subset = params_subset[params_subset['coef_name'] != '']

    return params_subset


def _coef_var_align_one(var_name: str
                        , coefs: pd.Series) -> pd.Series:
    """
    Worker function for `coef_var_align` manipulating single variable / transformation chain.

    :param var_name: final variable name
    :param coefs: regression coefficients (from statsmodels model object) to search through
    :return:
    """

    coef = coefs.loc[coefs.index == var_name]

    if len(coef) == 0:
        coef = [x for x in coefs.index if re.search(r'\b' + var_name + r'\b', x)]
        if len(coef) > 1:
            coef = coefs[coefs.index.isin(coef)][0:1]
        elif len(coef) == 1:
            coef = coefs[coefs.index.isin(coef)]
        else:
            coef = None

    return coef


"""
Visual utilities
"""


def palette_fill(entities_old: dict
                 , entities_new: dict | list
                 , palette_names: list | tuple | str = ('glasbey', )):
    """
    Preserve colours between entities in different functions.

    :param entities_old: dictionary of elements with already set colours
    :param entities_new: dictionary or list with elements to be checked with entities_old and added to it
        if not yet
    :param palette_names: colorcet palette names
    :return:
    """

    units_total = copy.deepcopy(entities_old)

    # get basic palette(s)

    if isinstance(palette_names, str):
        palette_names = [palette_names]

    palette_base = []
    for pl in palette_names:
        clr = getattr(cc, pl)
        if len(clr) > 0:
            palette_base.extend(clr)

    # remove already assigned colors and units

    if len(entities_old) > 0:
        # units_new = list(set(units_new) - set(units_old))
        entities_new = [i for i in entities_new if i not in entities_old.keys()]

        # palette_base = list(set(palette_base) - set(units_old.values()))
        palette_base = [i for i in palette_base if i not in entities_old.values()]

    # bind new units to colours

    for i, u in enumerate(entities_new):
        units_total[u] = palette_base[i]

    return units_total


