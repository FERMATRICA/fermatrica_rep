"""
Standard translators media budget -> media variable per media.

Translation interface is fixed among all translation functions, so some parameters could look "excessive"
in some functions. Do not remove them!
"""


import inspect
import pandas as pd
import re
import typing

from fermatrica_utils import select_eff, groupby_eff
from fermatrica.model.model_conf import ModelConf

from fermatrica_rep.options.define import OptionSettings


def basic(md: "ModelConf"
          , ds: pd.DataFrame
          , model_rep
          , var_key: str
          , var_name: str
          , val: float | int
          , ds_mask: pd.Series
          ):
    """
    Most standard / basic translation.
    Translate / apply budget to media variable 1 to 1, with respect to seasonality and media price.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    # get primitives

    media_price = md.trans_path_df.loc[md.trans_path_df['variable'] == var_name, 'price']

    if len(media_price) > 0:
        media_price = media_price.iloc[0]
    else:
        media_price = 1.0

    ssn_name = re.sub(r'^bdg_', 'opt_media_ssn_', var_key)

    # find seasonality correction

    ssn_cor = 1

    tmp = select_eff(ds, ['date', ssn_name]).loc[ds_mask, :].drop_duplicates()

    if (tmp.loc[:, 'date'].max() - tmp.loc[:, 'date'].min()).days <= 366:
        ssn_cor = tmp.loc[:, ssn_name].sum()
        if ssn_cor != 0:
            ssn_cor = 1 / ssn_cor
        else:
            ssn_cor = 1 / tmp.loc[:, ssn_name].shape[0]

    # run

    ds.loc[ds_mask, var_name] = ds.loc[ds_mask, ssn_name] * ssn_cor * val * 1e+6 / media_price

    # maybe switch to void function later
    return ds


def basic_fix(md: "ModelConf"
              , ds: pd.DataFrame
              , model_rep
              , var_key: str  # as in options
              , var_name: str  # as in dataset
              , val: float | int
              , ds_mask: pd.Series
              ):
    """
    Translate / apply budget to some 'fixed' variable 1 to 1, w/o seasonality, media price and Mln conversion.
    Use it for non-media variables mostly, such as constant market metrics etc.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    ds.loc[ds_mask, var_name] = val

    # maybe switch to void function later
    return ds


def price(md: "ModelConf"
          , ds: pd.DataFrame
          , model_rep
          , var_key: str  # as in options
          , var_name: str  # as in dataset
          , val: float | int
          , ds_mask: pd.Series
          ):
    """
    Standard price as fractions / ratios to basic price (price_..._back).

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    # run

    price_back_name = md.price_var + '_back'
    ds.loc[ds_mask, md.price_var] = ds.loc[ds_mask, price_back_name] * val

    # maybe switch to void function later
    return ds


def tv_complex(md: "ModelConf"
               , ds: pd.DataFrame
               , model_rep
               , var_key: str  # as in options
               , var_name: str  # as in dataset
               , val: float | int
               , ds_mask: pd.Series
               , clip_dur: float | int = 10
               , tv_type: str = 'rolik_nat'
               , media_prefix: str | list = 'ots'
               ):
    """
    Complex TV translation with respect to affinities and TV audience sizes.
    Use it as a worker function with extended interface to create final function with standard translation params.
    Some standard functions are defined below, for any changes it is recommended to define simple
    ad hoc function on the project level.

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :param clip_dur: TV clip duration (fixed over filtered data)
    :param tv_type: TV type string as TV variables contain. TV OTS and TV clip duration variables
        should be named using the same pattern, defaults to "rolik_nat"
    :param media_prefix: prefix setting media measure type, defaults to "ots"
    :return: main dataset with applied `val`
    """

    # get primitives

    media_price = md.trans_path_df.loc[md.trans_path_df['variable'] == var_key, 'price']
    aud_year = model_rep.aud_coefs['year'].max()

    var_name = re.split(r', *', var_name)
    sku_prefix = ''

    if len(var_name) == 3:
        umb_prefix = var_name[0]
        sku_coef = var_name[1]
        umb_coef = var_name[2]
    else:
        umb_prefix = var_name[0]
        sku_prefix = var_name[1]
        sku_coef = var_name[2]
        umb_coef = var_name[3]

    if sku_prefix != '' and not re.search(r'_$', sku_prefix):
        sku_prefix = sku_prefix + '_'

    if type(media_prefix) == list:
        media_prefix_base = media_prefix[0]
        media_prefix = media_prefix[1]
    else:
        media_prefix_base = media_prefix

    if len(media_price) > 0:
        media_price = media_price.iloc[0]
    else:
        media_price = 1.0

    ssn_name = re.sub(r'^bdg_', 'opt_media_ssn_', var_key)

    # find seasonality correction

    ssn_cor = 1

    tmp = select_eff(ds, ['date', ssn_name]).loc[ds_mask, :].drop_duplicates()

    if (tmp.loc[:, 'date'].max() - tmp.loc[:, 'date'].min()).days <= 366:
        ssn_cor = 1 / tmp.loc[:, ssn_name].sum()

    # run

    cln_tv_raw = ds.columns.tolist()

    cln_tv = [re.sub(r'^(' + sku_prefix + media_prefix_base + r')', sku_prefix + media_prefix, x) for x in cln_tv_raw
              if re.search(r'^' + sku_prefix + media_prefix_base + r'_(f|m|all).*' + tv_type, x) and not re.search(r'infl|softmax|gompert|trans|adst|\\binv|kids', x)]

    if len(cln_tv) == 0:
        cln_tv = [x for x in cln_tv_raw
                  if re.search(r'^' + sku_prefix + media_prefix + r'_(f|m|all).*' + tv_type, x) and not re.search(r'infl|softmax|gompert|trans|adst|\\binv|kids', x)]

    cln_tv.sort()

    cln_all = ds.columns.tolist()

    for col in cln_tv:

        # clip duration ('length')

        cl_clipl = re.sub(r'^(' + sku_prefix + media_prefix + r')_', sku_prefix + 'clipl_', col)
        cl_clipl_umb = re.sub(r'^(' + sku_prefix + media_prefix + r')_', umb_prefix + '_clipl_', col)

        # important for "additional" TV-variables

        if col not in cln_all:
            ds[col] = 0
            ds[cl_clipl] = 0
            ds[cl_clipl_umb] = 0

        cl_base = re.sub(r'^(' + sku_prefix + media_prefix + r')_', media_prefix_base + '_', col)

        ds.loc[ds_mask, cl_clipl] = clip_dur
        ds.loc[ds_mask, cl_clipl_umb] = clip_dur

        # per SKU / product

        ds.loc[ds_mask, col] = ds.loc[ds_mask, ssn_name] * ssn_cor * val * 1/clip_dur * 1e+6 / media_price * ds.loc[ds_mask, sku_coef] *\
                              model_rep.aud_coefs.loc[(model_rep.aud_coefs['variable'] == cl_base) & (model_rep.aud_coefs['year'] == aud_year), 'coef'].iloc[0]

        if 'tv_ssn' in ds.columns.tolist():
            ds.loc[ds_mask, col] = ds.loc[ds_mask, col] / ds.loc[ds_mask, 'tv_ssn']

        # per superbrand

        cl_umb = re.sub(sku_prefix + media_prefix, umb_prefix + '_' + media_prefix_base, col)

        rtrn = select_eff(ds, ['date', 'superbrand', col, umb_coef]).loc[ds_mask, :]
        rtrn.loc[:, cl_umb] = rtrn[col] * rtrn[umb_coef]
        rtrn = rtrn.groupby(['date', 'superbrand'])[cl_umb].sum()

        ds_tmp = select_eff(ds, ['date', 'superbrand'])
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name=cl_umb), how='left', on=['date', 'superbrand'],
                              sort=False, copy=False)

        ds.loc[ds_mask, cl_umb] = ds_tmp.loc[ds_mask, cl_umb]

    # maybe switch to void function later
    return ds


def nat_tv_complex(md: "ModelConf"
                   , ds: pd.DataFrame
                   , model_rep
                   , var_key: str  # as in options
                   , var_name: str  # as in dataset
                   , val: float | int
                   , ds_mask: pd.Series
                   ):
    """
    National TV: complex TV translation with respect to affinities and TV audience sizes.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    ds = tv_complex(md, ds, model_rep, var_key, var_name, val, ds_mask, clip_dur=10, tv_type='rolik_nat', media_prefix='ots')

    return ds


def reg_tv_complex(md: "ModelConf"
                   , ds: pd.DataFrame
                   , model_rep
                   , var_key: str  # as in options
                   , var_name: str  # as in dataset
                   , val: float | int
                   , ds_mask: pd.Series
                   ):
    """
    Regional TV: complex TV translation with respect to affinities and TV audience sizes.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    ds = tv_complex(md, ds, model_rep, var_key, var_name, val, ds_mask, clip_dur=10, tv_type='rolik_local', media_prefix='ots')

    return ds


def nat_tv_sponsor_complex(md: "ModelConf"
                           , ds: pd.DataFrame
                           , model_rep
                           , var_key: str  # as in options
                           , var_name: str  # as in dataset
                           , val: float | int
                           , ds_mask: pd.Series
                           ):
    """
    National TV sponsor: complex TV translation with respect to affinities and TV audience sizes.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    ds = tv_complex(md, ds, model_rep, var_key, var_name, val, ds_mask, clip_dur=5, tv_type='sponsor_nat', media_prefix='ots')

    return ds


def brnd_nat_tv_complex(md: "ModelConf"
                        , ds: pd.DataFrame
                        , model_rep
                        , var_key: str  # as in options
                        , var_name: str  # as in dataset
                        , val: float | int
                        , ds_mask: pd.Series
                        ):
    """
    National TV extended: complex TV translation with respect to affinities and TV audience sizes.

    Translation interface is fixed among all translation functions, so some parameters could look "excessive"
    in some functions. Do not remove them!

    :param md: ModelConf object (part of Model object, used here for performance efficiency)
    :param ds: dataset
    :param model_rep: ModelRep object (export objects).
    :param var_key: media tool key, as in options. E.g. "dgt_olv"
    :param var_name: variable name to translate to, as in dataset `ds`. E.g. "digital_olv_imp"
    :param val: value to set to `var_name` over filtered part of dataset
    :param ds_mask: mask to filter dataset to apply `val`
    :return: main dataset with applied `val`
    """

    ds = tv_complex(md, ds, model_rep, var_key, var_name, val, ds_mask, clip_dur=10, tv_type='rolik_nat', media_prefix=['ots', 'tmp_ots'])

    return ds


