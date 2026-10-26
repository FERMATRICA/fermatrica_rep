"""
Calculate adstock (decay, carry-over) for number of variables
"""

import copy

from fermatrica.model.model import Model
from fermatrica.model.transform import *

from fermatrica_rep.basics import _filter_df
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode, _pack_results
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel
from fermatrica_rep.meta_model.meta_model import MetaModel


def adstocks_data(model: "Model | ModelNode | MetaModel | list"
                  , ds: pd.DataFrame | list | None = None
                  , model_rep: "ModelRep | list | None" = None
                  , *, brands: list | None = None
                  , brands_var: str = 'superbrand'
                  , filters: dict | None = None
                  , cln_meas: list | tuple = ()
                  , cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef')
                  , n: int = 50
                  , if_scaled: bool = True
                  ) -> pd.DataFrame | list | dict:
    """
    Calculate data for adstocks plot: rate of decay per variable.
    Be careful selecting `cln_meas`: only variables immediately before decay transformation should be listed.


    :param model: Model, ModelNode, MetaModel, or list of Model objects (deprecated)
    :param ds: dataset or list of datasets (optional if ModelNode passed)
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects (optional if ModelNode passed)
    :param brands: simple filter values, applied to `brands_var`. When `brands`, `brands_var`
        and `filters` are all None, defaults to ``{'superbrand': [model.conf.target_superbrand]}``
        for each node (adstock impulse must be computed on one brand).
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param cln_meas: list of measurements columns names (to calculate adstock from)
    :param cln_dim: list of dimension columns names
    :param n: number of observations per column
    :param if_scaled: to scale dividing by MAX value per measurement column
    :return:
    """

    mm = _convert_to_metamodel(model, ds=ds, model_rep=model_rep)

    results_dict = {}
    for name, node in mm.nodes.items():
        if brands is None and filters is None:
            brand_filters = {'superbrand': [node.model.conf.target_superbrand]}
        else:
            brand_filters = filters
        results_dict[name] = _adstocks_data_worker_inner(
            model=node.model, ds=node.ds,
            brands=brands, brands_var=brands_var, filters=brand_filters,
            cln_meas=cln_meas,
            cln_dim=cln_dim, n=n, if_scaled=if_scaled)

    if isinstance(model, MetaModel):
        names = list(mm.nodes.keys())
    elif isinstance(model, list):
        names = True
    else:
        names = None
    return _pack_results(list(results_dict.values()), names)


def _adstocks_data_worker_inner(model, ds, brands, brands_var, filters, cln_meas, cln_dim, n, if_scaled):
    """
    Calculate adstock decay rates via impulse-response test on adstock/dwblp transforms.

    Creates a unit impulse (1.0 at t=0, 0 otherwise) in each ``cln_meas`` column,
    runs only the adstock and dwblp transform functions, and returns the decay trace.

    :param model: Model object (deep-copied internally; params filtered to adstock/dwblp only)
    :param ds: dataset providing column structure, superbrand, bs_key, and dates.
        Values in ``cln_meas`` columns are overwritten by the impulse.
    :param brands: simple filter values, applied to `brands_var`
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param cln_meas: media variable column names to calculate adstock for
    :param cln_dim: dimension columns preserved in the impulse dataframe
    :param n: number of time periods (rows) to trace
    :param if_scaled: if True, divide each column by its maximum value
    :return: DataFrame indexed by time step with adstock/dwblp output columns
    """
    model = copy.deepcopy(model)
    model_conf = model.conf

    model_conf.params = model_conf.params[(model_conf.params['fun'].str.contains('adstock|dwblp')) &
                                          (model_conf.params['variable'].isin(cln_meas))].copy()

    tmp = _filter_df(ds, brands=brands, brands_var=brands_var, filters=filters)
    tmp = tmp[list(cln_dim) + list(cln_meas)].copy()

    tmp = tmp.loc[tmp["bs_key"] == tmp["bs_key"].value_counts().reset_index(drop=False).bs_key[0]]
    tmp = tmp.iloc[:n].reset_index(drop=False)

    for col in cln_meas:
        tmp.loc[:, col] = 0.0
        tmp.loc[tmp.index[0], col] = 1.0

    not_used, tmp = transform(ds=tmp, model=model, set_start=False, if_by_ref=False)

    df_tmp = tmp.copy()
    df_tmp = df_tmp.sort_values(by='date')
    df_tmp = df_tmp.loc[:, df_tmp.columns.str.contains('adstock|dwblp')]

    if if_scaled:
        for col in df_tmp.columns:
            df_tmp[col] = df_tmp[col] / df_tmp[col].max()

    return df_tmp
