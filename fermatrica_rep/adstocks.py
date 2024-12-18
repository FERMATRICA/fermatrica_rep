"""
Calculate adstock (decay, carry-over) for number of variables
"""

from fermatrica.model.model import Model
from fermatrica.model.transform import *

from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.options.calc import option_report


def adstocks_data(model: "Model | list"
                  , ds: pd.DataFrame | list
                  , model_rep: "ModelRep | list"
                  , superbrand: str
                  , cln_meas: list | tuple
                  , option_dict: dict | list = None
                  , option_settings: "OptionSettings" = None
                  , cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef')
                  , n: int = 50
                  , if_scaled: bool = True
                  ) -> pd.DataFrame:
    """
    Calculate data for adstocks plot: rate of decay per variable.
    Be careful selecting `cln_meas`: only variables immediately before decay transformation should be listed.


    :param model: Model object or list of Model objects
    :param ds: dataset or list of datasets
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param superbrand: umbrella brand name
    :param option_dict: budget option / scenario to calculate as dictionary or list of option dictionaries
    :param option_settings: OptionSettings object (option setting: target period etc.) or list of OptionSettings
    :param cln_meas: list of measurements columns names (to calculate adstock from)
    :param cln_dim: list of dimension columns names
    :param n: number of observations per column
    :param if_scaled: to scale dividing by MAX value per measurement column
    :return:
    """

    if isinstance(model, list) and not (option_dict and option_settings):
        fermatrica_error("Some of the ['option_set', 'option_settings'] agrs are not set. Please provide these parameters in the case of multi-level models")

    if isinstance(model_rep, list):

        dt_pred = [None] * len(model_rep)
        option_summary = [None] * len(model_rep)

        for i, val in enumerate(model_rep):

            # fuse output of previous model into the current one

            if i > 0:

                if pd.notna(model_rep[i].adhoc_code_src) and len(model_rep[i].adhoc_code_src) > 0:

                    for k, v in model_rep[i].adhoc_code_src.items():
                        fr_cur_name = inspect.currentframe().f_globals['__name__']
                        import_module_from_string(k, v, fr_cur_name)

                    fun_name = 'code_py.adhoc.reporting.pred_merge_' + str(i)

                    if (fun_name in locals() or fun_name in globals() or
                            (len(fun_name.split('.')) > 1 and getattr(globals()['.'.join(fun_name.split('.')[:-1])],
                                                                      fun_name.split('.')[-1], False))):
                        ds[i] = eval(fun_name)(ds_cur=ds[i]
                                               , dt_pred_prev=dt_pred[i - 1])

            ds[i], dt_pred[i], option_summary[i] = option_report(model=model[i]
                                                                 , ds=ds[i]
                                                                 , model_rep=model_rep[i]
                                                                 , option_dict=option_dict
                                                                 , option_settings=option_settings)

    if isinstance(model, list):
        model = model[-1]

    model = copy.deepcopy(model)
    model_conf = model.conf

    model_conf.params = model_conf.params[(model_conf.params['fun'].str.contains('adstock|dwblp')) &
                                          (model_conf.params['variable'].isin(cln_meas))].copy()

    if isinstance(ds, list):
        ds = ds[-1]

    tmp = ds.loc[(ds['superbrand'] == superbrand), cln_dim + cln_meas].copy()

    tmp = tmp.loc[tmp["bs_key"] == tmp["bs_key"].value_counts().reset_index(drop=False).bs_key[0]]
    tmp = tmp.iloc[:n].reset_index(drop=False)

    for col in cln_meas:
        tmp[col].values[:] = 0.0
        tmp[col].values[0] = 1.0

    not_used, tmp = transform(ds=tmp
                              , model=model
                              , set_start=False
                              , if_by_ref=False)

    df_tmp = tmp.copy()
    df_tmp = df_tmp.sort_values(by='date')
    df_tmp = df_tmp.loc[:, df_tmp.columns.str.contains('adstock|dwblp')]

    if if_scaled:
        for col in df_tmp.columns:
            df_tmp[col] = df_tmp[col] / df_tmp[col].max()

    return df_tmp
