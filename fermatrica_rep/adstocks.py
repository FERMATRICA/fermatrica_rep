"""
Calculate adstock (decay, carry-over) for number of variables
"""


from fermatrica.model.model import Model
from fermatrica.model.transform import *


def adstocks_data(model: "Model"
                  , ds: pd.DataFrame
                  , superbrand: str
                  , cln_meas: list | tuple
                  , cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef')
                  , n: int = 50
                  , if_scaled: bool = True
                  ) -> pd.DataFrame:
    """
    Calculate data for adstocks plot: rate of decay per variable.
    Be careful selecting `cln_meas`: only variables immediately before decay transformation should be listed.


    :param model: Model object
    :param ds: dataset
    :param superbrand: umbrella brand name
    :param cln_meas: list of measurements columns names (to calculate adstock from)
    :param cln_dim: list of dimension columns names
    :param n: number of observations per column
    :param if_scaled: to scale dividing by MAX value per measurement column
    :return:
    """

    model = copy.deepcopy(model)
    model_conf = model.conf

    model_conf.params = model_conf.params[(model_conf.params['fun'].str.contains('adstock|dwblp')) &
                                                     (model_conf.params['variable'].isin(cln_meas))].copy()

    #

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
