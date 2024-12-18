"""

"""


import logging
import lzma
import os
import numpy as np
import pandas as pd
import re

from line_profiler_pycharm import profile

from fermatrica_utils import pandas_tree_final_child, StableClass, listdir_abs
from fermatrica import fermatrica_error, FermatricaError, model_load_ext


class MetaModel(StableClass):
    """
    MetaModel keeps together number of models. Models are connected as a chain now and to be connected in the future
    as a graph.
    """

    _path: str
    conf: "MetaModelConf"
    obj: "MetaModelObj"

    def __init__(
            self,
            path: str | None = None,
            model_graph_ds: pd.DataFrame | None = None,
            dss: dict | None = None,
            models: dict | None = None,
            model_reps: dict | None = None,
            if_stable: bool = True
    ):
        """
        Initialise instance

        """

        # check if folder is present

        if path is not None and not os.path.exists(path):
            fermatrica_error(path + ': path to model folder is not found. ' +
                             'To create model from scratch pass `None` as path argument')

        self._path = path

        # load config

        if self._path is not None:

            model_files = listdir_abs(self._path)

            if len([x for x in model_files if os.path.basename(x) in ['meta_conf.xlsx', 'meta_model_conf.xlsx']]) > 0:

                meta_conf_path = [x for x in model_files if os.path.basename(x) in ['meta_conf.xlsx', 'meta_model_conf.xlsx']][0]
                self.conf = MetaModelConf(path=meta_conf_path)

            else:
                msg = "Selected metamodel doesn't contain metamodel definition file"
                fermatrica_error(msg)

        else:
            self.conf = MetaModelConf(model_graph_ds=model_graph_ds)

        # load objects

        self.obj = MetaModelObj(meta_model_conf=self.conf
                                , dss=dss
                                , models=models
                                , model_reps=model_reps
                                , if_stable=if_stable)

        # prevent from creating new attributes
        # use instead of __slots__ to allow dynamic attribute creation during initialisation
        if if_stable:
            self._init_finish()


class MetaModelConf(StableClass):
    """
    MetaModelConf saves info about metamodel structure: how different nodes are connected as directed graph
    """

    path: str
    model_graph: pd.DataFrame
    start_nodes: list
    finish_nodes: list
    nodes_order: list

    def __init__(
            self,
            path: str | None = None,
            model_graph_ds: pd.DataFrame | None = None,
            if_stable: bool = True
    ):
        """
        Initialise instance

        """

        if path is not None and not os.path.exists(path):
            fermatrica_error(path + ': path to model definition file or model folder is not found. ' +
                             'To create model from scratch pass `None` as path argument')

        # save path to use in MetaModelObj

        self.path = path

        # read / set model graph

        if self.path is not None:
            self.conf_read()
        else:
            self.model_graph = model_graph_ds

        # additional

        self.set_order()

        if if_stable:
            self._init_finish()


    def conf_read(self):

        try:
            self.model_graph = pd.read_excel(self.path
                                             , sheet_name='model_graph'
                                             , engine='openpyxl'
                                             , dtype=str)

        except Exception as err:
            msg = "Error in reading meta_conf file: " + str(err)
            raise FermatricaError(msg)


    def set_order(self):

        start_nodes = self.model_graph['left'].unique()
        finish_nodes = self.model_graph['right'].unique()

        self.start_nodes = [x for x in start_nodes if x not in finish_nodes]
        self.finish_nodes = [x for x in finish_nodes if x not in start_nodes]

        # maybe change later to more sophisticated approach
        self.model_graph['order']= self.model_graph.index
        self.model_graph.sort_values(by='order', inplace=True)

        self.nodes_order = []

        for node in self.model_graph['left']:
            if node not in self.nodes_order:
                self.nodes_order.append(node)

        for node in self.model_graph['right']:
            if node not in self.nodes_order:
                self.nodes_order.append(node)


class MetaModelObj(StableClass):
    """

    """

    dss: dict | None
    models: dict | None
    model_reps: dict | None

    def __init__(
            self,
            meta_model_conf: "MetaModelConf",
            dss: pd.DataFrame | None = None,
            models: dict | None = None,
            model_reps: dict | None = None,
            if_stable: bool = True
    ):
        """
        Initialise instance

        """

        if meta_model_conf.path is not None:
            self.load_node(meta_model_conf)
        else:
            self.dss = dss
            self.models = models
            self.model_reps = model_reps

        if self.dss is None:
            fermatrica_error('Error in loading MetaModel objects: missed data')
        elif self.models is None:
            fermatrica_error('Error in loading MetaModel objects: missed models')
        elif self.model_reps is None:
            fermatrica_error('Error in loading MetaModel objects: missed reporting settings (ModelRep) objects')

        if if_stable:
            self._init_finish()


    def load_node(self,
                  meta_model_conf: "MetaModelConf"):
        """

        """

        self.dss = {}
        self.models = {}
        self.model_reps = {}

        for node in meta_model_conf.nodes_order:

            path_current = os.path.join(os.path.dirname(meta_model_conf.path), node)
            if not os.path.exists(path_current):
                fermatrica_error(path_current + ': path to model folder is not found. ' +
                                 'To create model from scratch pass `None` as path argument')

            self.models[node], self.dss[node], return_state = model_load_ext(path_current, missed_stop=False)

            pth = os.path.join(path_current, 'model_rep.pkl.lzma')
            if os.path.exists(pth):
                with lzma.open(pth, 'rb') as handle:
                    self.model_reps[node] = pd.compat.pickle_compat.load(handle)
            else:
                fermatrica_error('ModelRep objects were not provided for MetaModel.' + \
                                 'As for now explicit ModelRep are reqired if model is defined via MetaModel')

