"""
Model export settings. Define, _how_ to report: language, colours, export adhoc source code,
coefs, translation etc.

ModelRep is deliberately separated from Model object (_what_ to report).
"""


import inspect
import logging
import pandas as pd
import typing
from importlib import resources

from fermatrica_utils import StableClass

from fermatrica_rep.basics import palette_fill
from fermatrica_rep.options.calc import option_translate_long as otl


class ModelRep(StableClass):
    """
    Model Reporting settings.
    """

    vis_dict: pd.DataFrame
    language: str
    trans_dict: dict | None
    palette_sbr: dict
    palette_tools: dict
    adhoc_code_src: dict | None
    option_translate_long: typing.Callable
    option_summary_adhoc: typing.Callable | None
    aud_coefs: pd.DataFrame | None
    ssn_dict: dict | None
    media_prices: dict | None

    def __init__(
            self
            , ds: pd.DataFrame
            , vis_dict: pd.DataFrame | None = None
            , trans_dict: dict | None = None
            , language: str = 'english'
            , tools_list: list | tuple | None = None
            , palette_sbr_names: list | tuple = ('glasbey',)
            , palette_tool_names: list | tuple = ('glasbey',)
            , adhoc_code: list | None = None
            , option_translate_long_fn: "typing.Callable | None" = None
            , option_summary_adhoc_fn: "typing.Callable | None" = None
            , ssn_dict: dict | None = None
    ):
        """
        Initialise ModelRep instance.

        :param ds: main dataset
        :param vis_dict: visual dictionary (if None the default one is used)
        :param trans_dict: translation dictionary (how to "translate" options into independent variables)
        :param language: "english" and "russian" are supported via default vis_dict
        :param tools_list: list of marketing tools to be reported
        :param palette_sbr_names: `colorcet` palettes to be used for umbrella brands (string names)
        :param palette_tool_names: `colorcet` palettes to be used for marketing tools (string names)
        :param adhoc_code: adhoc code (Python loaded modules) required by model to be reported
        :param option_translate_long_fn: function to perform translation in long term (by default built-in
            function is used)
        :param option_summary_adhoc_fn: adhoc function to summarize options (by default built-in
            function is used)
        :param ssn_dict: seasonality dictionary keyed by scope name, used by translators
            to distribute budgets over time periods
        """

        if vis_dict is None:
            with resources.path("fermatrica_rep.res.dict", "vis_dict.xlsx") as xlsxio:
                vis_dict = pd.read_excel(xlsxio)

        if option_translate_long_fn is None:
            option_translate_long_fn = otl

        self.vis_dict = vis_dict
        self.set_language(language)
        self.fill_colours(ds, tools_list, palette_sbr_names, palette_tool_names)
        self.trans_dict = trans_dict
        self.fill_adhoc_code_src(adhoc_code)
        self.option_translate_long = option_translate_long_fn
        self.option_summary_adhoc = option_summary_adhoc_fn
        self.aud_coefs = None
        self.media_prices = None
        self.ssn_dict = ssn_dict

    def __repr__(self) -> str:
        trans_keys = list(self.trans_dict.get('function', {}).keys()) if self.trans_dict else []
        sbr_count = len(self.palette_sbr) if self.palette_sbr else 0
        tool_count = len(self.palette_tools) if self.palette_tools else 0
        has_ssn = self.ssn_dict is not None and len(self.ssn_dict) > 0 if self.ssn_dict else False
        has_mp = self.media_prices is not None and len(self.media_prices) > 0 if self.media_prices else False
        parts = [f"lang={self.language}"]
        if trans_keys:
            parts.append(f"trans={trans_keys}")
        parts.append(f"sbr={sbr_count}")
        parts.append(f"tools={tool_count}")
        if has_ssn:
            parts.append("ssn+")
        if has_mp:
            parts.append("mp+")
        if self.option_summary_adhoc is not None:
            parts.append("summary_adhoc+")
        return f"ModelRep({', '.join(parts)})"

    def __str__(self) -> str:
        lines = [f"ModelRep:"]
        lines.append(f"  language: {self.language}")
        lines.append(f"  vis_dict: {self.vis_dict.shape[0]} rows x {self.vis_dict.shape[1]} cols")
        if self.trans_dict:
            trans_fns = self.trans_dict.get('function', {})
            lines.append(f"  trans_dict: {len(trans_fns)} keys ({list(trans_fns.keys())})")
        else:
            lines.append(f"  trans_dict: (none)")
        lines.append(f"  palette_sbr: {len(self.palette_sbr) if self.palette_sbr else 0} colours")
        lines.append(f"  palette_tools: {len(self.palette_tools) if self.palette_tools else 0} colours")
        if self.adhoc_code_src:
            lines.append(f"  adhoc_code: {list(self.adhoc_code_src.keys())}")
        lines.append(f"  option_translate_long: {getattr(self.option_translate_long, '__name__', self.option_translate_long)}")
        if self.option_summary_adhoc is not None:
            lines.append(f"  option_summary_adhoc: {getattr(self.option_summary_adhoc, '__name__', self.option_summary_adhoc)}")
        if self.ssn_dict is not None and len(self.ssn_dict) > 0:
            lines.append(f"  ssn_dict: {len(self.ssn_dict)} keys ({list(self.ssn_dict.keys())})")
        if self.media_prices is not None and len(self.media_prices) > 0:
            lines.append(f"  media_prices: {len(self.media_prices)} keys ({list(self.media_prices.keys())})")
        return "\n".join(lines)

    def set_language(self
                     , language: str):
        """
        Set export language.

        :param language: "english" and "russian" are supported via default vis_dict
        :return: void
        """

        language = language.lower().strip()
        if language in ['english', 'russian']:
            self.language = language
        else:
            logging.warning('Language ' + language + ' is not supported. Reset to the default value: English')
            self.language = 'english'

    def fill_colours(self
                     , ds: pd.DataFrame
                     , tools_list: list | tuple = ()
                     , palette_sbr_names: list | tuple = ('glasbey',)
                     , palette_tool_names: list | tuple = ('glasbey',)
                     ):
        """
        Fill export colours with provided `colorcet` palettes.

        :param ds: main dataset
        :param tools_list: list of marketing tools to be reported
        :param palette_sbr_names: `colorcet` palettes to be used for umbrella brands (string name)
        :param palette_tool_names: `colorcet` palettes to be used for marketing tools (string name)
        :return: void
        """

        self.palette_sbr = {}
        if 'superbrand' in ds.columns:
            self.fill_colours_sbr(ds['superbrand'].unique(), palette_sbr_names)

        self.palette_tools = {}
        if pd.notna(tools_list):
            self.fill_colours_tools(tools_list, palette_tool_names)

    def fill_colours_sbr(self
                         , entities_new: dict
                         , palette_names: list | tuple = ('glasbey',)):
        """
        Fill superbrand (umbrella brand) names.

        :param entities_new: dictionary or list with elements to be checked with entities_old and added to it
            if not yet
        :param palette_names: colorcet palette names
        :return: void
        """

        if isinstance(palette_names, str):
            palette_names = [palette_names]

        self.palette_sbr = palette_fill(self.palette_sbr, entities_new, palette_names)

    def fill_colours_tools(self
                           , entities_new: dict | list
                           , palette_names: list | tuple = ('glasbey',)):
        """
        Fill marketing tool colour palette names.

        :param entities_new: dictionary or list with elements to be checked with entities_old and added to it
            if not yet
        :param palette_names: colorcet palette names
        :return: void
        """

        if isinstance(palette_names, str):
            palette_names = [palette_names]

        self.palette_tools = palette_fill(self.palette_tools, entities_new, palette_names)

    def fill_adhoc_code_src(self
                            , adhoc_code: list | None):
        """
        Extract source code from the adhoc modules and keep it as string dictionary.

        :param adhoc_code: adhoc code (Python loaded modules) required by model to be reported
        :return: void
        """

        self.adhoc_code_src = {}

        if isinstance(adhoc_code, list) and len(adhoc_code) > 0:

            for obj in adhoc_code:
                self.adhoc_code_src[obj.__name__] = inspect.getsource(obj)



