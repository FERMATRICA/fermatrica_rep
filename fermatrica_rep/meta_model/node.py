"""
ModelNode — unit of computation bundling (Model, ds, ModelRep).

A ModelNode represents one level in the MetaModel DAG.
"""


import inspect
import logging
import os
import tomllib

import pandas as pd

from fermatrica_utils import import_module_from_string, read_pickle_compat, StableClass
from fermatrica import (
    fermatrica_error,
    model_load_ext,
    predict_ext,
    transform,
)
from fermatrica.model.model_conf import ModelConf

from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.options.scope import OptionScope, find_scope
from fermatrica_rep.options.translators import *


# Translators referenced by ``trans_dict`` but not resolvable in this process.
# Warned once per unique name so multi-option / budget-grid runs stay quiet
# while still surfacing the misconfiguration.
_warned_missing_translators: set[str] = set()

# Columns created by a translator call that were not present in ``ds`` before it.
# Translation is expected to overwrite an existing model-input column; a brand-new
# column means the written value is likely never consumed by the model (``ds.loc[
# mask, col] = ...`` silently appends a column full of NaN when ``col`` is absent).
# Warned once per (budget item, column) so multi-option / budget-grid runs stay
# quiet while still surfacing the misconfiguration.
_warned_new_translation_cols: set[tuple[str, str]] = set()


def _pack_results(results: list, names: list[str] | None | bool):
    """Pack worker results into dict, list, or single value.

    ``names=True`` forces list return (backward compat for old list API).
    ``names=None`` unwraps to single if ``len(results) == 1``.
    ``names=list[str]`` returns ``dict`` keyed by name.
    """
    if names is True:
        return results
    if names is not None:
        return dict(zip(names, results))
    if len(results) == 1:
        return results[0]
    return results


def _get_price_var(model_conf: ModelConf) -> str | None:
    """
    Extract the price variable name from a ModelConf, if defined.

    Returns ``None`` for models without a price variable (volume-only
    models). The price variable is used to ensure the column exists in
    the dataset and to compute value-based metrics alongside volume
    in summaries.

    Parameters
    ----------
    model_conf : ModelConf
        Model configuration object, typically ``model.conf``.

    Returns
    -------
    str or None
        The price variable column name, or ``None`` if not configured.
    """
    if (
        hasattr(model_conf, "price_var")
        and pd.notna(model_conf.price_var)
        and model_conf.price_var != ""
    ):
        return model_conf.price_var
    return None


class ModelNode(StableClass):
    """
    Bundles one model level: (Model, ds, ModelRep).

    Provides `compute()` which iterates scopes, translates budgets into media
    variables, runs transform + predict, and returns summaries.

    Optional ``scope_names`` restricts which scopes this node participates in
    during ``translate()`` / ``compute_after_translate()`` — useful when
    different nodes in a MetaModel use disjoint scope sets.
    """

    model: object
    ds: pd.DataFrame
    model_rep: ModelRep
    dt_pred: pd.DataFrame | None
    scope_names: set[str] | None = None
    static: bool = False
    _display_only: bool = False

    def __init__(
        self,
        model=None,
        ds: pd.DataFrame = None,
        model_path: str | None = None,
        options_path: str | None = None,
        model_rep: ModelRep | None = None,
        dt_pred: pd.DataFrame | None = None,
        missed_stop: bool = False,
        scope_names: set[str] | None = None,
        static: bool = False,
        _display_only: bool = False,
        **model_rep_kwargs,
    ):
        """
        Initialise a ModelNode.

        Bundles a trained model, its dataset, and reporting settings into
        a single computation unit. ModelRep may be passed explicitly or
        auto-constructed from keyword arguments (``language``,
        ``trans_dict``, etc.).

        Parameters
        ----------
        model : object, optional
            Trained fermatrica model instance. Optional when ``model_path``
            is provided. Required when ``model_path`` is ``None``.
        ds : pd.DataFrame, optional
            Dataset used for transform and predict. Optional when
            ``model_path`` is provided and the model directory contains a
            saved dataset. When ``model_path`` is given, an explicit
            ``ds`` **overrides** the saved (post-transform) dataset — pass
            the raw data so the no-arg refresh (``node.compute()``)
            transforms from raw.
        model_path : str or None, optional
            Path to a saved model directory. When provided, the model and
            dataset are auto-loaded via ``model_load_ext``. Cannot be used
            together with ``model``.
        options_path : str or None, optional
            Path to an options XLSX file. When provided and
            ``model_rep.media_prices`` is unset, the optional ``media_prices``
            sheet is loaded automatically. An already supplied
            ``model_rep.media_prices`` value is preserved.
        missed_stop : bool, default False
            Propagated to ``model_load_ext`` when ``model_path`` is used.
        scope_names : set[str] or None, optional
            Names of scopes this node participates in. When ``None``
            (default), all scopes passed to ``translate()`` /
            ``compute_after_translate()`` apply. When set, only scopes whose
            ``scope_name`` is in this set are translated and summarized —
            others are skipped. Useful for multi-node MetaModels where
            different nodes use disjoint scope sets (e.g. per-target UBA vs
            per-region sales scopes).
        static : bool, default False
            Node whose model has no option-driven media RHS. Its
            ``translate()`` is a no-op and no ``trans_dict`` is required,
            but the node still computes per option — it may consume dynamic
            upstream predictions via DAG merges.
        _display_only : bool, default False
            Internal bypass for visual-only nodes that carry a ``ModelRep``
            but no model (constructed by ``_convert_to_metamodel`` for
            plot dispatch). Such nodes must never be computed.
        model_rep : ModelRep or None, optional
            Pre-built reporting configuration. If ``None``, one is created
            from ``**model_rep_kwargs`` using ``ModelRep(ds, **kwargs)``.
        dt_pred : pd.DataFrame or None, optional
            Pre-computed prediction DataFrame. If ``None``, prediction
            will be computed on first call to ``compute()``.
        **model_rep_kwargs
            Keyword arguments forwarded to ``ModelRep`` when
            ``model_rep`` is ``None``, e.g. ``language``, ``trans_dict``.

        Raises
        ------
        FermatricaError
            If both ``model_path`` and ``model`` are provided, if ``ds``
            cannot be resolved, or if neither a model nor ``model_path``
            is given (unless ``_display_only`` is set).
        """
        if model_path is not None and model is not None:
            fermatrica_error("Cannot specify both model_path and model")

        if model is None and model_path is None and not _display_only:
            fermatrica_error(
                "ModelNode requires a model — pass model= or model_path="
            )

        if model_path is not None:
            model, loaded_ds, _ = model_load_ext(model_path, missed_stop=missed_stop)
            if ds is None:
                ds = loaded_ds

        if ds is None:
            fermatrica_error("ds must be provided")

        self.model = model
        self.ds = ds.copy()
        self.dt_pred = dt_pred
        self.scope_names = set(scope_names) if scope_names is not None else None
        self.static = static
        self._display_only = _display_only
        if model_rep is not None:
            self.model_rep = model_rep
        else:
            self.model_rep = ModelRep(ds, **model_rep_kwargs)
        if options_path is not None and getattr(self.model_rep, "media_prices", None) is None:
            from fermatrica_rep.options.define import media_prices_read

            self.model_rep.media_prices = media_prices_read(options_path)
        self._init_finish()

    def __repr__(self) -> str:
        conf = getattr(self.model, 'conf', None)
        target = getattr(conf, 'Y_var', '?')
        mtype = getattr(conf, 'model_type', '?')
        ds_shape = f"({self.ds.shape[0]}, {self.ds.shape[1]})"
        scopes = f", scopes={sorted(self.scope_names)}" if self.scope_names is not None else ""
        static = ", static=True" if self.static else ""
        return f"ModelNode(target={target}, type={mtype}, ds={ds_shape}{scopes}{static})"

    def __str__(self) -> str:
        conf = getattr(self.model, 'conf', None)
        lines = ["ModelNode:"]
        lines.append(f"  target:  {getattr(conf, 'Y_var', '?')}")
        lines.append(f"  type:    {getattr(conf, 'model_type', '?')}")
        lines.append(f"  ds:      {self.ds.shape[0]} rows x {self.ds.shape[1]} cols")
        if self.static:
            lines.append("  static:  True (translation skipped)")
        if hasattr(self.model, '_path') and self.model._path:
            lines.append(f"  path:    {self.model._path}")
        if self.scope_names is not None:
            lines.append(f"  scopes:  {sorted(self.scope_names)}")
        lines.append(f"  model_rep: language={self.model_rep.language}, trans_dict={'present' if self.model_rep.trans_dict else '(none)'}")
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str, model_rep_path: str | None = None):
        """
        Load a ModelNode from a saved model directory on disk.

        Reads the model and dataset via ``model_load_ext`` and the
        ModelRep from ``model_rep.pkl.lzma`` in the same directory
        (or a custom path). ``model_rep.pkl.lzma`` is optional — when
        absent (modelling-stage saves) a default ModelRep is created
        from the dataset (English, palettes derived from ``ds``) and a
        warning is logged.

        Parameters
        ----------
        path : str
            Directory path containing the saved model artifacts
            (as produced by :meth:`ModelNode.save`).
        model_rep_path : str or None, optional
            Explicit path to the ``model_rep.pkl.lzma`` file. When
            ``None`` (default), looks for ``model_rep.pkl.lzma``
            inside ``path``.

        Returns
        -------
        ModelNode
            Fully initialised node with model, dataset, and ModelRep.
        """
        model, ds, _ = model_load_ext(path, missed_stop=False)

        if model_rep_path is None:
            model_rep_path = os.path.join(path, "model_rep.pkl.lzma")
        if os.path.exists(model_rep_path):
            model_rep = read_pickle_compat(model_rep_path)
        else:
            from fermatrica_rep.meta_model.meta_model import _auto_model_rep

            logging.warning(
                f"ModelRep not found at {model_rep_path} — using default ModelRep"
            )
            model_rep = _auto_model_rep(ds, None)

        node = cls(model=model, ds=ds, model_rep=model_rep)

        toml_path = os.path.join(path, "meta_conf.toml")
        if os.path.exists(toml_path):
            with open(toml_path, "rb") as handle:
                conf = tomllib.load(handle)
            settings = conf.get("node_settings", {})
        else:
            settings = {}

        if len(settings) == 1:
            entry = next(iter(settings.values()))
            if entry is not None and "scope_names" in entry:
                node.scope_names = set(entry["scope_names"])
            if entry is not None and "static" in entry:
                node.static = bool(entry["static"])

        return node


    def save(self, path: str = '', save_tag: str | None = None):
        """
        Save the node's model, dataset, and ModelRep to disk.

        Thin wrapper over ``model.save`` that passes ``self.ds`` and
        ``self.model_rep``, so both ``dt_p.pkl.zstd`` and
        ``model_rep.pkl.lzma`` are always written. This lets
        ``ModelNode.load`` / ``MetaModel.load`` restore the full reporting
        configuration.

        :param path: path to the up-directory, where model directory to be created
            (``''`` defaults to ``code_py/model_data``)
        :param save_tag: specific tag to name model directory; if None,
            time stamp to be used
        :return: path to the saved model directory
        """
        return self.model.save(self.ds, model_rep=self.model_rep, path=path, save_tag=save_tag)


    def compute(
        self,
        option=None,
        scopes: list[OptionScope] | None = None,
        allow_past: bool = False,
        if_exact: bool = False,
        *,
        return_full: bool = False,
    ) -> pd.DataFrame | dict[str, dict | tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Refresh the model fit, or translate budgets and derive summaries.

        When ``option`` is ``None`` (default), runs a single
        transform + predict on ``self.ds`` with ``set_start=False``
        (refresh — the initial ``transform(set_start=True)`` bootstrap
        remains a core modelling step) and stores the results on the node.
        ``scopes`` / ``return_full`` are ignored on this path.

        With an ``option``, translates budgets into media variables for all
        scopes, then runs a single transform + predict and derives per-scope
        summaries. Convenience method that chains :meth:`translate` and
        :meth:`compute_after_translate`. When used directly on a single
        ``ModelNode``, scopes are translated cumulatively — each scope inherits
        the previous scope's media values and adstock. This is the standard
        entry point; call the two sub-methods separately only when the caller
        needs to interleave custom operations between translation and compute.

        ``self.ds`` and ``self.dt_pred`` are always updated after the
        transform + predict, regardless of ``return_full``.

        Args:
            option: Option object with budget allocations keyed by scope name.
                Must provide ``iter_scopes()`` yielding ``(scope_name,
                budget_items)`` tuples. ``None`` runs the no-arg refresh path.
            scopes: List of ``OptionScope`` objects defining date ranges,
                entity filters, and other scope parameters. Scope names are
                matched against ``option.iter_scopes()`` during translation.
            allow_past: If ``True``, the scope mask built by
                ``OptionScope.build_mask`` includes historical periods
                (``listed`` ≤ 3). Default ``False`` restricts to ``listed ==
                4`` only.
            if_exact: If ``True``, skip long-term translation beyond the scope
                end date. Use for exact-period scenarios where post-scope media
                should not be projected forward.
            return_full: If ``True``, each scope entry is a 3-tuple
                ``(ds, dt_pred, summary)``. If ``False`` (default), each
                scope entry is just the ``summary`` dict. Follows the same
                convention as ``fermatrica.fit_predict``.

        Returns:
            When ``option`` is ``None``: the ``dt_pred`` DataFrame produced by
            the refresh (also stored on ``self.dt_pred``).

            With an ``option``: dict mapping ``{scope_name: summary}`` when
            ``return_full=False`` (default), or
            ``{scope_name: (ds, dt_pred, summary)}`` when ``return_full=True``.

            ``ds`` is the transformed dataset (a copy made after
            transform/predict — safe to inspect without mutating
            ``self.ds``). ``dt_pred`` is the full prediction DataFrame
            (shared across all scopes — only the per-scope summary slices
            differ). ``summary`` is a dict with keys such as
            ``pred_exact_vol``, ``pred_long_vol``, ``growth_vol``, etc.
        """
        if option is None:
            return self._refresh()
        self.translate(option, scopes, allow_past=allow_past, if_exact=if_exact)
        return self.compute_after_translate(option, scopes, return_full=return_full)

    def _refresh(self) -> pd.DataFrame:
        """
        Run transform (``set_start=False``) + predict and store the result.

        Uses the node's current ``ds`` as the transform input, then stores
        the transformed dataset and prediction back on the node. Returns the
        ``dt_pred`` DataFrame.

        ``_check_price_var`` / ``_load_adhoc_code`` are idempotent — safe to
        re-run even when already executed during :meth:`translate`.
        """
        if not getattr(self.model.obj, 'models', None) or 'main' not in self.model.obj.models:
            fermatrica_error(
                "Model is not fitted — run `fm.fit_predict(ds, model, if_fit=True)` "
                "before `node.compute()`"
            )
        self._check_price_var(self.model.conf, self.ds)
        self._load_adhoc_code()
        model, ds = transform(ds=self.ds, model=self.model, set_start=False, if_by_ref=False)
        dt_pred = predict_ext(model, ds)
        self.ds = ds
        self.dt_pred = dt_pred
        return dt_pred

    def translate(
        self,
        option,
        scopes: list[OptionScope] | None = None,
        allow_past: bool = False,
        if_exact: bool = False,
    ):
        """
        Apply per-scope budget translations and media seasonality into
        ``self.ds``.

        Iterates scopes sorted by ``OptionScope.date_start`` (not by the
        order returned by ``option.iter_scopes()``), applying each scope's
        seasonality correction and translating budget values into media
        variables directly on ``self.ds``. No copies are made between scopes
        — later scopes **cumulatively inherit** the translated media values
        and adstock carryover from earlier scopes.

        Sorting by start date matters for ``_translate_long``: each scope's
        budget is projected forward beyond its ``date_end``, so a scope with
        an earlier period would otherwise overwrite the media of a later
        scope when its ``_translate_long`` runs after it. Processing scopes
        chronologically ensures the latest (target) scope's translation and
        long-term projection win.

        No transform or predict is performed (fast, mutations only). Designed
        as Phase 1 of :meth:`MetaModel.compute` so predecessor ``dt_pred``
        merges happen before any heavy computation.

        Raises ``FermatricaError`` at computation time when the option
        carries budgets for the node's scopes but ``model_rep.trans_dict``
        is not set — budgets would otherwise silently remain untranslated.
        Empty budget items are allowed without a ``trans_dict``.

        Args:
            option: Option object with budget allocations keyed by scope name.
            scopes: List of ``OptionScope`` objects. Each scope is looked up
                by name; unmatched scope entries in the option are silently
                skipped.
            allow_past: Forwarded to ``OptionScope.build_mask``. When
                ``False`` (default), only future rows (``listed == 4``)
                receive translated values.
            if_exact: If ``True``, skip :meth:`_translate_long` — budget
                values are applied only to the exact scope period, not
                projected forward.

        For a static node this is a no-op: its model has no media RHS, so
        budgets and seasonality do not apply to it.
        """
        if self.static:
            return
        model_conf = self.model.conf
        self._check_price_var(model_conf, self.ds)
        self._load_adhoc_code()

        bdg_ssn_cols = self._build_bdg_ssn_cols()

        # Sort scopes chronologically by their start date so that a scope
        # translated later (with a later date_start) is not overwritten by
        # _translate_long of an earlier-ending scope processed after it.
        scopes_sorted: list[tuple[OptionScope, dict[str, float]]] = []

        for scope_name, budget_items in option.iter_scopes():
            if self.scope_names is not None and scope_name not in self.scope_names:
                continue
            scope = self._find_scope(scopes, scope_name)
            if scope is None:
                logging.warning(
                    f"Scope '{scope_name}' not found in provided scopes — skipping"
                )
                continue
            scopes_sorted.append((scope, budget_items))

        scopes_sorted.sort(key=lambda x: x[0].date_start)

        if self.model_rep.trans_dict is None and any(
            budget_items for _, budget_items in scopes_sorted
        ):
            missing_scopes = [
                s.scope_name for s, budget_items in scopes_sorted if budget_items
            ]
            fermatrica_error(
                "No translation dictionary (trans_dict) is set on the node, but "
                f"budgets are provided for scope(s): {', '.join(missing_scopes)}. "
                "Budgets cannot be translated into media variables. "
                "Wire a translation sheet via MetaModel(compute_context=...) "
                "or set node.model_rep.trans_dict before compute()."
            )

        for scope, budget_items in scopes_sorted:
            self.ds = self._apply_media_ssn(self.ds, scope, bdg_ssn_cols)
            ds_mask = scope.build_mask(self.ds, allow_past=allow_past)
            self.ds = self._translate_budget(self.ds, model_conf, budget_items, ds_mask, scope)

            if not if_exact:
                self.ds = self._translate_long(self.ds, model_conf, budget_items, scope, allow_past)

    def compute_after_translate(
        self,
        option,
        scopes: list[OptionScope] | None = None,
        *,
        return_full: bool = False,
    ) -> dict[str, dict | tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Run a single transform + predict on ``self.ds``, then derive
        per-scope summaries from the shared prediction.

        Assumes ``self.ds`` has already been mutated by :meth:`translate`
        (or equivalent cumulative translation). Runs ``transform`` and
        ``predict_ext`` exactly once, then slices the single ``dt_pred``
        via each scope's ``date_start`` / ``date_end`` and entity filters
        to produce per-scope summaries.

        Designed as Phase 2 of :meth:`MetaModel.compute` — called after
        all predecessor ``dt_pred`` values have been merged via edge-specific
        merge functions.

        ``self.ds`` and ``self.dt_pred`` are always updated, regardless of
        ``return_full``.

        Args:
            option: Option object. Only used for ``iter_scopes()`` to
                enumerate scope names and budget items for summarization.
            scopes: List of ``OptionScope`` objects defining date ranges,
                entity filters, and reference periods for summaries.
            return_full: If ``True``, each scope entry is a 3-tuple
                ``(ds, dt_pred, summary)``. If ``False`` (default), each
                scope entry is just the ``summary`` dict.

        Returns:
            dict mapping ``{scope_name: summary}`` when ``return_full=False``
            (default), or ``{scope_name: (ds, dt_pred, summary)}`` when
            ``return_full=True``.

            ``ds`` is a **copy** of the transform output (safe to inspect).
            ``dt_pred`` is the full prediction DataFrame (identical for
            every scope — shared reference). ``summary`` is a dict with
            keys ``pred_exact_vol``, ``pred_long_vol``,
            ``pred_ref_vol``, ``growth_vol``, and their ``_val``
            counterparts when a price variable is defined.
        """
        dt_pred = self._refresh()
        ds = self.ds

        results = {}
        model_conf = self.model.conf
        for scope_name, budget_items in option.iter_scopes():
            if self.scope_names is not None and scope_name not in self.scope_names:
                continue
            scope = self._find_scope(scopes, scope_name)
            if scope:
                summary = self._summarize(model_conf, budget_items, scope, dt_pred)
                if return_full:
                    results[scope_name] = (ds.copy(), dt_pred, summary)
                else:
                    results[scope_name] = summary

        return results

    def _find_scope(
        self,
        scopes: list[OptionScope] | None,
        scope_name: str,
    ) -> OptionScope | None:
        """
        Look up an OptionScope by name from a list of scopes.

        Used during translation to match budget items (keyed by scope
        name) to their corresponding OptionScope objects.

        Parameters
        ----------
        scopes : list[OptionScope] or None
            List of scopes to search. ``None`` returns ``None``
            immediately (caller handles the missing scope warning).
        scope_name : str
            Scope name to match against
            :attr:`OptionScope.scope_name`.

        Returns
        -------
        OptionScope or None
            The matching scope, or ``None`` if not found or scopes
            is ``None``.
        """
        return find_scope(scopes, scope_name)

    def _check_price_var(self, model_conf: ModelConf, ds: pd.DataFrame):
        """
        Ensure the price variable column exists in the dataset.

        If the model defines a price variable (via
        ``model_conf.price_var``) but the column is missing from the
        dataset, this method adds it with a default value of ``1.0`` and
        logs a warning. This guards against downstream errors in summary
        computation when deriving value-based metrics.

        Parameters
        ----------
        model_conf : ModelConf
            Model configuration object.
        ds : pd.DataFrame
            Dataset to check and potentially modify in place.
        """
        price_var = _get_price_var(model_conf)
        if price_var is not None and price_var not in ds.columns:
            ds[price_var] = 1.0
            logging.warning(
                f"Price variable '{price_var}' not in dataset — defaulting to 1.0"
            )

    def _apply_media_ssn(
        self, ds: pd.DataFrame, scope: OptionScope, bdg_ssn_cols: set[str]
    ) -> pd.DataFrame:
        """
        Apply media seasonality coefficients to the dataset for a given
        scope.

        Looks up the seasonality table from ``model_rep.ssn_dict`` using
        ``scope.seasonality`` as the key (falling back to
        ``scope.scope_name``). Calls ``media_ssn_apply`` to multiply
        media variables by their seasonal coefficients within the scope's
        date range.

        Does nothing if no seasonality dictionary is available or if the
        lookup key is not found.

        Parameters
        ----------
        ds : pd.DataFrame
            Dataset to modify in place. Media columns are multiplied by
            seasonal coefficients.
        scope : OptionScope
            Scope providing the date range and seasonality lookup key.

        Returns
        -------
        pd.DataFrame
            Reference to the same ``ds`` object (mutated in place for
            performance — the caller receives it for chaining).
        """
        if (
            hasattr(self.model_rep, "ssn_dict")
            and pd.notna(self.model_rep.ssn_dict)
            and len(self.model_rep.ssn_dict) > 0
        ):
            from fermatrica_rep.options.define import media_ssn_apply

            lookup_key = scope.seasonality if scope.seasonality else scope.scope_name
            if lookup_key in self.model_rep.ssn_dict:
                dt_ssn = self.model_rep.ssn_dict[lookup_key]
                ds = media_ssn_apply(
                    ds=ds,
                    dt_ssn=dt_ssn,
                    date_start=scope.date_start,
                    date_end=scope.date_end,
                    bdg_ssn_cols=bdg_ssn_cols,
                )
        return ds

    def _build_bdg_ssn_cols(self) -> set[str]:
        """
        Build a set of bare seasonality column names (without prefix)
        that correspond to budget items.

        Derived from ``model_rep.trans_dict`` — each key starting with
        ``bdg_`` is stripped to its bare name (e.g. ``bdg_tv`` → ``tv``).
        """

        trans_dict = self.model_rep.trans_dict
        if trans_dict is None:
            return set()

        bdg_cols: set[str] = set()
        for k in trans_dict.get("function", {}):
            if k.startswith("bdg_"):
                bdg_cols.add(k[4:])
        return bdg_cols

    def _load_adhoc_code(self):
        """
        Import adhoc user modules registered on
        ``model_rep.adhoc_code_src``.

        Adhoc modules may define custom translator functions
        (``adhoc_*``) or merge functions that are referenced by name in
        the translation dictionary or model graph. These must be imported
        into the current module's namespace before dispatch.

        Idempotency is keyed on the process-local module globals, not on
        per-node instance state: ``import_module_from_string`` registers the
        full dotted module name as a key in this module's ``globals()``, so
        ``k in globals()`` means "already imported in this process". Gating on
        the globals keeps the check correct when a node crosses a process
        boundary — a spawn Pool worker unpickles the node but starts with a
        fresh node.py namespace, so adhoc modules must be re-imported there.

        The imported source is this node's pickled snapshot (``adhoc_code_src``
        captured at ``ModelRep`` construction), not the live project adhoc
        files. Nodes of one MetaModel may have been saved at different
        development stages, so the same adhoc module can be imported from a
        different (older) source than the current ``code_py/adhoc/`` code —
        check that saved sources are up to date in all nodes before relying on
        a loaded model.

        Does nothing if ``adhoc_code_src`` is empty or unset.
        """
        if (
            pd.notna(self.model_rep.adhoc_code_src)
            and len(self.model_rep.adhoc_code_src) > 0
        ):
            for k, v in self.model_rep.adhoc_code_src.items():
                if k in globals():
                    continue
                fr_cur_name = inspect.currentframe().f_globals["__name__"]
                import_module_from_string(k, v, fr_cur_name)

    def _translate_budget(
        self,
        ds: pd.DataFrame,
        model_conf: ModelConf,
        budget_items: dict[str, float],
        ds_mask: pd.Series,
        scope: OptionScope | None = None,
    ) -> pd.DataFrame:
        """
        Dispatch budget items to their registered translator functions.

        For each budget item (e.g. ``"bdg_tv": 50``):

        1. Looks up the translator function name and target variable
           from ``model_rep.trans_dict``.
        2. Resolves the effective media price by combining the base
           price from ``trans_dict`` with per-scope or per-tool
           inflation multipliers.
        3. Calls the translator function via ``eval()``, passing the
           computed ``media_price`` and the scope mask.

        Budget items not registered in ``trans_dict["function"]`` are skipped.
        Items whose translator functions cannot be resolved are skipped with a
        one-time warning (per unique function name, per process).

        Parameters
        ----------
        ds : pd.DataFrame
            Dataset to modify in place. Media columns receive budget
            values on rows where ``ds_mask`` is ``True``.
        model_conf : ModelConf
            Model configuration (passed through to translator functions).
        budget_items : dict[str, float]
            Budget items keyed by item name, e.g.
            ``{"bdg_tv": 50.0}``. Values are in millions.
        ds_mask : pd.Series
            Boolean mask identifying rows to apply budgets to. Built by
            :meth:`OptionScope.build_mask`.
        scope : OptionScope or None, optional
            Scope used to resolve per-tool inflation multipliers from
            ``media_prices``. When ``None``, only the base price is used.

        Returns
        -------
        pd.DataFrame
            Reference to ``ds`` (mutated in place).
        """
        trans_dict = self.model_rep.trans_dict
        if trans_dict is None:
            return ds

        media_prices = getattr(self.model_rep, "media_prices", {})
        if media_prices is None:
            media_prices = {}

        cols_known = set(ds.columns)

        for k, val in budget_items.items():
            if k not in trans_dict.get("function", {}):
                continue

            fun_name = trans_dict["function"][k]
            var_name = trans_dict["variable"][k]

            if (
                fun_name not in locals()
                and fun_name not in globals()
                and not (
                    len(fun_name.split(".")) > 1
                    and getattr(
                        globals().get(".".join(fun_name.split(".")[:-1]), None),
                        fun_name.split(".")[-1],
                        False,
                    )
                )
            ):
                if fun_name not in _warned_missing_translators:
                    _warned_missing_translators.add(fun_name)
                    logging.warning(
                        f"Translator '{fun_name}' for budget item '{k}' not found "
                        f"in node module globals — skipping budget item"
                    )
                continue

            base_price = trans_dict.get("base_price", {}).get(k, 1.0)
            if scope is not None:
                scope_prices = media_prices.get(scope.scope_name, {})
                multiplier = scope_prices.get(k, scope.inflation_multiplier)
                final_price = base_price * multiplier
            else:
                final_price = base_price

            ds = eval(fun_name)(model_conf, ds, self.model_rep, k, var_name, val, ds_mask, media_price=final_price)

            cols_now = set(ds.columns)
            new_cols = cols_now - cols_known
            if new_cols:
                for col in sorted(new_cols):
                    key = (k, col)
                    if key not in _warned_new_translation_cols:
                        _warned_new_translation_cols.add(key)
                        logging.warning(
                            f"Translation of budget item '{k}' by translator "
                            f"'{fun_name}' created new column '{col}' "
                            f"(declared target variable: '{var_name}'). "
                            "Translation is expected to overwrite an existing "
                            "dataset column — if this new column is not read by "
                            "the model, the budget is silently unused."
                        )
            cols_known = cols_now

        return ds

    def _translate_long(
        self,
        ds: pd.DataFrame,
        model_conf: ModelConf,
        budget_items: dict[str, float],
        scope: OptionScope,
        allow_past: bool,
    ) -> pd.DataFrame:
        """
        Apply the same budget items beyond the scope end date into the
        forecast horizon.

        Builds a mask from ``scope.date_end + 1 day`` to the last
        future row in the dataset and delegates to
        :meth:`_translate_budget`. This enables long-term forecasting
        where a one-year budget scenario is projected forward through
        all available future periods.

        Parameters
        ----------
        ds : pd.DataFrame
            Dataset to modify in place.
        model_conf : ModelConf
            Model configuration.
        budget_items : dict[str, float]
            Same budget items used for the exact-period translation.
        scope : OptionScope
            The scope whose ``date_end`` defines the long-term start.
        allow_past : bool
            Forwarded to the mask builder. Typically ``False``.

        Returns
        -------
        pd.DataFrame
            Reference to ``ds`` (mutated in place).
        """
        long_start = scope.date_end + pd.DateOffset(1)
        long_end = ds.loc[ds["listed"] == 4, "date"].max()
        ds_mask = (ds["date"] >= long_start) & (ds["date"] <= long_end)
        for col, values in scope.filters.items():
            if col in ds.columns:
                ds_mask = ds_mask & ds[col].isin(values)
        if not allow_past:
            ds_mask = ds_mask & ds["listed"].isin([4])
        ds = self._translate_budget(ds, model_conf, budget_items, ds_mask, scope)

        return ds

    def _summarize(
        self,
        model_conf: ModelConf,
        budget_items: dict[str, float],
        scope: OptionScope,
        dt_pred: pd.DataFrame,
    ) -> dict:
        """
        Derive per-scope summary metrics from a shared prediction.

        Computes three slices of ``dt_pred`` by intersecting date ranges
        with the scope's entity filter:

        - **Exact period**: ``date_start`` -> ``date_end``. KPI aggregated
          via ``sum`` or ``mean`` per ``model_conf.summary_type``.
        - **Long period**: ``date_start`` -> last future date. Used for
          long-term forecast metrics when the scope end is before the
          data horizon. Falls back to exact if scope already covers the
          full horizon.
        - **Reference period**: ``ref_date_start`` -> ``ref_date_end``.
          Used to compute growth (forecast / reference).

        Both volume (``predicted``) and value (``predicted * price_var``)
        variants are produced when a price variable is configured.

        Reference aggregates (``pred_ref_*``, ``ref_*``) follow the same
        ``summary_type`` aggregation as the exact period, so growth
        compares like with like.

        Growth is computed as ``pred_exact / ref`` for both volume and
        value, with a safety floor of ``1e-10`` to avoid division by zero.

        If ``model_rep.option_summary_adhoc`` is set, the summary is
        replaced by the adhoc function's return value.

        Parameters
        ----------
        model_conf : ModelConf
            Model configuration (provides ``summary_type`` and
            ``price_var``).
        budget_items : dict[str, float]
            Budget items for this scope (passed to adhoc summary).
        scope : OptionScope
            Scope defining the date ranges and entity filter.
        dt_pred : pd.DataFrame
            Full prediction DataFrame with columns ``date``, ``listed``,
            ``predicted``, ``observed``, and any entity filter columns.

        Returns
        -------
        dict
            Summary with keys:
            ``pred_exact_vol``, ``pred_exact_val`` (if price var exists),
            ``pred_long_vol``, ``pred_long_val`` (if price var exists),
            ``pred_ref_vol``, ``pred_ref_val`` (if price var exists),
            ``ref_vol``, ``ref_val`` (if price var exists),
            ``growth_vol``, ``growth_val`` (if price var exists).
        """
        summary_type = getattr(model_conf, "summary_type", "sum")
        date_max = dt_pred["date"].max()
        date_obs_max = dt_pred.loc[dt_pred["listed"].isin([2, 3]), "date"].max()

        summary: dict = {}

        entity_mask = scope.build_entity_mask(dt_pred)

        exact_mask = entity_mask & (
            (dt_pred["date"] >= scope.date_start)
            & (dt_pred["date"] <= scope.date_end)
            & (dt_pred["listed"].isin([2, 3, 4]))
        )

        if summary_type == "fin":
            date_end = dt_pred.loc[dt_pred["listed"].isin([2, 3, 4]), "date"].unique()
            dlt = abs(date_end - scope.date_end)
            ind = dlt.argmin()
            date_end = date_end[ind]
            exact_mask = exact_mask & (dt_pred["date"] == date_end)

        if summary_type in ("mean", "mean_fin"):
            summary["pred_exact_vol"] = dt_pred.loc[exact_mask, "predicted"].mean()
        else:
            summary["pred_exact_vol"] = dt_pred.loc[exact_mask, "predicted"].sum()

        price_var = _get_price_var(model_conf)
        if price_var is not None:
            price_val = dt_pred.loc[exact_mask, "predicted"] * dt_pred.loc[exact_mask, model_conf.price_var]
            if summary_type in ("mean", "mean_fin"):
                summary["pred_exact_val"] = price_val.mean()
            else:
                summary["pred_exact_val"] = price_val.sum()

        if scope.date_end < date_max:
            long_mask = entity_mask & (
                (dt_pred["date"] >= scope.date_start)
                & (dt_pred["listed"].isin([2, 3, 4]))
            )
            if summary_type in ("mean", "mean_fin"):
                summary["pred_long_vol"] = dt_pred.loc[long_mask, "predicted"].mean()
            else:
                summary["pred_long_vol"] = dt_pred.loc[long_mask, "predicted"].sum()

            if price_var is not None:
                price_val = dt_pred.loc[long_mask, "predicted"] * dt_pred.loc[long_mask, model_conf.price_var]
                if summary_type in ("mean", "mean_fin"):
                    summary["pred_long_val"] = price_val.mean()
                else:
                    summary["pred_long_val"] = price_val.sum()
        else:
            summary["pred_long_vol"] = summary["pred_exact_vol"]
            summary["pred_long_val"] = summary.get("pred_exact_val")

        ref_mask = entity_mask & (
            (dt_pred["date"] >= scope.ref_date_start)
            & (dt_pred["date"] <= scope.ref_date_end)
            & (dt_pred["listed"].isin([2, 3, 4]))
        )

        if summary_type in ("mean", "mean_fin"):
            summary["pred_ref_vol"] = dt_pred.loc[ref_mask, "predicted"].mean()
        else:
            summary["pred_ref_vol"] = dt_pred.loc[ref_mask, "predicted"].sum()

        if price_var is not None:
            price_val = dt_pred.loc[ref_mask, "predicted"] * dt_pred.loc[ref_mask, model_conf.price_var]
            if summary_type in ("mean", "mean_fin"):
                summary["pred_ref_val"] = price_val.mean()
            else:
                summary["pred_ref_val"] = price_val.sum()

        if scope.ref_date_end <= date_obs_max:
            if summary_type in ("mean", "mean_fin"):
                summary["ref_vol"] = dt_pred.loc[ref_mask, "observed"].mean()
            else:
                summary["ref_vol"] = dt_pred.loc[ref_mask, "observed"].sum()
            if price_var is not None:
                price_val = dt_pred.loc[ref_mask, "observed"] * dt_pred.loc[ref_mask, model_conf.price_var]
                if summary_type in ("mean", "mean_fin"):
                    summary["ref_val"] = price_val.mean()
                else:
                    summary["ref_val"] = price_val.sum()
        else:
            obs_ref_mask = ref_mask & (dt_pred["date"] <= date_obs_max)
            pred_ref_mask = ref_mask & (dt_pred["date"] > date_obs_max)
            ref_vol_series = pd.concat([
                dt_pred.loc[obs_ref_mask, "observed"],
                dt_pred.loc[pred_ref_mask, "predicted"],
            ])
            if summary_type in ("mean", "mean_fin"):
                summary["ref_vol"] = ref_vol_series.mean()
            else:
                summary["ref_vol"] = ref_vol_series.sum()
            if price_var is not None:
                ref_val_series = pd.concat([
                    dt_pred.loc[obs_ref_mask, "observed"] * dt_pred.loc[obs_ref_mask, model_conf.price_var],
                    dt_pred.loc[pred_ref_mask, "predicted"] * dt_pred.loc[pred_ref_mask, model_conf.price_var],
                ])
                if summary_type in ("mean", "mean_fin"):
                    summary["ref_val"] = ref_val_series.mean()
                else:
                    summary["ref_val"] = ref_val_series.sum()

        ref_vol = float(summary.get("ref_vol", 1.0))
        summary["growth_vol"] = float(summary["pred_exact_vol"]) / max(ref_vol, 1e-10)

        if "pred_exact_val" in summary and "ref_val" in summary:
            ref_val = float(summary["ref_val"])
            summary["growth_val"] = float(summary["pred_exact_val"]) / max(ref_val, 1e-10)

        if (
            pd.notna(self.model_rep.option_summary_adhoc)
            and self.model_rep.option_summary_adhoc is not None
        ):
            summary = self.model_rep.option_summary_adhoc(
                model_conf, self.model_rep, budget_items, scope, dt_pred
            )

        return summary
