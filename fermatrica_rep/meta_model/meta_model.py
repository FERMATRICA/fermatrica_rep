"""
MetaModel — DAG-based multi-model container.

Keeps together multiple ModelNodes connected as a directed acyclic graph.
ModelNodes are executed in topological order. Predecessor outputs are merged
via edge-specific merge functions before a node runs.
"""


import datetime
import importlib
import inspect
import logging
import os
import re
import shutil
import tomllib
import typing
import warnings
from importlib import resources

import pandas as pd

from fermatrica_utils import (
    import_module_from_string,
    StableClass,
)
from fermatrica import fermatrica_error

from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.model_rep import ModelRep

_default_vis_dict: pd.DataFrame | None = None


class MetaModel(StableClass):
    """
    Multi-model DAG container.

    ``nodes`` accepts either a ``dict[str, ModelNode]`` (multi-node) or a
    single ``ModelNode`` (default name ``"model"``).  Pass ``edges`` when
    multiple nodes form a DAG with dependencies.

    ``edges`` is a list of ``(left, right, merge_fn)`` tuples where
    ``merge_fn`` is a callable, a string name (resolved at compute time), or
    ``None`` for parallel independent branches.
    """

    nodes: dict[str, ModelNode]
    edges: list[tuple[str, str, typing.Callable | str | None]]

    def __init__(
        self,
        nodes: dict[str, ModelNode] | ModelNode,
        edges: list[tuple[str, str, typing.Callable | str | None]] | None = None,
        compute_context: "ComputeContext | str | None" = None,
    ):
        """
        Initialise a MetaModel.

        ``nodes`` accepts either a ``dict[str, ModelNode]`` (multi-node)
        or a single ``ModelNode`` (auto-named ``"model"``). Pass
        ``edges`` when multiple nodes form a DAG with dependencies.

        When ``compute_context`` is provided (an :class:`ComputeContext`
        or a path to a context XLSX file), each node's ``ModelRep`` is
        wired at construction time: ``trans_dict`` (per-node
        ``translation_<node>`` sheet, or the single shared
        ``translation`` sheet when exactly one exists), ``ssn_dict`` and
        ``media_prices`` are assigned only where the node does not already
        carry an explicit value. Scopes are **not** stored on the
        MetaModel — pass them to :meth:`compute` as the ``scopes``
        argument (e.g. ``ctx.scopes``) so different scope sets can be
        applied to the same model.

        Parameters
        ----------
        nodes : dict[str, ModelNode] or ModelNode
            Node(s) to include. A single ``ModelNode`` is wrapped as
            ``{"model": node}`` automatically.
        edges : list[tuple[str, str, Callable or str or None]] or None, optional
            Dependency edges as ``(left, right, merge_fn)`` tuples.
            ``merge_fn`` is a callable, a string name resolved at
            compute time, or ``None`` for parallel independent branches.
        compute_context : ComputeContext or str or None, optional
            Pre-built :class:`ComputeContext` or a path to a context
            XLSX file (loaded via :func:`fermatrica_rep.options.context.load_context`).
        """
        if isinstance(nodes, ModelNode):
            nodes = {"model": nodes}
        self.nodes = nodes
        self.edges = edges if edges is not None else []
        if compute_context is not None:
            from fermatrica_rep.options.context import load_context

            if isinstance(compute_context, str):
                compute_context = load_context(compute_context)

            for node_name, node in self.nodes.items():
                mr = node.model_rep
                if mr.trans_dict is None and not node.static:
                    trans_dict = compute_context.trans_dicts.get(node_name)
                    if trans_dict is None and len(compute_context.trans_dicts) == 1:
                        trans_dict = next(iter(compute_context.trans_dicts.values()))
                    if trans_dict is None and len(compute_context.trans_dicts) > 1:
                        fermatrica_error(
                            f"Node '{node_name}' has no matching translation sheet in "
                            f"{compute_context.path}. Found sheets: "
                            f"{sorted(compute_context.trans_dicts.keys())}. "
                            "Name the node to match a 'translation_<node>' sheet "
                            "or pass trans_dict explicitly."
                        )
                    mr.trans_dict = trans_dict
                if mr.ssn_dict is None:
                    mr.ssn_dict = compute_context.ssn_dict
                if getattr(mr, "media_prices", None) is None:
                    mr.media_prices = compute_context.media_prices
        self._validate()
        self._init_finish()

    def __repr__(self) -> str:
        node_names = list(self.nodes.keys())
        return f"MetaModel(nodes={len(node_names)}: {node_names}, edges={len(self.edges)})"

    def __str__(self) -> str:
        lines = [f"MetaModel: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        if self.edges:
            lines.append("  Graph:")
            for left, right, merge_fn in self.edges:
                edge_str = f"    {left} -> {right}"
                if merge_fn is not None:
                    label = merge_fn if isinstance(merge_fn, str) else getattr(merge_fn, '__name__', merge_fn)
                    edge_str += f"  [{label}]"
                lines.append(edge_str)
        else:
            lines.append("  (single node, no edges)")
        order = self._topological_order()
        lines.append(f"  Order: {' -> '.join(order)}")
        return "\n".join(lines)

    def _validate(self):
        """
        Validate node and edge consistency.

        Raises FermatricaError if the graph is malformed.
        """
        node_names = set(self.nodes.keys())
        n_nodes = len(node_names)

        if n_nodes == 0:
            fermatrica_error("MetaModel must contain at least one node.")

        if n_nodes > 1 and not self.edges:
            fermatrica_error(
                f"MetaModel has {n_nodes} nodes but no edges provided. "
                "Edges are required when more than one node is present."
            )

        if not self.edges:
            return

        for left, right, _ in self.edges:
            if left not in node_names:
                fermatrica_error(
                    f"Edge left node '{left}' is not in MetaModel nodes: {sorted(node_names)}"
                )
            if right not in node_names:
                fermatrica_error(
                    f"Edge right node '{right}' is not in MetaModel nodes: {sorted(node_names)}"
                )
            if left == right:
                fermatrica_error(f"Self-loop detected: '{left}' -> '{right}'.")

        seen_edges = set()
        for edge in self.edges:
            key = (edge[0], edge[1])
            if key in seen_edges:
                fermatrica_error(f"Duplicate edge: '{key[0]}' -> '{key[1]}'.")
            seen_edges.add(key)

        referenced = set()
        for left, right, _ in self.edges:
            referenced.add(left)
            referenced.add(right)

        orphans = node_names - referenced
        if orphans:
            fermatrica_error(
                f"Nodes not referenced in any edge: {sorted(orphans)}. "
                f"All nodes must appear in at least one edge when edges are provided."
            )

        adj = {n: [] for n in node_names}
        for left, right, _ in self.edges:
            adj[left].append(right)
            adj[right].append(left)

        start = next(iter(node_names))
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(visited) != n_nodes:
            unreachable = node_names - visited
            fermatrica_error(
                f"MetaModel graph is not a single connected DAG. "
                f"Unreachable node(s): {sorted(unreachable)}."
            )

    @classmethod
    def _load_multi(cls, path: str) -> "MetaModel":
        """
        Load a multi-model DAG from a directory with a meta configuration.

        Reads ``meta_conf.toml`` (the single-file format written by
        :meth:`MetaModel.save`), then loads each node's model + dataset +
        ModelRep from its corresponding subdirectory. Nodes are created in
        the order they appear in the graph (start nodes first, then
        nodes referenced in ``right``).

        Legacy saves written as ``meta_conf.xlsx`` /
        ``meta_model_conf.xlsx`` (``model_graph`` sheet) remain readable —
        a warning is logged and the content is converted to the same
        structure.

        Node folders may be named ``<name>`` or ``model_<name>`` (the
        layout written by :meth:`MetaModel.save`); a leading ``model_``
        prefix on graph names is stripped so saved nodes load with clean
        keys. ``model_rep.pkl.lzma`` is optional — when absent a default
        ``ModelRep`` is auto-created from the dataset (with a warning),
        matching the modelling-stage save layout. Per-node ``scope_names``
        are restored from the ``[node_settings.<name>]`` tables.

        Parameters
        ----------
        path : str
            Directory containing the meta config file and one subdirectory
            per node. Each subdirectory must contain the model artifacts
            (as saved by ``ModelNode.save``) and, optionally,
            ``model_rep.pkl.lzma``.

        Returns
        -------
        MetaModel
            Fully initialized with all nodes and edges.

        Raises
        ------
        FermatricaError
            If the path is missing, the meta config file is not found, or
            a node folder referenced in the graph is missing.
        """
        if not os.path.exists(path):
            fermatrica_error(f"Path not found: {path}")

        graph, settings = _read_meta_conf(path)

        left_series = []
        right_series = []
        for entry in graph:
            left = entry.get("left")
            right = entry.get("right")
            if (
                not isinstance(left, str) or not isinstance(right, str)
                or not left.strip() or not right.strip()
            ):
                fermatrica_error(
                    "Invalid [[graph]] entry in meta config: 'left' and 'right' "
                    f"must be non-empty strings. Found: {entry}"
                )
            left_series.append(left.strip())
            right_series.append(right.strip())

        start_nodes_set = set(left_series)
        finish_nodes_set = set(right_series)
        nodes_order = [n for n in start_nodes_set if n not in finish_nodes_set]
        for n in right_series:
            if n not in nodes_order:
                nodes_order.append(n)

        name_map = {orig: _clean_node_name(orig) for orig in nodes_order}

        nodes = {}
        for orig_name in nodes_order:
            node_key = name_map[orig_name]
            node_path = _resolve_node_dir(path, orig_name)
            if node_path is None:
                fermatrica_error(
                    f"Node folder not found for '{orig_name}' in {path}. "
                    "Expected '<name>' or 'model_<name>' subdirectory."
                )

            nodes[node_key] = ModelNode.load(node_path)

        for node_key, node in nodes.items():
            entry = settings.get(node_key)
            if entry is not None and "scope_names" in entry:
                node.scope_names = set(entry["scope_names"])
            if entry is not None and "static" in entry:
                node.static = bool(entry["static"])

        edges = []
        for entry in graph:
            raw_left = entry["left"].strip()
            raw_right = entry["right"].strip()
            left = name_map.get(raw_left, _clean_node_name(raw_left))
            right = name_map.get(raw_right, _clean_node_name(raw_right))
            merge_val = entry.get("merge_fn")
            merge_fn = str(merge_val).strip() if merge_val is not None else None
            edges.append((left, right, merge_fn or None))

        mm = cls(nodes=nodes, edges=edges)
        return mm

    def save(self, path: str = '', save_tag: str | None = None,
             include_context_path: str | None = None) -> str:
        """
        Save the MetaModel (all nodes + edges + settings) to disk.

        Mirrors ``Model.save`` naming conventions: multi-node DAGs are
        written to ``path/meta_<save_tag>/`` with one ``model_<node_name>/``
        subdirectory per node plus a single ``meta_conf.toml`` (DAG edges
        as ``[[graph]]`` tables and per-node ``scope_names`` as
        ``[node_settings.<name>]`` tables). Single-node MetaModels
        delegate to :meth:`ModelNode.save` (plain
        ``path/model_<save_tag>/``) and write the same ``meta_conf.toml``
        into the node directory so that :meth:`MetaModel.load` can
        auto-detect the layout.

        ``dt_pred`` is not persisted — call ``mm.compute()`` (no-arg
        refresh) after loading to rebuild predictions. Seasonality
        (``ssn_dict``) and ``media_prices`` persist automatically via the
        per-node ``model_rep.pkl.lzma`` pickle.

        Adhoc module sources (``adhoc_code_src``) are snapshots taken at
        ``ModelRep`` construction time and pickled per node. Nodes of a
        MetaModel may have been saved at different development stages, so
        the same adhoc module can be pickled with different source across
        nodes — at compute time the first-imported snapshot wins
        process-wide. Before relying on a saved MetaModel, check that
        every node's saved adhoc sources are up to date with the current
        ``code_py/adhoc/`` files.

        Parameters
        ----------
        path : str, default ''
            Path to the up-directory where the model directory is to be
            created. ``''`` defaults to ``code_py/model_data`` under the
            current working directory (same default as ``Model.save``).
        save_tag : str or None, optional
            Specific tag to name the model directory. When ``None``, a
            timestamp is generated exactly like ``Model.save``
            (``YYYY-MM-DD_HHMMSS``).
        include_context_path : str or None, optional
            Path to a context XLSX file. When provided, the file is
            **copied** (``shutil.copy2``) into the saved directory as
            ``context.xlsx`` (no sheet rewriting). Useful to make the
            folder self-contained for downstream reporting.

        Returns
        -------
        str
            Path to the saved directory: ``meta_<save_tag>/`` for a
            multi-node DAG, or the single node's ``model_<save_tag>/``
            directory.

        Raises
        ------
        FermatricaError
            If a callable merge function cannot be re-imported from its
            module (lambdas, closures, and ``__main__``-defined functions
            are not serializable).
        """
        if path == '':
            path = os.path.join(os.getcwd(), 'code_py', 'model_data')

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            logging.warning(
                'Directory to save MetaModel is not found: ' + path +
                " - and created from scratch. You might delete it by hand if wasn't your intention"
            )

        if save_tag is None:
            save_tag = re.sub(' ', '_', datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            save_tag = re.sub(r':', '', save_tag)
        _validate_save_name(save_tag, "save_tag")

        if len(self.nodes) == 1 and not self.edges:
            node_name = next(iter(self.nodes.keys()))
            _validate_save_name(node_name, "node name")
            node = next(iter(self.nodes.values()))
            node_dir = node.save(path=path, save_tag=save_tag)
            _write_meta_conf(node_dir, [], self._node_settings_map())
            if include_context_path:
                shutil.copy2(include_context_path, os.path.join(node_dir, "context.xlsx"))
            return node_dir

        meta_dir = os.path.join(path, "meta_" + save_tag)
        if os.path.exists(meta_dir):
            fermatrica_error(
                f"Directory '{meta_dir}' already exists. "
                "Choose a different save_tag or remove the existing directory first."
            )

        for node_name in self.nodes:
            _validate_save_name(node_name, "node name")

        graph_rows = []
        for left, right, merge_fn in self.edges:
            graph_rows.append({
                "left": left,
                "right": right,
                "merge_fn": _serialize_merge_fn(merge_fn),
            })

        os.makedirs(meta_dir, exist_ok=True)

        try:
            _write_meta_conf(meta_dir, graph_rows, self._node_settings_map())

            for node_name, node in self.nodes.items():
                node.save(path=meta_dir, save_tag=node_name)

            if include_context_path:
                shutil.copy2(include_context_path, os.path.join(meta_dir, "context.xlsx"))
        except Exception:
            shutil.rmtree(meta_dir, ignore_errors=True)
            raise

        return meta_dir

    def _node_settings_map(self) -> dict:
        """
        Map node name to the per-node settings dict.

        Nodes with ``scope_names`` set and static nodes are included.
        Persisted via :func:`_write_meta_conf` and restored by
        :meth:`_load_multi` (multi-node) and ``ModelNode.load``
        (single-node) to restore node participation in scopes and the
        static (no-translation) flag.
        """
        node_settings: dict = {}
        for name, node in self.nodes.items():
            entry = {}
            if node.scope_names is not None:
                entry["scope_names"] = sorted(node.scope_names)
            if node.static:
                entry["static"] = True
            if entry:
                node_settings[name] = entry
        return node_settings

    @classmethod
    def from_node(cls, node: ModelNode, name: str = "model") -> "MetaModel":
        """
        Create a single-node MetaModel from a ModelNode.

        Parameters
        ----------
        node : ModelNode
            The node to wrap.
        name : str
            Node name in the resulting MetaModel. Defaults to
            ``"model"``. An empty or whitespace-only name raises
            ``FermatricaError``.

        Returns
        -------
        MetaModel
            A single-node MetaModel containing the given node.
        """
        if not name or not name.strip():
            fermatrica_error("MetaModel.from_node requires a non-empty name")
        if name.strip() == "model":
            return cls(nodes=node)
        return cls(nodes={name.strip(): node})

    @classmethod
    def load(cls, path: str) -> "MetaModel":
        """
        Load a MetaModel from a directory.

        Auto-detects the directory type:
        - Contains ``meta_conf.toml`` with node subdirectories (or a
          legacy ``meta_conf.xlsx`` / ``meta_model_conf.xlsx``)
          → multi-model DAG via :meth:`_load_multi`.
        - ``meta_conf.toml`` without subdirectories → single model
          (the TOML is read by ``ModelNode.load`` to restore
          ``scope_names``), node name ``"model"``.
        - No config file, no subdirectories → single model, node name
          ``"model"`` via ``ModelNode.load``.
        - No config file but has subdirectories → raises
          ``FermatricaError`` (looks like a DAG missing its config).

        Parameters
        ----------
        path : str
            Directory path containing the model artifacts.

        Returns
        -------
        MetaModel
            Fully initialised MetaModel with all nodes and edges.

        Raises
        ------
        FermatricaError
            If the path is missing, is not a directory, or the layout
            is ambiguous (subdirectories without meta config).
        """
        if not os.path.exists(path):
            fermatrica_error(f"Path not found: {path}")
        if not os.path.isdir(path):
            fermatrica_error(f"Path is not a directory: {path}")

        has_toml = os.path.isfile(os.path.join(path, "meta_conf.toml"))
        legacy_names = ["meta_conf.xlsx", "meta_model_conf.xlsx"]
        has_legacy_conf = any(
            os.path.isfile(os.path.join(path, f)) for f in legacy_names
        )

        subdirs = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
            and not d.startswith("__")
        ]

        if has_toml and not subdirs:
            node = ModelNode.load(path)
            return cls(nodes=node)

        if has_toml or has_legacy_conf:
            return cls._load_multi(path)

        if subdirs:
            fermatrica_error(
                f"No meta_conf.toml found in '{path}', but directory "
                f"contains subdirectories: {subdirs}. "
                f"This looks like a multi-model DAG. Add meta_conf.toml."
            )

        node = ModelNode.load(path)
        return cls(nodes=node)

    def compute(
        self,
        option=None,
        scopes: list | None = None,
        allow_past: bool | None = None,
        if_exact: bool = False,
        *,
        return_full: bool = False,
    ) -> dict[str, pd.DataFrame | dict[str, dict | tuple]]:
        """
        Refresh all nodes in the DAG, or execute budget options across them.

        When ``option`` is ``None`` (default), walks nodes in topological
        order, merges each predecessor's ``dt_pred`` via edge-specific merge
        functions, and runs the no-arg refresh (:meth:`ModelNode.compute`)
        on every node. Returns ``{node_name: dt_pred}``. ``scopes`` /
        ``return_full`` are ignored on this path. ``allow_past`` defaults to
        ``True`` on the refresh path (via the ``None`` sentinel) and is
        forwarded to the merge functions.

        With an ``option``, uses a two-phase design to avoid redundant
        transform/predict work:

        1. **Translate** — walk nodes in topological order, calling
           ``node.translate()`` on each. Budget values are translated into
           media variables cumulatively (each scope inherits prior scopes'
           values). No transform or predict runs — this phase only mutates
           ``node.ds``.
        2. **Compute** — walk nodes again in topological order. For each
           node, merge predecessor ``dt_pred`` via edge-specific merge
           functions (from ``meta_conf.toml``), then call
           ``node.compute_after_translate()`` which runs a single
           ``transform`` + ``predict`` and derives per-scope summaries
           from the shared prediction.

        Phase separation matters for multi-model DAGs: predecessor nodes'
        translations complete before their predictions are merged into
        downstream nodes, ensuring downstream nodes see the *actual*
        predecessor output, not stale data.

        ``node.ds`` and ``node.dt_pred`` are always updated on every node,
        regardless of ``return_full``.

        Args:
            option: Option object with budget allocations. Must provide
                ``iter_scopes()`` yielding ``(scope_name, budget_items)``.
                ``None`` runs the no-arg refresh path.
            scopes: List of ``OptionScope`` objects defining date ranges,
                entity filters, reference periods, and inflation. Matched
                by scope name during translation. Required on the option
                path (scopes are not stored on the MetaModel — the same
                model can be computed under different scope sets).
            allow_past: Forwarded to ``OptionScope.build_mask`` and merge
                functions. When ``False`` (default on the option path), only
                future rows (``listed == 4``) are affected by budget
                translation. On the no-arg refresh path, ``None`` (default)
                resolves to ``True``.
            if_exact: If ``True``, skip long-term translation — budget
                values are applied only to each scope's exact period
                without forward projection.
            return_full: If ``True``, each scope entry is a 3-tuple
                ``(ds, dt_pred, summary)``. If ``False`` (default), each
                scope entry is just the ``summary`` dict. Follows the same
                convention as ``fermatrica.fit_predict``.

        Returns:
            ``{node_name: dt_pred}`` when ``option`` is ``None``.

            With an ``option``: ``{node_name: {scope_name: summary}}`` when
            ``return_full=False`` (default), or
            ``{node_name: {scope_name: (ds, dt_pred, summary)}}`` when
            ``return_full=True``.

            Each node entry maps scope names to either a summary dict or a
            3-tuple. Summary dict keys include volume/value metrics such as
            ``pred_exact_vol``, ``pred_long_vol``, ``growth_vol``, etc.

        Note:
            Predecessor merge passes the **full** predecessor ``dt_pred``
            (all periods), not a per-scope slice. All existing merge
            functions in the codebase handle this safely.
        """
        order = self._topological_order()

        if option is None:
            allow_past = True if allow_past is None else allow_past
            results: dict[str, pd.DataFrame] = {}
            for node_name in order:
                node = self.nodes[node_name]
                self._merge_predecessors(node_name, allow_past)
                results[node_name] = node.compute()
            return results

        if scopes is None:
            fermatrica_error(
                "scopes argument is required on the option path "
                "(pass scopes=ctx.scopes)"
            )

        allow_past = False if allow_past is None else allow_past

        for node_name in order:
            self.nodes[node_name].translate(option, scopes, allow_past=allow_past, if_exact=if_exact)

        all_results: dict[str, dict] = {}
        for node_name in order:
            node = self.nodes[node_name]
            self._merge_predecessors(node_name, allow_past)
            all_results[node_name] = node.compute_after_translate(option, scopes, return_full=return_full)

        return all_results

    def _merge_predecessors(self, node_name: str, allow_past: bool):
        """
        Merge each predecessor's ``dt_pred`` into the node's dataset.

        For every direct predecessor of ``node_name`` with a registered
        merge function, loads the predecessor's adhoc code (if any) and
        applies the merge function to ``node.ds`` with the predecessor's
        ``dt_pred`` and the given ``allow_past``.

        Shared by the no-arg refresh path and the option path of
        :meth:`compute` so both DAGs merge identical data.
        """
        node = self.nodes[node_name]
        for pred_name in self._predecessors(node_name):
            merge_fn = self._edge_merge_fn(pred_name, node_name)
            if merge_fn:
                self._load_adhoc_from_node(node)
                node.ds = self._apply_merge_fn(
                    merge_fn,
                    ds_cur=node.ds,
                    dt_pred_prev=self.nodes[pred_name].dt_pred,
                    allow_past=allow_past,
                )

    def _topological_order(self) -> list[str]:
        """
        Return nodes in topological order (Kahn's algorithm).

        Start nodes (zero incoming edges) appear first, followed by
        nodes whose dependencies have been resolved, and finally sink
        nodes. Raises ``FermatricaError`` if the graph contains a
        cycle.

        Returns
        -------
        list[str]
            Node names in execution order.
        """
        in_degree: dict[str, int] = {}
        adj: dict[str, list[str]] = {}

        for node_name in self.nodes:
            in_degree[node_name] = 0
            adj[node_name] = []

        for left, right, _ in self.edges:
            if left in in_degree and right in in_degree:
                in_degree[right] += 1
                adj.setdefault(left, []).append(right)

        queue = [n for n in self.nodes if in_degree.get(n, 0) == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor in adj.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            fermatrica_error(
                "MetaModel graph contains a cycle. Cannot compute."
            )

        return result

    def _predecessors(self, node_name: str) -> list[str]:
        """
        Return the names of direct predecessors of a node.

        Parameters
        ----------
        node_name : str
            Name of the node whose predecessors to look up.

        Returns
        -------
        list[str]
            Names of nodes that have an edge pointing to ``node_name``.
            Empty list if the node has no predecessors.
        """
        return [left for left, right, _ in self.edges if right == node_name]

    def _edge_merge_fn(
        self, left: str, right: str
    ) -> typing.Callable | str | None:
        """
        Return the merge function for a given edge.

        Parameters
        ----------
        left : str
            Name of the source (predecessor) node.
        right : str
            Name of the target (successor) node.

        Returns
        -------
        Callable or str or None
            The merge function registered for the ``left -> right``
            edge. ``None`` if the edge has no merge function or does
            not exist.
        """
        for l, r, merge_fn in self.edges:
            if l == left and r == right:
                return merge_fn
        return None

    def _load_adhoc_from_node(self, node: ModelNode):
        """
        Import adhoc user modules registered on a node's ModelRep.

        Called before invoking edge-specific merge functions so that
        adhoc merge functions (prefixed ``code_py.adhoc.reporting.``)
        are available in the current module's namespace.

        Idempotency is keyed on the process-local module globals (see
        ``ModelNode._load_adhoc_code`` for the rationale): a spawn Pool
        worker inherits the pickled MetaModel but starts with a fresh
        ``fermatrica_rep.meta_model.node`` namespace, so adhoc modules
        must be re-imported there.

        The imported source is the node's pickled snapshot (taken at
        ``ModelRep`` construction time), not the live project adhoc
        files. Because imports are idempotent on the module name, the
        first node whose snapshot registers a module wins process-wide —
        a DAG whose nodes were saved at different development stages can
        silently run stale or mixed adhoc code. Verify that every node's
        saved adhoc sources are up to date before relying on a loaded
        MetaModel.

        Parameters
        ----------
        node : ModelNode
            The node whose ``model_rep.adhoc_code_src`` is imported.
        """
        if (
            pd.notna(node.model_rep.adhoc_code_src)
            and len(node.model_rep.adhoc_code_src) > 0
        ):
            for k, v in node.model_rep.adhoc_code_src.items():
                if k in globals():
                    continue
                fr_cur_name = inspect.currentframe().f_globals["__name__"]
                import_module_from_string(k, v, fr_cur_name)

    def _apply_merge_fn(
        self,
        merge_fn: typing.Callable | str,
        ds_cur: pd.DataFrame,
        dt_pred_prev: pd.DataFrame,
        allow_past: bool,
    ) -> pd.DataFrame:
        """
        Execute a merge function by resolving it to a callable.

        Supports three resolution strategies:
        1. Already callable — invoked directly.
        2. String name found in ``locals()`` or ``globals()`` —
           resolved via ``eval()``.
        3. Qualified dotted name (e.g. ``module.fn``) — resolved
           from the module object in ``globals()``.
        4. (Fallback) Scans all module values for an attribute
           matching the name — useful for adhoc functions imported
           at runtime.

        If none of the strategies resolve the function, a warning
        is logged and ``ds_cur`` is returned unchanged.

        Parameters
        ----------
        merge_fn : Callable or str
            Merge function to invoke. Passed ``ds_cur``,
            ``dt_pred_prev``, and ``allow_past``.
        ds_cur : pd.DataFrame
            Current node's dataset to merge into.
        dt_pred_prev : pd.DataFrame
            Predecessor node's prediction DataFrame.
        allow_past : bool
            Whether to include historical periods in the merge.

        Returns
        -------
        pd.DataFrame
            The merged dataset (modified in place or replaced).
        """
        if callable(merge_fn):
            return merge_fn(
                ds_cur=ds_cur,
                dt_pred_prev=dt_pred_prev,
                allow_past=allow_past,
            )

        if merge_fn in locals() or merge_fn in globals():
            fn = eval(merge_fn)
        elif (
            len(merge_fn.split(".")) > 1
            and getattr(
                globals().get(
                    ".".join(merge_fn.split(".")[:-1]), None
                ),
                merge_fn.split(".")[-1],
                False,
            )
        ):
            fn = eval(merge_fn)
        else:
            for v in globals().values():
                if (
                    inspect.ismodule(v)
                    and hasattr(v, merge_fn)
                    and callable(getattr(v, merge_fn))
                ):
                    fn = getattr(v, merge_fn)
                    break
            else:
                logging.warning(
                    f"Merge function '{merge_fn}' not found — skipping merge"
                )
                return ds_cur

        return fn(
            ds_cur=ds_cur,
            dt_pred_prev=dt_pred_prev,
            allow_past=allow_past,
        )


def _last_node_name(meta_model: MetaModel) -> str:
    """Return the name of the single sink (terminal) node in the DAG.

    A sink node is a node that never appears as ``left`` in any graph
    edge — meaning nothing depends on it. In a chain like
    ``awareness → bht → sales``, ``sales`` is the sink.

    Args:
        meta_model: The MetaModel whose DAG to inspect.

    Returns:
        The name of the single terminal node.

    Raises:
        FermatricaError: If the graph has multiple sink nodes (parallel
            independent branches with no common downstream consumer).
            Callers must pass ``summary_node`` explicitly in this case.
    """
    order = meta_model._topological_order()
    terminal_nodes = [
        n
        for n in order
        if not any(left == n for left, _, _ in meta_model.edges)
    ]

    if len(terminal_nodes) > 1:
        fermatrica_error(
            f"MetaModel has {len(terminal_nodes)} sink nodes ({terminal_nodes}). "
            "Pass summary_node explicitly to identify the output node."
        )

    return terminal_nodes[0] if terminal_nodes else order[-1]


def _resolve_node_dir(path: str, name: str) -> str | None:
    """Resolve a node's saved directory inside a meta directory.

    Tries ``path/<name>`` first, then ``path/model_<name>`` (the layout
    written by :meth:`MetaModel.save`). Returns ``None`` when neither
    exists.
    """
    direct = os.path.join(path, name)
    if os.path.isdir(direct):
        return direct
    prefixed = os.path.join(path, "model_" + name)
    if os.path.isdir(prefixed):
        return prefixed
    return None


def _clean_node_name(name: str) -> str:
    """Strip a leading ``model_`` prefix from a graph node name.

    ``MetaModel.save`` writes node folders as ``model_<node_name>/`` but
    keeps the graph names bare, so loading restores clean keys
    (``uba``/``sales``, not ``model_uba``/``model_sales``).
    """
    if isinstance(name, str) and name.startswith("model_"):
        return name[len("model_"):]
    return name


def _validate_save_name(name: str, label: str) -> str:
    """Validate a ``save_tag`` / node name used as a directory component.

    Names must be non-empty, contain no path separators, no ``..``, and no
    characters invalid in directory names — otherwise the saved tree could
    escape ``path`` or be un-loadable. Raises ``FermatricaError``.
    """
    if not name or not str(name).strip():
        fermatrica_error(f"{label} must be a non-empty string")
    if isinstance(name, str) and name != name.strip():
        fermatrica_error(f"{label} '{name}' must not have leading/trailing whitespace")
    invalid_chars = {os.sep, "/"}
    if os.altsep:
        invalid_chars.add(os.altsep)
    invalid_chars.update('< > : " | ? * \\'.split())
    if any(c in str(name) for c in invalid_chars) or str(name) in (".", ".."):
        fermatrica_error(
            f"{label} '{name}' contains invalid characters — it is used as a directory name"
        )
    return name


def _serialize_merge_fn(merge_fn: typing.Callable | str | None) -> str | None:
    """Serialize a merge function into its ``meta_conf.toml`` value.

    - ``None`` → ``None`` (omitted → parallel branch).
    - ``str`` → written as-is.
    - callable → qualified name ``f"{module}.{qualname}"``, verified to be
      re-importable. Lambdas, closures, and ``__main__``-defined functions
      raise ``FermatricaError`` (dropping a merge silently is worse).
    """
    if merge_fn is None or isinstance(merge_fn, str):
        return merge_fn

    qualname = f"{merge_fn.__module__}.{merge_fn.__qualname__}"
    module_name = merge_fn.__module__
    parts = merge_fn.__qualname__.split(".")
    try:
        obj = importlib.import_module(module_name)
        for part in parts:
            obj = getattr(obj, part)
        if not callable(obj):
            raise AttributeError
    except (ImportError, AttributeError) as exc:
        fermatrica_error(
            f"Merge function '{qualname}' cannot be re-imported from its module. "
            "MetaModel.save requires merge functions to be top-level, importable "
            f"callables. Original error: {exc}"
        )
    return qualname


def _toml_basic_string(value: str) -> str:
    """Encode ``value`` as a TOML basic string."""
    out = []
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\b":
            out.append("\\b")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _write_meta_conf(meta_dir: str, graph_rows: list[dict], node_settings: dict) -> str:
    """
    Write the DAG graph + per-node settings into a single ``meta_conf.toml``.

    Each edge becomes a ``[[graph]]`` table (``left``, ``right`` and an
    optional ``merge_fn``); each node with settings becomes a
    ``[node_settings.<name>]`` table carrying ``scope_names`` (when set)
    and ``static`` (when true). Returns the path of the written file.
    """
    lines: list[str] = []
    for row in graph_rows:
        lines.append("[[graph]]")
        lines.append(f"left = {_toml_basic_string(row['left'])}")
        lines.append(f"right = {_toml_basic_string(row['right'])}")
        if row.get("merge_fn"):
            lines.append(f"merge_fn = {_toml_basic_string(row['merge_fn'])}")
        lines.append("")
    for name, entry in node_settings.items():
        lines.append(f"[node_settings.{_toml_basic_string(name)}]")
        if entry.get("scope_names") is not None:
            names = sorted(entry["scope_names"])
            lines.append("scope_names = [" + ", ".join(_toml_basic_string(n) for n in names) + "]")
        if entry.get("static"):
            lines.append("static = true")
        lines.append("")

    path = os.path.join(meta_dir, "meta_conf.toml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _read_meta_conf(path: str) -> tuple[list[dict], dict]:
    """
    Read the DAG graph + per-node settings from a meta config file.

    Prefers the single-file ``meta_conf.toml``; falls back to the legacy
    ``meta_conf.xlsx`` / ``meta_model_conf.xlsx`` (``model_graph`` sheet),
    with a warning, for directories saved before the TOML format. Raises
    ``FermatricaError`` when neither is present.

    Returns a ``(graph, settings)`` pair: ``graph`` is a list of
    ``{"left", "right", "merge_fn"?}`` dicts, ``settings`` maps node name
    to a settings dict.
    """
    toml_path = os.path.join(path, "meta_conf.toml")
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as handle:
            conf = tomllib.load(handle)
        return conf.get("graph", []), conf.get("node_settings", {})

    legacy_names = ["meta_conf.xlsx", "meta_model_conf.xlsx"]
    found = [f for f in legacy_names if os.path.isfile(os.path.join(path, f))]
    if not found:
        fermatrica_error(
            f"No meta_conf.toml found in {path} "
            "(and no legacy meta_conf.xlsx / meta_model_conf.xlsx either)"
        )

    logging.warning(
        f"Reading legacy meta config from {found[0]} — re-save via MetaModel.save "
        "to upgrade to meta_conf.toml."
    )
    model_graph = pd.read_excel(
        os.path.join(path, found[0]), sheet_name="model_graph", engine="openpyxl", dtype=str
    )

    if "merge_fn" not in model_graph.columns:
        logging.warning(
            "column 'merge_fn' not found in model_graph sheet — assuming no merge functions"
        )

    graph = []
    for _, row in model_graph.iterrows():
        if pd.isna(row.get("left")) or pd.isna(row.get("right")):
            continue
        entry = {"left": str(row["left"]).strip(), "right": str(row["right"]).strip()}
        merge_val = row.get("merge_fn")
        if pd.notna(merge_val) and str(merge_val).strip():
            entry["merge_fn"] = str(merge_val).strip()
        graph.append(entry)

    return graph, {}


def _get_default_vis_dict() -> pd.DataFrame:
    """Return the packaged default visual dictionary, loaded once.

    ``ModelRep`` re-reads ``res/dict/vis_dict.xlsx`` from disk on every
    construction with ``vis_dict=None``; caching it avoids N identical
    full-file reads when auto-creating default reps for many nodes.
    """
    global _default_vis_dict
    if _default_vis_dict is None:
        with resources.path("fermatrica_rep.res.dict", "vis_dict.xlsx") as xlsxio:
            _default_vis_dict = pd.read_excel(xlsxio)
    return _default_vis_dict


def _auto_model_rep(ds: pd.DataFrame | None, dt_pred: pd.DataFrame | None) -> ModelRep:
    """
    Build a default ``ModelRep`` from the data available at conversion time.

    ``ds`` takes precedence over ``dt_pred`` (matching the historical
    ``ModelRep(ds)`` semantics for data-bearing calls such as
    ``extract_effect``); when only ``dt_pred`` is passed (fit plots), the rep
    is derived from it so palettes are non-empty. When neither is available,
    an empty default rep (language ``'english'``, empty palettes) is returned.
    """
    rep_ds = ds if ds is not None else dt_pred
    if rep_ds is None:
        rep_ds = pd.DataFrame()
    return ModelRep(rep_ds, vis_dict=_get_default_vis_dict().copy())


def _convert_to_metamodel(model, ds=None, model_rep=None, dt_pred=None, merge_fn_pattern=None):
    """
    Convert ``Model | ModelNode | MetaModel | ModelRep | list`` + optional
    overrides into a ``MetaModel``.

    Always produces a ``MetaModel``; the caller can iterate
    ``mm.nodes.items()`` and pass ``(node.model, node.ds,
    node.model_rep, node.dt_pred)`` to their worker.

    Parameters
    ----------
    merge_fn_pattern : str | None
        When ``model`` is a ``list`` and this is a string like
        ``"code_py.adhoc.reporting.pred_merge_{i}"``, sequential edges
        are created between consecutive list elements with merge function
        names interpolated from the pattern.  When ``None`` (default),
        no edges are created — nodes are independent.

    When ``model_rep`` is ``None``, a default ``ModelRep`` is auto-created
    from ``ds`` (falling back to ``dt_pred``, then an empty frame) via
    ``_auto_model_rep``.
    """
    if isinstance(model, MetaModel):
        return model

    if isinstance(model, list):
        warnings.warn(
            "Passing list of models is deprecated. Use MetaModel instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        nodes = {}
        for i, m in enumerate(model):
            if isinstance(m, ModelRep):
                nodes[f"model_{i}"] = ModelNode(
                    model=None, ds=pd.DataFrame(), model_rep=m, dt_pred=None,
                    _display_only=True,
                )
            else:
                _ds = ds[i] if isinstance(ds, (list, tuple)) else ds
                if _ds is None:
                    _ds = pd.DataFrame()
                _rep = model_rep[i] if isinstance(model_rep, (list, tuple)) else model_rep
                _pred = dt_pred[i] if isinstance(dt_pred, (list, tuple)) else dt_pred
                if _rep is None:
                    _rep = _auto_model_rep(_ds, _pred)
                nodes[f"model_{i}"] = ModelNode(model=m, ds=_ds, model_rep=_rep, dt_pred=_pred)

        edges = []
        if merge_fn_pattern:
            for i in range(len(model) - 1):
                edges.append((f"model_{i}", f"model_{i + 1}", merge_fn_pattern.format(i=i + 1)))

        return MetaModel(nodes=nodes, edges=edges)

    if isinstance(model, ModelNode):
        node = model
        if ds is not None or model_rep is not None or dt_pred is not None:
            node = ModelNode(
                model=node.model,
                ds=ds if ds is not None else node.ds,
                model_rep=model_rep if model_rep is not None else node.model_rep,
                dt_pred=dt_pred if dt_pred is not None else node.dt_pred,
                static=node.static,
                _display_only=node._display_only,
            )
        return MetaModel(nodes={"model": node})

    if isinstance(model, ModelRep):
        return MetaModel(nodes={
            "model": ModelNode(
                model=None, ds=pd.DataFrame(), model_rep=model, dt_pred=None,
                _display_only=True,
            )
        })

    if model_rep is None:
        model_rep = _auto_model_rep(ds, dt_pred)

    return MetaModel(nodes={"model": ModelNode(model=model, ds=ds if ds is not None else pd.DataFrame(), model_rep=model_rep, dt_pred=dt_pred)})

