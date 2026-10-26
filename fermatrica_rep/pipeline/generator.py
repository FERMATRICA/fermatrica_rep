"""
Pipeline generator — auto-generate options XLSX sheets, reporting scripts,
and adhoc stubs from existing ``translation_*`` sheets.
"""

import os
import re
import shutil

import pandas as pd
from fermatrica_utils import read_pickle_compat
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from fermatrica.basics.basics import fermatrica_error


def generate_pipeline(
    options_path: str,
    model_paths: dict[str, str],
    targets: dict[str, list[tuple[str, str]]],
    edges: list[tuple[str, str]] | None = None,
    project_root: str = '.',
    force: bool = False,
) -> None:
    """
    Auto-generate pipeline infrastructure from existing ``translation_*``
    sheets in an options XLSX file.

    Reads translation sheets (one per DAG node), inspects saved model
    predictions to detect scope years, and produces:

    - A new XLSX file (``context.xlsx``, derived by stripping
      ``_template`` / ``_templ`` from the input filename, written to
      ``{project_root}/model_data/options/``) with four generated
      sheets: ``scopes``, ``allocations``, ``media_prices``,
      ``seasonality``.
    - Adhoc stubs in ``{project_root}/adhoc/``: ``__init__.py``,
      ``translators.py``, ``reporting.py`` (with merge function stubs
      when edges are provided).
    - Per-node reporting scripts in ``{project_root}/scripts/``:
      ``reporting_{node}.py`` for each node. Nodes deeper in the DAG
      automatically include all their upstream predecessors so the
      script is self-contained.

    Example usage::

        fmr.generate_pipeline(
            options_path="options_template.xlsx",
            model_paths={
                "sales": "code_py/model_data_sales",
                "category": "code_py/model_data_category",
            },
            targets={
                "sales": [("superbrand", "brand_1")],
                "category": [("superbrand", "mock_brand")],
            },
            edges=[("category", "sales")],
            project_root="code_py",
            force=True,
        )

    :param options_path: Path to template XLSX with ``translation_*``
        sheets. Never modified; output goes to a new file.
    :param model_paths: ``{node_name: parent_dir}`` mapping. Each
        parent directory must contain at least one ``model_*``
        subdirectory with ``dt_p.pkl.zstd`` inside.
    :param targets: ``{node_name: [(apply_var, target_value), ...]}``.
    :param edges: Optional DAG edges as ``[(left_node, right_node), ...]``.
        Controls which upstream nodes each reporting script includes.
    :param project_root: Base directory for ``scripts/``, ``adhoc/``,
        and the output XLSX. Defaults to ``"."``.
    :param force: If ``False`` (default), raises an error when any
        output file already exists. If ``True``, overwrites silently.
    """
    node_names, trans_data = _read_translation_sheets(options_path)
    scope_years = _detect_scope_years(node_names, model_paths, targets, project_root)
    dt_scopes = _generate_scopes_df(scope_years)
    dt_alloc = _generate_allocations_df(trans_data, scope_years)
    dt_media = _generate_media_prices_df(trans_data, scope_years)
    dt_ssn = _generate_seasonality_df(trans_data, scope_years)
    edge_index = _edge_index_map(edges or [])
    predecessors = _compute_predecessors(node_names, edges or [], edge_index)
    output_xlsx = _output_xlsx_path(options_path, project_root)
    _validate_inputs(options_path, model_paths, targets, edges, project_root, force, node_names, predecessors)
    _write_xlsx(options_path, output_xlsx, dt_scopes, dt_alloc, dt_media, dt_ssn)
    _generate_adhoc_files(project_root, edges, force)
    _generate_reporting_scripts(output_xlsx, node_names, model_paths, predecessors, project_root, force)


def _edge_index_map(edges: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """
    Map each edge to its 1-based index in the global edges list.

    The index corresponds to ``pred_merge_N`` in ``adhoc/reporting.py``.

    :param edges: Global DAG edge list.
    :return: ``{edge: index}`` where index starts at 1.
    """
    return {edge: i for i, edge in enumerate(edges, 1)}


def _output_xlsx_path(options_path: str, project_root: str) -> str:
    """
    Derive the output XLSX filename by stripping ``_template`` /
    ``_templ`` from the input basename and placing it under
    ``{project_root}/model_data/options/``.

    :param options_path: Path to template XLSX file.
    :param project_root: Target base directory.
    :return: Clean output path.
    """
    basename = os.path.basename(options_path)
    basename = re.sub(r'_(template|templ)(?=\.xlsx$)', '', basename, flags=re.IGNORECASE)
    return os.path.join(project_root, 'model_data', 'options', basename)


def _compute_predecessors(
    node_names: list[str],
    edges: list[tuple[str, str]],
    edge_index: dict[tuple[str, str], int],
) -> dict[str, dict]:
    """
    For each node, compute the set of upstream nodes (transitive
    predecessors) that must be loaded so the node's reporting script
    is self-contained.

    :param node_names: All node names.
    :param edges: Global DAG edges.
    :param edge_index: ``{edge: global_index}`` from :func:`_edge_index_map`.
    :return: ``{node: {"nodes": [topo_sorted_list], "edges": [(l, r, idx)]}}``.
    """
    reverse: dict[str, set[str]] = {n: set() for n in node_names}
    for left, right in edges:
        reverse.setdefault(right, set()).add(left)
        reverse.setdefault(left, set())

    closure: dict[str, set[str]] = {n: set(preds) for n, preds in reverse.items()}
    changed = True
    while changed:
        changed = False
        for node in node_names:
            for p in list(closure[node]):
                for pp in closure.get(p, set()):
                    if pp not in closure[node]:
                        closure[node].add(pp)
                        changed = True

    result: dict[str, dict] = {}
    for node in node_names:
        all_nodes = closure[node] | {node}
        ordered = _topo_subset(all_nodes, edges)
        subset_edges = [
            (left, right, edge_index[(left, right)])
            for left, right in edges
            if left in all_nodes and right in all_nodes
        ]
        result[node] = {"nodes": ordered, "edges": subset_edges}
    return result


def _topo_subset(node_set: set[str], edges: list[tuple[str, str]]) -> list[str]:
    """
    Topologically sort a subset of DAG nodes.

    :param node_set: Nodes to include.
    :param edges: Full DAG edges (used for ordering).
    :return: Topologically ordered list of node names from ``node_set``.
    """
    in_degree: dict[str, int] = {n: 0 for n in node_set}
    adj: dict[str, list[str]] = {n: [] for n in node_set}
    for left, right in edges:
        if left in node_set and right in node_set:
            adj[left].append(right)
            in_degree[right] += 1

    queue = [n for n in node_set if in_degree[n] == 0]
    result = []
    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return result


def _resolve_path(pth: str, project_root: str) -> str:
    """
    Resolve a model parent directory path to an existing directory.

    :param pth: Raw path string from ``model_paths``.
    :param project_root: Fallback base directory.
    :return: Resolved path guaranteed to exist.
    """
    if os.path.isabs(pth) and os.path.isdir(pth):
        return pth
    if os.path.isdir(pth):
        return pth
    joined = os.path.join(project_root, pth)
    if os.path.isdir(joined):
        return joined
    fermatrica_error(f"Model parent directory not found: {pth}")


def _validate_inputs(
    options_path: str,
    model_paths: dict[str, str],
    targets: dict[str, list[tuple[str, str]]],
    edges: list[tuple[str, str]] | None,
    project_root: str,
    force: bool,
    node_names: list[str],
    predecessors: dict[str, dict],
) -> None:
    """
    Validate all inputs before generating any output.
    """
    if not os.path.exists(options_path):
        fermatrica_error(f"Options file not found: {options_path}")
    if not options_path.lower().endswith('.xlsx'):
        fermatrica_error(f"File must be .xlsx: {options_path}")

    xl_file = pd.ExcelFile(options_path, engine="openpyxl")
    all_sheets = xl_file.sheet_names
    xl_file.close()

    trans_sheets = [s for s in all_sheets if s.startswith('translation_')]
    if not trans_sheets:
        fermatrica_error(f"No translation_* sheets found in {options_path}")

    for name in node_names:
        if name not in model_paths:
            fermatrica_error(f"Node '{name}' missing from model_paths")
        if name not in targets:
            fermatrica_error(f"Node '{name}' missing from targets")

    for name, pth in model_paths.items():
        abs_pth = _resolve_path(pth, project_root)
        subdirs = [d for d in os.listdir(abs_pth)
                   if d.startswith('model_') and not d.endswith('_dash')
                   and os.path.isdir(os.path.join(abs_pth, d))]
        if not subdirs:
            fermatrica_error(f"No model_* subdirectories in {abs_pth}")
        latest = sorted(subdirs, reverse=True)[0]
        dt_p_path = os.path.join(abs_pth, latest, 'dt_p.pkl.zstd')
        if not os.path.exists(dt_p_path):
            fermatrica_error(f"Missing dt_p.pkl.zstd for node '{name}' in {abs_pth}/{latest}")

    for name, tgt_list in targets.items():
        if not tgt_list:
            fermatrica_error(f"Each node needs at least 1 target, got empty list for '{name}'")

    if edges:
        if len(node_names) < 2:
            fermatrica_error("Edges require at least 2 nodes")
        for left, right in edges:
            if left not in node_names:
                fermatrica_error(f"Unknown node '{left}' in edges")
            if right not in node_names:
                fermatrica_error(f"Unknown node '{right}' in edges")
            if left == right:
                fermatrica_error(f"Self-loop in edges: {left} -> {right}")
        adjacency = {name: [] for name in node_names}
        for left, right in edges:
            adjacency.setdefault(left, []).append(right)
        _check_cycles(adjacency, node_names)

    if not force:
        conflicts = []
        out_xlsx = _output_xlsx_path(options_path, project_root)
        if os.path.exists(out_xlsx):
            conflicts.append(out_xlsx)
        for name in node_names:
            script_path = os.path.join(project_root, 'scripts', f'reporting_{name}.py')
            if os.path.exists(script_path):
                conflicts.append(script_path)
        adhoc_init = os.path.join(project_root, 'adhoc', '__init__.py')
        if os.path.exists(adhoc_init):
            conflicts.append(adhoc_init)
        adhoc_reporting = os.path.join(project_root, 'adhoc', 'reporting.py')
        if os.path.exists(adhoc_reporting):
            conflicts.append(adhoc_reporting)
        adhoc_translators = os.path.join(project_root, 'adhoc', 'translators.py')
        if os.path.exists(adhoc_translators):
            conflicts.append(adhoc_translators)
        if conflicts:
            fermatrica_error(
                "Output files already exist. Use force=True to overwrite.\n"
                + "\n".join(conflicts)
            )


def _check_cycles(adjacency: dict[str, list[str]], node_names: list[str]) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in node_names}

    def dfs(u):
        color[u] = GRAY
        for v in adjacency.get(u, []):
            if color[v] == GRAY:
                fermatrica_error("DAG contains a cycle")
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for n in node_names:
        if color[n] == WHITE:
            dfs(n)


def _read_translation_sheets(options_path: str) -> tuple[list[str], dict[str, dict]]:
    xl_file = pd.ExcelFile(options_path, engine="openpyxl")
    trans_sheets = sorted(s for s in xl_file.sheet_names if s.startswith('translation_'))
    node_names = [s.replace('translation_', '', 1) for s in trans_sheets]

    trans_data: dict[str, dict] = {}
    for name, sheet in zip(node_names, trans_sheets):
        df = xl_file.parse(sheet_name=sheet)
        required_cols = {'item', 'function', 'variable'}
        missing = required_cols - set(df.columns)
        if missing:
            fermatrica_error(f"Sheet '{sheet}' missing required columns: {missing}")

        items = df['item'].dropna().tolist()
        trans_data[name] = {
            'all_items': list(items),
            'bdg_items': [i for i in items if str(i).startswith('bdg_')],
            'nobdg_items': [i for i in items if not str(i).startswith('bdg_')],
        }
    xl_file.close()
    return node_names, trans_data


def _detect_scope_years(
    node_names: list[str],
    model_paths: dict[str, str],
    targets: dict[str, list[tuple[str, str]]],
    project_root: str = '.',
) -> dict[str, dict]:
    scope_data: dict[str, dict] = {}

    for name in node_names:
        pth = _resolve_path(model_paths[name], project_root)
        dirs = [d for d in os.listdir(pth)
                if d.startswith('model_') and not d.endswith('_dash')
                and os.path.isdir(os.path.join(pth, d))]
        latest = sorted(dirs, reverse=True)[0]
        dt_p_path = os.path.join(pth, latest, 'dt_p.pkl.zstd')
        dt_p = read_pickle_compat(dt_p_path)

        future = dt_p[dt_p['listed'] == 4]
        if future.empty:
            fermatrica_error(f"No future data (listed==4) in model '{name}'")

        years = future['date'].dt.year.unique().tolist()
        for yr in sorted(years):
            for apply_var, target_value in targets[name]:
                scope_name = f"{target_value}_{yr}"
                if scope_name not in scope_data:
                    scope_data[scope_name] = {
                        'year': yr,
                        'target_value': target_value,
                        'apply_var': apply_var,
                        'nodes': {name},
                        'earliest_date': future['date'].min(),
                    }
                else:
                    existing = scope_data[scope_name]
                    if existing['apply_var'] != apply_var:
                        fermatrica_error(
                            f"Scope '{scope_name}' has conflicting apply_vars: "
                            f"'{existing['apply_var']}' (from {next(iter(existing['nodes']))}) "
                            f"vs '{apply_var}' (from {name})"
                        )
                    scope_data[scope_name]['nodes'].add(name)

    if not scope_data:
        fermatrica_error("No future data (listed==4) found in any model")
    return scope_data


def _generate_scopes_df(scope_data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for scope_name, info in scope_data.items():
        yr = info['year']
        first_year = min(info['year'] for info in scope_data.values())
        if yr == first_year:
            earliest = max(
                sd['earliest_date']
                for sn, sd in scope_data.items()
                if sd['year'] == first_year
            )
            date_start = earliest.strftime('%Y-%m-%d')
        else:
            date_start = f"{yr}-01-01"
        date_end = f"{yr}-12-31"
        ref_date_start = f"{int(yr) - 1}-01-01"
        ref_date_end = f"{int(yr) - 1}-12-31"

        rows.append({
            'scope_name': scope_name,
            'date_start': date_start,
            'date_end': date_end,
            'ref_date_start': ref_date_start,
            'ref_date_end': ref_date_end,
            'apply_vars': info['apply_var'],
            'target': info['target_value'],
            'plan_period': 'exact',
            'inflation_multiplier': 1.0,
            'seasonality': '',
        })
    return pd.DataFrame(rows, columns=[
        'scope_name', 'date_start', 'date_end',
        'ref_date_start', 'ref_date_end', 'apply_vars', 'target',
        'plan_period', 'inflation_multiplier', 'seasonality',
    ])


def _generate_allocations_df(
    trans_data: dict[str, dict],
    scope_data: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for scope_name, info in scope_data.items():
        all_items_set: set[str] = set()
        for node_name in info['nodes']:
            all_items_set.update(trans_data[node_name]['all_items'])
        for item in sorted(all_items_set):
            for opt_name in ('zero', 'predefined'):
                rows.append({
                    'option': opt_name,
                    'scope_name': scope_name,
                    'item': item,
                    'val': 0,
                })
    return pd.DataFrame(rows, columns=['option', 'scope_name', 'item', 'val'])


def _generate_media_prices_df(
    trans_data: dict[str, dict],
    scope_data: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for scope_name, info in scope_data.items():
        for node_name in info['nodes']:
            for item in sorted(trans_data[node_name]['all_items']):
                rows.append({
                    'scope_name': scope_name,
                    'item': item,
                    'multiplier': 1,
                })
    return pd.DataFrame(rows, columns=['scope_name', 'item', 'multiplier'])


def _generate_seasonality_df(
    trans_data: dict[str, dict],
    scope_data: dict[str, dict],
) -> pd.DataFrame:
    all_tool_cols: set[str] = set()
    scope_tools: dict[str, set[str]] = {}

    for scope_name, info in scope_data.items():
        bdg_set: set[str] = set()
        for node_name in info['nodes']:
            bdg_set.update(trans_data[node_name]['bdg_items'])
        tool_cols = {i.replace('bdg_', '', 1) for i in bdg_set}
        scope_tools[scope_name] = tool_cols
        all_tool_cols.update(tool_cols)

    tool_cols_sorted = sorted(all_tool_cols)
    columns = ['scope_name', 'month'] + tool_cols_sorted
    flat_val = 1.0 / 12.0

    rows = []
    for scope_name in scope_data:
        for month in range(1, 13):
            row_data = {'scope_name': scope_name, 'month': month}
            for tc in tool_cols_sorted:
                row_data[tc] = flat_val if tc in scope_tools[scope_name] else None
            rows.append(row_data)

    df = pd.DataFrame(rows, columns=columns)
    for tc in tool_cols_sorted:
        if all(pd.isna(df[tc])):
            df = df.drop(columns=[tc])
    return df


def _write_xlsx(
    options_path: str,
    output_xlsx: str,
    dt_scopes: pd.DataFrame,
    dt_alloc: pd.DataFrame,
    dt_media: pd.DataFrame,
    dt_ssn: pd.DataFrame,
) -> None:
    os.makedirs(os.path.dirname(output_xlsx) or '.', exist_ok=True)
    shutil.copy2(options_path, output_xlsx)

    wb = load_workbook(output_xlsx)
    for sheet_name in ('scopes', 'allocations', 'media_prices', 'seasonality'):
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]

    def _write_df_sheet(ws, df):
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.cell(row=1, column=col_idx, value=col_name)
        for row_idx, row in enumerate(df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        for col_idx in range(1, len(df.columns) + 1):
            max_width = max(
                len(str(df.columns[col_idx - 1])),
                max((len(str(row[col_idx - 1])) for row in df.itertuples(index=False)), default=0)
            ) + 2
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_width, 50)

    ws_scopes = wb.create_sheet('scopes', 0)
    _write_df_sheet(ws_scopes, dt_scopes)
    ws_alloc = wb.create_sheet('allocations')
    _write_df_sheet(ws_alloc, dt_alloc)
    ws_media = wb.create_sheet('media_prices')
    _write_df_sheet(ws_media, dt_media)
    ws_ssn = wb.create_sheet('seasonality')
    _write_df_sheet(ws_ssn, dt_ssn)

    wb.save(output_xlsx)


def _generate_adhoc_files(
    project_root: str,
    edges: list[tuple[str, str]] | None,
    force: bool,
) -> None:
    adhoc_dir = os.path.join(project_root, 'adhoc')
    os.makedirs(adhoc_dir, exist_ok=True)

    def _write(path, content):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    _write(os.path.join(adhoc_dir, '__init__.py'),
           "# Adhoc reporting and translator functions for this project's pipeline.\n")

    _write(os.path.join(adhoc_dir, 'translators.py'),
           '"""\n'
           'Adhoc translator functions.\n'
           '\n'
           'Define custom budget -> media translators here if needed.\n'
           "Referenced by the 'function' column in translation sheets.\n"
           '\n'
           'All highest-level translators must follow the standard signature:\n'
           '    def my_translator(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):\n'
           '        return ds\n'
           '"""\n')

    reporting_lines = ['"""\n']
    if edges:
        reporting_lines.append('Adhoc reporting and merge functions for MetaModel DAG.\n')
    else:
        reporting_lines.append('Adhoc reporting functions.\n')
    reporting_lines.append('"""\n')
    reporting_lines.append('import pandas as pd\n')
    reporting_lines.append('\n')
    if edges:
        for i, (left, right) in enumerate(edges, 1):
            reporting_lines.append('\n')
            reporting_lines.append(
                f'def pred_merge_{i}(ds_cur: pd.DataFrame, dt_pred_prev: pd.DataFrame, allow_past: bool = False):\n'
            )
            reporting_lines.append('    """\n')
            reporting_lines.append(
                f'    Merge predictions from upstream node ({left}) into current node ({right}) dataset.\n'
            )
            reporting_lines.append('\n')
            reporting_lines.append('    TODO: Implement your merge logic. Common patterns:\n')
            reporting_lines.append('\n')
            reporting_lines.append('    - Merge on date only (category -> SKU broadcast):\n')
            reporting_lines.append("        tmp = dt_pred_prev.loc[:, ['date', 'predicted']].copy()\n")
            reporting_lines.append("        ds_cur = pd.merge(ds_cur, tmp, how='left', on='date', sort=False)\n")
            reporting_lines.append('\n')
            reporting_lines.append("    - Merge on date + superbrand (brand-level -> brand-level):\n")
            reporting_lines.append("        tmp = dt_pred_prev.loc[:, ['date', 'superbrand', 'predicted']].copy()\n")
            reporting_lines.append("        ds_cur = pd.merge(ds_cur, tmp, how='left', on=['date', 'superbrand'], sort=False)\n")
            reporting_lines.append('\n')
            reporting_lines.append('    Args:\n')
            reporting_lines.append("        ds_cur: Current node's dataset (mutated in-place and returned).\n")
            reporting_lines.append("        dt_pred_prev: Predecessor node's full dt_pred (all periods).\n")
            reporting_lines.append('        allow_past: Whether to apply to historical rows (listed <= 3).\n')
            reporting_lines.append('\n')
            reporting_lines.append('    Returns:\n')
            reporting_lines.append('        Modified dataset.\n')
            reporting_lines.append('    """\n')
            reporting_lines.append('    return ds_cur\n')

    _write(os.path.join(adhoc_dir, 'reporting.py'), ''.join(reporting_lines))


def _generate_reporting_scripts(
    options_path: str,
    node_names: list[str],
    model_paths: dict[str, str],
    predecessors: dict[str, dict],
    project_root: str,
    force: bool,
) -> None:
    scripts_dir = os.path.join(project_root, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)

    for name in node_names:
        pred = predecessors[name]
        content = _per_node_script(
            options_path=options_path,
            node_name=name,
            model_paths=model_paths,
            included_nodes=pred["nodes"],
            node_edges=pred["edges"],
        )
        script_path = os.path.join(scripts_dir, f'reporting_{name}.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)


def _per_node_script(
    options_path: str,
    node_name: str,
    model_paths: dict[str, str],
    included_nodes: list[str],
    node_edges: list[tuple[str, str, int]],
) -> str:
    """
    Generate a self-contained reporting script for one DAG node.

    Loads the canonical template (``pipeline/templates/reporting_template.py``)
    and substitutes its ``{{ TOKEN }}`` placeholders via ``str.replace``.
    If the node has upstream predecessors they are included so the
    script builds a MetaModel covering the full chain up to this node.

    :param options_path: XLSX path baked into the script.
    :param node_name: This script's target node.
    :param model_paths: ``{node_name: parent_dir}`` mapping.
    :param included_nodes: Topologically sorted nodes to load (this
        node plus all transitive predecessors).
    :param node_edges: Edges subset as ``[(left, right, merge_idx)]``.
    :return: Complete Python source code as a string.
    """
    multi = len(included_nodes) > 1
    has_edges = bool(node_edges)

    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'templates',
        'reporting_template.py',
    )
    with open(template_path, encoding='utf-8') as f:
        template = f.read()

    adhoc_imports = 'import code_py.adhoc.translators'
    if has_edges:
        adhoc_imports += '\nfrom code_py.adhoc import reporting'

    load_block = _node_load_block(model_paths, included_nodes, has_edges)

    if multi:
        nodes_dict = ", ".join(f"'{n}': node_{n}" for n in included_nodes)
        edges_list = ", ".join(
            f'("{left}", "{right}", reporting.pred_merge_{idx})'
            for left, right, idx in node_edges
        )
        nodes_repr = f"{{{nodes_dict}}}"
        edges_repr = f"[{edges_list}]"
        last_node_key = node_name
    else:
        nodes_repr = "node"
        edges_repr = ""
        last_node_key = "model"
        # single-node MetaModels auto-name their node "model" and take no edges
        template = template.replace(', edges="{{ EDGES_LIST }}"', '')

    script = (template
              .replace('"{{ ADHOC_IMPORTS }}"', adhoc_imports)
              .replace('{{ OPTIONS_PATH }}', _esc(options_path))
              .replace('    "{{ NODE_LOAD_BLOCK }}"', load_block)
              .replace('"{{ NODES_DICT }}"', nodes_repr)
              .replace('"{{ EDGES_LIST }}"', edges_repr)
              .replace('{{ LAST_NODE }}', last_node_key))

    return script


def _node_load_block(
    model_paths: dict[str, str],
    included_nodes: list[str],
    has_edges: bool,
) -> str:
    """
    Build the ``{{ NODE_LOAD_BLOCK }}`` substitution for one script.

    Emits per-node latest ``model_*`` auto-detection followed by
    ``ModelNode(model_path=..., adhoc_code=..., language='english')`` —
    trans_dict / ssn_dict / media_prices are wired via the MetaModel
    ``compute_context`` and are no longer passed per node.

    :param model_paths: ``{node_name: parent_dir}`` mapping.
    :param included_nodes: Topologically sorted nodes to load.
    :param has_edges: Whether the script's DAG subset has edges.
    :return: Indented code block (4-space indent, safe to drop in place
        of the template's ``    "{{ NODE_LOAD_BLOCK }}"`` line).
    """
    lines = ['    adhoc_modules = [code_py.adhoc.translators]']
    if has_edges:
        lines.append('    adhoc_modules.append(code_py.adhoc.reporting)')

    last = included_nodes[-1]

    if len(included_nodes) == 1:
        n = included_nodes[0]
        lines.append(f'    pth = os.path.join("{_esc(model_paths[n])}")')
        lines.append('    saved_dirs = [d for d in os.listdir(pth)')
        lines.append("                      if d.startswith('model_') and not d.endswith('_dash') "
                     "and os.path.isdir(os.path.join(pth, d))]")
        lines.append('    saved_dirs.sort(reverse=True)')
        lines.append('    model_tag = saved_dirs[0].replace("model_", "")')
        lines.append('    pth_full = os.path.join(pth, "model_" + model_tag)')
        lines.append('    node = fmr.ModelNode(model_path=pth_full, adhoc_code=adhoc_modules,'
                     " language='english', missed_stop=True)")
        lines.append('    model_tag_last = model_tag')
    else:
        for n in included_nodes:
            lines.append(f'    pth_{n} = os.path.join("{_esc(model_paths[n])}")')
            lines.append(f'    saved_dirs_{n} = [d for d in os.listdir(pth_{n})')
            lines.append(f"                      if d.startswith('model_') and not d.endswith('_dash') "
                         f"and os.path.isdir(os.path.join(pth_{n}, d))]")
            lines.append(f'    saved_dirs_{n}.sort(reverse=True)')
            lines.append(f'    model_tag_{n} = saved_dirs_{n}[0].replace("model_", "")')
            lines.append(f'    pth_full_{n} = os.path.join(pth_{n}, "model_" + model_tag_{n})')
            lines.append(f'    node_{n} = fmr.ModelNode(model_path=pth_full_{n}, '
                         'adhoc_code=adhoc_modules, language="english", missed_stop=True)')
        lines.append(f'    model_tag_last = model_tag_{last}')

    return '\n'.join(lines)


def _esc(s: str) -> str:
    """
    Escape backslashes for safe embedding in generated Python string
    literals.

    :param s: Raw path string.
    :return: Path with ``\\`` doubled to ``\\\\``.
    """
    return s.replace('\\', '\\\\')
