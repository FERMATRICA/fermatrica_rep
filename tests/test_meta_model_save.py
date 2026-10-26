"""Tests for MetaModel.save / MetaModel.load round-trip.

Covers folder layout (``meta_<tag>/`` + ``model_<node>/`` +
``meta_conf.toml``), merge-function serialization, single-node
delegation, missing ModelRep auto-defaulting, ``scope_names`` round-trip,
legacy ``meta_conf.xlsx`` fallback,
``include_context_path`` copying, and timestamp defaults. Mock-based and
fast — the load half monkeypatches ``model_load_ext`` / ``ModelNode.load``.
"""
import datetime
import logging
import lzma
import os
import pickle
import re
import tomllib

import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class _MockConf:
    def __init__(self, name):
        self.name = name
        self.brands = []
        self.Y_var = 'kpi'
        self.model_type = 'OLS'


class _MockObj:
    pass


class _MockModel:
    """Minimal model whose ``save`` writes the four artifact files for real."""

    def __init__(self, name):
        self.conf = _MockConf(name)
        self.obj = _MockObj()

    def save(self, ds, model_rep=None, path='', save_tag=None):
        if path == '':
            path = os.path.join(os.getcwd(), 'code_py', 'model_data')
        if save_tag is None:
            save_tag = re.sub(' ', '_', datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            save_tag = re.sub(r':', '', save_tag)
        save_path = os.path.join(path, 'model_' + save_tag)
        os.makedirs(save_path, exist_ok=True)

        if ds is not None:
            ds.to_pickle(os.path.join(save_path, "dt_p.pkl.zstd"), compression="zstd")
        if model_rep is not None:
            with lzma.open(os.path.join(save_path, 'model_rep.pkl.lzma'), 'wb') as handle:
                pickle.dump(model_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame({'name': [self.conf.name]}).to_excel(
            os.path.join(save_path, 'model_conf.xlsx'), index=False)
        with lzma.open(os.path.join(save_path, 'model_objects.pkl.lzma'), 'wb') as handle:
            pickle.dump({'stub': True}, handle)
        return save_path


class _MockModelRep:
    language = 'english'
    trans_dict = None
    adhoc_code_src = None
    ssn_dict = None
    media_prices = None
    option_summary_adhoc = None


def _merge_uba_sales(ds_cur, dt_pred_prev, allow_past):
    """Module-level merge function — must stay importable for serialization."""
    return ds_cur


def _basic_ds(name='x'):
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='MS'),
        'superbrand': [name] * 3,
        'listed': [4] * 3,
    })


def _mock_node(name):
    return ModelNode(
        model=_MockModel(name),
        ds=_basic_ds(name),
        model_rep=_MockModelRep(),
    )


def _fake_model_load_ext(node_path, missed_stop=False):
    """Stand-in for ``model_load_ext``: reads ds back from the saved dir."""
    from fermatrica_utils import read_pickle_compat

    ds = read_pickle_compat(os.path.join(node_path, "dt_p.pkl.zstd"))
    return _MockModel('loaded'), ds, ''


def _read_meta_conf(meta_dir):
    with open(os.path.join(meta_dir, 'meta_conf.toml'), 'rb') as handle:
        return tomllib.load(handle)


def _write_legacy_xlsx_conf(meta_dir, graph):
    with pd.ExcelWriter(os.path.join(meta_dir, 'meta_conf.xlsx'), engine='openpyxl') as writer:
        graph.to_excel(writer, sheet_name='model_graph', index=False)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestMetaModelSave:

    def test_multi_node_layout(self, tmp_path, monkeypatch):
        n1 = _mock_node('uba')
        n2 = _mock_node('sales')
        mm = MetaModel(nodes={'uba': n1, 'sales': n2}, edges=[('uba', 'sales', _merge_uba_sales)])

        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        assert meta_dir == os.path.join(str(tmp_path), 'meta_test')
        assert os.path.isfile(os.path.join(meta_dir, 'meta_conf.toml'))
        assert os.path.isdir(os.path.join(meta_dir, 'model_uba'))
        assert os.path.isdir(os.path.join(meta_dir, 'model_sales'))
        for node_name in ('uba', 'sales'):
            node_dir = os.path.join(meta_dir, f'model_{node_name}')
            assert os.path.isfile(os.path.join(node_dir, 'model_conf.xlsx'))
            assert os.path.isfile(os.path.join(node_dir, 'model_objects.pkl.lzma'))
            assert os.path.isfile(os.path.join(node_dir, 'model_rep.pkl.lzma'))
            assert os.path.isfile(os.path.join(node_dir, 'dt_p.pkl.zstd'))

        conf = _read_meta_conf(meta_dir)
        assert len(conf['graph']) == 1
        assert conf['graph'][0]['left'] == 'uba'
        assert conf['graph'][0]['right'] == 'sales'
        assert conf['graph'][0]['merge_fn'] == f'{_merge_uba_sales.__module__}.{_merge_uba_sales.__qualname__}'

    def test_single_node_delegates(self, tmp_path):
        node = _mock_node('only')
        mm = MetaModel(nodes={'only': node})

        node_dir = mm.save(path=str(tmp_path), save_tag='solo')

        assert node_dir == os.path.join(str(tmp_path), 'model_solo')
        assert os.path.isfile(os.path.join(node_dir, 'meta_conf.toml'))
        assert os.path.isfile(os.path.join(node_dir, 'model_rep.pkl.lzma'))
        assert os.path.isfile(os.path.join(node_dir, 'dt_p.pkl.zstd'))

    def test_timestamp_default(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path))
        assert re.match(r'^meta_\d{4}-\d{2}-\d{2}_\d{6}$', os.path.basename(meta_dir))

    def test_merge_fn_none_writes_empty(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='t')
        conf = _read_meta_conf(meta_dir)
        assert 'merge_fn' not in conf['graph'][0]

    def test_merge_fn_string_passthrough(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', 'code_py.adhoc.reporting.pred_merge_1')])
        meta_dir = mm.save(path=str(tmp_path), save_tag='t')
        conf = _read_meta_conf(meta_dir)
        assert conf['graph'][0]['merge_fn'] == 'code_py.adhoc.reporting.pred_merge_1'

    def test_merge_fn_lambda_raises(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', lambda ds_cur, dt_pred_prev, allow_past: ds_cur)])
        with pytest.raises(FermatricaError, match="cannot be re-imported"):
            mm.save(path=str(tmp_path), save_tag='t')

    def test_merge_fn_closure_raises(self, tmp_path):
        def closure(ds_cur, dt_pred_prev, allow_past):
            return ds_cur

        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', closure)])
        with pytest.raises(FermatricaError, match="cannot be re-imported"):
            mm.save(path=str(tmp_path), save_tag='t')

    def test_scope_names_written(self, tmp_path):
        n1 = _mock_node('a')
        n1.scope_names = {'s1', 's2'}
        n2 = _mock_node('b')
        mm = MetaModel(nodes={'a': n1, 'b': n2}, edges=[('a', 'b', None)])

        meta_dir = mm.save(path=str(tmp_path), save_tag='t')

        conf = _read_meta_conf(meta_dir)
        assert conf['node_settings']['a']['scope_names'] == ['s1', 's2']

    def test_scope_names_absent_when_unset(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='t')
        conf = _read_meta_conf(meta_dir)
        assert 'node_settings' not in conf

    def test_include_context_path_copied(self, tmp_path):
        src = tmp_path / 'src_context.xlsx'
        pd.DataFrame({'item': ['bdg_tv']}).to_excel(str(src), index=False)
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])

        meta_dir = mm.save(path=str(tmp_path), save_tag='t',
                           include_context_path=str(src))
        assert os.path.isfile(os.path.join(meta_dir, 'context.xlsx'))
        assert os.path.getsize(os.path.join(meta_dir, 'context.xlsx')) == os.path.getsize(str(src))

    def test_include_context_path_single_node(self, tmp_path):
        src = tmp_path / 'src_context.xlsx'
        pd.DataFrame({'item': ['bdg_tv']}).to_excel(str(src), index=False)
        mm = MetaModel(nodes={'a': _mock_node('a')})

        node_dir = mm.save(path=str(tmp_path), save_tag='solo',
                           include_context_path=str(src))
        assert os.path.isfile(os.path.join(node_dir, 'context.xlsx'))
        assert not os.path.exists(os.path.join(os.path.dirname(node_dir), 'context.xlsx'))

    def test_existing_meta_dir_raises(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        mm.save(path=str(tmp_path), save_tag='test')
        with pytest.raises(FermatricaError, match="already exists"):
            mm.save(path=str(tmp_path), save_tag='test')

    def test_invalid_save_tag_raises(self, tmp_path):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        with pytest.raises(FermatricaError, match="invalid characters"):
            mm.save(path=str(tmp_path), save_tag='../escape')

    def test_invalid_node_name_raises(self, tmp_path):
        mm = MetaModel(nodes={'a/b': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a/b', 'b', None)])
        with pytest.raises(FermatricaError, match="invalid characters"):
            mm.save(path=str(tmp_path), save_tag='test')

    def test_failed_save_cleans_partial_dir(self, tmp_path):
        class _FailingSaveModel(_MockModel):
            def save(self, ds, model_rep=None, path='', save_tag=None):
                raise OSError('disk full')

        failing_node = ModelNode(
            model=_FailingSaveModel('a'),
            ds=_basic_ds('a'),
            model_rep=_MockModelRep(),
        )
        ok_node = _mock_node('b')
        mm = MetaModel(nodes={'a': failing_node, 'b': ok_node}, edges=[('a', 'b', None)])

        with pytest.raises(OSError):
            mm.save(path=str(tmp_path), save_tag='t')
        assert not os.path.exists(os.path.join(str(tmp_path), 'meta_t'))

    def test_merge_validation_before_dir_creation(self, tmp_path):
        def closure(ds_cur, dt_pred_prev, allow_past):
            return ds_cur

        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', closure)])
        with pytest.raises(FermatricaError, match="cannot be re-imported"):
            mm.save(path=str(tmp_path), save_tag='t')
        assert not os.path.exists(os.path.join(str(tmp_path), 'meta_t'))

    def test_static_flag_written(self, tmp_path):
        n1 = _mock_node('a')
        n1.static = True
        n2 = _mock_node('b')
        mm = MetaModel(nodes={'a': n1, 'b': n2}, edges=[('a', 'b', None)])

        meta_dir = mm.save(path=str(tmp_path), save_tag='t')

        conf = _read_meta_conf(meta_dir)
        assert conf['node_settings']['a']['static'] is True
        assert 'b' not in conf['node_settings']


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class TestMetaModelLoadMulti:

    def test_multi_node_round_trip(self, tmp_path, monkeypatch):
        n1 = _mock_node('uba')
        n2 = _mock_node('sales')
        mm = MetaModel(nodes={'uba': n1, 'sales': n2}, edges=[('uba', 'sales', _merge_uba_sales)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)

        assert set(mm2.nodes.keys()) == {'uba', 'sales'}
        assert len(mm2.edges) == 1
        left, right, merge_fn = mm2.edges[0]
        assert left == 'uba' and right == 'sales'
        assert merge_fn == f'{_merge_uba_sales.__module__}.{_merge_uba_sales.__qualname__}'
        assert mm2.nodes['uba'].model_rep is not None
        assert mm2.nodes['sales'].model_rep is not None

    def test_parallel_branches_load(self, tmp_path, monkeypatch):
        mm = MetaModel(
            nodes={'a': _mock_node('a'), 'b': _mock_node('b'), 'c': _mock_node('c')},
            edges=[('a', 'c', None), ('b', 'c', None)],
        )
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert set(mm2.nodes.keys()) == {'a', 'b', 'c'}
        assert mm2.edges == [('a', 'c', None), ('b', 'c', None)]

    def test_missing_model_rep_uses_default(self, tmp_path, monkeypatch, caplog):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')
        os.remove(os.path.join(meta_dir, 'model_a', 'model_rep.pkl.lzma'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        with caplog.at_level(logging.WARNING, logger='fermatrica_rep.meta_model.node'):
            mm2 = MetaModel.load(meta_dir)

        from fermatrica_rep.meta_model.model_rep import ModelRep
        assert isinstance(mm2.nodes['a'].model_rep, ModelRep)
        assert any('ModelRep not found' in r.message for r in caplog.records)

    def test_missing_node_folder_raises(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')
        import shutil
        shutil.rmtree(os.path.join(meta_dir, 'model_b'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        with pytest.raises(FermatricaError, match="Node folder not found"):
            MetaModel.load(meta_dir)

    def test_scope_names_restored(self, tmp_path, monkeypatch):
        n1 = _mock_node('a')
        n1.scope_names = {'s1', 's2'}
        n2 = _mock_node('b')
        mm = MetaModel(nodes={'a': n1, 'b': n2}, edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert mm2.nodes['a'].scope_names == {'s1', 's2'}
        assert mm2.nodes['b'].scope_names is None

    def test_empty_scope_names_round_trip(self, tmp_path, monkeypatch):
        n1 = _mock_node('a')
        n1.scope_names = set()
        n2 = _mock_node('b')
        mm = MetaModel(nodes={'a': n1, 'b': n2}, edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert mm2.nodes['a'].scope_names == set()

    def test_static_flag_restored(self, tmp_path, monkeypatch):
        n1 = _mock_node('a')
        n1.static = True
        n2 = _mock_node('b')
        mm = MetaModel(nodes={'a': n1, 'b': n2}, edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert mm2.nodes['a'].static is True
        assert mm2.nodes['b'].static is False

    def test_no_scopes_attribute_on_load(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert not hasattr(mm2, 'scopes')

    def test_legacy_model_prefixed_graph_names(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'uba': _mock_node('uba'), 'sales': _mock_node('sales')},
                       edges=[('uba', 'sales', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        conf_path = os.path.join(meta_dir, 'meta_conf.toml')
        with open(conf_path, encoding='utf-8') as handle:
            text = handle.read()
        with open(conf_path, 'w', encoding='utf-8') as handle:
            handle.write(text.replace('"uba"', '"model_uba"').replace('"sales"', '"model_sales"'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert set(mm2.nodes.keys()) == {'uba', 'sales'}
        assert mm2.edges == [('uba', 'sales', None)]

    def test_whitespace_padded_graph_names(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'uba': _mock_node('uba'), 'sales': _mock_node('sales')},
                       edges=[('uba', 'sales', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        conf_path = os.path.join(meta_dir, 'meta_conf.toml')
        with open(conf_path, encoding='utf-8') as handle:
            text = handle.read()
        with open(conf_path, 'w', encoding='utf-8') as handle:
            handle.write(text.replace('"uba"', '"uba   "').replace('"sales"', '"sales   "'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert set(mm2.nodes.keys()) == {'uba', 'sales'}
        assert mm2.edges == [('uba', 'sales', None)]

    def test_missing_merge_fn_column_warns_legacy(self, tmp_path, monkeypatch, caplog):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        os.remove(os.path.join(meta_dir, 'meta_conf.toml'))
        _write_legacy_xlsx_conf(meta_dir, pd.DataFrame({
            'left': ['a'], 'right': ['b'],
        }))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        with caplog.at_level(logging.WARNING, logger='fermatrica_rep.meta_model.meta_model'):
            mm2 = MetaModel.load(meta_dir)
        assert mm2.edges == [('a', 'b', None)]
        assert any("'merge_fn' not found" in r.message for r in caplog.records)
        assert any('legacy meta config' in r.message for r in caplog.records)

    def test_missing_graph_left_raises(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', None)])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        conf_path = os.path.join(meta_dir, 'meta_conf.toml')
        with open(conf_path, encoding='utf-8') as handle:
            text = handle.read()
        with open(conf_path, 'w', encoding='utf-8') as handle:
            handle.write(text.replace('left = "a"\n', ''))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        with pytest.raises(FermatricaError, match="Invalid .*graph.*entry"):
            MetaModel.load(meta_dir)

    def test_merge_fn_whitespace_normalized(self, tmp_path, monkeypatch):
        mm = MetaModel(nodes={'a': _mock_node('a'), 'b': _mock_node('b')},
                       edges=[('a', 'b', 'code_py.adhoc.reporting.pred_merge_1')])
        meta_dir = mm.save(path=str(tmp_path), save_tag='test')

        conf_path = os.path.join(meta_dir, 'meta_conf.toml')
        with open(conf_path, encoding='utf-8') as handle:
            text = handle.read()
        with open(conf_path, 'w', encoding='utf-8') as handle:
            handle.write(text.replace('pred_merge_1"', 'pred_merge_1 "'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(meta_dir)
        assert mm2.edges == [('a', 'b', 'code_py.adhoc.reporting.pred_merge_1')]


class TestMetaModelLoadSingle:

    def test_single_node_load_auto_detect(self, tmp_path, monkeypatch):
        node = _mock_node('only')
        mm = MetaModel(nodes={'only': node})
        node_dir = mm.save(path=str(tmp_path), save_tag='solo')

        mock_node = ModelNode(
            model=_MockModel('loaded'),
            ds=_basic_ds(),
            model_rep=_MockModelRep(),
        )
        monkeypatch.setattr(ModelNode, 'load', lambda p: mock_node)

        mm2 = MetaModel.load(node_dir)
        assert list(mm2.nodes.keys()) == ['model']
        assert mm2.nodes['model'] is mock_node
        assert len(mm2.edges) == 0

    def test_single_node_scope_names_round_trip(self, tmp_path, monkeypatch):
        node = _mock_node('only')
        node.scope_names = {'s1'}
        mm = MetaModel(nodes={'only': node})
        node_dir = mm.save(path=str(tmp_path), save_tag='solo')

        assert os.path.isfile(os.path.join(node_dir, 'meta_conf.toml'))

        monkeypatch.setattr(
            'fermatrica_rep.meta_model.node.model_load_ext', _fake_model_load_ext)

        mm2 = MetaModel.load(node_dir)
        assert list(mm2.nodes.keys()) == ['model']
        assert mm2.nodes['model'].scope_names == {'s1'}
