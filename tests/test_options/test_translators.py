import pytest
import pandas as pd
import numpy as np

from fermatrica_rep.options.translators import basic_fix, price, basic, tv_complex, nat_tv_complex, reg_tv_complex, nat_tv_sponsor_complex, brnd_nat_tv_complex


class MockModelConf:
    pass


class TestBasicFix:

    def test_sets_value_on_masked_rows(self):
        ds = pd.DataFrame({'var': [1.0, 2.0, 3.0, 4.0]})
        ds_mask = pd.Series([True, False, True, False])
        md = MockModelConf()
        result = basic_fix(md, ds, None, 'key', 'var', 999.0, ds_mask)
        assert result.loc[0, 'var'] == 999.0
        assert result.loc[1, 'var'] == 2.0
        assert result.loc[2, 'var'] == 999.0
        assert result.loc[3, 'var'] == 4.0

    def test_all_masked(self):
        ds = pd.DataFrame({'var': [1.0, 2.0]})
        ds_mask = pd.Series([True, True])
        result = basic_fix(MockModelConf(), ds, None, 'key', 'var', 100.0, ds_mask)
        assert (result['var'] == 100.0).all()

    def test_no_masked_rows(self):
        ds = pd.DataFrame({'var': [1.0, 2.0]})
        ds_mask = pd.Series([False, False])
        result = basic_fix(MockModelConf(), ds, None, 'key', 'var', 100.0, ds_mask)
        assert result['var'].tolist() == [1.0, 2.0]

    def test_float_val(self):
        ds = pd.DataFrame({'var': [0.0]})
        ds_mask = pd.Series([True])
        result = basic_fix(MockModelConf(), ds, None, 'key', 'var', 3.14, ds_mask)
        assert np.isclose(result.loc[0, 'var'], 3.14)

    def test_zero_val(self):
        ds = pd.DataFrame({'var': [5.0]})
        ds_mask = pd.Series([True])
        result = basic_fix(MockModelConf(), ds, None, 'key', 'var', 0.0, ds_mask)
        assert result.loc[0, 'var'] == 0.0

    def test_returns_dataframe(self):
        ds = pd.DataFrame({'var': [1.0, 2.0], 'other': ['a', 'b']})
        ds_mask = pd.Series([True, False])
        result = basic_fix(MockModelConf(), ds, None, 'key', 'var', 999.0, ds_mask)
        assert isinstance(result, pd.DataFrame)
        assert 'other' in result.columns


class TestPrice:

    def test_price_applied_via_ratio(self):
        md = MockModelConf()
        md.price_var = 'price'
        ds = pd.DataFrame({
            'price': [100.0, 200.0, 300.0],
            'price_back': [100.0, 100.0, 100.0]
        })
        ds_mask = pd.Series([True, True, False])
        result = price(md, ds, None, 'key', 'price', 0.8, ds_mask)
        assert np.isclose(result.loc[0, 'price'], 80.0)
        assert np.isclose(result.loc[1, 'price'], 80.0)
        assert np.isclose(result.loc[2, 'price'], 300.0)

    def test_price_increase(self):
        md = MockModelConf()
        md.price_var = 'price'
        ds = pd.DataFrame({
            'price': [100.0],
            'price_back': [100.0]
        })
        ds_mask = pd.Series([True])
        result = price(md, ds, None, 'key', 'price', 1.5, ds_mask)
        assert np.isclose(result.loc[0, 'price'], 150.0)

    def test_price_zero(self):
        md = MockModelConf()
        md.price_var = 'price'
        ds = pd.DataFrame({
            'price': [100.0],
            'price_back': [100.0]
        })
        ds_mask = pd.Series([True])
        result = price(md, ds, None, 'key', 'price', 0.0, ds_mask)
        assert np.isclose(result.loc[0, 'price'], 0.0)

    def test_custom_price_var_name(self):
        md = MockModelConf()
        md.price_var = 'price_rel'
        ds = pd.DataFrame({
            'price_rel': [1.0, 2.0],
            'price_rel_back': [1.0, 1.0]
        })
        ds_mask = pd.Series([True, False])
        result = price(md, ds, None, 'key', 'price_rel', 0.5, ds_mask)
        assert np.isclose(result.loc[0, 'price_rel'], 0.5)

    def test_masked_only(self):
        md = MockModelConf()
        md.price_var = 'price'
        ds = pd.DataFrame({
            'price': [100.0, 200.0],
            'price_back': [100.0, 200.0]
        })
        ds_mask = pd.Series([True, False])
        result = price(md, ds, None, 'key', 'price', 0.75, ds_mask)
        assert np.isclose(result.loc[0, 'price'], 75.0)
        assert np.isclose(result.loc[1, 'price'], 200.0)


class TestBasic:

    def test_basic_translation_with_seasonality(self):
        md = MockModelConf()
        ssn_vals = [0.1, 0.2, 0.3, 0.4]
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=4, freq='MS'),
            'opt_media_ssn_tv': ssn_vals,
            'brnd_tv_ots_clip': [0.0] * 4
        })
        ds_mask = pd.Series([True] * 4)
        result = basic(md, ds, None, 'bdg_tv', 'brnd_tv_ots_clip', 100.0, ds_mask)

        expected = [s * 100.0 * 1e6 / 1.0 for s in ssn_vals]
        np.testing.assert_array_almost_equal(result['brnd_tv_ots_clip'].values, expected)

    def test_media_price_defaults_to_1(self):
        md = MockModelConf()
        md.trans_path_df = pd.DataFrame({
            'variable': ['other_var'],
            'price': [0.1]
        })
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=2, freq='MS'),
            'opt_media_ssn_radio': [0.5, 0.5],
            'my_var': [0.0, 0.0]
        })
        ds_mask = pd.Series([True, True])
        result = basic(md, ds, None, 'bdg_radio', 'my_var', 100.0, ds_mask)

        expected = 0.5 * 1.0 * 100.0 * 1e6 / 1.0
        assert np.allclose(result['my_var'].values[0], expected)

    def test_long_period_no_seasonality_correction(self):
        md = MockModelConf()
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=4, freq='YS'),
            'opt_media_ssn_tv': [0.3, 0.7, 0.5, 0.9],
            'my_var': [0.0, 0.0, 0.0, 0.0]
        })
        ds_mask = pd.Series([True, False, False, True])
        ssn_vals = ds.loc[ds_mask, 'opt_media_ssn_tv']
        result = basic(md, ds, None, 'bdg_tv', 'my_var', 200.0, ds_mask)
        expected_0 = ssn_vals.iloc[0] * 200.0 * 1e6 / 1.0
        assert np.isclose(result.loc[0, 'my_var'], expected_0)


class MockModelRep:
    pass


class TestTvComplex:

    @pytest.fixture
    def md(self):
        mc = MockModelConf()
        mc.trans_path_df = pd.DataFrame({
            'variable': ['bdg_tv'],
            'price': [0.05]
        })
        return mc

    @pytest.fixture
    def model_rep(self):
        mr = MockModelRep()
        mr.aud_coefs = pd.DataFrame({
            'year': [2024, 2024, 2024],
            'variable': ['ots_f_rolik_nat', 'ots_m_rolik_nat', 'ots_all_rolik_nat'],
            'coef': [1.0, 1.2, 1.1]
        })
        return mr

    @pytest.fixture
    def ds(self):
        np.random.seed(42)
        n = 12
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'listed': [4] * n,
            'superbrand': ['brand_x'] * n,
            'bs_key': [1] * n,
            'ots_f_rolik_nat': [0.0] * n,
            'ots_m_rolik_nat': [0.0] * n,
            'ots_all_rolik_nat': [0.0] * n,
            'opt_media_ssn_tv': [0.1] * n,
            'sku_coef': [0.5] * n,
            'umb_coef': [0.8] * n,
        })

    def test_populates_ots_columns(self, md, model_rep, ds):
        ds_mask = pd.Series([True] * len(ds))
        result = tv_complex(md, ds, model_rep, 'bdg_tv',
                            'umb, sku_coef, umb_coef', 100.0, ds_mask,
                            clip_dur=10, tv_type='rolik_nat', media_prefix='ots')
        assert np.all(result.loc[ds_mask, 'ots_f_rolik_nat'] > 0)
        assert np.all(result.loc[ds_mask, 'ots_m_rolik_nat'] > 0)
        assert np.all(result.loc[ds_mask, 'ots_all_rolik_nat'] > 0)

    def test_unmasked_rows_unchanged(self, md, model_rep, ds):
        ds_mask = pd.Series([True, False] * (len(ds) // 2))
        original = ds.copy()
        result = tv_complex(md, ds, model_rep, 'bdg_tv',
                            'umb, sku_coef, umb_coef', 100.0, ds_mask,
                            clip_dur=10, tv_type='rolik_nat', media_prefix='ots')
        pd.testing.assert_series_equal(
            result.loc[~ds_mask, 'ots_f_rolik_nat'],
            original.loc[~ds_mask, 'ots_f_rolik_nat']
        )

    def test_clip_duration_set(self, md, model_rep, ds):
        ds_mask = pd.Series([True] * len(ds))
        result = tv_complex(md, ds, model_rep, 'bdg_tv',
                            'umb, sku_coef, umb_coef', 100.0, ds_mask,
                            clip_dur=15, tv_type='rolik_nat', media_prefix='ots')
        assert np.all(result.loc[ds_mask, 'clipl_f_rolik_nat'] == 15)

    def test_golden_ots_value(self, md, model_rep, ds):
        ds_mask = pd.Series([True] * len(ds))
        result = tv_complex(md, ds, model_rep, 'bdg_tv',
                            'umb, sku_coef, umb_coef', 100.0, ds_mask,
                            clip_dur=10, tv_type='rolik_nat', media_prefix='ots')
        ssn = 0.1
        val = 100.0
        clip_dur = 10
        sku_coef = 0.5
        aud_coef = 1.0
        expected = ssn * val * 1.0 / clip_dur * 1e6 / 1.0 * sku_coef * aud_coef
        np.testing.assert_allclose(result.loc[0, 'ots_f_rolik_nat'], expected, rtol=1e-10)

    def test_4part_var_name(self, md, model_rep, ds):
        ds['sku_ots_f_rolik_nat'] = [0.0] * len(ds)
        ds['sku_clipl_f_rolik_nat'] = [0.0] * len(ds)
        ds['sku_coef_sku'] = [0.3] * len(ds)
        ds_mask = pd.Series([True] * len(ds))
        result = tv_complex(md, ds, model_rep, 'bdg_tv',
                            'umb, sku_, sku_coef_sku, umb_coef', 100.0, ds_mask,
                            clip_dur=10, tv_type='rolik_nat', media_prefix='ots')
        assert np.all(result.loc[ds_mask, 'sku_ots_f_rolik_nat'] > 0)


class TestNatTvComplex:

    @pytest.fixture
    def md(self):
        mc = MockModelConf()
        mc.trans_path_df = pd.DataFrame({
            'variable': ['bdg_tv'],
            'price': [0.05]
        })
        return mc

    @pytest.fixture
    def model_rep(self):
        mr = MockModelRep()
        mr.aud_coefs = pd.DataFrame({
            'year': [2024], 'variable': ['ots_f_rolik_nat'], 'coef': [1.0]
        })
        return mr

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
            'listed': [4, 4], 'superbrand': ['brand_x'] * 2,
            'ots_f_rolik_nat': [0.0, 0.0],
            'opt_media_ssn_tv': [0.5, 0.5],
            'sku_coef': [0.5, 0.5], 'umb_coef': [0.8, 0.8],
        })

    def test_smoke_delegates_to_tv_complex(self, md, model_rep, ds):
        ds_mask = pd.Series([True, True])
        result = nat_tv_complex(md, ds, model_rep, 'bdg_tv',
                                'umb, sku_coef, umb_coef', 100.0, ds_mask)
        assert np.all(result.loc[ds_mask, 'ots_f_rolik_nat'] > 0)


class TestRegTvComplex:

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
            'listed': [4, 4], 'superbrand': ['brand_x'] * 2,
            'ots_f_rolik_local': [0.0, 0.0],
            'ots_m_rolik_local': [0.0, 0.0],
            'opt_media_ssn_tv': [0.5, 0.5],
            'sku_coef': [0.5, 0.5], 'umb_coef': [0.8, 0.8],
        })

    def test_smoke_regional_tv(self, md, model_rep, ds):
        ds_mask = pd.Series([True, True])
        result = reg_tv_complex(md, ds, model_rep, 'bdg_tv',
                                'umb, sku_coef, umb_coef', 100.0, ds_mask)
        assert np.all(result.loc[ds_mask, 'ots_f_rolik_local'] > 0)

    @pytest.fixture
    def md(self):
        mc = MockModelConf()
        mc.trans_path_df = pd.DataFrame({
            'variable': ['bdg_tv'], 'price': [0.05]
        })
        return mc

    @pytest.fixture
    def model_rep(self):
        mr = MockModelRep()
        mr.aud_coefs = pd.DataFrame({
            'year': [2024, 2024],
            'variable': ['ots_f_rolik_local', 'ots_m_rolik_local'],
            'coef': [1.0, 1.2]
        })
        return mr


class TestNatTvSponsorComplex:

    @pytest.fixture
    def md(self):
        mc = MockModelConf()
        mc.trans_path_df = pd.DataFrame({
            'variable': ['bdg_tv'], 'price': [0.05]
        })
        return mc

    @pytest.fixture
    def model_rep(self):
        mr = MockModelRep()
        mr.aud_coefs = pd.DataFrame({
            'year': [2024], 'variable': ['ots_f_sponsor_nat'], 'coef': [1.0]
        })
        return mr

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
            'listed': [4, 4], 'superbrand': ['brand_x'] * 2,
            'ots_f_sponsor_nat': [0.0, 0.0],
            'opt_media_ssn_tv': [0.5, 0.5],
            'sku_coef': [0.5, 0.5], 'umb_coef': [0.8, 0.8],
        })

    def test_smoke_sponsor(self, md, model_rep, ds):
        ds_mask = pd.Series([True, True])
        result = nat_tv_sponsor_complex(md, ds, model_rep, 'bdg_tv',
                                        'umb, sku_coef, umb_coef', 100.0, ds_mask)
        assert np.all(result.loc[ds_mask, 'ots_f_sponsor_nat'] > 0)


class TestBrndNatTvComplex:

    @pytest.fixture
    def md(self):
        mc = MockModelConf()
        mc.trans_path_df = pd.DataFrame({
            'variable': ['bdg_tv'], 'price': [0.05]
        })
        return mc

    @pytest.fixture
    def model_rep(self):
        mr = MockModelRep()
        mr.aud_coefs = pd.DataFrame({
            'year': [2024], 'variable': ['ots_f_rolik_nat'], 'coef': [1.0]
        })
        return mr

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
            'listed': [4, 4], 'superbrand': ['brand_x'] * 2,
            'ots_f_rolik_nat': [0.0, 0.0],
            'opt_media_ssn_tv': [0.5, 0.5],
            'sku_coef': [0.5, 0.5], 'umb_coef': [0.8, 0.8],
        })

    def test_smoke_brand_nat(self, md, model_rep, ds):
        ds_mask = pd.Series([True, True])
        result = brnd_nat_tv_complex(md, ds, model_rep, 'bdg_tv',
                                     'umb, sku_coef, umb_coef', 100.0, ds_mask)
        assert np.all(result.loc[ds_mask, 'tmp_ots_f_rolik_nat'] > 0)
