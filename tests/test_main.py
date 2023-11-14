# Copyright WillianFuks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Tests for module main.py. Fixtures comes from file conftest.py located at the same dir
of this file.
"""


import mock
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from causalimpact import CausalImpact
from causalimpact.misc import standardize

seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)


def test_default_causal_cto(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    new_rand_data = rand_data.set_index(pd.date_range('2020-01-01',
                                        periods=len(rand_data)))
    new_rand_data = tfp.sts.regularize_series(new_rand_data).astype(np.float32)
    pre_data = new_rand_data.iloc[pre_int_period[0]: pre_int_period[1] + 1, :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = new_rand_data.iloc[post_int_period[0]: post_int_period[1] + 1, :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'vi', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.Sum)
    design_matrix = ci.model.components[1].design_matrix.to_dense()
    assert_array_equal(
        design_matrix,
        pd.concat([normed_pre_data, normed_post_data]).astype(np.float32).iloc[:, 1:]
    )
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == rand_data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None


@pytest.mark.slow
def test_default_causal_cto_with_date_index(date_rand_data, pre_str_period,
                                            post_str_period):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    assert_frame_equal(ci.data, date_rand_data)
    assert ci.pre_period == pre_str_period
    assert ci.post_period == post_str_period
    new_date_rand_data = tfp.sts.regularize_series(date_rand_data).astype(np.float32)
    pre_data = new_date_rand_data.loc[pre_str_period[0]: pre_str_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = new_date_rand_data.loc[post_str_period[0]: post_str_period[1], :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'vi', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.Sum)
    design_matrix = ci.model.components[1].design_matrix.to_dense()
    assert_array_equal(
        design_matrix,
        pd.concat([normed_pre_data, normed_post_data]).astype(np.float32).iloc[:, 1:]
    )
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == date_rand_data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None


@pytest.mark.slow
def test_default_causal_cto_no_covariates(rand_data, pre_int_period, post_int_period):
    rand_data = pd.DataFrame(rand_data.iloc[:, 0])
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    new_rand_data = rand_data.set_index(pd.date_range('2020-01-01',
                                        periods=len(rand_data)))
    new_rand_data = tfp.sts.regularize_series(new_rand_data).astype(np.float32)
    pre_data = new_rand_data.iloc[pre_int_period[0]: pre_int_period[1] + 1, :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = new_rand_data.iloc[post_int_period[0]: post_int_period[1] + 1, :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'vi', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.Sum)
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == rand_data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None


def test_default_causal_cto_with_np_array(rand_data, pre_int_period, post_int_period):
    data = rand_data.values
    ci = CausalImpact(data, pre_int_period, post_int_period)
    assert_array_equal(ci.data, data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    data = pd.DataFrame(data)
    new_data = data.set_index(pd.date_range('2020-01-01', periods=len(data)))
    new_data = tfp.sts.regularize_series(new_data).astype(np.float32)
    pre_data = new_data.iloc[pre_int_period[0]: pre_int_period[1] + 1, :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = new_data.iloc[post_int_period[0]: post_int_period[1] + 1, :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'vi', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.Sum)
    design_matrix = ci.model.components[1].design_matrix.to_dense()
    assert_array_equal(
        design_matrix,
        pd.concat([normed_pre_data, normed_post_data]).astype(np.float32).iloc[:, 1:]
    )
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None


@pytest.mark.slow
def test_causal_cto_with_no_standardization(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model_args=dict(
        standardize=False, fit_method='vi'))
    assert ci.normed_pre_data is None
    assert ci.normed_post_data is None
    assert ci.mu_sig is None
    assert ci.p_value > 0 and ci.p_value < 1


@pytest.mark.slow
def test_causal_cto_with_seasons(date_rand_data, pre_str_period, post_str_period):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period,
                      model_args={'nseasons': 7, 'season_duration': 2,
                                  'fit_method': 'vi'})
    assert len(ci.model.components) == 3
    seasonal_component = ci.model.components[2]
    assert seasonal_component.num_seasons == 7
    assert seasonal_component.num_steps_per_season == 2


def test_plotter(monkeypatch, rand_data, pre_int_period, post_int_period):
    plotter_mock = mock.Mock()
    fit_mock = mock.Mock()
    process_mock = mock.Mock()
    summarize_mock = mock.Mock()
    monkeypatch.setattr('causalimpact.main.CausalImpact._fit_model', fit_mock)
    monkeypatch.setattr('causalimpact.main.CausalImpact._summarize_inferences',
                        summarize_mock)
    monkeypatch.setattr('causalimpact.main.CausalImpact._process_posterior_inferences',
                        process_mock)
    monkeypatch.setattr('causalimpact.main.plotter', plotter_mock)
    ci = CausalImpact(rand_data, pre_int_period, post_int_period,
                      model_args={'fit_method': 'vi'})
    ci.inferences = 'inferences'
    ci.pre_data = 'pre_data'
    ci.post_data = 'post_data'
    ci._mask = slice(0, len(ci.post_data))
    ci.plot()
    plotter_mock.plot.assert_called_with('inferences', 'pre_data', 'post_data',
                                         panels=['original', 'pointwise', 'cumulative'],
                                         figsize=(10, 7), show=True)


def test_summarizer(monkeypatch, rand_data, pre_int_period, post_int_period):
    summarizer_mock = mock.Mock()
    fit_mock = mock.Mock()
    process_mock = mock.Mock()
    summarize_mock = mock.Mock()
    monkeypatch.setattr('causalimpact.main.CausalImpact._fit_model', fit_mock)
    monkeypatch.setattr('causalimpact.main.CausalImpact._summarize_inferences',
                        summarize_mock)
    monkeypatch.setattr('causalimpact.main.CausalImpact._process_posterior_inferences',
                        process_mock)
    monkeypatch.setattr('causalimpact.main.summarizer', summarizer_mock)
    ci = CausalImpact(rand_data, pre_int_period, post_int_period,
                      model_args={'fit_method': 'vi'})
    ci.summary_data = 'summary_data'
    ci.p_value = 0.5
    ci.alpha = 0.05
    ci.summary()
    summarizer_mock.summary.assert_called_with('summary_data', 0.5, 0.05, 'summary', 2)

    with pytest.raises(ValueError) as excinfo:
        ci.summary(digits='1')
    assert str(excinfo.value) == ('Input value for digits must be integer. Received '
                                  '"<class \'str\'>" instead.')


def test_causal_cto_with_custom_model_and_seasons(rand_data, pre_int_period,
                                                  post_int_period):
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    observed_time_series = pre_data.iloc[:, 0].astype(np.float32)
    level = tfp.sts.LocalLevel(observed_time_series=observed_time_series)
    seasonal = tfp.sts.Seasonal(num_seasons=7, num_steps_per_season=2,
                                observed_time_series=observed_time_series)
    model = tfp.sts.Sum([level, seasonal], observed_time_series=observed_time_series)

    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=model,
                      model_args={'fit_method': 'vi'})

    assert len(ci.model.components) == 2
    assert isinstance(ci.model.components[0], tfp.sts.LocalLevel)
    assert isinstance(ci.model.components[1], tfp.sts.Seasonal)
    seasonal_component = ci.model.components[-1]
    assert seasonal_component.num_seasons == 7
    assert seasonal_component.num_steps_per_season == 2
    assert ci.inferences.index.dtype == rand_data.index.dtype


def test_default_causal_cto_vi_method(rand_data, pre_int_period, post_int_period):
    freq_rand_data = rand_data.set_index(
        pd.date_range(start='2020-01-01', periods=len(rand_data))
    ).astype(np.float32).asfreq(pd.offsets.DateOffset(days=1))
    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model_args=dict(
        fit_method='vi'))
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = freq_rand_data.iloc[pre_int_period[0]: pre_int_period[1] + 1, :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = freq_rand_data.iloc[post_int_period[0]: post_int_period[1] + 1, :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'vi', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.Sum)
    design_matrix = ci.model.components[1].design_matrix.to_dense()
    assert_array_equal(
        design_matrix,
        pd.concat([normed_pre_data, normed_post_data]).astype(np.float32).iloc[:, 1:]
    )
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == rand_data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None
    assert ci.model_kernel_results is None


def test_default_model_arma_data():
    data = pd.read_csv('tests/fixtures/arma_data.csv')
    data.iloc[70:, 0] += 5

    pre_period = [0, 69]
    post_period = [70, 99]

    ci = CausalImpact(data, pre_period, post_period)
    assert int(ci.summary_data['average']['actual']) == 105
    assert int(ci.summary_data['average']['predicted']) == 100
    assert int(ci.summary_data['average']['predicted_lower']) == 99
    assert int(ci.summary_data['average']['predicted_upper']) == 100
    assert int(ci.summary_data['average']['abs_effect']) == 5
    assert int(ci.summary_data['average']['abs_effect_lower']) == 4
    assert int(ci.summary_data['average']['abs_effect_upper']) == 5


def test_default_model_sparse_linear_regression_arma_data():
    data = pd.read_csv('tests/fixtures/arma_sparse_reg.csv')
    data.iloc[70:, 0] += 5

    pre_period = [0, 69]
    post_period = [70, 99]

    ci = CausalImpact(data, pre_period, post_period)
    samples = ci.model_samples

    # Weights are computed as per original TFP source code:
    # https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/sts/regression.py#L489-L494 # noqa: E501
    global_scale = (
        samples['SparseLinearRegression/_global_scale_noncentered'] *
        tf.sqrt(samples['SparseLinearRegression/_global_scale_variance']) * 0.1
    )
    local_scales = (
        samples['SparseLinearRegression/_local_scales_noncentered'] *
        tf.sqrt(samples['SparseLinearRegression/_local_scale_variances'])
    )
    weights = (
        samples['SparseLinearRegression/_weights_noncentered'] * local_scales *
        global_scale[..., tf.newaxis]
    )
    assert tf.abs(tf.reduce_mean(weights, axis=0).numpy()[1]) < 0.05


def test_data_no_freq():
    data = pd.read_csv('tests/fixtures/btc.csv', parse_dates=True, index_col='Date')
    training_start = "2020-12-01"
    training_end = "2021-02-05"
    treatment_start = "2021-02-08"
    treatment_end = "2021-02-09"
    pre_period = [training_start, training_end]
    post_period = [treatment_start, treatment_end]

    ci = CausalImpact(data, pre_period, post_period)

    freq_data = tfp.sts.regularize_series(data)
    pre_data = freq_data.loc[pre_period[0]: pre_period[1]]
    post_data = freq_data.loc[post_period[0]: post_period[1]]

    assert len(ci.inferences) == len(pre_data) + len(post_data)


def test_post_data_nan_values_removed():
    """
    Related to issue #51 where some points in post_data can end up having `NaN` values
    """
    data = pd.DataFrame(np.random.rand(100, 2))
    data.set_index(pd.date_range('20200101', periods=len(data)), inplace=True)
    data.iloc[:, 0].loc['20200302':] += 1
    _mask_ = data.index.isin(['20200303', '20200304', '20200404']) == False  # noqa: E712
    data = data[_mask_]
    expected_mask = np.array([True] * (39))  # 39 post data points
    expected_mask[1] = False  # '20200303'
    expected_mask[2] = False  # '20200304'
    expected_mask[33] = False  # '20200404'
    pre_period = ['20200101', '20200301']
    post_period = ['20200302', '20200409']
    ci = CausalImpact(data, pre_period, post_period)

    np.testing.assert_equal(ci._mask, expected_mask)
    assert '20200303' not in ci.inferences
    assert '20200304' not in ci.inferences
    assert '20200404' not in ci.inferences
    inferences = ci.inferences.loc[post_period[0]: post_period[1]]
    assert inferences.post_cum_y.isna().sum() == 0

    assert ci.summary_data['average']['actual'] > 0
    assert ci.summary_data['cumulative']['actual'] > 0
    assert ci.summary_data['average']['predicted'] > 0
    assert ci.summary_data['cumulative']['predicted'] > 0
    assert ci.summary_data['average']['abs_effect'] > 0
    assert ci.summary_data['cumulative']['abs_effect'] > 0


def test_custom_model_post_data_with_index_freq_holes():
    data = pd.DataFrame(np.random.rand(100, 2)).astype(np.float32)
    data.set_index(pd.date_range('20200101', periods=len(data)), inplace=True)
    pre_period = ['20200101', '20200301']
    post_period = ['20200302', '20200409']
    data = tfp.sts.regularize_series(data)
    pre_data = data.loc[pre_period[0]: pre_period[1]]
    data.iloc[:, 0].loc['20200302':] += 1
    _mask_ = data.index.isin(['20200303', '20200304', '20200404']) == False  # noqa: E712
    data = data[_mask_]
    # Apply regularize again to force design matrix to have proper shape
    reg_data = tfp.sts.regularize_series(data)
    expected_mask = np.array([True] * (39))  # 39 post data points
    expected_mask[1] = False  # '20200303'
    expected_mask[2] = False  # '20200304'
    expected_mask[33] = False  # '20200404'
    observed_time_series = pre_data.iloc[:, 0].astype(np.float32)
    level = tfp.sts.LocalLevel(observed_time_series=observed_time_series)
    design_matrix_data = reg_data.iloc[:, 1:].fillna(0).values.reshape(
        -1, data.shape[1] - 1)
    linear = tfp.sts.LinearRegression(design_matrix=design_matrix_data)
    model = tfp.sts.Sum([level, linear], observed_time_series=observed_time_series)
    ci = CausalImpact(data, pre_period, post_period, model=model,
                      model_args={'standardize': False})

    np.testing.assert_equal(ci._mask, expected_mask)
    assert '20200303' not in ci.inferences
    assert '20200304' not in ci.inferences
    assert '20200404' not in ci.inferences
    inferences = ci.inferences.loc[post_period[0]: post_period[1]]
    assert inferences.post_cum_y.isna().sum() == 0

    assert ci.summary_data['average']['actual'] > 0
    assert ci.summary_data['cumulative']['actual'] > 0
    assert ci.summary_data['average']['predicted'] > 0
    assert ci.summary_data['cumulative']['predicted'] > 0
    assert ci.summary_data['average']['abs_effect'] > 0
    assert ci.summary_data['cumulative']['abs_effect'] > 0
