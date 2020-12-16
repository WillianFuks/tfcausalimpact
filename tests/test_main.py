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


import numpy as np
import pandas as pd
import pytest
import tensorflow_probability as tfp
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

from causalimpact import CausalImpact
from causalimpact.misc import standardize


@pytest.mark.slow
def test_default_causal_cto(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'hmc', 'niter': 1000, 'prior_level_sd': 0.01,
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
    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1], :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'hmc', 'niter': 1000, 'prior_level_sd': 0.01,
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
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'hmc', 'niter': 1000, 'prior_level_sd': 0.01,
                             'season_duration': 1, 'nseasons': 1, 'standardize': True}
    assert isinstance(ci.model, tfp.sts.LocalLevel)
    assert ci.inferences is not None
    assert ci.inferences.index.dtype == rand_data.index.dtype
    assert ci.summary_data is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.model_args['niter'] == 1000
    assert ci.model_samples is not None


@pytest.mark.slow
def test_default_causal_cto_with_np_array(rand_data, pre_int_period, post_int_period):
    data = rand_data.values
    ci = CausalImpact(data, pre_int_period, post_int_period)
    assert_array_equal(ci.data, data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    data = pd.DataFrame(data)
    pre_data = data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = data.loc[post_int_period[0]: post_int_period[1], :]
    assert_frame_equal(ci.post_data, post_data)
    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)
    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'fit_method': 'hmc', 'niter': 1000, 'prior_level_sd': 0.01,
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_default_causal_cto_vi_method(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model_args=dict(
        fit_method='vi'))
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
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
