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


import mock
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import causalimpact.model as cimodel

tfd = tfp.distributions


def test_process_model_args():
    model_args = cimodel.process_model_args(dict(standardize=False))
    assert model_args['standardize'] is False

    model_args = cimodel.process_model_args(dict(standardize=True))
    assert model_args['standardize'] is True

    model_args = cimodel.process_model_args({})
    assert model_args['standardize'] is True

    with pytest.raises(ValueError) as excinfo:
        cimodel.process_model_args(dict(standardize='yes'))
    assert str(excinfo.value) == 'standardize argument must be of type bool.'

    model_args = cimodel.process_model_args(dict(niter=10))
    assert model_args['niter'] == 10

    model_args = cimodel.process_model_args({})
    assert model_args['niter'] == 1000

    with pytest.raises(ValueError) as excinfo:
        cimodel.process_model_args(dict(niter='yes'))
    assert str(excinfo.value) == 'niter argument must be of type int.'

    model_args = cimodel.process_model_args({})
    assert model_args['prior_level_sd'] == 0.01

    with pytest.raises(ValueError) as excinfo:
        cimodel.process_model_args(dict(prior_level_sd='test'))
    assert str(excinfo.value) == 'prior_level_sd argument must be of type float.'

    model_args = cimodel.process_model_args(dict(fit_method='hmc'))
    assert model_args['fit_method'] == 'hmc'

    model_args = cimodel.process_model_args(dict(fit_method='vi'))
    assert model_args['fit_method'] == 'vi'

    model_args = cimodel.process_model_args(dict())
    assert model_args['fit_method'] == 'vi'

    with pytest.raises(ValueError) as excinfo:
        model_args = cimodel.process_model_args(dict(fit_method='test'))
    assert str(excinfo.value) == 'fit_method can be either "hmc" or "vi".'

    model_args = cimodel.process_model_args(dict(nseasons=7))
    assert model_args['nseasons'] == 7

    model_args = cimodel.process_model_args({})
    assert model_args['nseasons'] == 1

    with pytest.raises(ValueError) as excinfo:
        model_args = cimodel.process_model_args(dict(nseasons='test'))
    assert str(excinfo.value) == 'nseasons argument must be of type int.'

    model_args = cimodel.process_model_args({})
    assert model_args['season_duration'] == 1

    model_args = cimodel.process_model_args(dict(nseasons=7, season_duration=24))
    assert model_args['season_duration'] == 24

    with pytest.raises(ValueError) as excinfo:
        model_args = cimodel.process_model_args(dict(season_duration='test'))
    assert str(excinfo.value) == 'season_duration argument must be of type int.'

    with pytest.raises(ValueError) as excinfo:
        model_args = cimodel.process_model_args(dict(season_duration=24))
    assert str(excinfo.value) == ('nseasons must be bigger than 1 when season_duration '
                                  'is also bigger than 1.')


def test_check_input_model():
    model = tfp.sts.Sum([tfp.sts.LocalLevel()])
    cimodel.check_input_model(model, None, None)

    model = tfp.sts.LocalLevel()
    cimodel.check_input_model(model, None, None)

    data = pd.DataFrame(np.random.rand(200, 2)).astype(np.float32)
    pre_data = data.iloc[:100, :]
    post_data = data.iloc[100:, :]
    model = tfp.sts.LinearRegression(design_matrix=data.iloc[:, 1].values.reshape(-1, 1))
    cimodel.check_input_model(model, pre_data, post_data)

    model = tfp.sts.LinearRegression(
        design_matrix=pre_data.iloc[:, 1].values.reshape(-1, 1)
    )
    with pytest.raises(ValueError) as excinfo:
        cimodel.check_input_model(model, pre_data, post_data)
    assert str(excinfo.value) == (
        'Customized Linear Regression Models must have total '
        'points equal to pre_data and post_data points and same number of covariates. '
        'Input design_matrix shape was (100, 1) and expected (200, 1) instead.'
    )

    model = tfp.sts.Sum([tfp.sts.LocalLevel(), tfp.sts.LinearRegression(
                        design_matrix=pre_data.iloc[:, 1].values.reshape(-1, 1))])
    with pytest.raises(ValueError) as excinfo:
        cimodel.check_input_model(model, pre_data, post_data)
    assert str(excinfo.value) == (
        'Customized Linear Regression Models must have total '
        'points equal to pre_data and post_data points and same number of covariates. '
        'Input design_matrix shape was (100, 1) and expected (200, 1) instead.'
    )
    with pytest.raises(ValueError) as excinfo:
        cimodel.check_input_model('test', None, None)
    assert str(excinfo.value) == 'Input model must be of type StructuralTimeSeries.'

    # tests dtype != float32
    data = pd.DataFrame(np.random.rand(200, 2))
    pre_data = data.iloc[:100, :]
    post_data = data.iloc[100:, :]
    model = tfp.sts.LinearRegression(design_matrix=data.iloc[:, 1].values.reshape(-1, 1))
    with pytest.raises(AssertionError):
        cimodel.check_input_model(model, pre_data, post_data)

    model = tfp.sts.LocalLevel(observed_time_series=pre_data.iloc[:, 0])
    with pytest.raises(AssertionError):
        cimodel.check_input_model(model, pre_data, post_data)

    model = tfp.sts.Sum(
        [tfp.sts.LinearRegression(design_matrix=data.iloc[:, 1].values.reshape(-1, 1)),
         tfp.sts.LocalLevel(observed_time_series=pre_data.iloc[:, 0])],
        observed_time_series=pre_data.iloc[:, 0]
    )
    with pytest.raises(AssertionError):
        cimodel.check_input_model(model, pre_data, post_data)


def test_build_default_model(rand_data, pre_int_period, post_int_period):
    prior_level_sd = 0.01
    pre_data = pd.DataFrame(rand_data.iloc[pre_int_period[0]: pre_int_period[1], 0])
    post_data = pd.DataFrame(rand_data.iloc[post_int_period[0]: post_int_period[1], 0])
    observed_time_series = pd.DataFrame(pre_data.iloc[:, 0]).astype(np.float32)
    model = cimodel.build_default_model(observed_time_series, pre_data, post_data,
                                        prior_level_sd, 1, 1)
    assert isinstance(model, tfp.sts.Sum)
    obs_prior = model.parameters[0].prior
    assert isinstance(obs_prior, tfd.TransformedDistribution)
    assert isinstance(obs_prior.bijector, tfp.bijectors.Power)
    assert isinstance(obs_prior.distribution, tfd.InverseGamma)
    level_prior = model.parameters[1].prior
    assert isinstance(level_prior, tfd.TransformedDistribution)
    assert isinstance(level_prior.bijector, tfp.bijectors.Power)
    assert isinstance(level_prior.distribution, tfd.InverseGamma)
    assert level_prior.dtype == tf.float32

    pre_data = pd.DataFrame(rand_data.iloc[pre_int_period[0]: pre_int_period[1], :])
    post_data = pd.DataFrame(rand_data.iloc[post_int_period[0]: post_int_period[1], :])
    observed_time_series = pd.DataFrame(pre_data.iloc[:, 0]).astype(np.float32)
    model = cimodel.build_default_model(observed_time_series, pre_data, post_data,
                                        prior_level_sd, 1, 1)
    assert isinstance(model, tfp.sts.Sum)
    obs_prior = model.parameters[0].prior
    assert isinstance(obs_prior, tfd.TransformedDistribution)
    assert isinstance(obs_prior.bijector, tfp.bijectors.Power)
    assert isinstance(obs_prior.distribution, tfd.InverseGamma)
    level_prior = model.parameters[1].prior
    assert isinstance(level_prior, tfd.TransformedDistribution)
    assert isinstance(level_prior.bijector, tfp.bijectors.Power)
    assert isinstance(level_prior.distribution, tfd.InverseGamma)
    assert level_prior.dtype == tf.float32
    linear = model.components[1]
    design_matrix = linear.design_matrix.to_dense()
    np.testing.assert_equal(pd.concat([pre_data, post_data]).iloc[:, 1:].values.astype(
                            np.float32),
                            design_matrix)
    assert design_matrix.dtype == tf.float32
    # test seasonal
    pre_data = pd.DataFrame(rand_data.iloc[pre_int_period[0]: pre_int_period[1], :])
    post_data = pd.DataFrame(rand_data.iloc[post_int_period[0]: post_int_period[1], :])
    observed_time_series = pd.DataFrame(pre_data.iloc[:, 0]).astype(np.float32)
    model = cimodel.build_default_model(observed_time_series, pre_data, post_data,
                                        prior_level_sd, 7, 2)
    assert isinstance(model, tfp.sts.Sum)
    obs_prior = model.parameters[0].prior
    assert isinstance(obs_prior, tfd.TransformedDistribution)
    assert isinstance(obs_prior.bijector, tfp.bijectors.Power)
    assert isinstance(obs_prior.distribution, tfd.InverseGamma)
    assert obs_prior.dtype == tf.float32
    level_prior = model.parameters[1].prior
    assert isinstance(level_prior, tfd.TransformedDistribution)
    assert isinstance(level_prior.bijector, tfp.bijectors.Power)
    assert isinstance(level_prior.distribution, tfd.InverseGamma)
    assert level_prior.dtype == tf.float32
    linear = model.components[1]
    design_matrix = linear.design_matrix.to_dense()
    np.testing.assert_equal(pd.concat([pre_data, post_data]).iloc[:, 1:].values.astype(
                            np.float32),
                            design_matrix)
    seasonal_component = model.components[-1]
    assert isinstance(seasonal_component, tfp.sts.Seasonal)
    assert seasonal_component.num_seasons == 7
    assert seasonal_component.num_steps_per_season == 2


def test_fit_model(monkeypatch):
    observed_time_series = 'observed_time_series'
    fit_mock = mock.Mock(return_value=('samples', 'kr'))
    vi_fit_mock = mock.Mock()

    class VarPost:
        def sample(self, *args):
            return f'{args[0]} var_post_samples'
    var_post = VarPost()
    surrogate_posterior_mock = mock.Mock(return_value=var_post)
    monkeypatch.setattr('causalimpact.model.tfp.sts.fit_with_hmc', fit_mock)
    monkeypatch.setattr('causalimpact.model.tfp.sts.build_factored_surrogate_posterior',
                        surrogate_posterior_mock)
    monkeypatch.setattr('causalimpact.model.Adam',
                        mock.Mock(return_value='optimizer'))

    monkeypatch.setattr('causalimpact.model.tfp.vi.fit_surrogate_posterior', vi_fit_mock)

    class Model:
        def joint_log_prob(self, observed_time_series):
            return f'target_log_prob of {observed_time_series}'
    model = Model()
    samples, kr = cimodel.fit_model(
        model=model,
        observed_time_series=observed_time_series,
        method='hmc'
    )
    fit_mock.assert_called_once_with(
        model=model, observed_time_series=observed_time_series
    )
    assert samples == 'samples'
    assert kr == 'kr'
    surrogate_posterior_mock.assert_not_called()

    samples, kr = cimodel.fit_model(
        model=model,
        observed_time_series=observed_time_series,
        method='vi'
    )
    surrogate_posterior_mock.assert_called_with(model=model)
    vi_fit_mock.assert_called_once_with(
        target_log_prob_fn='target_log_prob of observed_time_series',
        surrogate_posterior=var_post,
        optimizer='optimizer',
        num_steps=200
    )
    assert kr is None
    assert samples == '100 var_post_samples'

    with pytest.raises(ValueError) as excinfo:
        cimodel.fit_model(model=model, observed_time_series=observed_time_series,
                          method='test')
    assert str(excinfo.value) == (
        'Input method "test" not valid. Choose between "hmc" or "vi".'
    )
