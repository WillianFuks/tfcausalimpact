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


import pytest
import pandas as pd
import numpy as np

import tensorflow_probability as tfp

import causalimpact.model as cimodel


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
    assert model_args['niter'] == 100

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
    assert model_args['fit_method'] == 'hmc'

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

    data = pd.DataFrame(np.random.rand(200, 2))
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
