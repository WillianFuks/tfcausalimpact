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


# import os

# import mock
# import numpy as np
# import pandas as pd
# import pytest
# from numpy.testing import assert_array_equal
# from pandas.core.indexes.range import RangeIndex
from pandas.util.testing import assert_frame_equal

from causalimpact import CausalImpact
# from causalimpact.misc import standardize


def test_default_causal_cto(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    #assert ci.pre_period == pre_int_period
    #assert ci.post_period == post_int_period
    #pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    #assert_frame_equal(ci.pre_data, pre_data)

    #post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
    #assert_frame_equal(ci.post_data, post_data)

    #assert ci.alpha == 0.05
    #normed_pre_data, (mu, sig) = standardize(pre_data)
    #assert_frame_equal(ci.normed_pre_data, normed_pre_data)

    #normed_post_data = (post_data - mu) / sig
    #assert_frame_equal(ci.normed_post_data, normed_post_data)

    #assert ci.mu_sig == (mu[0], sig[0])
    #assert ci.model_args == {'standardize': True, 'nseasons': []}

    #assert isinstance(ci.model, UnobservedComponents)
    #assert_array_equal(ci.model.endog, normed_pre_data.iloc[:, 0].values.reshape(-1, 1))
    #assert_array_equal(ci.model.exog, normed_pre_data.iloc[:, 1:].values.reshape(
    #        -1,
    #        rand_data.shape[1] - 1
    #    )
    #)
    #assert ci.model.endog_names == 'y'
    #assert ci.model.exog_names == ['x1', 'x2']
    #assert ci.model.k_endog == 1
    #assert ci.model.level
    #assert ci.model.trend_specification == 'local level'

    #assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
    #assert ci.trained_model.nobs == len(pre_data)

    #assert ci.inferences is not None
    #assert ci.inferences.index.dtype == rand_data.index.dtype
    #assert ci.p_value > 0 and ci.p_value < 1
    #assert ci.n_sims == 1000


#def test_default_causal_cto_w_date(date_rand_data, pre_str_period, post_str_period):
#    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
#    assert_frame_equal(ci.data, date_rand_data)
#    assert ci.pre_period == pre_str_period
#    assert ci.post_period == post_str_period
#    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1], :]
#    assert_frame_equal(ci.pre_data, pre_data)
#
#    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1], :]
#    assert_frame_equal(ci.post_data, post_data)
#
#    assert ci.alpha == 0.05
#    normed_pre_data, (mu, sig) = standardize(pre_data)
#    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
#
#    normed_post_data = (post_data - mu) / sig
#    assert_frame_equal(ci.normed_post_data, normed_post_data)
#
#    assert ci.mu_sig == (mu[0], sig[0])
#    assert ci.model_args == {'standardize': True, 'nseasons': []}
#
#    assert isinstance(ci.model, UnobservedComponents)
#    assert_array_equal(ci.model.endog, normed_pre_data.iloc[:, 0].values.reshape(-1, 1))
#    assert_array_equal(ci.model.exog, normed_pre_data.iloc[:, 1:].values.reshape(
#            -1,
#            date_rand_data.shape[1] - 1
#        )
#    )
#    assert ci.model.endog_names == 'y'
#    assert ci.model.exog_names == ['x1', 'x2']
#    assert ci.model.k_endog == 1
#    assert ci.model.level
#    assert ci.model.trend_specification == 'local level'
#
#    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
#    assert ci.trained_model.nobs == len(pre_data)
#
#    assert ci.inferences is not None
#    assert ci.inferences.index.dtype == date_rand_data.index.dtype
#    assert ci.p_value > 0 and ci.p_value < 1
#    assert ci.n_sims == 1000
#
#
#def test_default_causal_cto_no_exog(rand_data, pre_int_period, post_int_period):
#    rand_data = pd.DataFrame(rand_data.iloc[:, 0])
#    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
#    assert_frame_equal(ci.data, rand_data)
#    assert ci.pre_period == pre_int_period
#    assert ci.post_period == post_int_period
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    assert_frame_equal(ci.pre_data, pre_data)
#
#    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
#    assert_frame_equal(ci.post_data, post_data)
#
#    assert ci.alpha == 0.05
#    normed_pre_data, (mu, sig) = standardize(pre_data)
#    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
#
#    normed_post_data = (post_data - mu) / sig
#    assert_frame_equal(ci.normed_post_data, normed_post_data)
#
#    assert ci.mu_sig == (mu[0], sig[0])
#    assert ci.model_args == {'standardize': True, 'nseasons': []}
#
#    assert isinstance(ci.model, UnobservedComponents)
#    assert_array_equal(ci.model.endog, normed_pre_data.iloc[:, 0].values.reshape(-1, 1))
#    assert ci.model.exog is None
#    assert ci.model.endog_names == 'y'
#    assert ci.model.exog_names is None
#    assert ci.model.k_endog == 1
#    assert ci.model.level
#    assert ci.model.trend_specification == 'local level'
#
#    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
#    assert ci.trained_model.nobs == len(pre_data)
#
#    assert ci.inferences is not None
#    assert ci.inferences.index.dtype == rand_data.index.dtype
#    assert ci.p_value > 0 and ci.p_value < 1
#    assert ci.n_sims == 1000
#
#
#def test_default_causal_cto_w_np_array(rand_data, pre_int_period, post_int_period):
#    data = rand_data.values
#    ci = CausalImpact(data, pre_int_period, post_int_period)
#    assert_array_equal(ci.data, data)
#    assert ci.pre_period == pre_int_period
#    assert ci.post_period == post_int_period
#    pre_data = pd.DataFrame(data[pre_int_period[0]: pre_int_period[1] + 1, :])
#    assert_frame_equal(ci.pre_data, pre_data)
#
#    post_data = pd.DataFrame(data[post_int_period[0]: post_int_period[1] + 1, :])
#    post_data.index = RangeIndex(start=len(pre_data), stop=len(rand_data))
#    assert_frame_equal(ci.post_data, post_data)
#
#    assert ci.alpha == 0.05
#    normed_pre_data, (mu, sig) = standardize(pre_data)
#    assert_frame_equal(ci.normed_pre_data, normed_pre_data)
#
#    normed_post_data = (post_data - mu) / sig
#    assert_frame_equal(ci.normed_post_data, normed_post_data)
#
#    assert ci.mu_sig == (mu[0], sig[0])
#    assert ci.model_args == {'standardize': True, 'nseasons': []}
#
#    assert isinstance(ci.model, UnobservedComponents)
#    assert_array_equal(ci.model.endog, normed_pre_data.iloc[:, 0].values.reshape(-1, 1))
#    assert_array_equal(ci.model.exog, normed_pre_data.iloc[:, 1:].values.reshape(
#            -1,
#            data.shape[1] - 1
#        )
#    )
#    assert ci.model.endog_names == 'y'
#    assert ci.model.exog_names == [1, 2]
#    assert ci.model.k_endog == 1
#    assert ci.model.level
#    assert ci.model.trend_specification == 'local level'
#
#    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
#    assert ci.trained_model.nobs == len(pre_data)
#
#    assert ci.inferences is not None
#    assert ci.inferences.index.dtype == rand_data.index.dtype
#    assert ci.p_value > 0 and ci.p_value < 1
#    assert ci.n_sims == 1000
#
#
#def test_causal_cto_w_no_standardization(rand_data, pre_int_period, post_int_period):
#    ci = CausalImpact(rand_data, pre_int_period, post_int_period, standardize=False)
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    assert ci.normed_pre_data is None
#    assert ci.normed_post_data is None
#    assert ci.mu_sig is None
#    assert_array_equal(ci.model.endog, pre_data.iloc[:, 0].values.reshape(-1, 1))
#    assert_array_equal(ci.model.exog, pre_data.iloc[:, 1:].values.reshape(
#            -1,
#            rand_data.shape[1] - 1
#        )
#    )
#    assert ci.p_value > 0 and ci.p_value < 1
#    assert ci.inferences.index.dtype == rand_data.index.dtype
#
#
#def test_causal_cto_w_seasons(date_rand_data, pre_str_period, post_str_period):
#    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period,
#                      nseasons=[{'period': 4}, {'period': 3}])
#    assert ci.model.freq_seasonal_periods == [4, 3]
#    assert ci.model.freq_seasonal_harmonics == [2, 1]
#
#    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period,
#                      nseasons=[{'period': 4, 'harmonics': 1},
#                                {'period': 3, 'harmonis': 1}])
#    assert ci.model.freq_seasonal_periods == [4, 3]
#    assert ci.model.freq_seasonal_harmonics == [1, 1]
#    assert ci.inferences.index.dtype == date_rand_data.index.dtype
#
#
#def test_causal_cto_w_custom_model_and_seasons(rand_data, pre_int_period,
#                                               post_int_period):
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:],
#                                 freq_seasonal=[{'period': 4}, {'period': 3}])
#
#    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=model)
#
#    assert ci.model.freq_seasonal_periods == [4, 3]
#    assert ci.model.freq_seasonal_harmonics == [2, 1]
#    assert ci.inferences.index.dtype == rand_data.index.dtype
#
#
#def test_causal_cto_w_custom_model(rand_data, pre_int_period, post_int_period):
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:])
#
#    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=model)
#
#    assert ci.model.endog_names == 'y'
#    assert ci.model.exog_names == ['x1', 'x2']
#    assert ci.model.k_endog == 1
#    assert ci.model.level
#    assert ci.model.trend_specification == 'local level'
#
#    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
#    assert ci.trained_model.nobs == len(pre_data)
#    assert ci.inferences.index.dtype == rand_data.index.dtype
#
#
#def test_causal_cto_raises_on_None_input(rand_data, pre_int_period, post_int_period):
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(None, pre_int_period, post_int_period)
#    assert str(excinfo.value) == 'data input cannot be empty'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, None, post_int_period)
#    assert str(excinfo.value) == 'pre_period input cannot be empty'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, None)
#    assert str(excinfo.value) == 'post_period input cannot be empty'
#
#
#def test_invalid_data_input_raises():
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact('test', [0, 5], [5, 10])
#    assert str(excinfo.value) == 'Could not transform input data to pandas DataFrame.'
#
#    data = [1, 2, 3, 4, 5, 6, 2 + 1j]
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(data, [0, 3], [3, 6])
#    assert str(excinfo.value) == 'Input data must contain only numeric values.'
#
#    data = np.random.randn(10, 2)
#    data[0, 1] = np.nan
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(data, [0, 3], [3, 6])
#    assert str(excinfo.value) == 'Input data cannot have NAN values.'
#
#
#def test_invalid_response_raises():
#    data = np.random.rand(100, 2)
#    data[:, 0] = np.ones(len(data)) * np.nan
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(data, [0, 50], [50, 100])
#    assert str(excinfo.value) == 'Input response cannot have just Null values.'
#
#    data[0:2, 0] = 1
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(data, [0, 50], [50, 100])
#    assert str(excinfo.value) == ('Input response must have more than 3 non-null points '
#                                  'at least.')
#
#    data[0:3, 0] = 1
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(data, [0, 50], [50, 100])
#    assert str(excinfo.value) == 'Input response cannot be constant.'
#
#
#def test_invalid_alpha_raises(rand_data, pre_int_period, post_int_period):
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, alpha=1)
#    assert str(excinfo.value) == 'alpha must be of type float.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, alpha=2.)
#    assert str(excinfo.value) == (
#        'alpha must range between 0 (zero) and 1 (one) inclusive.'
#    )
#
#
#def test_custom_model_input_validation(rand_data, pre_int_period, post_int_period):
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, model='test')
#    assert str(excinfo.value) == 'Input model must be of type UnobservedComponents.'
#
#    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
#                               exog=rand_data.iloc[:101, 1:])
#    ucm.level = False
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
#    assert str(excinfo.value) == 'Model must have level attribute set.'
#
#    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
#                               exog=rand_data.iloc[:101, 1:])
#    ucm.exog = None
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
#    assert str(excinfo.value) == 'Model must have exog attribute set.'
#
#    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
#                               exog=rand_data.iloc[:101, 1:])
#    ucm.data = None
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
#    assert str(excinfo.value) == 'Model must have data attribute set.'
#
#
#def test_kwargs_validation(rand_data, pre_int_period, post_int_period):
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period,
#                     standardize='yes')
#    assert str(excinfo.value) == 'Standardize argument must be of type bool.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period,
#                     standardize=False, nseasons=[7])
#    assert str(excinfo.value) == (
#        'nseasons must be a list of dicts with the required key "period" and the '
#        'optional key "harmonics".'
#        )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period,
#                     standardize=False, nseasons=[{'test': 8}])
#    assert str(excinfo.value) == 'nseasons dicts must contain the key "period" defined.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, pre_int_period, post_int_period,
#                     standardize=False, nseasons=[{'period': 4, 'harmonics': 3}])
#    assert str(excinfo.value) == (
#        'Total harmonics must be less or equal than periods divided by 2.')
#
#
#def test_periods_validation(rand_data, date_rand_data):
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [5, 10], [4, 7])
#    assert str(excinfo.value) == (
#        'Values in training data cannot be present in the '
#        'post-intervention data. Please fix your pre_period value to cover at most one '
#        'point less from when the intervention happened.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180101', '20180201'],
#                     ['20180110', '20180210'])
#    assert str(excinfo.value) == (
#        'Values in training data cannot be present in the '
#        'post-intervention data. Please fix your pre_period value to cover at most one '
#        'point less from when the intervention happened.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [5, 10], [15, 11])
#    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [5, 10], [10, 15])
#    assert str(excinfo.value) == ('post_period first value (10) must be bigger than the '
#                                  'second value of pre_period (10).')
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180101', '20180110'],
#                     ['20180115', '20180111'])
#    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [0, 2], [15, 11])
#    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180101', '20180102'],
#                     ['20180115', '20180111'])
#    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [5, 0], [15, 11])
#    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180105', '20180101'],
#                     ['20180115', '20180111'])
#    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180105', '20180110'],
#                     ['20180110', '20180115'])
#    assert str(excinfo.value) == ('post_period first value (20180110) must be bigger '
#                                  'than the second value of pre_period (20180110).')
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, 0, [15, 11])
#    assert str(excinfo.value) == 'Input period must be of type list.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, '20180101', ['20180115', '20180130'])
#    assert str(excinfo.value) == 'Input period must be of type list.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [0, 10, 30], [15, 11])
#    assert str(excinfo.value) == (
#        'Period must have two values regarding the beginning '
#        'and end of the pre and post intervention data.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [0, None], [15, 11])
#    assert str(excinfo.value) == 'Input period cannot have `None` values.'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [0, 5.5], [15, 11])
#    assert str(excinfo.value) == 'Input must contain either int, str or pandas Timestamp'
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [-2, 10], [11, 20])
#    assert str(excinfo.value) == (
#        '-2 not present in input data index.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [0, 10], [11, 2000])
#    assert str(excinfo.value) == (
#        '2000 not present in input data index.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, ['20180101', '20180110'],
#                     ['20180111', '20180130'])
#    assert str(excinfo.value) == (
#        '20180101 not present in input data index.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20180101', '20180110'],
#                     ['20180111', '20200130'])
#    assert str(excinfo.value) == ('20200130 not present in input data index.')
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, ['20170101', '20180110'],
#                     ['20180111', '20180120'])
#    assert str(excinfo.value) == ('20170101 not present in input data index.')
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(rand_data, [pd.Timestamp('20180101'), pd.Timestamp('20180110')],
#                     [pd.Timestamp('20180111'), pd.Timestamp('20180130')])
#    assert str(excinfo.value) == (
#        '20180101 not present in input data index.'
#    )
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, [pd.Timestamp('20180101'),
#                     pd.Timestamp('20180110')], [pd.Timestamp('20180111'),
#                     pd.Timestamp('20200130')])
#    assert str(excinfo.value) == ('20200130 not present in input data index.')
#
#    with pytest.raises(ValueError) as excinfo:
#        CausalImpact(date_rand_data, [pd.Timestamp('20170101'),
#                     pd.Timestamp('20180110')], [pd.Timestamp('20180111'),
#                     pd.Timestamp('20180120')])
#    assert str(excinfo.value) == ('20170101 not present in input data index.')
#
#
#def test_string_index_with_no_date_formatted(rand_data, pre_int_period, post_int_period):
#    rand_data.set_index(rand_data.index.map(str), inplace=True)
#    pre_period = ['0', '60']
#    post_period = ['61', '90']
#
#    _ = CausalImpact(rand_data, pre_period, post_period)
#
#
#def test_default_causal_inferences(fix_path):
#    np.random.seed(1)
#    data = pd.read_csv(os.path.join(fix_path, 'google_data.csv'))
#    del data['t']
#
#    pre_period = [0, 60]
#    post_period = [61, 90]
#
#    ci = CausalImpact(data, pre_period, post_period)
#    assert int(ci.summary_data['average']['actual']) == 156
#    assert int(ci.summary_data['average']['predicted']) == 129
#    assert int(ci.summary_data['average']['predicted_lower']) == 124
#    assert int(ci.summary_data['average']['predicted_upper']) == 134
#    assert int(ci.summary_data['average']['abs_effect']) == 27
#    assert round(ci.summary_data['average']['abs_effect_lower'], 1) == 21.6
#    assert int(ci.summary_data['average']['abs_effect_upper']) == 31
#    assert round(ci.summary_data['average']['rel_effect'], 1) == 0.2
#    assert round(ci.summary_data['average']['rel_effect_lower'], 2) == 0.17
#    assert round(ci.summary_data['average']['rel_effect_upper'], 2) == 0.25
#
#    assert int(ci.summary_data['cumulative']['actual']) == 4687
#    assert int(ci.summary_data['cumulative']['predicted']) == 3876
#    assert int(ci.summary_data['cumulative']['predicted_lower']) == 3729
#    assert int(ci.summary_data['cumulative']['predicted_upper']) == 4040
#    assert int(ci.summary_data['cumulative']['abs_effect']) == 810
#    assert int(ci.summary_data['cumulative']['abs_effect_lower']) == 646
#    assert int(ci.summary_data['cumulative']['abs_effect_upper']) == 957
#    assert round(ci.summary_data['cumulative']['rel_effect'], 1) == 0.2
#    assert round(ci.summary_data['cumulative']['rel_effect_lower'], 2) == 0.17
#    assert round(ci.summary_data['cumulative']['rel_effect_upper'], 2) == 0.25
#
#    assert round(ci.p_value, 1) == 0.0
#    assert ci.inferences.index.dtype == data.index.dtype
#
#
#def test_default_causal_inferences_w_date(fix_path):
#    np.random.seed(1)
#    data = pd.read_csv(os.path.join(fix_path, 'google_data.csv'))
#    data['date'] = pd.to_datetime(data['t'])
#    data.index = data['date']
#    del data['t']
#    del data['date']
#
#    pre_period = ['2016-02-20 22:41:20', '2016-02-20 22:51:20']
#    post_period = ['2016-02-20 22:51:30', '2016-02-20 22:56:20']
#
#    ci = CausalImpact(data, pre_period, post_period)
#    assert int(ci.summary_data['average']['actual']) == 156
#    assert int(ci.summary_data['average']['predicted']) == 129
#    assert int(ci.summary_data['average']['predicted_lower']) == 124
#    assert int(ci.summary_data['average']['predicted_upper']) == 134
#    assert int(ci.summary_data['average']['abs_effect']) == 27
#    assert round(ci.summary_data['average']['abs_effect_lower'], 1) == 21.6
#    assert int(ci.summary_data['average']['abs_effect_upper']) == 31
#    assert round(ci.summary_data['average']['rel_effect'], 1) == 0.2
#    assert round(ci.summary_data['average']['rel_effect_lower'], 2) == 0.17
#    assert round(ci.summary_data['average']['rel_effect_upper'], 2) == 0.25
#
#    assert int(ci.summary_data['cumulative']['actual']) == 4687
#    assert int(ci.summary_data['cumulative']['predicted']) == 3876
#    assert int(ci.summary_data['cumulative']['predicted_lower']) == 3729
#    assert int(ci.summary_data['cumulative']['predicted_upper']) == 4040
#    assert int(ci.summary_data['cumulative']['abs_effect']) == 810
#    assert int(ci.summary_data['cumulative']['abs_effect_lower']) == 646
#    assert int(ci.summary_data['cumulative']['abs_effect_upper']) == 957
#    assert round(ci.summary_data['cumulative']['rel_effect'], 1) == 0.2
#    assert round(ci.summary_data['cumulative']['rel_effect_lower'], 2) == 0.17
#    assert round(ci.summary_data['cumulative']['rel_effect_upper'], 2) == 0.25
#
#    assert round(ci.p_value, 1) == 0.0
#    assert ci.inferences.index.dtype == data.index.dtype
#
#
#def test_default_causal_inferences_w_str_date(fix_path):
#    np.random.seed(1)
#    data = pd.read_csv(os.path.join(fix_path, 'volks_data.csv'), header=0, sep=' ',
#                       index_col='Date')
#
#    pre_period = [np.min(data.index.values), pd.Timestamp('2015-09-13')]
#    post_period = [pd.Timestamp('2015-09-20'), np.max(data.index.values)]
#
#    ci = CausalImpact(data, pre_period, post_period)
#    assert int(ci.summary_data['average']['actual']) == 126
#    assert int(ci.summary_data['average']['predicted']) == 171
#    assert int(ci.summary_data['average']['predicted_lower']) == 165
#    assert int(ci.summary_data['average']['predicted_upper']) == 177
#    assert int(ci.summary_data['average']['abs_effect']) == -44
#    assert round(ci.summary_data['average']['abs_effect_lower'], 1) == -50.4
#    assert int(ci.summary_data['average']['abs_effect_upper']) == -39
#    assert round(ci.summary_data['average']['rel_effect'], 1) == -0.3
#    assert round(ci.summary_data['average']['rel_effect_lower'], 2) == -0.29
#    assert round(ci.summary_data['average']['rel_effect_upper'], 2) == -0.23
#
#    assert int(ci.summary_data['cumulative']['actual']) == 10026
#    assert int(ci.summary_data['cumulative']['predicted']) == 13574
#    assert int(ci.summary_data['cumulative']['predicted_lower']) == 13113
#    assert int(ci.summary_data['cumulative']['predicted_upper']) == 14004
#    assert int(ci.summary_data['cumulative']['abs_effect']) == -3548
#    assert int(ci.summary_data['cumulative']['abs_effect_lower']) == -3977
#    assert int(ci.summary_data['cumulative']['abs_effect_upper']) == -3087
#    assert round(ci.summary_data['cumulative']['rel_effect'], 1) == -0.3
#    assert round(ci.summary_data['cumulative']['rel_effect_lower'], 2) == -0.29
#    assert round(ci.summary_data['cumulative']['rel_effect_upper'], 2) == -0.23
#
#    assert round(ci.p_value, 1) == 0.0
#    assert ci.inferences.index.dtype == data.index.dtype
#
#
#def test_default_model_fit(rand_data, pre_int_period, post_int_period, monkeypatch):
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    fit_mock = mock.Mock()
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:])
#
#    model.fit = fit_mock
#
#    construct_mock = mock.Mock(return_value=model)
#
#    monkeypatch.setattr('causalimpact.main.CausalImpact._get_default_model',
#                        construct_mock)
#    monkeypatch.setattr('causalimpact.main.CausalImpact._process_posterior_inferences',
#                        mock.Mock())
#
#    CausalImpact(rand_data, pre_int_period, post_int_period)
#    model.fit.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.012), (None, None), (None, None)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, disp=True)
#    model.fit.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.012), (None, None), (None, None)],
#        disp=True,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, disp=True,
#                 prior_level_sd=0.1)
#    model.fit.assert_called_with(
#        bounds=[(None, None), (0.1 / 1.2, 0.1 * 1.2), (None, None), (None, None)],
#        disp=True,
#        prior_level_sd=0.1,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, disp=True,
#                 prior_level_sd=None)
#    model.fit.assert_called_with(
#        bounds=[(None, None), (None, None), (None, None), (None, None)],
#        disp=True,
#        prior_level_sd=None,
#        nseasons=[],
#        standardize=True
#    )
#
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:],
#                                 freq_seasonal=[{'period': 3}])
#
#    model.fit = fit_mock
#
#    construct_mock = mock.Mock(return_value=model)
#
#    monkeypatch.setattr('causalimpact.main.CausalImpact._get_default_model',
#                        construct_mock)
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, disp=True,
#                 prior_level_sd=0.001, nseasons=[{'period': 3}])
#    model.fit.assert_called_with(
#        bounds=[(None, None), (0.001 / 1.2, 0.001 * 1.2), (None, None), (None, None),
#                (None, None)],
#        disp=True,
#        prior_level_sd=0.001,
#        nseasons=[{'period': 3}],
#        standardize=True
#    )
#
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel')
#
#    model.fit = fit_mock
#
#    construct_mock = mock.Mock(return_value=model)
#
#    monkeypatch.setattr('causalimpact.main.CausalImpact._get_default_model',
#                        construct_mock)
#
#    new_data = pd.DataFrame(np.random.randn(200, 1), columns=['y'])
#    CausalImpact(new_data, pre_int_period, post_int_period, disp=False)
#    model.fit.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
#
#
#def test_custom_model_fit(rand_data, pre_int_period, post_int_period, monkeypatch):
#    fit_mock = mock.Mock()
#    monkeypatch.setattr('causalimpact.main.CausalImpact._process_posterior_inferences',
#                        mock.Mock())
#
#    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:])
#
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2), (None, None), (None, None)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model, disp=True)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2), (None, None), (None, None)],
#        disp=True,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model, disp=True,
#                 prior_level_sd=0.01)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2), (None, None), (None, None)],
#        disp=True,
#        prior_level_sd=0.01,
#        nseasons=[],
#        standardize=True
#    )
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model, disp=True,
#                 prior_level_sd=None)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (None, None), (None, None), (None, None)],
#        disp=True,
#        prior_level_sd=None,
#        nseasons=[],
#        standardize=True
#    )
#
#    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
#                                 exog=pre_data.iloc[:, 1:], freq_seasonal=[{'period': 3}])
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model, disp=True,
#                 prior_level_sd=0.001)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.001 / 1.2, 0.001 * 1.2), (None, None), (None, None),
#                (None, None)],
#        disp=True,
#        prior_level_sd=0.001,
#        nseasons=[],
#        standardize=True
#    )
#
#    model = UnobservedComponents(
#        endog=pre_data.iloc[:, 0],
#        level=True,
#        exog=pre_data.iloc[:, 1],
#        trend=True,
#        seasonal=3,
#        stochastic_level=True
#    )
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model, disp=True,
#                 prior_level_sd=0.001)
#    fit_mock.assert_called_with(
#        bounds=[(0.001 / 1.2, 0.001 * 1.2), (None, None), (None, None)],
#        disp=True,
#        prior_level_sd=0.001,
#        nseasons=[],
#        standardize=True
#    )
#
#    new_pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], ['y', 'x1']]
#    model = UnobservedComponents(endog=new_pre_data.iloc[:, 0], level='llevel',
#                                 exog=new_pre_data.iloc[:, 1:])
#
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model,
#                 disp=False)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2), (None, None)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
#
#    model = UnobservedComponents(endog=new_pre_data.iloc[:, 0], level='dtrend',
#                                 exog=new_pre_data.iloc[:, 1:])
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model,
#                 disp=False)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (None, None)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
#
#    model = UnobservedComponents(endog=new_pre_data.iloc[:, 0], level='lltrend',
#                                 exog=new_pre_data.iloc[:, 1:])
#    model.fit = fit_mock
#
#    CausalImpact(rand_data, pre_int_period, post_int_period, model=model,
#                 disp=False)
#    fit_mock.assert_called_with(
#        bounds=[(None, None), (0.01 / 1.2, 0.01 * 1.2), (None, None), (None, None)],
#        disp=False,
#        nseasons=[],
#        standardize=True
#    )
