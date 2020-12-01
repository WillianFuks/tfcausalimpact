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
import tensorflow_probability as tfp


import causalimpact.data as cidata


# from numpy.testing import assert_array_equal
# from pandas.core.indexes.range import RangeIndex
# from pandas.util.testing import assert_frame_equal


def test_format_input_data(rand_data):
    data = cidata.format_input_data(rand_data)
    pd.testing.assert_frame_equal(data, rand_data)

    with pytest.raises(ValueError) as excinfo:
        cidata.format_input_data('test')
    assert str(excinfo.value) == 'Could not transform input data to pandas DataFrame.'

    data = [1, 2, 3, 4, 5, 6, 2 + 1.j]
    with pytest.raises(ValueError) as excinfo:
        cidata.format_input_data(data)
    assert str(excinfo.value) == 'Input data must contain only numeric values.'

    data = np.random.randn(10, 2)
    data[0, 1] = np.nan
    with pytest.raises(ValueError) as excinfo:
        cidata.format_input_data(data)
    assert str(excinfo.value) == 'Input data cannot have NAN values.'


def test_process_pre_post_data(rand_data, date_rand_data):
    pre_data, post_data = cidata.process_pre_post_data(rand_data, [0, 10], [11, 20])
    pd.testing.assert_frame_equal(pre_data, rand_data.loc[0:10, :])
    pd.testing.assert_frame_equal(post_data, rand_data.loc[11:20, :])

    pre_data, post_data = cidata.process_pre_post_data(
        date_rand_data,
        ['20200101', '20200110'], ['20200111', '20200120']
    )
    pd.testing.assert_frame_equal(pre_data, date_rand_data.iloc[0:10, :])
    pd.testing.assert_frame_equal(post_data, date_rand_data.iloc[10:20, :])

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [5, 10], [4, 7])
    assert str(excinfo.value) == (
        'Values in training data cannot be present in the '
        'post-intervention data. Please fix your pre_period value to cover at most one '
        'point less from when the intervention happened.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200101', '20200201'],
                                     ['20200110', '20200210'])
    assert str(excinfo.value) == (
        'Values in training data cannot be present in the '
        'post-intervention data. Please fix your pre_period value to cover at most one '
        'point less from when the intervention happened.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [5, 10], [15, 11])
    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [5, 10], [10, 15])
    assert str(excinfo.value) == ('post_period first value (10) must be bigger than the '
                                  'second value of pre_period (10).')

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200101', '20200110'],
                                     ['20200115', '20200111'])
    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [0, 2], [15, 11])
    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200101', '20200102'],
                                     ['20200115', '20200111'])
    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [5, 0], [15, 11])
    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200105', '20200101'],
                                     ['20200115', '20200111'])
    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200105', '20200110'],
                                     ['20200110', '20200115'])
    assert str(excinfo.value) == ('post_period first value (20200110) must be bigger '
                                  'than the second value of pre_period (20200110).')

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, 0, [15, 11])
    assert str(excinfo.value) == 'Input period must be of type list.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, '20200101',
                                     ['20200115', '20200130'])
    assert str(excinfo.value) == 'Input period must be of type list.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [0, 10, 30], [15, 11])
    assert str(excinfo.value) == (
        'Period must have two values regarding the beginning '
        'and end of the pre and post intervention data.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [0, None], [15, 11])
    assert str(excinfo.value) == 'Input period cannot have `None` values.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [0, 5.5], [15, 11])
    assert str(excinfo.value) == 'Input must contain either int, str or pandas Timestamp'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [-2, 10], [11, 20])
    assert str(excinfo.value) == '-2 not present in input data index.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, [0, 10], [11, 2000])
    assert str(excinfo.value) == '2000 not present in input data index.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(rand_data, ['20200101', '20200110'],
                                     ['20200111', '20200130'])
    assert str(excinfo.value) == (
        '20200101 not present in input data index.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20200101', '20200110'],
                                     ['20200111', '20220130'])
    assert str(excinfo.value) == ('20220130 not present in input data index.')

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(date_rand_data, ['20190101', '20200110'],
                                     ['20200111', '20200120'])
    assert str(excinfo.value) == ('20190101 not present in input data index.')

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(
            rand_data,
            [pd.Timestamp('20200101'), pd.Timestamp('20200110')],
            [pd.Timestamp('20200111'), pd.Timestamp('20200130')]
        )
    assert str(excinfo.value) == (
        '20200101 not present in input data index.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(
            date_rand_data,
            [pd.Timestamp('20200101'), pd.Timestamp('20200110')],
            [pd.Timestamp('20200111'), pd.Timestamp('20220130')]
        )
    assert str(excinfo.value) == ('20220130 not present in input data index.')

    with pytest.raises(ValueError) as excinfo:
        cidata.process_pre_post_data(
            date_rand_data,
            [pd.Timestamp('20190101'), pd.Timestamp('20200110')],
            [pd.Timestamp('20200111'), pd.Timestamp('20200120')]
        )
    assert str(excinfo.value) == ('20190101 not present in input data index.')


def test_validate_y():
    data = pd.Series(np.ones(100) * np.nan)
    with pytest.raises(ValueError) as excinfo:
        cidata.validate_y(data)
    assert str(excinfo.value) == 'Input response cannot have just Null values.'

    data[0:2] = 1
    with pytest.raises(ValueError) as excinfo:
        cidata.validate_y(data)
    assert str(excinfo.value) == ('Input response must have more than 3 non-null points '
                                  'at least.')

    data[0:3] = 1
    with pytest.raises(ValueError) as excinfo:
        cidata.validate_y(data)
    assert str(excinfo.value) == 'Input response cannot be constant.'


def test_convert_index_to_datetime():
    data = pd.DataFrame(np.random.rand(100, 2))
    assert isinstance(data.index, pd.core.indexes.range.RangeIndex)
    data = cidata.convert_index_to_datetime(data)
    assert isinstance(data.index, pd.core.indexes.range.RangeIndex)

    data.set_index(pd.date_range(start='20200101', periods=100), inplace=True)
    data.set_index(data.index.map(lambda x: str(x)[:-9]), inplace=True)
    assert isinstance(data.index.values[0], str)

    data = cidata.convert_index_to_datetime(data)
    assert isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex)

    data = pd.DataFrame([1, 2, 3])
    data.set_index(pd.Series(['A', 'B', 'C']), inplace=True)
    assert isinstance(data.index.values[0], str)
    data = cidata.convert_index_to_datetime(data)
    assert isinstance(data.index.values[0], str)


def test_convert_date_period_to_int(date_rand_data):
    result = cidata.convert_date_period_to_int(['20200101', '20200102'], date_rand_data)
    assert result == [0, 1]


def test_alpha_input():
    alpha = cidata.process_alpha(alpha=0.5)
    assert alpha == 0.5

    with pytest.raises(ValueError) as excinfo:
        cidata.process_alpha(alpha=1)
    assert str(excinfo.value) == 'alpha must be of type float.'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_alpha(alpha=2.)
    assert str(excinfo.value) == (
        'alpha must range between 0 (zero) and 1 (one) inclusive.'
    )

    with pytest.raises(ValueError) as excinfo:
        cidata.process_alpha(alpha=-0.5)
    assert str(excinfo.value) == (
        'alpha must range between 0 (zero) and 1 (one) inclusive.'
    )


def test_process_input_data(rand_data, pre_int_period, post_int_period, date_rand_data,
                            pre_str_period, post_str_period, monkeypatch):
    for type_ in ['int', 'str']:
        cur_data = rand_data if type_ == 'int' else date_rand_data
        cur_pre_period = pre_int_period if type_ == 'int' else pre_str_period
        cur_post_period = post_int_period if type_ == 'int' else post_str_period

        results = cidata.process_input_data(cur_data, cur_pre_period, cur_post_period,
                                            None, {}, 0.05)

        pd.testing.assert_frame_equal(results['data'], cur_data)
        assert results['pre_period'] == cur_pre_period
        assert results['post_period'] == cur_post_period
        pd.testing.assert_frame_equal(
            results['pre_data'],
            cur_data.loc[cur_pre_period[0]: cur_pre_period[1], :]
        )
        pd.testing.assert_frame_equal(
            results['post_data'],
            cur_data.loc[cur_post_period[0]: cur_post_period[1], :]
        )

        np.testing.assert_almost_equal(
            results['normed_pre_data'].mean().values,
            np.array([0, 0, 0])
        )
        np.testing.assert_almost_equal(
            results['normed_pre_data'].std().values,
            np.array([1, 1, 1]),
            decimal=2
        )

        assert results['model'] is None
        assert results['model_args'] == {
            'standardize': True,
            'prior_level_sd': 0.01,
            'niter': 100,
            'nseasons': 1,
            'season_duration': 1,
            'fit_method': 'hmc'
        }
        assert results['alpha'] == 0.05

        # tests user input model setting
        model = tfp.sts.LocalLevel()
        results = cidata.process_input_data(cur_data, cur_pre_period, cur_post_period,
                                            model, {}, 0.05)
        assert isinstance(results['model'], tfp.sts.LocalLevel)

        # test normalization of data not set
        results = cidata.process_input_data(cur_data, cur_pre_period, cur_post_period,
                                            None, {'standardize': False}, 0.05)
        assert results['normed_pre_data'] is None
        assert results['normed_post_data'] is None

    # test Exceptions
    with pytest.raises(ValueError) as excinfo:
        cidata.process_input_data(None, pre_int_period, post_int_period, None, {},
                                  0.05)
    assert str(excinfo.value) == 'data input argument cannot be empty'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_input_data(rand_data, None, post_int_period, None, {}, 0.05)
    assert str(excinfo.value) == 'pre_period input argument cannot be empty'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_input_data(rand_data, pre_int_period, None, None, {}, 0.05)
    assert str(excinfo.value) == 'post_period input argument cannot be empty'

    with pytest.raises(ValueError) as excinfo:
        cidata.process_input_data(None, None, post_int_period, None, {}, 0.05)
    assert str(excinfo.value) == 'data, pre_period input arguments cannot be empty'

    # testing calls
    format_input_data_mock = mock.Mock()
    format_input_data_mock.return_value = 'processed_data'
    process_pre_post_data_mock = mock.Mock()
    process_pre_post_data_mock.return_value = ['pre_data', 'post_data']
    process_alpha_mock = mock.Mock()
    process_model_args_mock = mock.Mock()
    process_model_args_mock.return_value = {'standardize': True}
    check_input_model_mock = mock.Mock()
    standardize_pre_and_post_data_mock = mock.Mock()
    standardize_pre_and_post_data_mock.return_value = ('normed_pre_data',
                                                       'normed_post_data',
                                                       'mu_sig')

    monkeypatch.setattr('causalimpact.data.format_input_data', format_input_data_mock)
    monkeypatch.setattr('causalimpact.data.process_pre_post_data',
                        process_pre_post_data_mock)
    monkeypatch.setattr('causalimpact.data.process_alpha', process_alpha_mock)
    monkeypatch.setattr('causalimpact.data.cimodel.process_model_args',
                        process_model_args_mock)
    monkeypatch.setattr('causalimpact.data.cimodel.check_input_model',
                        check_input_model_mock)
    monkeypatch.setattr('causalimpact.data.standardize_pre_and_post_data',
                        standardize_pre_and_post_data_mock)

    cidata.process_input_data('input_data', pre_int_period, post_int_period, 'model',
                              {}, 0.05)

    format_input_data_mock.assert_called_once_with('input_data')
    process_pre_post_data_mock.assert_called_once_with('processed_data', pre_int_period,
                                                       post_int_period)
    process_alpha_mock.assert_called_once_with(0.05)
    process_model_args_mock.assert_called_once_with({})
    check_input_model_mock.assert_called_once_with('model', 'pre_data', 'post_data')
    standardize_pre_and_post_data_mock.assert_called_once_with('pre_data', 'post_data')
