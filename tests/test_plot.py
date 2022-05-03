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
Tests for module plot.py. Module matplotlib is not required as it's mocked accordingly.
"""


import mock
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import causalimpact.plot as plotter


@pytest.fixture
def inferences(rand_data):
    df = pd.DataFrame(np.random.rand(len(rand_data), 9))
    df.columns = [
        'complete_preds_means',
        'complete_preds_lower',
        'complete_preds_upper',
        'point_effects_means',
        'point_effects_lower',
        'point_effects_upper',
        'post_cum_effects_means',
        'post_cum_effects_lower',
        'post_cum_effects_upper'
    ]
    return df


def test_build_data():
    pre_data = pd.DataFrame([0, 1, np.nan])
    # `post_data` is assumed to not have `NaN` values already
    post_data = pd.DataFrame([3, 4], index=[3, 4])
    inferences = pd.DataFrame([0, 1, 2, 3, 4])

    pre_data, post_data, inferences = plotter.build_data(pre_data, post_data, inferences)

    expected_pre_data = pd.DataFrame([0, 1]).astype(np.float64)
    pd.testing.assert_frame_equal(pre_data, expected_pre_data)

    expected_post_data = pd.DataFrame([3, 4], index=[3, 4]).astype(np.float64)
    pd.testing.assert_frame_equal(post_data, expected_post_data)

    expected_inferences = pd.DataFrame([0, 1, 3, 4],
                                       index=[0, 1, 3, 4]).astype(np.float64)
    pd.testing.assert_frame_equal(inferences, expected_inferences)


def test_plot_original_panel(rand_data, pre_int_period, post_int_period, inferences,
                             monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1]]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['original'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
        ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == 'k'
    assert ax_args[0][1] == {'label': 'y'}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences['complete_preds_means'].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {'color': 'orangered', 'ls': 'dashed', 'label': 'Predicted'}

    ax_mock.axvline.assert_called_with(pre_int_period[1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['complete_preds_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['complete_preds_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_original_panel_gap_data(rand_data, pre_int_gap_period, post_int_gap_period,
                                      inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_gap_period[0]: pre_int_gap_period[1]]
    post_data = rand_data.loc[post_int_gap_period[0]: post_int_gap_period[1]]

    pre_data = pre_data.set_index(pd.RangeIndex(start=0, stop=len(pre_data)))
    post_data = post_data.set_index(pd.RangeIndex(start=len(pre_data),
                                    stop=len(pre_data) + len(post_data)))

    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['original'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
        ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == 'k'
    assert ax_args[0][1] == {'label': 'y'}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences['complete_preds_means'].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {'color': 'orangered', 'ls': 'dashed', 'label': 'Predicted'}

    ax_mock.axvline.assert_called_with(pre_data.index[-1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['complete_preds_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['complete_preds_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_original_panel_date_index(date_rand_data, pre_str_period, post_str_period,
                                        inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    inferences = inferences.set_index(pre_post_index)
    plotter.plot(inferences, pre_data, post_data, panels=['original'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
        ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == 'k'
    assert ax_args[0][1] == {'label': 'y'}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences['complete_preds_means'].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {'color': 'orangered', 'ls': 'dashed', 'label': 'Predicted'}

    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['complete_preds_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['complete_preds_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_original_panel_gap_date_index(date_rand_data, pre_str_gap_period,
                                            post_str_gap_period, inferences,
                                            monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_gap_period[0]: pre_str_gap_period[1]]
    post_data = date_rand_data.loc[post_str_gap_period[0]: post_str_gap_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['original'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
        ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == 'k'
    assert ax_args[0][1] == {'label': 'y'}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences['complete_preds_means'].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {'color': 'orangered', 'ls': 'dashed', 'label': 'Predicted'}

    idx_value = pre_post_index.get_loc(pre_str_gap_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['complete_preds_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['complete_preds_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_original_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                post_str_period, inferences,
                                                monkeypatch):
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20], inplace=True)
    plot_mock = mock.Mock()
    pre_data = dd.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = dd.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['original'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
        ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == 'k'
    assert ax_args[0][1] == {'label': 'y'}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences['complete_preds_means'].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {'color': 'orangered', 'ls': 'dashed', 'label': 'Predicted'}

    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['complete_preds_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['complete_preds_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_pointwise_panel(rand_data, pre_int_period, post_int_period, inferences,
                              monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1]]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['point_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Point Effects', 'ls': 'dashed',
                             'color': 'orangered'}
    ax_mock.axvline.assert_called_with(pre_int_period[1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_pointwise_panel_gap_data(rand_data, pre_int_gap_period,
                                       post_int_gap_period, inferences,
                                       monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_gap_period[0]: pre_int_gap_period[1]]
    post_data = rand_data.loc[post_int_gap_period[0]: post_int_gap_period[1]]

    pre_data = pre_data.set_index(pd.RangeIndex(start=0, stop=len(pre_data)))
    post_data = post_data.set_index(pd.RangeIndex(start=len(pre_data),
                                    stop=len(pre_data) + len(post_data)))

    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['point_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Point Effects', 'ls': 'dashed',
                             'color': 'orangered'}
    ax_mock.axvline.assert_called_with(pre_data.index[-1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_pointwise_panel_date_index(date_rand_data, pre_str_period, post_str_period,
                                         inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(pre_post_index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['point_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Point Effects', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_pointwise_panel_gap_date_index(date_rand_data, pre_str_gap_period,
                                             post_str_gap_period, inferences,
                                             monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_gap_period[0]: pre_str_gap_period[1]]
    post_data = date_rand_data.loc[post_str_gap_period[0]: post_str_gap_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['point_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Point Effects', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_gap_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_pointwise_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                 post_str_period, inferences,
                                                 monkeypatch):
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20], inplace=True)
    plot_mock = mock.Mock()
    pre_data = dd.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = dd.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['point_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Point Effects', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_cumulative_panel(rand_data, pre_int_period, post_int_period, inferences,
                               monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1]]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['post_cum_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Cumulative Effect', 'ls': 'dashed',
                             'color': 'orangered'}
    ax_mock.axvline.assert_called_with(pre_int_period[1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_cumulative_panel_gap_data(rand_data, pre_int_gap_period,
                                        post_int_gap_period, inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_gap_period[0]: pre_int_gap_period[1]]
    post_data = rand_data.loc[post_int_gap_period[0]: post_int_gap_period[1]]

    pre_data = pre_data.set_index(pd.RangeIndex(start=0, stop=len(pre_data)))
    post_data = post_data.set_index(pd.RangeIndex(start=len(pre_data),
                                    stop=len(pre_data) + len(post_data)))

    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['post_cum_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Cumulative Effect', 'ls': 'dashed',
                             'color': 'orangered'}
    ax_mock.axvline.assert_called_with(pre_data.index[-1], c='gray', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_cumulative_panel_date_index(date_rand_data, pre_str_period,
                                          post_str_period, inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(pre_post_index)
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['post_cum_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Cumulative Effect', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_cumulative_panel_gap_date_index(date_rand_data, pre_str_gap_period,
                                              post_str_gap_period, inferences,
                                              monkeypatch):
    plot_mock = mock.Mock()
    pre_data = date_rand_data.loc[pre_str_gap_period[0]: pre_str_gap_period[1]]
    post_data = date_rand_data.loc[post_str_gap_period[0]: post_str_gap_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['post_cum_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Cumulative Effect', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_gap_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_cumulative_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                  post_str_period, inferences,
                                                  monkeypatch):
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20], inplace=True)
    plot_mock = mock.Mock()
    pre_data = dd.loc[pre_str_period[0]: pre_str_period[1]]
    post_data = dd.loc[post_str_period[0]: post_str_period[1]]
    pre_post_index = pre_data.index.union(post_data.index)
    inferences = inferences.set_index(
        pd.date_range(start=pre_post_index[0], periods=len(inferences))
    )
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index[1:], ax_args[0][0][0])
    assert_array_equal(
        inferences['post_cum_effects_means'][1:],
        ax_args[0][0][1]
    )
    assert ax_args[0][1] == {'label': 'Cumulative Effect', 'ls': 'dashed',
                             'color': 'orangered'}
    idx_value = pre_post_index.get_loc(pre_str_period[1])
    ax_mock.axvline.assert_called_with(pre_post_index[idx_value], c='gray',
                                       linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'].iloc[1:])
    assert ax_args[1] == {'color': (1.0, 0.4981, 0.0549), 'alpha': 0.4}

    ax_mock.grid.assert_called_with(True, color='gainsboro')
    ax_mock.legend.assert_called()
    plot_mock.return_value.show.assert_called_once()


def test_plot_multi_panels(rand_data, pre_int_period, post_int_period, inferences,
                           monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1]]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1]]
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['original', 'pointwise'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    ax_mock = plot_mock.return_value.subplot.return_value
    assert ax_mock.plot.call_count == 3
    assert ax_mock.axvline.call_count == 2
    assert ax_mock.grid.call_count == 2
    assert ax_mock.legend.call_count == 2
    assert ax_mock.fill_between.call_count == 2
    assert ax_mock.axhline.call_count == 1
    assert plot_mock.return_value.setp.call_count == 1

    plot_mock.reset_mock()
    ax_mock.reset_mock()
    plotter.plot(inferences, pre_data, post_data, panels=['original', 'cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    ax_mock = plot_mock.return_value.subplot.return_value
    assert ax_mock.plot.call_count == 3
    assert ax_mock.axvline.call_count == 2
    assert ax_mock.grid.call_count == 2
    assert ax_mock.legend.call_count == 2
    assert ax_mock.fill_between.call_count == 2
    assert ax_mock.axhline.call_count == 1
    assert plot_mock.return_value.setp.call_count == 1

    plot_mock.reset_mock()
    ax_mock.reset_mock()
    plotter.plot(inferences, pre_data, post_data, panels=['original', 'pointwise',
                 'cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    ax_mock = plot_mock.return_value.subplot.return_value
    assert ax_mock.plot.call_count == 4
    assert ax_mock.axvline.call_count == 3
    assert ax_mock.grid.call_count == 3
    assert ax_mock.legend.call_count == 3
    assert ax_mock.fill_between.call_count == 3
    assert ax_mock.axhline.call_count == 2
    assert plot_mock.return_value.setp.call_count == 2

    plot_mock.reset_mock()
    ax_mock.reset_mock()
    plotter.plot(inferences, pre_data, post_data, panels=['pointwise', 'cumulative'])
    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    ax_mock = plot_mock.return_value.subplot.return_value
    assert ax_mock.plot.call_count == 2
    assert ax_mock.axvline.call_count == 2
    assert ax_mock.grid.call_count == 2
    assert ax_mock.legend.call_count == 2
    assert ax_mock.fill_between.call_count == 2
    assert ax_mock.axhline.call_count == 2
    assert plot_mock.return_value.setp.call_count == 1


def test_plot_raises_wrong_input_panel(rand_data, pre_int_period, post_int_period,
                                       inferences, monkeypatch):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1]]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1]]
    monkeypatch.setattr('causalimpact.plot.get_plotter', plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=['cumulative'])

    with pytest.raises(ValueError) as excinfo:
        plotter.plot(inferences, rand_data, rand_data, panels=['test'])
    assert str(excinfo.value) == (
        '"test" is not a valid panel. Valid panels are: '
        '"original", "pointwise", "cumulative".'
    )


def test_plot_original_panel_gap_data_show_is_false(
    rand_data, pre_int_gap_period, post_int_gap_period, inferences, monkeypatch
):
    plot_mock = mock.Mock()
    pre_data = rand_data.loc[pre_int_gap_period[0]: pre_int_gap_period[1]]
    post_data = rand_data.loc[post_int_gap_period[0]: post_int_gap_period[1]]

    pre_data = pre_data.set_index(pd.RangeIndex(start=0, stop=len(pre_data)))
    post_data = post_data.set_index(pd.RangeIndex(start=len(pre_data),
                                    stop=len(pre_data) + len(post_data)))

    pre_post_index = pre_data.index.union(post_data.index)
    monkeypatch.setattr("causalimpact.plot.get_plotter", plot_mock)
    plotter.plot(inferences, pre_data, post_data, panels=["original"], show=False)

    plot_mock.assert_called_once()
    plot_mock.return_value.figure.assert_called_with(figsize=(10, 7))
    plot_mock.return_value.subplot.assert_any_call(1, 1, 1)
    ax_mock = plot_mock.return_value.subplot.return_value
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(pre_post_index, ax_args[0][0][0])
    assert_array_equal(
        pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]), ax_args[0][0][1]
    )
    assert ax_args[0][0][2] == "k"
    assert ax_args[0][1] == {"label": "y"}
    assert_array_equal(pre_post_index[1:], ax_args[1][0][0])
    assert_array_equal(inferences["complete_preds_means"].iloc[1:], ax_args[1][0][1])
    assert ax_args[1][1] == {"color": "orangered", "ls": "dashed", "label": "Predicted"}

    ax_mock.axvline.assert_called_with(pre_data.index[-1], c="gray", linestyle="--")

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], pre_post_index[1:])
    assert_array_equal(ax_args[0][1], inferences["complete_preds_lower"].iloc[1:])
    assert_array_equal(ax_args[0][2], inferences["complete_preds_upper"].iloc[1:])
    assert ax_args[1] == {"color": (1.0, 0.4981, 0.0549), "alpha": 0.4}

    ax_mock.grid.assert_called_with(True, color="gainsboro")
    ax_mock.legend.assert_called()
    # If `show == False` then `plt.show()` should not have been called
    plot_mock.return_value.show.assert_not_called()
