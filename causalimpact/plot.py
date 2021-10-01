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
Plots the analysis obtained in causal impact algorithm.
"""


import numpy as np
import pandas as pd


def plot(
    inferences: pd.DataFrame,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    panels=['original', 'pointwise', 'cumulative'],
    figsize=(10, 7),
    show=True
) -> None:
    """Plots inferences results related to causal impact analysis.

    Args
    ----
      panels: list.
        Indicates which plot should be considered in the graphics.
      figsize: tuple.
        Changes the size of the graphics plotted.
      show: bool.
        If true, runs plt.show(), i.e., displays the figure.
        If false, it gives acess to the axis, i.e., the figure can be saved
        and the style of the plot can be modified by getting the axis with
        `ax = plt.gca()` or the figure with `fig = plt.gcf()`.
        Defaults to True.
    Raises
    ------
      RuntimeError: if inferences were not computed yet.
    """
    plt = get_plotter()
    plt.figure(figsize=figsize)
    valid_panels = ['original', 'pointwise', 'cumulative']
    for panel in panels:
        if panel not in valid_panels:
            raise ValueError(
                '"{}" is not a valid panel. Valid panels are: {}.'.format(
                    panel, ', '.join(['"{}"'.format(e) for e in valid_panels])
                )
            )
    pre_data, post_data, inferences = build_data(pre_data, post_data, inferences)
    pre_post_index = pre_data.index.union(post_data.index)

    post_period_init = post_data.index[0]
    intervention_idx = pre_post_index.get_loc(post_period_init)
    n_panels = len(panels)
    ax = plt.subplot(n_panels, 1, 1)
    idx = 1
    color = (1.0, 0.4981, 0.0549)
    # The operation `iloc[1:]` is used mainly to remove the uncertainty associated to the
    # predictions of the first points. As the predictions follow
    # `P(z[t] | y[1...t-1], z[1...t-1])` the very first point ends up being quite noisy
    # as there's no previous point observed.
    if 'original' in panels:
        ax.plot(
            pre_post_index,
            pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]),
            'k',
            label='y'
        )
        ax.plot(
            pre_post_index[1:],
            inferences['complete_preds_means'].iloc[1:],
            color='orangered',
            ls='dashed',
            label='Predicted'
        )
        ax.axvline(pre_post_index[intervention_idx - 1], c='gray', linestyle='--')
        ax.fill_between(
            pre_post_index[1:],
            inferences['complete_preds_lower'].iloc[1:],
            inferences['complete_preds_upper'].iloc[1:],
            color=color,
            alpha=0.4
        )
        ax.legend()
        ax.grid(True, color='gainsboro')
        if idx != n_panels:
            plt.setp(ax.get_xticklabels(), visible=False)
        idx += 1
    if 'pointwise' in panels:
        ax = plt.subplot(n_panels, 1, idx, sharex=ax)
        ax.plot(
            pre_post_index[1:],
            inferences['point_effects_means'].iloc[1:],
            ls='dashed',
            color='orangered',
            label='Point Effects'
        )
        ax.axvline(pre_post_index[intervention_idx - 1], c='gray', linestyle='--')
        ax.fill_between(
            pre_post_index[1:],
            inferences['point_effects_lower'].iloc[1:],
            inferences['point_effects_upper'].iloc[1:],
            color=color,
            alpha=0.4
        )
        ax.axhline(y=0, color='gray')
        ax.legend()
        ax.grid(True, color='gainsboro')
        if idx != n_panels:
            plt.setp(ax.get_xticklabels(), visible=False)
        idx += 1
    if 'cumulative' in panels:
        ax = plt.subplot(n_panels, 1, idx, sharex=ax)
        ax.plot(
            pre_post_index[1:],
            inferences['post_cum_effects_means'].iloc[1:],
            ls='dashed',
            color='orangered',
            label='Cumulative Effect'
        )
        ax.axvline(pre_post_index[intervention_idx - 1], c='gray', linestyle='--')
        ax.fill_between(
            pre_post_index[1:],
            inferences['post_cum_effects_lower'].iloc[1:],
            inferences['post_cum_effects_upper'].iloc[1:],
            color=color,
            alpha=0.4
        )
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.legend()
        ax.grid(True, color='gainsboro')
    if show:
        plt.show()


def build_data(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    inferences: pd.DataFrame
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Input data may contain NaN points due TFP requirement for a valid frequency set. As
    it breaks the plotting API this function removes those points.
    """
    if isinstance(inferences.index, pd.RangeIndex):
        pre_data = pre_data.set_index(pd.RangeIndex(start=0, stop=len(pre_data)))
        post_data = post_data.set_index(pd.RangeIndex(start=len(pre_data),
                                        stop=len(pre_data) + len(post_data)))
    pre_data_null_index = pre_data[pre_data.iloc[:, 0].isnull()].index
    post_data_null_index = post_data[post_data.iloc[:, 0].isnull()].index

    pre_data = pre_data.drop(pre_data_null_index).astype(np.float64)
    post_data = post_data.drop(post_data_null_index).astype(np.float64)
    inferences = inferences.drop(
        pre_data_null_index.union(post_data_null_index)
    ).astype(np.float64)
    return pre_data, post_data, inferences


def get_plotter():  # pragma: no cover
    """As some environments do not have matplotlib then we import the library through
    this method which prevents import exceptions.

    Returns
    -------
      plotter: `matplotlib.pyplot`.
    """
    import matplotlib.pyplot as plt
    return plt
