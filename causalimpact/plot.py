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


import pandas as pd


def plot(
    inferences: pd.DataFrame,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    panels=['original', 'pointwise', 'cumulative'],
    figsize=(15, 12)
):
    """Plots inferences results related to causal impact analysis.

    Args
    ----
      panels: list.
        Indicates which plot should be considered in the graphics.
      figsize: tuple.
        Changes the size of the graphics plotted.

    Raises
    ------
      RuntimeError: if inferences were not computed yet.
    """
    plt = get_plotter()
    _ = plt.figure(figsize=figsize)
    valid_panels = ['original', 'pointwise', 'cumulative']
    for panel in panels:
        if panel not in valid_panels:
            raise ValueError(
                '"{}" is not a valid panel. Valid panels are: {}.'.format(
                    panel, ', '.join(['"{}"'.format(e) for e in valid_panels])
                )
            )
    post_period_init = post_data.index[0]
    intervention_idx = inferences.index.get_loc(post_period_init)
    n_panels = len(panels)
    ax = plt.subplot(n_panels, 1, 1)
    idx = 1
    if 'original' in panels:
        ax.plot(pd.concat([pre_data.iloc[:, 0], post_data.iloc[:, 0]]), 'k', label='y')
        ax.plot(inferences['preds_means'], 'b--', label='Predicted')
        ax.axvline(inferences.index[intervention_idx - 1], c='k', linestyle='--')
        ax.fill_between(
            pre_data.index.union(post_data.index),
            inferences['preds_lower'],
            inferences['preds_upper'],
            facecolor='blue',
            interpolate=True,
            alpha=0.25
        )
        ax.grid(True, linestyle='--')
        ax.legend()
        if idx != n_panels:
            plt.setp(ax.get_xticklabels(), visible=False)
        idx += 1
    if 'pointwise' in panels:
        ax = plt.subplot(n_panels, 1, idx, sharex=ax)
        ax.plot(inferences['point_effects'], 'b--', label='Point Effects')
        ax.axvline(inferences.index[intervention_idx - 1], c='k', linestyle='--')
        ax.fill_between(
            inferences['point_effects'].index,
            inferences['point_effects_lower'],
            inferences['point_effects_upper'],
            facecolor='blue',
            interpolate=True,
            alpha=0.25
        )
        ax.axhline(y=0, color='k', linestyle='--')
        ax.grid(True, linestyle='--')
        ax.legend()
        if idx != n_panels:
            plt.setp(ax.get_xticklabels(), visible=False)
        idx += 1
    if 'cumulative' in panels:
        ax = plt.subplot(n_panels, 1, idx, sharex=ax)
        ax.plot(inferences['post_cum_effects'], 'b--',
                label='Cumulative Effect')
        ax.axvline(inferences.index[intervention_idx - 1], c='k', linestyle='--')
        ax.fill_between(
            inferences['post_cum_effects'].index,
            inferences['post_cum_effects_lower'],
            inferences['post_cum_effects_upper'],
            facecolor='blue',
            interpolate=True,
            alpha=0.25
        )
        ax.grid(True, linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend()
    plt.show()


def get_plotter():  # pragma: no cover
    """As some environments do not have matplotlib then we import the library through
    this method which prevents import exceptions.

    Returns
    -------
      plotter: `matplotlib.pyplot`.
    """
    import matplotlib.pyplot as plt
    return plt