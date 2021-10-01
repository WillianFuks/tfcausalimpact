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
Uses the posterior distribution to prepare inferences for the Causal Impact summary and
plotting functionalities.
"""


from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from causalimpact.misc import maybe_unstandardize

tfd = tfp.distributions


def get_lower_upper_percentiles(alpha: float) -> List[float]:
    """
    Returns the lower and upper quantile values for the chosen `alpha` value.

    Args
    ----
      alpha: float
         Sets the size of the credible interval. If `alpha=0.05` then extracts the
         95% credible interval for forecasts.

    Returns
    -------
      List[float]
          First value is the lower quantile and second value is upper.
    """
    return [alpha * 100. / 2., 100 - alpha * 100. / 2.]


def compile_posterior_inferences(
    original_index: pd.core.indexes.base.Index,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    one_step_dist: tfd.Distribution,
    posterior_dist: tfd.Distribution,
    mu_sig: Optional[Tuple[float, float]],
    alpha: float = 0.05,
    niter: int = 1000
) -> pd.DataFrame:
    """
    Uses the posterior distribution of the structural time series probabilistic
    model to run predictions and forecasts for observed data. Results are stored for
    later usage on the summary and plotting functionalities.

    Args
    ----
      original_index: pd.core.indexes.base.Index
          Original index from input data. If it's a `RangeIndex` then cast inferences
          index to be of the same type.
      pre_data: pd.DataFrame
          This is the original input data, that is, it's not standardized.
      post_data: pd.DataFrame
          Same as `pre_data`.
          This is the original input data, that is, it's not standardized.
      one_step_dist: tfd.Distribution
          Uses posterior parameters to run one-step-prediction on past observed data.
      posterior_dist: tfd.Distribution
          Uses posterior parameters to run forecasts on post intervention data.
      mu_sig: Optional[Tuple[float, float]]
          First value is the mean used for standardization and second value is the
          standard deviation.
      alpha: float
          Sets credible interval size.
      niter: int
          Total mcmc samples to sample from the posterior structural model.

    Returns
    -------
      inferences: pd.DataFrame
          Final dataframe with all data related to one-step predictions and forecasts.
    """
    lower_percen, upper_percen = get_lower_upper_percentiles(alpha)
    # Integrates pre and post index for cumulative index data.
    cum_index = build_cum_index(pre_data.index, post_data.index)
    # We create a pd.Series with a single 0 (zero) value to work as the initial value
    # when computing the cumulative inferences. Without this value the plotting of
    # cumulative data breaks at the initial point.
    zero_series = pd.Series([0])
    simulated_pre_ys = one_step_dist.sample(niter)  # shape (niter, n_train_timestamps, 1)
    simulated_pre_ys = maybe_unstandardize(
        np.squeeze(simulated_pre_ys.numpy()),
        mu_sig
    )  # shape (niter, n_forecasts)
    simulated_post_ys = posterior_dist.sample(niter)  # shape (niter, n_forecasts, 1)
    simulated_post_ys = maybe_unstandardize(
        np.squeeze(simulated_post_ys.numpy()),
        mu_sig
    )  # shape (niter, n_forecasts)
    # Pre inference
    pre_preds_means = one_step_dist.mean()
    pre_preds_means = pd.Series(
        np.squeeze(
            maybe_unstandardize(pre_preds_means, mu_sig)
        ),
        index=pre_data.index
    )
    pre_preds_lower, pre_preds_upper = np.percentile(
        simulated_pre_ys,
        [lower_percen, upper_percen],
        axis=0
    )
    pre_preds_lower = pd.Series(pre_preds_lower, index=pre_data.index)
    pre_preds_upper = pd.Series(pre_preds_upper, index=pre_data.index)
    # Post inference
    post_preds_means = posterior_dist.mean()
    post_preds_means = pd.Series(
        np.squeeze(
            maybe_unstandardize(post_preds_means, mu_sig)
        ),
        index=post_data.index
    )
    post_preds_lower, post_preds_upper = np.percentile(
        simulated_post_ys,
        [lower_percen, upper_percen],
        axis=0
    )
    post_preds_lower = pd.Series(post_preds_lower, index=post_data.index)
    post_preds_upper = pd.Series(post_preds_upper, index=post_data.index)
    # Concatenations
    complete_preds_means = pd.concat([pre_preds_means, post_preds_means])
    complete_preds_lower = pd.concat([pre_preds_lower, post_preds_lower])
    complete_preds_upper = pd.concat([pre_preds_upper, post_preds_upper])
    # Cumulative
    post_cum_y = np.cumsum(post_data.iloc[:, 0])
    post_cum_y = pd.concat([zero_series, post_cum_y], axis=0)
    post_cum_y.index = cum_index
    post_cum_preds_means = np.cumsum(post_preds_means)
    post_cum_preds_means = pd.concat([zero_series, post_cum_preds_means])
    post_cum_preds_means.index = cum_index
    post_cum_preds_lower, post_cum_preds_upper = np.percentile(
        np.cumsum(simulated_post_ys, axis=1),
        [lower_percen, upper_percen],
        axis=0
    )
    # Sets index properly
    post_cum_preds_lower = pd.Series(
        np.squeeze(
            np.concatenate([[0], post_cum_preds_lower])
        ),
        index=cum_index
    )
    post_cum_preds_upper = pd.Series(
        np.squeeze(
            np.concatenate([[0], post_cum_preds_upper])
        ),
        index=cum_index
    )
    # Using a net value of data to accomodate cases where there're gaps between pre
    # and post intervention periods.
    net_data = pd.concat([pre_data, post_data])
    # Point effects
    point_effects_means = net_data.iloc[:, 0] - complete_preds_means
    point_effects_upper = net_data.iloc[:, 0] - complete_preds_lower
    point_effects_lower = net_data.iloc[:, 0] - complete_preds_upper
    post_point_effects_means = post_data.iloc[:, 0] - post_preds_means
    # Cumulative point effects analysis
    post_cum_effects_means = np.cumsum(post_point_effects_means)
    post_cum_effects_means = pd.concat([zero_series, post_cum_effects_means])
    post_cum_effects_means.index = cum_index
    post_cum_effects_lower, post_cum_effects_upper = np.percentile(
        np.cumsum(post_data.iloc[:, 0].values - simulated_post_ys, axis=1),
        [lower_percen, upper_percen],
        axis=0
    )
    # Sets index properly.
    post_cum_effects_lower = pd.Series(
        np.squeeze(
            np.concatenate([[0], post_cum_effects_lower])
        ),
        index=cum_index
    )
    post_cum_effects_upper = pd.Series(
        np.squeeze(
            np.concatenate([[0], post_cum_effects_upper])
        ),
        index=cum_index
    )

    inferences = pd.concat(
        [
            complete_preds_means,
            complete_preds_lower,
            complete_preds_upper,
            post_preds_means,
            post_preds_lower,
            post_preds_upper,
            post_cum_y,
            post_cum_preds_means,
            post_cum_preds_lower,
            post_cum_preds_upper,
            point_effects_means,
            point_effects_lower,
            point_effects_upper,
            post_cum_effects_means,
            post_cum_effects_lower,
            post_cum_effects_upper
        ],
        axis=1
    )
    inferences.columns = [
        'complete_preds_means',
        'complete_preds_lower',
        'complete_preds_upper',
        'post_preds_means',
        'post_preds_lower',
        'post_preds_upper',
        'post_cum_y',
        'post_cum_preds_means',
        'post_cum_preds_lower',
        'post_cum_preds_upper',
        'point_effects_means',
        'point_effects_lower',
        'point_effects_upper',
        'post_cum_effects_means',
        'post_cum_effects_lower',
        'post_cum_effects_upper'
    ]
    if isinstance(original_index, pd.RangeIndex):
        inferences.set_index(pd.RangeIndex(start=0, stop=len(inferences)), inplace=True)
    return inferences


def build_cum_index(
    pre_data_index: Union[pd.core.indexes.range.RangeIndex,
                          pd.core.indexes.datetimes.DatetimeIndex],
    post_data_index: Union[pd.core.indexes.range.RangeIndex,
                           pd.core.indexes.datetimes.DatetimeIndex]
) -> Union[pd.core.indexes.range.RangeIndex,
           pd.core.indexes.datetimes.DatetimeIndex]:
    """
    As the cumulative data has one more data point (the first point is a zero),
    we add to the post-intervention data the first index of the pre-data right at the
    beginning of the index. This helps in the plotting functionality.

    Args
    ----
      pre_data_index: Union[pd.core.indexes.range.RangeIndex,
                            pd.core.indexes.datetimes.DatetimeIndex]
      post_data_index: Union[pd.core.indexes.range.RangeIndex,
                             pd.core.indexes.datetimes.DatetimeIndex]

    Returns
    -------
      Union[pd.core.indexes.range.RangeIndex, pd.core.indexes.datetimes.DatetimeIndex]
          `post_data_index` extended with the latest index value from `pre_data`.
    """
    # In newer versions of Numpy/Pandas, the union operation between indices returns
    # an Index with `dtype=object`. We, therefore, create this variable in order to
    # restore the original value which is used later on by the plotting interface.
    index_dtype = post_data_index.dtype
    new_idx = post_data_index.union([pre_data_index[-1]])
    new_idx = new_idx.astype(index_dtype)
    return new_idx


def summarize_posterior_inferences(
    post_preds_means: pd.core.series.Series,
    post_data: pd.DataFrame,
    simulated_ys: Union[np.array, tf.Tensor],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    After running the posterior inferences compilation, this function aggregates the
    results and gets the final interpretation for the Causal Impact results, such as
    the expected absolute impact of the given intervention and its credible interval.

    Args
    ----
      post_preds_means: pd.core.series.Series
          Forecats means of post intervention data.
      post_data: pd.DataFrame
      simulated_ys: Union[np.array, tf.tensor]
          Array of simulated forecasts for response `y` extract from running mcmc samples
          from the posterior `P(z | y)`.
      alpha: float

    Returns
    -------
      summary: pd.DataFrame
          Summary data which is used in the `summary` functionality.
    """
    lower_percen, upper_percen = get_lower_upper_percentiles(alpha)
    # Compute the mean of metrics
    mean_post_y = post_data.iloc[:, 0].mean()
    mean_post_pred = post_preds_means.mean()
    mean_post_pred_lower, mean_post_pred_upper = np.percentile(
        simulated_ys.mean(axis=1), [lower_percen, upper_percen]
    )
    # Compute the sum of metrics
    sum_post_y = post_data.iloc[:, 0].sum()
    sum_post_pred = post_preds_means.sum()
    sum_post_pred_lower, sum_post_pred_upper = np.percentile(
        simulated_ys.sum(axis=1), [lower_percen, upper_percen]
    )
    # Causal Impact analysis metrics
    abs_effect = mean_post_y - mean_post_pred
    abs_effect_lower = mean_post_y - mean_post_pred_upper
    abs_effect_upper = mean_post_y - mean_post_pred_lower
    # Sum
    sum_abs_effect = sum_post_y - sum_post_pred
    sum_abs_effect_lower = sum_post_y - sum_post_pred_upper
    sum_abs_effect_upper = sum_post_y - sum_post_pred_lower
    # Relative
    rel_effect = abs_effect / mean_post_pred
    rel_effect_lower = abs_effect_lower / mean_post_pred
    rel_effect_upper = abs_effect_upper / mean_post_pred
    # Sum relative
    sum_rel_effect = sum_abs_effect / sum_post_pred
    sum_rel_effect_lower = sum_abs_effect_lower / sum_post_pred
    sum_rel_effect_upper = sum_abs_effect_upper / sum_post_pred

    summary = [
        [mean_post_y, sum_post_y],
        [mean_post_pred, sum_post_pred],
        [mean_post_pred_lower, sum_post_pred_lower],
        [mean_post_pred_upper, sum_post_pred_upper],
        [abs_effect, sum_abs_effect],
        [abs_effect_lower, sum_abs_effect_lower],
        [abs_effect_upper, sum_abs_effect_upper],
        [rel_effect, sum_rel_effect],
        [rel_effect_lower, sum_rel_effect_lower],
        [rel_effect_upper, sum_rel_effect_upper]
    ]
    summary = pd.DataFrame(
        summary,
        columns=['average', 'cumulative'],
        index=[
            'actual',
            'predicted',
            'predicted_lower',
            'predicted_upper',
            'abs_effect',
            'abs_effect_lower',
            'abs_effect_upper',
            'rel_effect',
            'rel_effect_lower',
            'rel_effect_upper'
        ]
    )
    return summary


def compute_p_value(
    simulated_ys: Union[np.array, tf.Tensor],
    post_data_sum: float
) -> float:
    """
    Computes the p-value for the hypothesis testing that there's signal in the
    observed data. The computation follows the same idea as the one implemented in the
    origina R package which consists of simulating with the fitted parameters several
    time series for the post-intervention period and counting how many either surpass the
    total summation of `y` (positive relative effect) or how many falls under its
    summation (negative relative effect).

    Args
    ----
      simulated_ys: Union[np.array, tf.Tensor]
          Forecast simulations for value of `y` extracted from `P(z | y)`.
      post_data_sum: float
          sum of post intervention data.

    Returns
    -------
      p_value: float
          Ranging between 0 and 1, represents the likelihood of obtaining the observed
          data by random chance.
    """
    sim_sum = tf.reduce_sum(simulated_ys, axis=1)
    # The minimum value between positive and negative signals reveals how many times
    # either the summation of the simulation could surpass ``y_post_sum`` or be
    # surpassed by the same (in which case it means the sum of the simulated time
    # series is bigger than ``y_post_sum`` most of the time, meaning the signal in
    # this case reveals the impact caused the response variable to decrease from what
    # was expected had no effect taken place.
    signal = min(
        np.sum(sim_sum > post_data_sum),
        np.sum(sim_sum < post_data_sum)
    )
    return signal / (len(simulated_ys) + 1)
