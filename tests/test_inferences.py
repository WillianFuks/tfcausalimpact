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


import numpy as np
import pandas as pd
import tensorflow as tf

import causalimpact.inferences as inferrer
from causalimpact.misc import get_z_score, maybe_unstandardize


def test_get_lower_upper_percentiles():
    lower_percen, upper_percen = inferrer.get_lower_upper_percentiles(0.05)
    assert [lower_percen, upper_percen] == [2.5, 97.5]


def test_maybe_unstandardize():
    data = pd.DataFrame(np.arange(0, 10))
    results = inferrer.maybe_unstandardize(data)
    pd.testing.assert_frame_equal(results, data)

    mu_sig = (1, 2)
    results = inferrer.maybe_unstandardize(data, mu_sig)
    pd.testing.assert_frame_equal(results, data * 2 + 1)


def test_build_cum_index():
    data = pd.DataFrame(np.arange(0, 10))
    pre_data = data.iloc[:4]
    post_data = data.iloc[6:]
    new_index = inferrer.build_cum_index(pre_data.index, post_data.index)
    np.testing.assert_equal(new_index, np.array([3, 6, 7, 8, 9]))


def test_compile_posterior_inferences():
    data = pd.DataFrame(np.arange(10))
    pre_data = data.iloc[:3]
    post_data = data.iloc[7:]
    one_step_mean = 3
    one_step_stddev = 1.5
    posterior_mean = 7.5
    posterior_stddev = 1.5
    alpha = 0.05
    mu = 1
    sig = 2
    mu_sig = (mu, sig)
    niter = 10

    class OneStepDist:
        def mean(self):
            return np.ones((len(pre_data), 1)) * one_step_mean

        def stddev(self):
            return np.ones((len(pre_data), 1)) * one_step_stddev

    class PosteriorDist:
        def sample(self, niter):
            tmp = tf.convert_to_tensor(
                np.tile(np.arange(start=7.1, stop=10.1, step=1), (niter, 1)) +
                np.arange(niter).reshape(-1, 1),
                dtype=np.float32
            )
            tmp = tmp[..., tf.newaxis]
            return tmp

        def mean(self):
            return np.ones((len(post_data), 1)) * posterior_mean

        def stddev(self):
            return np.ones((len(post_data), 1)) * posterior_stddev

    one_step_dist = OneStepDist()
    posterior_dist = PosteriorDist()
    inferences = inferrer.compile_posterior_inferences(
        pre_data,
        post_data,
        one_step_dist,
        posterior_dist,
        mu_sig,
        alpha=alpha,
        niter=niter
    )

    expected_index = np.array([0, 1, 2, 7, 8, 9])
    # test complete_preds_means
    expec_complete_preds_means = pd.DataFrame(
        data=np.array([7, 7, 7, 16, 16, 16]),
        index=expected_index,
        dtype=np.float64,
        columns=['complete_preds_means']
    )
    pd.testing.assert_series_equal(
        expec_complete_preds_means['complete_preds_means'],
        inferences['complete_preds_means']
    )
    # test complete_preds_lower
    pre_preds_lower = (
        np.array([1, 1, 1]) * one_step_mean -
        get_z_score(1 - alpha / 2) * one_step_stddev
    ) * sig + mu
    pre_preds_lower[np.abs(pre_preds_lower) > np.mean(pre_preds_lower) +
                    2 * np.std(pre_preds_lower)] = np.nan
    post_preds_lower = (
        np.array([1, 1, 1]) * posterior_mean -
        get_z_score(1 - alpha / 2) * posterior_stddev
    ) * sig + mu
    expec_complete_preds_lower = np.concatenate([pre_preds_lower, post_preds_lower])
    expec_complete_preds_lower = pd.DataFrame(
        data=expec_complete_preds_lower,
        index=expected_index,
        dtype=np.float64,
        columns=['complete_preds_lower']
    )
    pd.testing.assert_series_equal(
        expec_complete_preds_lower['complete_preds_lower'],
        inferences['complete_preds_lower']
    )
    # test complete_preds_upper
    pre_preds_upper = (
        np.array([1, 1, 1]) * one_step_mean +
        get_z_score(1 - alpha / 2) * one_step_stddev
    ) * sig + mu
    pre_preds_upper[np.abs(pre_preds_upper) > np.mean(pre_preds_upper) +
                    2 * np.std(pre_preds_upper)] = np.nan
    post_preds_upper = (
        np.array([1, 1, 1]) * posterior_mean +
        get_z_score(1 - alpha / 2) * posterior_stddev
    ) * sig + mu
    expec_complete_preds_upper = np.concatenate([pre_preds_upper, post_preds_upper])
    expec_complete_preds_upper = pd.DataFrame(
        data=expec_complete_preds_upper,
        index=expected_index,
        dtype=np.float64,
        columns=['complete_preds_upper']
    )
    pd.testing.assert_series_equal(
        expec_complete_preds_upper['complete_preds_upper'],
        inferences['complete_preds_upper']
    )
    # test post_preds_means
    expec_post_preds_means = pd.DataFrame(
        data=np.array([np.nan] * 3 + [posterior_mean * sig + mu] * len(pre_data)),
        index=expected_index,
        dtype=np.float64,
        columns=['post_preds_means']
    )
    pd.testing.assert_series_equal(
        expec_post_preds_means['post_preds_means'],
        inferences['post_preds_means']
    )
    # test post_preds_lower
    post_preds_lower = (
        np.array([np.nan] * 3 + [1, 1, 1]) * posterior_mean -
        get_z_score(1 - alpha / 2) * posterior_stddev
    ) * sig + mu
    expec_post_preds_lower = pd.DataFrame(
        data=post_preds_lower,
        index=expected_index,
        dtype=np.float64,
        columns=['post_preds_lower']
    )
    pd.testing.assert_series_equal(
        expec_post_preds_lower['post_preds_lower'],
        inferences['post_preds_lower']
    )
    # test post_preds_upper
    post_preds_upper = (
        np.array([np.nan] * 3 + [1, 1, 1]) * posterior_mean +
        get_z_score(1 - alpha / 2) * posterior_stddev
    ) * sig + mu
    expec_post_preds_upper = pd.DataFrame(
        data=post_preds_upper,
        index=expected_index,
        dtype=np.float64,
        columns=['post_preds_upper']
    )
    pd.testing.assert_series_equal(
        expec_post_preds_upper['post_preds_upper'],
        inferences['post_preds_upper']
    )
    # test post_cum_Y
    post_cum_y = np.concatenate([[np.nan] * (len(pre_data) - 1) + [0],
                                np.cumsum(post_data.iloc[:, 0])])
    expec_post_cum_y = pd.DataFrame(
        data=post_cum_y,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_y']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_y['post_cum_y'],
        inferences['post_cum_y']
    )
    # test post_cum_preds_means
    expec_post_cum_preds_means = np.cumsum(expec_post_preds_means)
    expec_post_cum_preds_means.rename(
        columns={'post_preds_means': 'post_cum_preds_means'},
        inplace=True
    )
    expec_post_cum_preds_means['post_cum_preds_means'][len(pre_data) - 1] = 0
    pd.testing.assert_series_equal(
        expec_post_cum_preds_means['post_cum_preds_means'],
        inferences['post_cum_preds_means']
    )
    # test post_cum_preds_lower
    post_cum_preds_lower, post_cum_preds_upper = np.percentile(
        np.cumsum(maybe_unstandardize(np.squeeze(posterior_dist.sample(niter)), mu_sig),
                  axis=1),
        [100 * alpha / 2, 100 - 100 * alpha / 2],
        axis=0
    )
    post_cum_preds_lower = np.concatenate(
        [np.array([np.nan] * (len(pre_data) - 1) + [0]),
         post_cum_preds_lower]
    )
    expec_post_cum_preds_lower = pd.DataFrame(
        data=post_cum_preds_lower,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_preds_lower']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_preds_lower['post_cum_preds_lower'],
        inferences['post_cum_preds_lower']
    )
    # test post_cum_preds_upper
    post_cum_preds_upper = np.concatenate(
        [np.array([np.nan] * (len(pre_data) - 1) + [0]),
         post_cum_preds_upper]
    )
    expec_post_cum_preds_upper = pd.DataFrame(
        data=post_cum_preds_upper,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_preds_upper']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_preds_upper['post_cum_preds_upper'],
        inferences['post_cum_preds_upper']
    )
    # test point_effects_means
    net_data = pd.concat([pre_data, post_data])
    expec_point_effects_means = net_data.iloc[:, 0] - inferences['complete_preds_means']
    expec_point_effects_means = pd.DataFrame(
        data=expec_point_effects_means,
        index=expected_index,
        dtype=np.float64,
        columns=['point_effects_means']
    )
    pd.testing.assert_series_equal(
        expec_point_effects_means['point_effects_means'],
        inferences['point_effects_means']
    )
    # test point_effects_lower
    expec_point_effects_lower = net_data.iloc[:, 0] - inferences['complete_preds_upper']
    expec_point_effects_lower = pd.DataFrame(
        data=expec_point_effects_lower,
        index=expected_index,
        dtype=np.float64,
        columns=['point_effects_lower']
    )
    pd.testing.assert_series_equal(
        expec_point_effects_lower['point_effects_lower'],
        inferences['point_effects_lower']
    )
    # test point_effects_upper
    expec_point_effects_upper = net_data.iloc[:, 0] - inferences['complete_preds_lower']
    expec_point_effects_upper = pd.DataFrame(
        data=expec_point_effects_upper,
        index=expected_index,
        dtype=np.float64,
        columns=['point_effects_upper']
    )
    pd.testing.assert_series_equal(
        expec_point_effects_upper['point_effects_upper'],
        inferences['point_effects_upper']
    )
    # test post_cum_effects_means
    post_effects_means = post_data.iloc[:, 0] - inferences['post_preds_means']
    post_effects_means.iloc[len(pre_data) - 1] = 0
    expec_post_cum_effects_means = np.cumsum(post_effects_means)
    expec_post_cum_effects_means = pd.DataFrame(
        data=expec_post_cum_effects_means,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_effects_means']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_effects_means['post_cum_effects_means'],
        inferences['post_cum_effects_means']
    )
    # test post_cum_effects_lower
    post_cum_effects_lower, post_cum_effects_upper = np.percentile(
        np.cumsum(post_data.iloc[:, 0].values - maybe_unstandardize(np.squeeze(
            posterior_dist.sample(niter)), mu_sig),
                  axis=1),
        [100 * alpha / 2, 100 - 100 * alpha / 2],
        axis=0
    )
    post_cum_effects_lower = np.concatenate(
        [np.array([np.nan] * (len(pre_data) - 1) + [0]),
         post_cum_effects_lower]
    )
    expec_post_cum_effects_lower = pd.DataFrame(
        data=post_cum_effects_lower,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_effects_lower']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_effects_lower['post_cum_effects_lower'],
        inferences['post_cum_effects_lower']
    )
    # test post_cum_effects_upper
    post_cum_effects_upper = np.concatenate(
        [np.array([np.nan] * (len(pre_data) - 1) + [0]),
         post_cum_effects_upper]
    )
    expec_post_cum_effects_upper = pd.DataFrame(
        data=post_cum_effects_upper,
        index=expected_index,
        dtype=np.float64,
        columns=['post_cum_effects_upper']
    )
    pd.testing.assert_series_equal(
        expec_post_cum_effects_upper['post_cum_effects_upper'],
        inferences['post_cum_effects_upper']
    )


def test_summarize_posterior_inference():
    post_preds_means = pd.Series([1, 2, 3])
    post_data = pd.DataFrame([1.1, 2.2, 3.3])
    simulated_ys = (
        np.tile(np.arange(start=7, stop=10, step=1), (10, 1)) +
        np.arange(10).reshape(-1, 1)
    )
    alpha = 0.05
    lower_percen, upper_percen = [100 * alpha / 2, 100 - 100 * alpha / 2]
    pred_lower, pred_upper = np.percentile(simulated_ys.mean(axis=1),
                                           [lower_percen, upper_percen])
    pred_cum_lower, pred_cum_upper = np.percentile(simulated_ys.sum(axis=1),
                                                   [lower_percen, upper_percen])

    summary = inferrer.summarize_posterior_inferences(post_preds_means, post_data,
                                                      simulated_ys, alpha)

    np.testing.assert_almost_equal(
        summary['average']['actual'],
        post_data.mean().values
    )
    np.testing.assert_almost_equal(
        summary['average']['predicted'],
        post_preds_means.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['predicted_lower'],
        pred_lower
    )
    np.testing.assert_almost_equal(
        summary['average']['predicted_upper'],
        pred_upper
    )
    np.testing.assert_almost_equal(
        summary['average']['abs_effect'],
        post_data.mean().values - post_preds_means.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['abs_effect_lower'],
        post_data.mean().values - pred_upper.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['abs_effect_upper'],
        post_data.mean().values - pred_lower.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['rel_effect'],
        (post_data.mean().values - post_preds_means.mean()) / post_preds_means.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['rel_effect_lower'],
        (post_data.mean().values - pred_upper.mean()) / post_preds_means.mean()
    )
    np.testing.assert_almost_equal(
        summary['average']['rel_effect_upper'],
        (post_data.mean().values - pred_lower.mean()) / post_preds_means.mean()
    )

    np.testing.assert_almost_equal(
        summary['cumulative']['actual'],
        post_data.values.sum()
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['predicted'],
        post_preds_means.sum()
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['predicted_lower'],
        pred_cum_lower
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['predicted_upper'],
        pred_cum_upper
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['abs_effect'],
        post_data.sum().values - post_preds_means.sum()
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['abs_effect_lower'],
        post_data.sum().values - pred_cum_upper
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['abs_effect_upper'],
        post_data.sum().values - pred_cum_lower
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['rel_effect'],
        (post_data.sum().values - post_preds_means.sum()) / post_preds_means.sum()
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['rel_effect_lower'],
        (post_data.sum().values - pred_cum_upper) / post_preds_means.sum()
    )
    np.testing.assert_almost_equal(
        summary['cumulative']['rel_effect_upper'],
        (post_data.sum().values - pred_cum_lower) / post_preds_means.sum()
    )


def test_compute_p_value():
    simulated_ys = (
        np.tile(np.arange(start=7, stop=10, step=1), (10, 1)) +
        np.arange(10).reshape(-1, 1)
    )
    post_data_sum = 46
    p_value = inferrer.compute_p_value(simulated_ys, post_data_sum)
    assert p_value == 2 / 11
