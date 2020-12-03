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
Uses the posterior distribution to prepare inferences for the Causal Impact summary
and plotting functionalities.
"""


import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from typing import List, Tuple, Optional, Union

from causalimpact.misc import get_z_score, unstandardize


tfd = tfp.distributions


class Inferences:
    def get_lower_upper_percentiles(self, alpha: float) -> List[float]:
        """
        Returns the lower and upper quantile values for the chosen `alpha` value.

        Args
        ----
          alpha: float
             Sets the size of the confidence interval. If `alpha=0.05` then extracts the
             95% confidence interval for forecasts.

        Returns
        -------
          List[float]
              First value is the lower quantile and second value is upper.
        """
        return [alpha * 100. / 2., 100 - alpha * 100. / 2.]

    def maybe_unstardardize(
        self, data: pd.DataFrame,
        mu_sig: Tuple[float, float]
    ) -> pd.DataFrame:
        """
        If input data was standardized this method is used to bring back data to its
        original values. The parameter `mu_sig` from holds the values used for
        standardizing (average and std, respectively) the response variable `y`. In case
        `mu_sig` is `None`, it means no standardization was applied; in this case, we
        just return data itself.

        Args
        ----
          self:
            mu_sig: Tuple[float, float]
                First value is the mean and second is the standard deviation used for
                normalizing the response variable `y`.
          data: pd.DataFrame
              Input dataframe to apply unstardization.

        Returns
        -------
          pd.DataFrame
              returns original input `data` if `mu_sig` is None and the "unstardardized"
              data otherwise.
        """
        if mu_sig is None:
            return data
        return unstandardize(data, self.mu_sig)

    def compile_posterior_inferences(
        self,
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
          pre_data: pd.DataFrame
          post_data: pd.DataFrame
          one_step_dist: tfd.Distribution
              Uses posterior parameters to run one-step-prediction on past observed data.
          posterior_dist: tfd.Distribution
              Uses posterior parameters to run forecasts on post intervention data.
          mu_sig: Optional[Tuple[float, float]]
              First value is the mean used for standardization and second value is the
              standard deviation.
          alpha: float
              Sets confidence interval size.
          niter: int
              Total mcmc samples to sample from the posterior structural model.

        Returns
        -------
          pd.DataFrame
              Final dataframe with all data related to one-step predictions and forecasts
        """
        lower_percen, upper_percen = self.get_lower_upper_percentiles(alpha)
        z_score = get_z_score(1 - alpha / 2)
        # We create a pd.Series with a single 0 (zero) value to work as the initial value
        # when computing the cumulative inferences. Without this value the plotting of
        # cumulative data breaks at the initial point.
        zero_series = pd.Series([0])
        simulated_ys = posterior_dist.sample(niter)
        # pre inference
        pre_preds_means = one_step_dist.mean()
        pre_preds_stds = one_step_dist.stddev()
        pre_preds_lower = pd.Series(
            self.maybe_unstardardize(pre_preds_means - z_score * pre_preds_stds, mu_sig),
            index=pre_data.index
        )
        pre_preds_upper = pd.Series(
            self.maybe_unstardardize(pre_preds_means + z_score * pre_preds_stds, mu_sig),
            index=pre_data.index
        )
        pre_preds = pd.Series(
            self.maybe_unstardardize(pre_preds_means, mu_sig),
            index=pre_data.index
        )
        # post inference
        post_preds_means = posterior_dist.mean()
        post_preds_stds = posterior_dist.stddev()
        post_preds_lower = pd.Series(
            self.maybe_unstardardize(
                post_preds_means - z_score * post_preds_stds,
                mu_sig
            ),
            index=post_data.index
        )
        post_preds_upper = pd.Series(
            self.maybe_unstardardize(
                post_preds_means + z_score * post_preds_stds,
                mu_sig
            ),
            index=post_data.index
        )
        post_preds = pd.Series(
            self.maybe_unstardardize(post_preds_means, mu_sig),
            index=post_data.index
        )
        # concatenations
        complete_preds = pd.concat([pre_preds, post_preds])
        complete_preds_lower = pd.concat([pre_preds_lower, post_preds_lower])
        complete_preds_upper = pd.concat([pre_preds_upper, post_preds_upper])
        # cumulative
        post_cum_y = np.cumsum(post_data.iloc[:, 0])
        post_cum_y = pd.concat([zero_series, post_cum_y], axis=0)
        post_cum_y.index = self._build_cum_index(pre_data.index, post_data.index)
        post_cum_pred = np.cumsum(post_preds)
        post_cum_pred = pd.concat([zero_series, post_cum_pred])
        post_cum_pred.index = self._build_cum_index(pre_data.index, post_data.index)
        post_cum_pred_lower, post_cum_pred_upper = np.percentile(
            np.cumsum(simulated_ys, axis=1),
            [lower_percen, upper_percen],
            axis=0
        )
        # Sets index properly.
        post_cum_pred_lower = pd.Series(
            np.concatenate([[0], post_cum_pred_lower]),
            index=self._build_cum_index()
        )
        post_cum_pred_upper = pd.Series(
            np.concatenate([[0], post_cum_pred_upper]),
            index=self._build_cum_index()
        )
        # Using a net value of data to accomodate cases where there's gaps between
        # pre and post intervention periods.
        net_data = pd.concat([self.pre_data, self.post_data])

        # Effects analysis.
        point_effects = net_data.iloc[:, 0] - preds
        point_effects_lower = net_data.iloc[:, 0] - preds_upper
        point_effects_upper = net_data.iloc[:, 0] - preds_lower
        post_point_effects = self.post_data.iloc[:, 0] - post_preds

        # Cumulative Effects analysis.
        post_cum_effects = np.cumsum(post_point_effects)
        post_cum_effects = pd.concat([zero_series, post_cum_effects])
        post_cum_effects.index = self._build_cum_index()
        post_cum_effects_lower, post_cum_effects_upper = np.percentile(
            np.cumsum(self.post_data.iloc[:, 0].values - self.simulated_y, axis=1),
            [lower, upper],
            axis=0
        )

        # Sets index properly.
        post_cum_effects_lower = pd.Series(
            np.concatenate([[0], post_cum_effects_lower]),
            index=self._build_cum_index()
        )
        post_cum_effects_upper = pd.Series(
            np.concatenate([[0], post_cum_effects_upper]),
            index=self._build_cum_index()
        )

        self.inferences = pd.concat(
            [
                post_cum_y,
                preds,
                post_preds,
                post_preds_lower,
                post_preds_upper,
                preds_lower,
                preds_upper,
                post_cum_pred,
                post_cum_pred_lower,
                post_cum_pred_upper,
                point_effects,
                point_effects_lower,
                point_effects_upper,
                post_cum_effects,
                post_cum_effects_lower,
                post_cum_effects_upper
            ],
            axis=1
        )

        self.inferences.columns = [
            'post_cum_y',
            'preds',
            'post_preds',
            'post_preds_lower',
            'post_preds_upper',
            'preds_lower',
            'preds_upper',
            'post_cum_pred',
            'post_cum_pred_lower',
            'post_cum_pred_upper',
            'point_effects',
            'point_effects_lower',
            'point_effects_upper',
            'post_cum_effects',
            'post_cum_effects_lower',
            'post_cum_effects_upper'
        ]

    def _build_cum_index(
        self,
        pre_data_index: Union[pd.core.indexes.range, pd.core.indexes.datetimes],
        post_data_index: Union[pd.core.indexes.range, pd.core.indexes.datetimes]
    ) -> Union[pd.core.indexes.range, pd.core.indexes.datetimes]:
        """
        As the cumulative data has one more data point (the first point is a zero),
        we add to the post-intervention data the first index of the pre-data right at the
        beginning of the index. This helps in the plotting functionality.

        Args
        ----
          pre_data_index: Union[pd.core.indexes.range, pd.core.indexes.datetimes]
          post_data_index: Union[pd.core.indexes.range, pd.core.indexes.datetimes]

        Returns
        -------
          Union[pd.core.indexes.range, pd.core.indexes.datetimes]
              `post_data_index` extended with the latest index value from `pre_data`.
        """
        # In newer versions of Numpy/Pandas, the union operation between indices returns
        # an Index with `dtype=object`. We, therefore, create this variable in order to
        # restore the original value which is used later on by the plotting interface.
        index_dtype = post_data_index.dtype
        new_idx = post_data_index.union([pre_data_index[-1]])
        new_idx = new_idx.astype(index_dtype)
        return new_idx

    def _summarize_posterior_inferences(self):
        """
        After running the posterior inferences compilation, this method aggregates
        the results and gets the final interpretation for the causal impact results, such
        as what is the expected absolute impact of the given intervention.
        """
        lower_percentile, upper_percentile = self.lower_upper_percentile(alpha)
        infers = self.inferences

        # Compute the mean of metrics.
        mean_post_y = self.post_data.iloc[:, 0].mean()
        mean_post_pred = infers['post_preds'].mean()
        mean_post_pred_lower, mean_post_pred_upper = np.percentile(
            self.simulated_y.mean(axis=1), [lower, upper])

        # Compute the sum of metrics.
        sum_post_y = self.post_data.iloc[:, 0].sum()
        sum_post_pred = infers['post_preds'].sum()
        sum_post_pred_lower, sum_post_pred_upper = np.percentile(
            self.simulated_y.sum(axis=1), [lower, upper])

        # Causal Impact analysis metrics.
        abs_effect = mean_post_y - mean_post_pred
        abs_effect_lower = mean_post_y - mean_post_pred_upper
        abs_effect_upper = mean_post_y - mean_post_pred_lower

        sum_abs_effect = sum_post_y - sum_post_pred
        sum_abs_effect_lower = sum_post_y - sum_post_pred_upper
        sum_abs_effect_upper = sum_post_y - sum_post_pred_lower

        rel_effect = abs_effect / mean_post_pred
        rel_effect_lower = abs_effect_lower / mean_post_pred
        rel_effect_upper = abs_effect_upper / mean_post_pred

        sum_rel_effect = sum_abs_effect / sum_post_pred
        sum_rel_effect_lower = sum_abs_effect_lower / sum_post_pred
        sum_rel_effect_upper = sum_abs_effect_upper / sum_post_pred

        # Prepares all this data into a DataFrame for later retrieval, such as when
        # running the `summary` method.
        summary_data = [
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

        self.summary_data = pd.DataFrame(
            summary_data,
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
        # We also save the p-value which will be used in `summary` as well.
        self.p_value = self._compute_p_value()

    def _compute_p_value(self, n_sims=1000):
        """
        Computes the p-value for the hypothesis testing that there's signal in the
        observed data. The computation follows the same idea as the one implemented in R
        by Google which consists of simulating with the fitted parameters several time
        series for the post-intervention period and counting how many either surpass the
        total summation of `y` (in case there's positive relative effect) or how many
        falls under its summation (in which case there's negative relative effect).

        For a better understanding of how this solution was obtained, this discussion was
        used as the main guide:

        https://stackoverflow.com/questions/51881148/simulating-time-series-with-unobserved-components-model/

        Args
        ----
          n_sims: int.
              Representing how many simulations to run for computing the p-value.

        Returns
        -------
          p_value: float.
              Ranging between 0 and 1, represents the likelihood of obtaining the observed
              data by random chance.
        """
        y_post_sum = self.post_data.iloc[:, 0].sum()
        sim_sum = self.simulated_y.sum(axis=1)
        # The minimum value between positive and negative signals reveals how many times
        # either the summation of the simulation could surpass ``y_post_sum`` or be
        # surpassed by the same (in which case it means the sum of the simulated time
        # series is bigger than ``y_post_sum`` most of the time, meaning the signal in
        # this case reveals the impact caused the response variable to decrease from what
        # was expected had no effect taken place.
        signal = min(np.sum(sim_sum > y_post_sum), np.sum(sim_sum < y_post_sum))
        return signal / (self.n_sims + 1)
