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
Main class definition for running Causal Impact analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

import causalimpact.data as cidata
import causalimpact.inferences as inferrer
import causalimpact.model as cimodel
import causalimpact.plot as plotter
import causalimpact.summary as summarizer
from causalimpact.misc import maybe_unstandardize


class CausalImpact():
    """
    Main class used to run the Causal Impact algorithm implemented by Google as
    described in the offical
    [paper](https://google.github.io/CausalImpact/CausalImpact.html).

    The algorithm basically fits a structural state space model to observed data `y` and
    uses Bayesian inferencing to find the posterior P(z|y) where `z` represents for the
    chosen model parameters (such as level, trend, season, and so on).

    In this package, the fitting method can be either 'Hamitonian Monte Carlo' or 'hmc'
    for short (more accurate algorithm but slower) or 'Variational Inference' or 'vi'
    (faster but less accurate), both available on Tensorflow Probability.

    Args
    ----
      data: Union[np.array, pd.DataFrame]
          First column must contain the `y` value whose future values will be forecasted
          while the remaining data contains the covariates `X` that are used in the
          linear regression component of the model (supposing that there's a linear
          regression otherwise `X` is not specified).
          If `data` it's a pandas DataFrame, its index can be defined either as a
          `RangeIndex`, `Index` or `DateTimeIndex`.
          In case of the second, then a conversion to `DateTime` type is automatically
          performed; in case of failure, the original index is kept untouched.
      pre_period: Union[List[int], List[str], List[pd.Timestamp]]
          A list of size two containing either `int`, `str` or `pd.Timestamp` values
          that references the range from beginning to end to be used in the
          pre-intervention data.
          As an example, valid inputs are:
            - [0, 30]
            - ['20200101', '20200130']
            - [pd.to_datetime('20200101'), pd.to_datetime('20200130')]
            - [pd.Timestamp('20200101'), pd.Timestamp('20200130')]
          The latter can be used only if the input `data` is a pandas DataFrame whose
          index is based on datetime values.
      post_period: Union[List[int], List[str], List[pd.Timestamp]]
          The same as `pre_period` but references where the post-intervention
          data begins and ends. This is the data that will be compared against the
          counter-factual forecasts.
      model: Optional[tfp.sts.StructuralTimeSeries]
          If `None` then a default `tfp.sts.LocalLevel` model is internally built
          otherwise use the input `model` for fitting and forecasting.
      model_args: Dict[str, Any]
          Sets general variables for building and running the state space model. Possible
          values are:
            standardize: bool
                If `True`, standardizes data to have zero mean and unitary standard
                deviation.
            prior_level_sd: Optional[float]
                Prior value for the local level standard deviation. If `None` then an
                automatic optimization of the local level is performed. This is
                recommended when there's uncertainty about what prior value is
                appropriate for the data.
                In general, if the covariates are expected to be good descriptors of the
                observed response then this value can be low (such as the default of
                0.01). In cases when the linear regression is not quite expected to fully
                explain the observed data, the value 0.1 can be used.
            fit_method: str
                Which method to use for the Bayesian algorithm. Can be either 'vi'
                (default) or 'hmc' (more precision but much slower).
            nseasons: int
              Specifies the duration of the period of the seasonal component; if input
              data is specified in terms of days, then choosing nseasons=7 adds a weekly
              seasonal effect.
            season_duration: int
              Specifies how many data points each value in season spans over. A good
              example to understand this argument is to consider a hourly data as input.
              For modeling a weekly season on this data, one can specify `nseasons=7` and
              season_duration=24 which means each value that builds the season component
              is repeated for 24 data points. Default value is 1 which means the season
              component spans over just 1 point (this in practice doesn't change
              anything). If this value is specified and bigger than 1 then `nseasons`
              must be specified and bigger than 1 as well.
      alpha: float
          A float that ranges between 0 and 1 indicating the significance level that
          will be used when statistically testing for signal presencen in the post-
          intervention period.

    Returns
    -------
      Causal Impact object with inferences, summary and plotting functionalities.

    Examples
    --------

      Imput data can be a `numpy.array`:

      ```python
      import numpy as np


      data = np.random.rand(100, 2)
      pre_period = [0, 69]
      post_period = [70, 99]

      ci = CausalImpact(data, pre_period, post_period)
      print(ci.summary())
      print(ci.summary('report'))
      ci.plot()
      ```

      Using pandas DataFrames:

      ```python
      df = pd.DataFrame('tests/fixtures/arma_data.csv')
      pre_period = [0, 69]
      post_period = [70, 99]
      ci = CausalImpact(df, pre_period, post_period)
      print(ci.summary())
      ```

      Using pandas DataFrames with pandas timestamps:

      ```python
      df = pd.DataFrame('tests/fixtures/arma_data.csv')
      df = df.set_index(pd.date_range(start='20200101', periods=len(data)))
      pre_period = [pd.to_datetime('20200101'), pd.to_datetime('20200311')]
      post_period = [pd.to_datetime('20200312'), pd.to_datetime('20200410')]
      ci = CausalImpact(df, pre_period, post_period)
      print(ci.summary())
      ```

      Using a weekly seasonal component on daily data:

      ```python
      df = pd.DataFrame('tests/fixtures/arma_data.csv')
      df = df.set_index(pd.date_range(start='20200101', periods=len(data)))
      pre_period = ['20200101', '20200311']
      post_period = ['20200312', '20200410']
      ci = CausalImpact(df, pre_period, post_period, model_args={'nseasons': 7})
      print(ci.summary())
      ```

      Using a weekly seasonal component on hourly data:

      ```python
      df = pd.DataFrame('tests/fixtures/arma_data.csv')
      df = df.set_index(pd.date_range(start='20200101', periods=len(data), freq='H'))
      pre_period = ['20200101 00:00:00', '20200311 23:00:00']
      post_period = ['20200312 00:00:00', '20200410 23:00:00']
      ci = CausalImpact(df, pre_period, post_period, model_args={'nseasons': 7,
                        'season_duration': 24})
      print(ci.summary())
      ```

      Using a customized model:

      ```python
      import tensorflow_probability as tfp


      pre_y = data[:70, 0]
      pre_X = data[:70, 1:]
      obs_series = data.iloc[:, 0]
      local_linear = tfp.sts.LocalLinearTrend(observed_time_series=obs_series)
      seasonal = tfp.sts.Seasonal(nseasons=7, observed_time_series=obs_series)
      model = tfp.sts.Sum([local_linear, seasonal], observed_time_series=obs_series)

      ci = CausalImpact(data, pre_period, post_period, model=model)
      print(ci.summary())
      ```
    """
    def __init__(
        self,
        data: Union[np.array, pd.DataFrame],
        pre_period: Union[List[int], List[str], List[pd.Timestamp]],
        post_period: Union[List[int], List[str], List[pd.Timestamp]],
        model: Optional[tfp.sts.StructuralTimeSeries] = None,
        model_args: Dict[str, Any] = {},
        alpha: float = 0.05
    ):
        processed_input = cidata.process_input_data(data, pre_period, post_period,
                                                    model, model_args, alpha)
        self.data = data
        self.pre_period = processed_input['pre_period']
        self.post_period = processed_input['post_period']
        self.pre_data = processed_input['pre_data']
        self.post_data = processed_input['post_data']
        self.alpha = processed_input['alpha']
        self.model_args = processed_input['model_args']
        self.model = processed_input['model']
        self.normed_pre_data = processed_input['normed_pre_data']
        self.normed_post_data = processed_input['normed_post_data']
        self.observed_time_series = processed_input['observed_time_series']
        self.mu_sig = processed_input['mu_sig']
        self._fit_model()
        self._process_posterior_inferences()
        self._summarize_inferences()

    def plot(
        self,
        panels: List[str] = ['original', 'pointwise', 'cumulative'],
        figsize: Tuple[int] = (10, 7),
        show: bool = True
    ) -> None:
        """
        Plots the graphic with results associated to Causal Impact.

        Args
        ----
          panels: List[str]
              Which graphics to plot. 'original' plots the original data, forecasts means
              and credible intervals related to the fitted model.
              'pointwise' plots the point wise differences between observed data and
              predictions. Finally, 'cumulative' is a cumulative summation over real
              data and its forecasts.
          figsize: Tuple[int]
              Sets the width and height of the figure to plot.
          show: bool
            If `True` then plots the figure by running `plt.plot()`.
            If `False` then nothing will be plotted which allows for accessing and
            manipulating the figure and axis of the plot, i.e., the figure can be saved
            and the styling can be modified. To get the axis, just run:
            `import matplotlib.pyplot as plt; ax = plt.gca()` or the figure:
            `fig = plt.gcf()`. Defaults to `True`.
        """
        plotter.plot(self.inferences, self.pre_data, self.post_data, panels=panels,
                     figsize=figsize, show=show)

    def summary(self, output: str = 'summary', digits: int = 2) -> str:
        """
        Builds and prints the summary report.

        Args
        ----
          output: str
              Can be either "summary" or "report". The first is a simpler output just
              informing general metrics such as expected absolute or relative effect.
          digits: int
              Defines the number of digits after the decimal point to round. For
              `digits=2`, value 1.566 becomes 1.57.

        Returns
        -------
          summary: str
              Contains results of the causal impact analysis.

        Raises
        ------
          ValueError: If input `output` is not either 'summary' or 'report'.
                      If input `digits` is not of type integer.
        """
        if not isinstance(digits, int):
            raise ValueError(
                f'Input value for digits must be integer. Received "{type(digits)}" '
                'instead.'
            )
        result = summarizer.summary(self.summary_data, self.p_value, self.alpha,
                                    output, digits)
        return result

    def _fit_model(self) -> None:
        """
        Use observed data `Y` to find the posterior `P(Z|Y)` where `Z` represents the
        structural components that were used for building the model (such as local level
        factor or seasonal components).
        """
        model_samples, model_kernel_results = cimodel.fit_model(
            self.model,
            self.observed_time_series,
            self.model_args['fit_method'],
        )
        self.model_samples = model_samples
        self.model_kernel_results = model_kernel_results

    def _summarize_inferences(self) -> None:
        """
        After processing predictions and forecasts, use these values to build the
        summary data used for reporting and plotting. Computes the estimated p-value
        for determining if the impact is statistically significant or not.
        """
        post_preds_means = self.inferences['post_preds_means']
        post_data_sum = self.post_data.iloc[:, 0].sum()
        niter = self.model_args['niter']
        simulated_ys = maybe_unstandardize(
            np.squeeze(self.posterior_dist.sample(niter).numpy()),
            self.mu_sig
        )
        self.summary_data = inferrer.summarize_posterior_inferences(post_preds_means,
                                                                    self.post_data,
                                                                    simulated_ys,
                                                                    self.alpha)
        self.p_value = inferrer.compute_p_value(simulated_ys, post_data_sum)

    def _process_posterior_inferences(self) -> None:
        """
        Run `inferrer` to process data forecasts and predictions. Results feeds the
        summary table as well as the plotting functionalities.
        """
        num_steps_forecast = len(self.post_data)
        self.one_step_dist = cimodel.build_one_step_dist(self.model,
                                                         self.observed_time_series,
                                                         self.model_samples)
        self.posterior_dist = cimodel.build_posterior_dist(self.model,
                                                           self.observed_time_series,
                                                           self.model_samples,
                                                           num_steps_forecast)
        self.inferences = inferrer.compile_posterior_inferences(
            self.data.index,
            self.pre_data,
            self.post_data,
            self.one_step_dist,
            self.posterior_dist,
            self.mu_sig,
            self.alpha,
            self.model_args['niter']
        )
