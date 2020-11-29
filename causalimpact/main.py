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
Main class definition for running causal impact analysis.
"""

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import causalimpact.data as cidata

from causalimpact.inferences import Inferences
from typing import Union, List, Dict, Any, Optional
# from causalimpact.misc import standardize
from causalimpact.plot import Plot
from causalimpact.summary import Summary

# distributions module is available on __init__.py or in path
# `tensorflow_probability.python.distributions` that's why it's required to be brought
# from `tfp` otherwise Python can't see it.
tfd = tfp.distributions


class CausalImpact():
    """
    Main class used to run the Causal Impact algorithm implemented by Google as
    described in their offical
    [paper.](https://google.github.io/CausalImpact/CausalImpact.html)

    The algorithm basically fits a structural state space model to observed data `y` and
    uses Bayesian inferencing to find the posterior P(z|y) where `z` represents for the
    chosen model parameters (such as level, trend, season, etc...).

    In this package, the fitting method can be either 'Hamitonian Monte Carlo' (more
    accurate) or 'Variational Inference' (faster but less accurate), both available on
    Tensorflow Probability.

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
      model_args: Optional[Dict[str, Any]]
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
      CausalImpact object with inferences, summary and plotting functionalities.

    Examples:
    ---------
      >>> import numpy as np
      >>> from statsmodels.tsa.statespace.structural import UnobservedComponents
      >>> from statsmodels.tsa.arima_process import ArmaProcess

      >>> np.random.seed(12345)
      >>> ar = np.r_[1, 0.9]
      >>> ma = np.array([1])
      >>> arma_process = ArmaProcess(ar, ma)
      >>> X = 100 + arma_process.generate_sample(nsample=100)
      >>> y = 1.2 * X + np.random.normal(size=100)
      >>> data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
      >>> pre_period = [0, 69]
      >>> post_period = [70, 99]

      >>> ci = CausalImpact(data, pre_period, post_period)
      >>> ci.summary()
      >>> ci.summary('report')
      >>> ci.plot()

      Using pandas DataFrames:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period)

      Using pandas DataFrames with pandas timestamps:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = [pd.to_datetime('20180101'), pd.to_datetime('20180311')]
      >>> post_period = [pd.to_datetime('20180312'), pd.to_datetime('20180410')]
      >>> ci = CausalImpact(df, pre_period, post_period)

      Using automatic local level optimization:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period, prior_level_sd=None)

      Using seasonal components:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period, nseasons=[{'period': 7}])

      Using a customized model:

      >>> pre_y = data[:70, 0]
      >>> pre_X = data[:70, 1:]
      >>> ucm = UnobservedComponents(endog=pre_y, level='llevel', exog=pre_X)
      >>> ci = CausalImpact(data, pre_period, post_period, model=ucm)
    """
    def __init__(
        self,
        data: Union[np.array, pd.DataFrame],
        pre_period: Union[List[int], List[str], List[pd.Timestamp]],
        post_period: Union[List[int], List[str], List[pd.Timestamp]],
        model: Optional[tfp.sts.StructuralTimeSeries] = None,
        model_args: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05,
        **kwargs: Dict[str, Any]
    ):
        processed_input = cidata.process_input_data(data, pre_period, post_period,
                                                    model, model_args, alpha, **kwargs)
        self.inferrer = Inferences(n_sims=kwargs.get('n_sims', 1000))
        self.summarizer = Summary()
        self.plotter = Plot()
        self.data = data
        self.pre_period = processed_input['pre_period']
        self.post_period = processed_input['post_period']
        self.pre_data = processed_input['pre_data']
        self.post_data = processed_input['post_data']
        self.alpha = processed_input['alpha']
        self.model_args = processed_input['model_args']
        self.model = processed_input['model']
        self.normed_pre_data = None
        self.normed_post_data = None
        self.mu_sig = None
        self._fit_model()
        # self._process_posterior_inferences()

    @property
    def model_args(self):
        """
        Gets the general settings used to guide the creation of the Causal model.

        Returns
        -------
          dict:
            standardize: bool.
        """
        return self._model_args

    @model_args.setter
    def model_args(self, value):
        """
        Sets general settings for how to build the Causal model.

        Args
        ----
          value: dict
              standardize: bool.
              nseasons: list of dicts.
        """
        if value.get('standardize'):
            self._standardize_pre_post_data()
        self._model_args = value

    @property
    def model(self):
        """
        Gets UnobservedComponents model that will be used for computing the Causal
        Impact algorithm.
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Sets model object.

        Args
        ----
          value: `UnobservedComponents`.
        """
        if value is None:
            self._model = self._get_default_model()
        else:
            self._model = value

    def _fit_model(self) -> None:
        """
        Use observed data `y` to find the posterior `P(z | y)` where `z` represents the
        structural components that were used for building the model (such as local level
        factor or seasonal components).
        """
        fit_args = self._process_fit_args()
        self.trained_model = self.model.fit(**fit_args)

    def _standardize_pre_post_data(self):
        """
        Applies normal standardization in pre and post data, based on mean and std of
        pre-data (as it's used for training our model). Sets new values for
        `self.pre_data`, `self.post_data`, `self.mu_sig`.
        """
        self.normed_pre_data, (mu, sig) = standardize(self.pre_data)
        self.normed_post_data = (self.post_data - mu) / sig
        self.mu_sig = (mu[0], sig[0])

    def _process_posterior_inferences(self):
        """
        Uses the trained model to make predictions for the post-intervention (or test
        data) period by invoking the class `Inferences` to process the forecasts. All
        data related to predictions, point effects and cumulative responses will be
        processed here.
        """
        self._compile_posterior_inferences()
        self._summarize_posterior_inferences()

    def _get_default_model(self):
        """Constructs default local level unobserved states model using input data and
        `self.model_args`.

        Returns
        -------
          model: `UnobservedComponents` built using pre-intervention data as training
              data.
        """
        data = self.pre_data if self.normed_pre_data is None else self.normed_pre_data
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:] if data.shape[1] > 1 else None
        freq_seasonal = self.model_args.get('nseasons')
        # model = UnobservedComponents(endog=y, level='llevel', exog=X,
                                     # freq_seasonal=freq_seasonal)
        # return model


    def _process_fit_args(self):
        """
        Process the input that will be used in the fitting process for the model.

        Args
        ----
          self:
            model: 
                If `None` them it means the fitting process will work with default model.
                Process level information of customized model otherwise.
            model_args: dict.
                Input args for general options of the model. All keywords defined
                in `scipy.optimize.minimize` can be used here. For more details,
                please refer to:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

              disp: bool.
                  Whether to display the logging of the `statsmodels` fitting process or
                  not. Defaults to `False` which means not display any logging.

              prior_level_sd: float.
                  Prior value to be used as reference for the fitting process.

        Returns
        -------
          model_args: dict
              The arguments that will be used in the `fit` method.
        """
        fit_args = self.model_args.copy()
        fit_args.setdefault('disp', False)
        level_sd = fit_args.get('prior_level_sd', 0.01)
        n_params = len(self.model.param_names)
        level_idx = [idx for (idx, name) in enumerate(self.model.param_names) if
                     name == 'sigma2.level']
        bounds = [(None, None)] * n_params
        if level_idx:  # If chosen model do not have level defined then this is None.
            level_idx = level_idx[0]
            # We make the maximum relative variation be up to 20% in order to simulate
            # an approximate behavior of the respective algorithm implemented in R.
            bounds[level_idx] = (
                level_sd / 1.2 if level_sd is not None else None,
                level_sd * 1.2 if level_sd is not None else None
            )
        fit_args.setdefault('bounds', bounds)
        return fit_args




    def _format_input_data(self, data):
        """
        Validates and formats input data.

        Args
        ----
          data: Union[np.array, pd.DataFrame]

        Returns
        -------
          data: pd.DataFrame
              Validated data to be used in Causal Impact algorithm.

        Raises
        ------
          ValueError: if input `data` is non-convertible to pandas DataFrame.
                      if input `data` has non-numeric values.
                      if input `data` has less than 3 points.
                      if input covariates have NAN values.
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except ValueError:
                raise ValueError(
                    'Could not transform input data to pandas DataFrame.'
                )
        self._validate_y(data.iloc[:, 0])
        # Must contain only numeric values
        if not data.applymap(np.isreal).values.all():
            raise ValueError('Input data must contain only numeric values.')
        # Covariates cannot have NAN values
        if data.shape[1] > 1:
            if data.iloc[:, 1:].isna().values.any():
                raise ValueError('Input data cannot have NAN values.')
        # If index is a string of dates, try to convert it to datetimes which helps
        # in plotting.
        data = self._convert_index_to_datetime(data)
        return data
