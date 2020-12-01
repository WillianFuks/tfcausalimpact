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
Module responsible for all functions related to processing model certification and
creation.
"""


import pandas as pd
import numpy as np

from typing import Dict, Any, Union

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

# K Local Level Prior Sample Size
# This is equal to the original [R package](https://github.com/google/CausalImpact/blob/07b60e1bf5c9c8d74e31ea602db39d7256a53b6f/R/impact_model.R#L25) # noqa: E501
kLocalLevelPriorSampleSize = 32


def process_model_args(model_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process general parameters related to how Causal Impact will be implemented, such
    as standardization procedure or the addition of seasonal components to the model.

    Args
    ----
      model_args:
        standardize: bool
            If `True`, standardize data so result has zero mean and unitary standard
            deviation.
        prior_level_sd: float
            Standard deviation that sets initial local level distribution. Default
            value is 0.01 which means the linear regression is expected to explain
            well the observed data. In cases where this is not expected, then it's also
            possible to use the value 0.1. Still, this value will increase considerably
            the extension of the random walk variance modeling data which can lead to
            unreliable predictions (this might indicate that better covariates are
            required).
        fit_method: str
            Which method to use when fitting the structural time series model. Can be
            either `hmc` which stands for "Hamiltonian Monte Carlo" or "vi", i.e.,
            "variational inference". The first is slower but more accurate whilst the
            latter is the opposite. Defaults to `hmc` which prioritizes accuracy.
        niter: int
            How many iterations to run either for `hmc` or `vi` algorithms.
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

    Returns
    -------
        Dict[str, Any]
          standardize: bool
          prior_level_sd: float
          niter: int
          fit_method: str
          nseasons: int
          season_duration: int

    Raises
    ------
      ValueError: if standardize is not of type `bool`.
                  if nseasons doesn't follow the pattern [{str key: number}].
    """
    standardize = model_args.get('standardize', True)
    if not isinstance(standardize, bool):
        raise ValueError('standardize argument must be of type bool.')
    model_args['standardize'] = standardize

    prior_level_sd = model_args.get('prior_level_sd', 0.01)
    if not isinstance(prior_level_sd, float):
        raise ValueError('prior_level_sd argument must be of type float.')
    model_args['prior_level_sd'] = prior_level_sd

    niter = model_args.get('niter', 100)
    if not isinstance(niter, int):
        raise ValueError('niter argument must be of type int.')
    model_args['niter'] = niter

    fit_method = model_args.get('fit_method', 'hmc')
    if fit_method not in {'hmc', 'vi'}:
        raise ValueError('fit_method can be either "hmc" or "vi".')
    model_args['fit_method'] = fit_method

    nseasons = model_args.get('nseasons', 1)
    if not isinstance(nseasons, int):
        raise ValueError('nseasons argument must be of type int.')
    model_args['nseasons'] = nseasons

    season_duration = model_args.get('season_duration', 1)
    if not isinstance(season_duration, int):
        raise ValueError('season_duration argument must be of type int.')
    if nseasons <= 1 and season_duration > 1:
        raise ValueError('nseasons must be bigger than 1 when season_duration is also '
                         'bigger than 1.')
    model_args['season_duration'] = season_duration

    return model_args


def check_input_model(
    model: tfp.sts.StructuralTimeSeries,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame
) -> None:
    """
    Checkes whether input model was properly built and is ready to be run. This function
    is only invoked if the client sent a customized input model. Various assertions are
    performed to guarantee it has been created appropriately, such as each component
    should have `len(pre_data)` points for the argument `observed_time_series`. In case
    the component is of type `tfp.sts.LinearRegression` or `SparseLinearRegression` then
    the design matrix must have
    `shape = (len(pre_data) + len(post_data), cols(pre_data) - 1)` which allows not only
    to fit the model as well as to run the forecasts.

    Args
    ----
      model: StructuralTimeSeries
          Can be either default `LocalLevel` or user specified generic model.
      pre_data: pd.DataFrame

    Raises
    ------
      ValueError: if model is not of appropriate type.
                  if model is built without appropriate observed time series data.
    """
    def _check_linear_component(component):
        if isinstance(
            component,
            (tfp.sts.LinearRegression, tfp.sts.SparseLinearRegression)
        ):
            covariates_shape = (len(pre_data) + len(post_data),
                                len(pre_data.columns) - 1)
            if component.design_matrix.shape != covariates_shape:
                raise ValueError(
                    'Customized Linear Regression Models must have total '
                    'points equal to pre_data and post_data points and '
                    'same number of covariates. Input design_matrix shape was '
                    f'{component.design_matrix.shape} and expected '
                    f'{(len(pre_data) + len(post_data), len(pre_data.columns) -1)} '
                    'instead.'
                )
    if not isinstance(model, tfp.sts.StructuralTimeSeries):
        raise ValueError('Input model must be of type StructuralTimeSeries.')
    if isinstance(model, tfp.sts.Sum):
        for component in model.components:
            _check_linear_component(component)
    else:
        _check_linear_component(model)


def build_default_model(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    prior_level_sd: float
) -> tfp.sts.StructuralTimeSeries:
    """
    When input model is `None` then proceeds to build a default `tfp.sts.LocalLevel`
    model. If input data has covariates then also adds a `tfp.sts.LinearRegression`
    component.

    The level_prior follows `1 / prior_level_sd ** 2 ~ Gamma(a, b)` according to
    the original [BOOM](https://github.com/steve-the-bayesian/BOOM/blob/63f08a708153c8405b809405fa1ab5ed7193d648/Interfaces/python/R/R/bayes.py#L4:L12) package.  # noqa: E501
    This is achieved by using the InverseGamma(a, b) and a [bijector](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/) # noqa: E501
    transformation for the square root operator.

    Args
    ----
      pre_data: pd.DataFrame
      post_data: pd.DataFrame
      prior_level_sd: float
          Sets an initial estimation for the standard deviation 'sigma' of the local
          level prior. The bigger this value is, the wider is expected to be the random
          walk extension on forecasts.

    Returns
    -------
      model: tfp.sts.StructuralTimeSeries
          A `tfp.sts.LocalLevel` default model with possible another
          `tfp.sts.LinearRegression` component representing the covariates.
    """
    sample_size = kLocalLevelPriorSampleSize
    df = sample_size
    a = df / 2
    ss = sample_size * prior_level_sd ** 2
    b = ss / 2

    variance_prior = tfd.InverseGamma(a, b)

    # model = tfp.sts.Sum([tfp.sts.LocalLevel(observed_time_series=pre_data)])
    pass


class SquareRootTransform(tfb.Bijector):
    """
    Compute `Y = g(X) = X ** (1 / 2) which transforms variance into standard deviation.
    Main reference for building this bijector is the original [PowerTransform](https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/bijectors/power_transform.py) # noqa: E501
    """
    def __init__(
        self,
        validate_args: bool = False,
        parameters: Dict[str, Any] = None,
        name: str = 'square_root_transform'
    ):
        """
        Args
        ----
          validate_args: bool
              Indicates whether arguments should be checked for correctness.
          parameters: Dict[str, Any]
              Locals dict captured by subclass constructor, to be used for copy/slice
              re-instantiation operators.
          name: str
              Name given to ops managed by this object.
        """
        parameters = dict(locals()) if parameters is None else parameters
        with tf.name_scope(name) as name:
            super().__init__(forward_min_event_ndims=0, validate_args=validate_args,
                             parameters=parameters, name=name)

    def _forward(self, x: Union[float, np.array, tf.Tensor]) -> tf.Tensor:
        """
        Implements the forward pass `G` as given by `Y = G(X)`. In this case, it's a
        simple square root of X.

        Args
        ----
          x: Union[float, np.array, tf.Tensor])
              Variable `X` to receive the transformation.

        Returns
        -------
          X: tf.Tensor
              Square root of `x`.
        """
        return tf.sqrt(x)

    def _inverse(self, y: Union[float, np.array, tf.Tensor]) -> tf.Tensor:
        """
        Implements G^-1(y).

        Args
        ----
          y: Union[float, np.array, tf.Tensor]
              Values to be transformed back. In this case, they will be squared.

        Returns
        -------
          y: tf.Tensor
              Squared `y`.
        """
        return tf.square(y)
