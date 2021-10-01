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


from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
        niter: int
            How many iterations to run either for `hmc` or `vi` algorithms.
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
            latter is the opposite. Defaults to `vi` which prioritizes performance.
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
      ValueError: if `standardize` is not of type `bool`.
                  if `prior_level_sd` is not `float`.
                  if `niter` is not `int`.
                  if `fit_method` not in {'hmc', 'vi'}.
                  if `nseasons` is not `int`.
                  if `season_duration` is not `int`.
                  if `season_duration` is bigger than 1 and `nseasons` is 1.
    """
    standardize = model_args.get('standardize', True)
    if not isinstance(standardize, bool):
        raise ValueError('standardize argument must be of type bool.')
    model_args['standardize'] = standardize

    prior_level_sd = model_args.get('prior_level_sd', 0.01)
    if not isinstance(prior_level_sd, float):
        raise ValueError('prior_level_sd argument must be of type float.')
    model_args['prior_level_sd'] = prior_level_sd

    niter = model_args.get('niter', 1000)
    if not isinstance(niter, int):
        raise ValueError('niter argument must be of type int.')
    model_args['niter'] = niter

    fit_method = model_args.get('fit_method', 'vi')
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
    The model must be built with data of dtype=tf.float32 or np.float32 as otherwise an
    error will be thrown when fitting the markov chains.

    Args
    ----
      model: StructuralTimeSeries
          Can be either default `LocalLevel` or user specified generic model.
      pre_data: pd.DataFrame

    Raises
    ------
      ValueError: if model is not of appropriate type.
                  if model is built without appropriate observed time series data.
                  if model components don't have dtype=tf.float32 or np.float32
    """
    def _check_component(component):
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
            assert component.design_matrix.dtype == tf.float32
        else:
            for parameter in component.parameters:
                assert parameter.prior.dtype == tf.float32

    if not isinstance(model, tfp.sts.StructuralTimeSeries):
        raise ValueError('Input model must be of type StructuralTimeSeries.')
    if isinstance(model, tfp.sts.Sum):
        for component in model.components:
            _check_component(component)
    else:
        _check_component(model)


def build_inv_gamma_sd_prior(sigma_guess: float) -> tfd.Distribution:
    """
    helper function to build the sd_prior distribution for standard deviation
    modeling.

    Args
    ----
      sigma_guess: float
          Initial guess of where the standard deviation of the parameter is located.

    Returns
    -------
      tfd.Distribution
          InverseGamma distribution modeling the standard deviation.
    """
    sample_size = kLocalLevelPriorSampleSize
    df = sample_size
    a = np.float32(df / 2)
    ss = sample_size * sigma_guess ** 2
    b = np.float32(ss / 2)
    return tfd.InverseGamma(a, b)


def build_bijector(dist: tfd.Distribution) -> tfd.Distribution:
    """
    helper function for building final bijector given sd_prior. The bijector is
    implemented through the `tfd.TransformedDistribution` class.

    Args
    ----
      dist: tfd.Distribution
          Distribution to receive the transformation `G(X)`.

    Returns
    -------
      new_dist: tfd.Distribution
          New distribution given by `y = G(X)`.
    """
    bijector = SquareRootBijector()
    new_dist = tfd.TransformedDistribution(dist, bijector)
    return new_dist


def build_default_model(
    observed_time_series: pd.DataFrame,
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame,
    prior_level_sd: float,
    nseasons: int,
    season_duration: int
) -> tfp.sts.StructuralTimeSeries:
    """
    When input model is `None` then proceeds to build a default `tfp.sts.LocalLevel`
    model. If input data has covariates then also adds a `tfp.sts.SparseLinearRegression`
    component.

    The level_prior follows `1 / prior_level_sd ** 2 ~ Gamma(a, b)` according to
    the original [BOOM](https://github.com/steve-the-bayesian/BOOM/blob/63f08a708153c8405b809405fa1ab5ed7193d648/Interfaces/python/R/R/bayes.py#L4:L12) package.  # noqa: E501
    This is achieved by using the InverseGamma(a, b) and a [bijector](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/) # noqa: E501
    transformation for the square root operator.

    As for the linear regressor, the `tfp.sts.SparseLinearRegression` operation is similar
    to the spike-and-slab from the original R package; main difference is that it follows
    instead a horseshoe distribution which tends to penalize less the meaningful weights
    in the shrinking process.[https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/sts/regression.py#L265-L523] # noaq: E501

    Args
    ----
      observed_time_series: pd.DataFrame
      pre_data: pd.DataFrame
      post_data: pd.DataFrame
      prior_level_sd: float
          Sets an initial estimation for the standard deviation 'sigma' of the local
          level prior. The bigger this value is, the wider is expected to be the random
          walk extension on forecasts.
      nseasons: int
      season_duration: int

    Returns
    -------
      model: tfp.sts.Sum
          A `tfp.sts.LocalLevel` default model with possible another
          `tfp.sts.SparseLinearRegression` and `tfp.sts.Seasonal` components representing
          covariates and seasonal patterns.
    """
    components = []
    # use `values` to avoid batching dims
    obs_sd = observed_time_series.std(skipna=True, ddof=0).values[0]
    sd_prior = build_inv_gamma_sd_prior(prior_level_sd)
    sd_prior = build_bijector(sd_prior)
    # This is an approximation to simulate the bsts package from R. It's expected that
    # given a few data points the posterior will converge appropriately given this
    # distribution, that's why it's divided by 2.
    obs_prior = build_inv_gamma_sd_prior(obs_sd / 2)
    obs_prior = build_bijector(obs_prior)
    level_component = tfp.sts.LocalLevel(
        level_scale_prior=sd_prior,
        observed_time_series=observed_time_series
    )
    components.append(level_component)
    # If it has more than 1 column then it has covariates X so add a linear regressor
    # component.
    if len(pre_data.columns) > 1:
        # We need to concatenate both pre and post data as this will allow the linear
        # regressor component to use the post data when running forecasts. As first
        # column is supposed to have response variable `y` then we filter out just the
        # remaining columns for the `X` value.
        complete_data = pd.concat([pre_data, post_data]).astype(np.float32)
        # Set NaN values to zero so to not break TFP linear regression
        complete_data.fillna(0, inplace=True)
        linear_component = tfp.sts.SparseLinearRegression(
            design_matrix=complete_data.iloc[:, 1:]
        )
        components.append(linear_component)
    if nseasons > 1:
        seasonal_component = tfp.sts.Seasonal(
            num_seasons=nseasons,
            num_steps_per_season=season_duration,
            observed_time_series=observed_time_series
        )
        components.append(seasonal_component)
    # Model must be built with `tfp.sts.Sum` so to add the observed noise `epsilon`
    # parameter.
    model = tfp.sts.Sum(components, observed_time_series=observed_time_series,
                        observation_noise_scale_prior=obs_prior)
    return model


def fit_model(
    model: tfp.sts.StructuralTimeSeries,
    observed_time_series: pd.DataFrame,
    method: str = 'hmc'
) -> Tuple[Union[List[tf.Tensor], Dict[str, tf.Tensor]], Optional[Dict[str, Any]]]:
    """
    Run the Markovian Monte Carlo fitting process for finding the posterior `P(z | y)`
    where z represents the structural components of the input state space model. Two
    main methods can be used, either `hmc` which stands for 'Hamiltonian Monte Carlo'
    and `vi` standing for 'Variational Inference'. The first method is expected to be
    more accurate while less performante whereas the second is the opposite, that is,
    faster but less accurate.

    Args
    ----
      model: tfp.sts.StructuralTimeSeries
          Structural time series model built to explain the observed data. It may
          contain several components such as local level, seasons and so on.
      observed_time_series: pd.DataFrame
          Contains the pre-period response variable `y`.
      method: str
          Either 'hmc' or 'vi' which selects which fitting process to run.

    Returns
    -------
      (samples, kernel_results): Tuple[Union[List[tf.Tensor], Dict[str, tf.Tensor]],
                                       Dict[str, Any]]

    Raises
    ------
      ValueError: If input method is invalid.
    """
    if method == 'hmc':
        # this method does not need to be wrapped in a `tf.function` context as the
        # internal sampling method already is:
        # https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/sts/fitting.py#L422 # noqa: E501
        # https://github.com/tensorflow/probability/issues/348
        samples, kernel_results = tfp.sts.fit_with_hmc(
            model=model,
            observed_time_series=observed_time_series,
        )
        return samples, kernel_results
    elif method == 'vi':
        optimizer = tf.optimizers.Adam(learning_rate=0.1)
        variational_steps = 200  # Hardcoded for now
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

        @tf.function()
        def _run_vi():  # pragma: no cover
            tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=model.joint_log_prob(
                    observed_time_series=observed_time_series
                ),
                surrogate_posterior=variational_posteriors,
                optimizer=optimizer,
                num_steps=variational_steps
            )
            # Don't sample too much as varitional inference method is built aiming for
            # performance first.
            samples = variational_posteriors.sample(100)
            return samples, None
        return _run_vi()
    else:
        raise ValueError(
            f'Input method "{method}" not valid. Choose between "hmc" or "vi".'
        )


def build_one_step_dist(
    model: tfp.sts.StructuralTimeSeries,
    observed_time_series: pd.DataFrame,
    parameter_samples: Union[List[tfd.Distribution], Dict[str, tfd.Distribution]]
) -> tfd.Distribution:  # pragma: no cover
    """
    Builds one step distribution for pre-intervention data given samples from the
    posterior `P(z | y)`.

    Args
    ----
      model: tfp.StructuralTimeSeries
      observed_time_series: pd.DataFrame
          Corresponds to the `y` value.
      parameter_samples: Union[List[tfd.Distribution], Dict[str, tfd.Distribution]]
          samples from the posterior for each state component in `model`.

    Returns
    -------
      one_step_dist: tfd.Distribution
    """
    return tfp.sts.one_step_predictive(
        model=model,
        observed_time_series=observed_time_series,
        parameter_samples=parameter_samples
    )


def build_posterior_dist(
    model: tfp.sts.StructuralTimeSeries,
    observed_time_series: pd.DataFrame,
    parameter_samples: Union[List[tfd.Distribution], Dict[str, tfd.Distribution]],
    num_steps_forecast: int
) -> tfd.Distribution:  # pragma: no cover
    """
    Builds the distribution for post-intervention data given samples from the
    posterior `P(z | y)`.

    Args
    ----
      model: tfp.StructuralTimeSeries
      observed_time_series: pd.DataFrame
          Corresponds to the `y` value.
      parameter_samples: Union[List[tfd.Distribution], Dict[str, tfd.Distribution]]
          samples from the posterior for each state component in `model`.
      num_steps_forecast: int
          How many time steps to forecast into the future. These will be compared against
          the real value of `y` to extract the estimation of impact.

    Returns
    -------
      posterior_dist: tfd.Distribution
    """
    return tfp.sts.forecast(
        model=model,
        observed_time_series=observed_time_series,
        parameter_samples=parameter_samples,
        num_steps_forecast=num_steps_forecast
    )


class SquareRootBijector(tfb.Bijector):
    """
    Compute `Y = g(X) = X ** (1 / 2) which transforms variance into standard deviation.
    Main reference for building this bijector is the original [PowerTransform](https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/bijectors/power_transform.py) # noqa: E501
    """
    def __init__(
        self,
        validate_args: bool = False,
        name: str = 'square_root_bijector'
    ):
        """
        Args
        ----
          validate_args: bool
              Indicates whether arguments should be checked for correctness.
          name: str
              Name given to ops managed by this object.
        """
        # Without these `parameters` the code won't be compatible with future versions
        # of tfp:
        # https://github.com/tensorflow/probability/issues/1202
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super().__init__(
                forward_min_event_ndims=0,
                validate_args=validate_args,
                parameters=parameters,
                name=name)

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

    def _inverse_log_det_jacobian(self, y: tf.Tensor) -> tf.Tensor:
        """
        When transforming from `P(X)` to `P(Y)` it's necessary to compute the log of the
        determinant of the Jacobian matrix for each correspondent function `G` which
        accounts for the volumetric transformations on each domain.

        The inverse log determinant is given by:

        `ln(|J(G^-1(Y)|) = ln(|J(Y ** 2)|) = ln(|2 * Y|) = ln(2 * Y)`

        Args
        ----
          y: tf.Tensor

        Returns
        -------
          tf.Tensor
        """
        return tf.math.log(2 * y)

    def _forward_log_det_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the volumetric change when moving forward from `P(X)` to `P(Y)`, given
        by:

        `ln(|J(G(X))|) = ln(|J(sqrt(X))|) = ln(|(1 / 2) * X ** (-1 / 2)|) =
                       = (-1 / 2) * ln(4.0 * X)

        Args
        ----
          x: tf.Tensor

        Returns
        -------
          tf.tensor
        """
        return -0.5 * tf.math.log(4.0 * x)
