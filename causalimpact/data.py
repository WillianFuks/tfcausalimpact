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
Module responsible for processing data that feeds the causal impact algorithm.
"""


from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

import causalimpact.model as cimodel
from causalimpact.misc import standardize


def process_input_data(
    data: Union[np.array, pd.DataFrame],
    pre_period: Union[List[int], List[str], List[pd.Timestamp]],
    post_period: Union[List[int], List[str], List[pd.Timestamp]],
    model: Optional[tfp.sts.StructuralTimeSeries],
    model_args: Dict[str, Any],
    alpha: float
) -> Dict[str, Any]:
    """
    Checks and formats input data for running the Causal Impact algorithm.

    Args
    ----
      data: Union[np.array, pd.DataFrame]
          First column is the response variable `y` and other columns correspond to
          the covariates `X`.
      pre_period: Union[List[int], List[str], List[pd.Timestamp]]
          List with initial and final points to consider for pre-intervention period.
      post_period: Union[List[int], List[str], List[pd.Timestamp]]
          List with initial and final points to consider for post-intervention period.
      model: Optional[tfd.LinearGaussianStateSpaceModel]
          If `None` then uses default `tfp.sts.LocalLevel` model otherwise it should
          be a `tfd.LinearGaussianStateSpaceModel` specified by client.
      model_args: Dict[str, Any]
          Sets general variables for building and running the state space model. Possible
          values are:
            niter: int
                Number of samples to draw from posterior `P(z | y)`.
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
          Used for two-sided statistical test on comparison between counter-factual
          and observed series.

    Returns
    -------
      Dict[str, Any]:
        data: pd.DataFrame
            Validated data, first column is `y` and the others are the `X` covariates.
        pre_period: List[int]
        post_period: List[int]
        pre_data: pd.DataFrame
            Data sliced using `pre_period` values.
        post_data: pd.DataFrame
        normed_pre_data: pd.DataFrame
            If `standardize==True` then this is the result of the input data being
            normalized.
        observed_time_series: pd.DataFrame
            The input time series used for the TFP API. Main change is the index now
            is mandatory to be of type `DatetimeIndex` and have a valid frequency.
        normed_post_data: pd.DataFrame
        model: tfp.sts.StructuralTimeSeries
            `tfp.sts.StructuralTimeSeries` validated input model.
        model_args: Dict[str, Any]
            Dict containing general information related to how to fit and run the
            structural time series model.
        alpha: float
        mu_sig: Tuple[float, float]
            Mean and standard deviation used to normalize just the response variable `y`.

    Raises
    ------
      ValueError: if input arguments is `None`.
    """
    _check_empty_inputs(locals())
    data = format_input_data(data)
    pre_data, post_data = process_pre_post_data(data, pre_period, post_period)
    alpha = process_alpha(alpha)
    model_args = cimodel.process_model_args(model_args if model_args else {})
    normed_data = (
        standardize_pre_and_post_data(pre_data, post_data) if model_args['standardize']
        else (None, None, None)
    )
    # if operation `iloc` returns a pd.Series, cast it back to pd.DataFrame
    observed_time_series = _build_observed_time_series(
        pre_data if normed_data[0] is None else normed_data[0]
    )
    if model:
        cimodel.check_input_model(model, pre_data, post_data)
    else:
        model = cimodel.build_default_model(
            observed_time_series,
            normed_data[0] if model_args['standardize'] else pre_data,
            normed_data[1] if model_args['standardize'] else post_data,
            model_args['prior_level_sd'],
            model_args['nseasons'],
            model_args['season_duration']
        )
    return {
        'data': data,
        'pre_period': pre_period,
        'post_period': post_period,
        'pre_data': pre_data,
        'post_data': post_data,
        'normed_pre_data': normed_data[0],
        'normed_post_data': normed_data[1],
        'observed_time_series': observed_time_series,
        'model': model,
        'model_args':  model_args,
        'alpha': alpha,
        'mu_sig': normed_data[2]
    }


def _build_observed_time_series(pre_data: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that works as a mocking point for unit tests.
    """
    # if operation `iloc` returns a pd.Series, cast it back to pd.DataFrame
    observed_time_series = pd.DataFrame(pre_data.iloc[:, 0])
    return observed_time_series



def _check_empty_inputs(inputs: Dict[str, Any]) -> None:
    """
    Works as a validator for whether input values for Causal Impact contains empty
    values.

    Args
    ----
      inputs: Dict[str, Any]
          Basically contains the `locals()` variables as received by CausalImpact.
    """
    none_args = sorted(
        [arg for arg in inputs if inputs[arg] is None and arg != 'model']
    )
    if none_args:
        raise ValueError(
            f'{", ".join(none_args)} '
            f'input argument{"s" if len(none_args) > 1 else ""} cannot be empty'
        )


def standardize_pre_and_post_data(
    pre_data: pd.DataFrame,
    post_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float]]:
    """
    Applies standardization in pre and post data, based on mean and standard deviation
    of `pre_data` (as it's used for training the causal impact model).

    Args
    ----
      pre_data: pd.DataFrame
          data selected to be the pre-intervention dataset of causal impact.
      post_data: pd.DataFrame

    Returns
    -------
      Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float]]
        `pre_data` and `post_data` normalized along with the mean and variance used for
        response variable `y` only.
    """
    normed_pre_data, (mu, sig) = standardize(pre_data)
    normed_post_data = (post_data - mu) / sig
    mu_sig = (mu.iloc[0], sig.iloc[0])
    return (normed_pre_data, normed_post_data, mu_sig)


def format_input_data(data: Union[np.array, pd.DataFrame]) -> pd.DataFrame:
    """
    Validates and formats input data.

    Args
    ----
      data: Union[np.array, pd.DataFrame]
          First column is the response variable `y` and other columns correspond to
          the covariates `X`.

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
    validate_y(data.iloc[:, 0])
    # must contain only numeric values
    if not data.applymap(np.isreal).values.all():
        raise ValueError('Input data must contain only numeric values.')
    # covariates cannot have NAN values
    if data.shape[1] > 1:
        if data.iloc[:, 1:].isna().values.any():
            raise ValueError('Input data cannot have NAN values.')
    # If index is a string of dates, try to convert it to datetimes which helps
    # in plotting
    data = convert_index_to_datetime(data)
    # TFP linear regressors only work with float32 data so cast it here already
    data = data.astype(np.float32)
    return data


def process_pre_post_data(
    data: pd.DataFrame,
    pre_period: Union[List[int], List[str], List[pd.Timestamp]],
    post_period: Union[List[int], List[str], List[pd.Timestamp]]
) -> List[pd.DataFrame]:
    """
    Checks `pre_period` and `post_period` to return input data sliced accordingly to
    each period.

    Args
    ----
      data: pd.DataFrame
          First column is the response variable `y` and other columns correspond to
          the covariates `X`. This input data has already been validated as appropriate
          for the causal impact algorithm.
      pre_period: Union[List[int], List[str], List[pd.Timestamp]]
      post_period: Union[List[int], List[str], List[pd.Timestamp]]

    Returns
    -------
      result: List[pd.DataFrame]
          First value is pre-intervention data and second value is post-intervention.

    Raises
    ------
      ValueError: if pre_period last value is bigger than post intervention period.
    """
    checked_pre_period = process_period(pre_period, data)
    checked_post_period = process_period(post_period, data)
    if checked_pre_period[1] > checked_post_period[0]:
        raise ValueError(
            'Values in training data cannot be present in the post-intervention '
            'data. Please fix your pre_period value to cover at most one point less '
            'from when the intervention happened.'
        )
    if checked_pre_period[1] < checked_pre_period[0]:
        raise ValueError('pre_period last number must be bigger than its first.')
    if checked_pre_period[1] - checked_pre_period[0] < 3:
        raise ValueError('pre_period must span at least 3 time points.')
    if checked_post_period[1] < checked_post_period[0]:
        raise ValueError('post_period last number must be bigger than its first.')
    if checked_post_period[0] <= checked_pre_period[1]:
        raise ValueError(f'post_period first value ({post_period[0]}) must '
                         'be bigger than the second value of pre_period '
                         f'({pre_period[1]}).')
    # Force data to have date index type as required by TFP >= 0.14.0
    if isinstance(data.index, pd.RangeIndex):
        data = data.set_index(pd.date_range(start='2020-01-01', periods=len(data)))
    # Add +1 to make slicing inclusive on both ends as `iloc` doesn't include last value
    pre_data, post_data = [
        data.iloc[checked_pre_period[0]: checked_pre_period[1] + 1, :],
        data.iloc[checked_post_period[0]: checked_post_period[1] + 1, :]
    ]
    pre_data = tfp.sts.regularize_series(pre_data)
    post_data = tfp.sts.regularize_series(post_data)
    return pre_data, post_data


def validate_y(y: pd.Series) -> None:
    """
    Validates if input response variable is correct and doesn't contain invalid input.

    Args
    ----
      y: pd.Series
          Response variable sent in input data in first column.

    Raises
    ------
      ValueError: if values in `y` are Null.
                  if less than 3 (three) non-null values in `y` (as in this case
                      we can't even train a model).
                  if `y` is constant (in this case it doesn't make much sense to
                    make predictions as the time series doesn't change in the
                    training phase.
    """
    if np.all(y.isna()):
        raise ValueError('Input response cannot have just Null values.')
    if y.notna().values.sum() < 3:
        raise ValueError('Input response must have more than 3 non-null '
                         'points at least.')
    if y.std(skipna=True, ddof=0) == 0:
        raise ValueError('Input response cannot be constant.')


def convert_index_to_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """
    If input data has index of string dates, i.e, '20200101', '20200102' and so on, try
    to convert it to datetime specifically, which results in
    Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-02 00:00:00') ...

    Args
    ----
      data: pd.DataFrame
          Input data used in causal impact analysis.

    Returns
    -------
      data: pd.DataFrame
          Same input data with potentially new index of type DateTime.
    """
    if isinstance(data.index.values[0], str):
        try:
            data.set_index(pd.to_datetime(data.index), inplace=True)
        except ValueError:
            pass
    return data


def process_period(
    period: Union[List[int], List[str], List[pd.Timestamp]],
    data: pd.DataFrame
) -> List[int]:
    """
    Validates period inputs.

    Args
    ----
      period: Union[List[int], List[str], List[pd.Timestamp]]
      data: pd.DataFrame.
          Input Causal Impact data.

    Returns
    -------
      period: List[int]
          Validated period list.

    Raises
    ------
      ValueError: if input `period` is not of type list.
                  if input doesn't have two elements.
                  if period date values are not present in data.
    """
    if not isinstance(period, list):
        raise ValueError('Input period must be of type list.')
    if len(period) != 2:
        raise ValueError(
            'Period must have two values regarding the beginning and end of '
            'the pre and post intervention data.'
        )
    none_args = [d for d in period if d is None]
    if none_args:
        raise ValueError('Input period cannot have `None` values.')
    if not (
        (isinstance(period[0], int) and isinstance(period[1], int)) or
        (isinstance(period[1], str) and isinstance(period[1], str)) or
        (isinstance(period[1], pd.Timestamp) and isinstance(period[1], pd.Timestamp))
    ):
        raise ValueError(
            'Input periods must contain either int, str or pandas Timestamp'
        )
    # check whether the input period is indeed present in the input data index
    for point in period:
        if point not in data.index:
            if isinstance(point, pd.Timestamp):
                point = point.strftime('%Y%m%d')
            raise ValueError("{point} not present in input data index.".format(
                point=str(point)
                )
            )
    if isinstance(period[0], str) or isinstance(period[0], pd.Timestamp):
        period = convert_date_period_to_int(period, data)
    return period


def convert_date_period_to_int(
    period: Union[List[int], List[str], List[pd.Timestamp]],
    data: pd.DataFrame
) -> List[int]:
    """
    Converts string values from `period` to integer offsets from `data`.

    Args
    ----
      period: Union[List[int], List[str], List[pd.Timestamp]]
      data: pd.DataFrame
          `data` index is either a `str` of a pd.Timestamp type.

    Returns
    -------
      period: List[int]
          Where each value is the correspondent integer based value in `data` index.
    """
    result = []
    for date in period:
        offset = data.index.get_loc(date)
        result.append(offset)
    return result


def process_alpha(alpha: float) -> float:
    """
    Asserts input `alpha` is appropriate to be used in the model.

    Args
    ----
      alpha: float
          Ranges from 0 up to 1 indicating level of significance to assert when
          testing for presence of signal in post-intervention data.

    Returns
    -------
      alpha: float
          Validated `alpha` value.

    Raises
    ------
      ValueError: if alpha is not float.
                  if alpha is not between 0. and 1.
    """
    if not isinstance(alpha, float):
        raise ValueError('alpha must be of type float.')
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha must range between 0 (zero) and 1 (one) inclusive.'
        )
    return alpha
