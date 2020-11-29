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


import pandas as pd
import numpy as np
import tensorflow_probability as tfp

from typing import Union, List, Dict, Any, Optional


def process_input_data(
    data: Union[np.array, pd.DataFrame],
    pre_period: Union[List[int], List[str], List[pd.Timestamp]],
    post_period: Union[List[int], List[str], List[pd.Timestamp]],
    model: Optional[tfp.sts.StructuralTimeSeries],
    model_args: Optional[Dict[str, Any]],
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
          Used for two-sided statistical test on comparison between counter-factual
          and observed series.

    Returns
    -------
      Dict[str, Any]:
        data: pd.DataFrame
            Validated data, first column is `y` and the others are the `X` covariates.
        pre_data: pd.DataFrame
            Data sliced using `pre_period` values.
        post_data: pd.DataFrame
        model: Optional[tfp.sts.StructuralTimeSeries]
            Either `None` or `tfp.sts.StructuralTimeSeries` validated input model.
        model_args: Dict[str, Any]
            Dict containing general information related to how to fit and run the
            structural time series model.
        alpha: float

    Raises
    ------
      ValueError: if input arguments is `None`.
    """
    locals_ = locals()
    required_args = ['data', 'pre_period', 'post_period', 'alpha']
    none_input_args = [arg for arg in required_args if locals_[arg] is None]
    if any(locals_[arg] is None for arg in required_args):
        raise ValueError(
            f'{", ".join(none_input_args)} '
            f'input argument{"s" if len(none_input_args) > 1 else ""} cannot be empty'
        )
    processed_data = format_input_data(data)
    pre_data, post_data = process_pre_post_data(processed_data, pre_period, post_period)
    alpha = process_alpha(alpha)
    model_args = process_model_args(model_args if model_args else {})
    if model:
        check_input_model(model)
    return {
        'data': processed_data,
        'pre_period': pre_period,
        'post_period': post_period,
        'pre_data': pre_data,
        'post_data': post_data,
        'model': model,
        'model_args':  model_args,
        'alpha': alpha
    }


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
    # in plotting.
    data = convert_index_to_datetime(data)
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

    result = [
        data.loc[pre_period[0]: pre_period[1], :],
        data.loc[post_period[0]: post_period[1], :]
    ]
    return result


def validate_y(y):
    """
    Validates if input response variable is correct and doesn't contain invalid input.

    Args
    ----
      y: pd.Series.
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


def convert_index_to_datetime(data):
    """
    If input data has index of string dates, i.e, '20200101', '20200102' and so on, try
    to convert it to datetime specifically, which results in
    Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-02 00:00:00')

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


def process_period(period, data):
    """
    Validates period inputs.

    Args
    ----
      period: .
      data: pd.DataFrame.
          Input Causal Impact data.

    Returns
    -------
      period: list.
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
        raise ValueError('Input must contain either int, str or pandas Timestamp')
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


def convert_date_period_to_int(period, data):
    """
    Converts string values from `period` to integer offsets from `data`.

    Args
    ----
      period: Union[List[str], List[pd.Timestamp]]
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
            Contains processed input args.

    Raises
    ------
      ValueError: if standardize is not of type `bool`.
                  if nseasons doesn't follow the pattern [{str key: number}].
    """
    standardize = model_args.get('standardize', True)
    if not isinstance(standardize, bool):
        raise ValueError('Standardize argument must be of type bool.')
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
    model_args['niter'] = niter

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


def check_input_model(model: tfp.sts.StructuralTimeSeries) -> None:
    """
    Checkes whether input model was properly built and is ready to be run.

    Args
    ----
      model: StructuralTimeSeries
          Can be either default `LocalLevel` or user specified generic model.

    Raises
    ------
      ValueError: if model is not of appropriate type
                  if model is built without observed time series data.
    """
    if not isinstance(model, tfp.sts.StructuralTimeSeries):
        raise ValueError('Input model must be of type StructuralTimeSeries.')
    # if not model.batch_shape.num_elements() > 0:
        # raise ValueError('Input model must contain observed time series data.')
