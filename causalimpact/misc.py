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


"""Miscellaneous functions to help in the implementation of Causal Impact."""


from typing import Tuple

import scipy.stats as stats
import pandas as pd


def standardize(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Applies standardization to input data. Result should have mean zero and standard
    deviation of one.

    Args
    ----
      data: pd.DataFrame

    Returns
    -------
      Tuple[pd.DataFrame, Tuple[float, float]]
        data: pd.DataFrame
            standardized data with zero mean and std of one.
        Tuple[float, float]
          mean and standard deviation used on each column of input data to make
          standardization. These values should be used to obtain the original dataframe.

    Raises
    ------
      ValueError: if data has only one value.
    """
    if data.shape[0] == 1:
        raise ValueError('Input data must have more than one value')
    mu = data.mean(skipna=True)
    std = data.std(skipna=True, ddof=0)
    data = (data - mu) / std.fillna(1)
    return [data, (mu, std)]


def unstandardize(data: pd.DataFrame, mus_sigs: Tuple[float, float]) -> pd.DataFrame:
    """
    Applies the inverse transformation to return to original data using `mus_sigs` as
    reference. Final result should have mean `mu` and standard deviation `std` both
    present in `mus_sigs`.

    Args
    ----
      data: pd.DataFrame
      mus_sigs: Tuple[float, float]
          tuple where first value is the mean used for the standardization and
                second value is the respective standard deviation.

    Returns
    -------
      data: pd.DataFrame
          input data with mean and standard deviation given by `mus_sigs`.
    """
    mu, sig = mus_sigs
    data = (data * sig) + mu
    return data


def get_z_score(p):
    """
    Returns the correspondent z-score with probability area p.

    Args
    ----
      p: float
          ranges between 0 and 1 representing the probability area to convert.

    Returns
    -------
      float
          The z-score correspondent of p.
    """
    return stats.norm.ppf(p)
