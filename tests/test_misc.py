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
import pytest

from causalimpact.misc import get_z_score, standardize, unstandardize


def test_basic_standardize():
    data = {
        'c1': [1, 4, 8, 9, 10],
        'c2': [4, 8, 12, 16, 20]
    }
    data = pd.DataFrame(data)
    result, (mu, sig) = standardize(data)

    np.testing.assert_array_almost_equal(
        np.zeros(data.shape[1]),
        result.mean().values
    )

    np.testing.assert_array_almost_equal(
        np.ones(data.shape[1]),
        result.std(ddof=0).values
    )


def test_standardize_with_integer_column_names():
    # https://github.com/WillianFuks/tfcausalimpact/issues/17
    data = {
        'c1': [1, 4, 8, 9, 10],
        0: [4, 8, 12, 16, 20]
    }
    data = pd.DataFrame(data)
    result, (mu, sig) = standardize(data)

    np.testing.assert_array_almost_equal(
        np.zeros(data.shape[1]),
        result.mean().values
    )

    np.testing.assert_array_almost_equal(
        np.ones(data.shape[1]),
        result.std(ddof=0).values
    )


def test_standardize_w_various_distinct_inputs():
    test_data = [[1, 2, 1], [1, np.nan, 3], [10, 20, 30]]
    test_data = [pd.DataFrame(data, dtype="float") for data in test_data]
    for data in test_data:
        result, (mu, sig) = standardize(data)
        pd.testing.assert_frame_equal(unstandardize(result, (mu, sig)), data)


def test_standardize_raises_single_input():
    with pytest.raises(ValueError):
        standardize(pd.DataFrame([1]))


def test_get_z_score():
    assert get_z_score(0.5) == 0.
    assert round(float(get_z_score(0.9177)), 2) == 1.39
