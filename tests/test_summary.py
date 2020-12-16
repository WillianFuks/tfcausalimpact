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
Tests for summary.py module.
"""


import os

import pandas as pd
import pytest

import causalimpact.summary as summarizer


@pytest.fixture
def summary_data():
    data = [
        [5.123, 10.456],
        [4.234, 9.567],
        [3.123, 8.456],
        [6.234, 9.567],
        [3.123, 10.456],
        [2.234, 4.567],
        [6.123, 9.456],
        [0.2345, 0.5678],
        [0.1234, 0.4567],
        [0.2345, 0.5678]
    ]
    data = pd.DataFrame(
        data,
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
    return data


def test_summary_raises(summary_data):
    with pytest.raises(ValueError):
        summarizer.summary(summary_data, 0.1, output='test')


def test_output_summary_single_digit(summary_data, fix_path):
    summarizer.summary_data = summary_data
    alpha = 0.1
    p_value = 0.459329

    result = summarizer.summary(summary_data, p_value, alpha, digits=1)
    expected = open(os.path.join(
                    fix_path, 'test_output_summary_single_digit')).read().strip()
    assert result == expected


def test_report_summary_single_digit(summary_data, fix_path):
    # detected positive signal but with no significance.
    alpha = 0.1
    p_value = 0.5
    summary_data['average']['rel_effect'] = 0.41
    summary_data['average']['rel_effect_lower'] = -0.30
    summary_data['average']['rel_effect_upper'] = 0.30

    result = summarizer.summary(summary_data, p_value, alpha=alpha, output='report',
                                digits=1)
    expected = open(os.path.join(
                    fix_path, 'test_report_summary_single_digit')).read().strip()
    assert result == expected


def test_output_summary_1(summary_data, fix_path):
    alpha = 0.1
    p_value = 0.459329

    result = summarizer.summary(summary_data, p_value, alpha=alpha)
    expected = open(os.path.join(fix_path, 'test_output_summary_1')).read().strip()
    assert result == expected


def test_report_summary_1(summary_data, fix_path):
    # detected positive signal but with no significance.
    alpha = 0.1
    p_value = 0.5
    summary_data['average']['rel_effect'] = 0.41
    summary_data['average']['rel_effect_lower'] = -0.30
    summary_data['average']['rel_effect_upper'] = 0.30

    result = summarizer.summary(summary_data, p_value, alpha=alpha, output='report')
    expected = open(os.path.join(fix_path, 'test_report_summary_1')).read().strip()
    assert result == expected


def test_report_summary_2(summary_data, fix_path):
    # detected positive signal with significance.
    alpha = 0.1
    p_value = 0.05
    summary_data['average']['rel_effect'] = 0.41
    summary_data['average']['rel_effect_lower'] = 0.434
    summary_data['average']['rel_effect_upper'] = 0.234

    result = summarizer.summary(summary_data, p_value, alpha=alpha, output='report')
    expected = open(os.path.join(fix_path, 'test_report_summary_2')).read().strip()
    assert result == expected


def test_report_summary_3(summary_data, fix_path):
    # detected negative signal but with no significance.
    summary_data['average']['rel_effect'] = -0.343
    summary_data['average']['rel_effect_lower'] = -0.434
    summary_data['average']['rel_effect_upper'] = 0.234
    alpha = 0.1
    p_value = 0.5

    result = summarizer.summary(summary_data, p_value, alpha=alpha, output='report')
    expected = open(os.path.join(fix_path, 'test_report_summary_3')).read().strip()
    assert result == expected


def test_report_summary_4(summary_data, fix_path):
    # detected negative signal with significance.
    summary_data['average']['rel_effect'] = -0.343
    summary_data['average']['rel_effect_lower'] = -0.434
    summary_data['average']['rel_effect_upper'] = -0.234
    alpha = 0.1
    p_value = 0.05

    result = summarizer.summary(summary_data, p_value, alpha=alpha, output='report')
    expected = open(os.path.join(fix_path, 'test_report_summary_4')).read().strip()
    assert result == expected
