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
Summarizes performance information inferred in post-inferences compilation process.
"""


import os

import pandas as pd
from jinja2 import Template

from causalimpact.misc import get_z_score

_here = os.path.dirname(os.path.abspath(__file__))
summary_tmpl_path = os.path.join(_here, 'summary', 'templates', 'summary')
report_tmpl_path = os.path.join(_here, 'summary', 'templates', 'report')

SUMMARY_TMPL = Template(open(summary_tmpl_path).read())
REPORT_TMPL = Template(open(report_tmpl_path).read())


def summary(
    summary_data: pd.DataFrame,
    p_value: float,
    alpha: float = 0.05,
    output: str = 'summary',
    digits: int = 2
) -> str:
    """
    Returns final results from causal impact analysis, such as absolute observed effect,
    the relative effect between prediction and observed variable, cumulative performances
    in post-intervention period among other metrics.

    Args
    ----
      summary_data: pd.DataFrame
          Contains information such as means and cumulatives averages.
      p_value: float
          p-value test for testing presence of signal in data.
      alpha: float
          Sets credible interval width.
      output: str
          Can be either "summary" or "report". The first is a simpler output just
          informing general metrics such as expected absolute or relative effect.
      digits: int
          Defines the number of digits after the decimal point to round. For `digits=2`,
          value 1.566 becomes 1.57.

    Returns
    -------
      summary: str
          Contains results of the causal impact analysis.

    Raises
    ------
      ValueError: If input `output` is not either 'summary' or 'report'.
    """
    if output not in {'summary', 'report'}:
        raise ValueError('Please choose either summary or report for output.')
    if output == 'summary':
        summary = SUMMARY_TMPL.render(
            summary=summary_data.to_dict(),
            alpha=alpha,
            z_score=get_z_score(1 - alpha / 2.),
            p_value=p_value,
            digits=digits
        )
    else:
        summary = REPORT_TMPL.render(
            summary=summary_data.to_dict(),
            alpha=alpha,
            p_value=p_value,
            digits=digits
        )
    return summary
