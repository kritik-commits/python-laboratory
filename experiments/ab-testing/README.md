\# A/B Testing with Python



\## Overview



This project implements a two-proportion z-test to compare conversion rates between a control group and a variant group.



\## Business Problem



A business changes its landing page and wants to know whether the new version improves conversion rates.



\## Experiment Data



| Metric | Control | Variant |

|---|---:|---:|

| Visitors | 1,000 | 1,000 |

| Conversions | 120 | 145 |

| Conversion Rate | 12.0% | 14.5% |



\## Methodology



\- Calculate conversion rates for both groups.

\- Calculate the pooled conversion rate.

\- Compute the z-statistic.

\- Calculate the two-sided p-value.

\- Compare the result against a significance level of 0.05.



\## Project Structure



\- `ab\_test.py` — Statistical test implementation.

\- `test\_ab\_test.py` — Unit tests for the implementation.



\## Run the Project



Run the analysis:



```bash

py ab\_test.py

```



Run the tests:



```bash

py -m unittest -v

```



\## Important Note



The example uses synthetic data. Statistical significance does not automatically establish business impact or causation.



