```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

rng = np.random.default_rng(12345)
```

```python
df = pd.DataFrame({'years': range(1987, 2021), 
                   'TP': 0, 'FP': 0, 'FN': 0}).set_index('years')
```

```python
df.loc[1988, 'TP'] = 1
df.loc[1989, 'FP'] = 1
df.loc[1996, 'FP'] = 1
df.loc[2002, 'FP'] = 1
df.loc[2004, 'FP'] = 1
df.loc[2012, 'FN'] = 1
df.loc[2016, 'FN'] = 1
df.loc[2017, 'FN'] = 1
df.loc[2018, 'TP'] = 1
df.loc[2019, 'TP'] = 1
df.loc[2020, 'TP'] = 2
```

```python
def calc_precision(TP, FP):
    return TP / (TP + FP)

def calc_recall(TP, FN):
    return TP / (TP + FN)
```

```python
# Actual precision and recall
df_sum = df.sum()
precision = calc_precision(df_sum.TP, df_sum.FP)
recall = calc_recall(df_sum.TP, df_sum.FN)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

```python
from statsmodels.stats.proportion import proportion_confint

proportion_confint(df_sum.TP, df_sum.TP + df_sum.FP, method='wilson')
```

```python
n_bootstrap = 10000
df_results = pd.DataFrame(columns=['precision', 'recall'])
for i in range(n_bootstrap):
    df_rs = df.sample(frac=1, replace=True, random_state=rng.bit_generator).sum()
    df_results = df_results.append({
        'precision': calc_precision(df_rs.TP, df_rs.FP),
        'recall': calc_recall(df_rs.TP, df_rs.FN)
    }, ignore_index=True)
df_results = df_results.dropna()    
```

```python
# Check that mean is close to what's expected
df_results.mean()
```

```python
df_results.hist()
```

```python
# Can ask e.g. how confident are we that precision or recall will be > 50%?
(df_results > 0.5).sum() / 10000
```

```python
# Get a confidence interval
df_results.quantile(0.05)
```

```python

```
