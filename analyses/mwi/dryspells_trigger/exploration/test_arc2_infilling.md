```python
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

dd = os.getenv("AA_DATA_DIR")
df = pd.read_csv(os.path.join(dd, 'public', 'exploration', 'mwi', 'arc2', 'mwi_arc2_precip_long_raw.csv'),
                 parse_dates = ["date"])
df = df[["ADM2_EN", "date", "mean_cell"]]
df = df[~np.isnan(df.mean_cell)].reset_index(drop=True)
```

Let's randomly drop dates of data to then just simply test the most accurate infilling methods.

```python
np.random.seed(5)
drop_index = np.random.choice(df.shape[0], size = 10000, replace = False)
df['mean_cell_test'] = df['mean_cell']
df.loc[drop_index, 'mean_cell_test'] = np.NaN
```

Now just some simple testing:

```python
df['prev_value'] = df.groupby('ADM2_EN')['mean_cell_test'].transform(lambda x: x.ffill())
df['interpolated'] = df.groupby('ADM2_EN')['mean_cell_test'].transform(lambda x: x.interpolate())
df['prev_3_values'] = df.groupby('ADM2_EN')['mean_cell_test'].transform(lambda x: x.fillna(x.rolling(4, min_periods=1).mean()))

df_metr = df[np.isnan(df.mean_cell_test)]
```

And let's just have a look at MAE for a quick check.

```python
prev_mae = mean_absolute_error(df_metr.mean_cell, df_metr.prev_value)
inter_mae = mean_absolute_error(df_metr.mean_cell, df_metr.interpolated)
prev_3_mae = mean_absolute_error(df_metr.mean_cell, df_metr.prev_3_values)

print(
    f"""The MAE scores are:
    Previous value: {np.round(prev_mae,1)}
    Interpolation: {np.round(inter_mae,1)}
    Previous 3 values: {np.round(prev_3_mae,1)}
    """
)
```

I think the baseline is essentially that none of these methods are doing particularly well. However, let's look just at situations where the actual observed rainfall is relatively small.

```python
df_small = df_metr[df_metr.mean_cell <= 2]

prev_mae = mean_absolute_error(df_small.mean_cell, df_small.prev_value)
inter_mae = mean_absolute_error(df_small.mean_cell, df_small.interpolated)
prev_3_mae = mean_absolute_error(df_small.mean_cell, df_small.prev_3_values)

print(
    f"""The MAE scores when true precipation was <= 2 are:
    Previous value: {np.round(prev_mae,1)}
    Interpolation: {np.round(inter_mae,1)}
    Previous 3 values: {np.round(prev_3_mae,1)}
    """
)
```

And since MAE is less sensitive to large errors, do we still see the same results when looking at RMSE.

```python
prev_rmse = mean_squared_error(df_small.mean_cell, df_small.prev_value, squared=False)
inter_rmse = mean_squared_error(df_small.mean_cell, df_small.interpolated, squared=False)
prev_3_rmse = mean_squared_error(df_small.mean_cell, df_small.prev_3_values, squared=False)

print(
    f"""The RMSE scores are:
    Previous value: {np.round(prev_rmse,1)}
    Interpolation: {np.round(inter_rmse,1)}
    Previous 3 values: {np.round(prev_3_rmse,1)}
    """
)
```

Overall, it looks like interpolation is performing better than the other simple methods tested. 
