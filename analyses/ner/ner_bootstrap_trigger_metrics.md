### Confidence interval trigger

This notebook tests the confidence we have in the performance metrics of the Niger trigger in different months.

More explanation about the idea of bootstrapping can be found [here](https://docs.google.com/presentation/d/1MyyZXg1roeAwmImNUdDtOtuUCVeXBt3jhZGtybxVWYk/edit#slide=id.g102d7723583_0_22).

The confidence interval is set at 90%, i.e. between the 5% and 95% percentile. 

We first walk through the methodology with plots for one month. 
Thereafter, we compute the dataframe across all months. This is then used as input to the trigger report.

## Load libraries and define function

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import pandas as pd
import numpy as np

from pathlib import Path
import os
import sys

# from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
```

```python
rng = np.random.default_rng(12345)

path_mod = f"{Path(os.path.abspath('')).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
config = Config()
iso3 = "ner"
country_data_exploration_dir = (
    Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
)
trigger_perf_path = country_data_exploration_dir / "trigger_performance"
input_path = trigger_perf_path / "historical_activations_trigger_v1.csv"
input_path_multi = trigger_perf_path / "historical_activ_multi_bad_years.xlsx"
```

```python
def calc_far(TP, FP):
    return FP / (TP + FP)


def calc_var(TP, FP):
    return TP / (TP + FP)


def calc_det(TP, FN):
    return TP / (TP + FN)


def calc_mis(TP, FN):
    return FN / (TP + FN)


def calc_acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def calc_atv(TP, TN, FP, FN):
    return (TP + FP) / (TP + TN + FP + FN)
```

## Load data

```python
# load data of when trigger activated for each month
df_all = pd.read_csv(input_path, index_col=0)
```

```python
# change df_all to more machine usable format
df_all = df_all.loc[:, df_all.columns.str.endswith(".1")]
df_all = df_all.iloc[1:]
df_all.columns = df_all.columns.str[:-2]
df_all.index.rename("year", inplace=True)
df_all.index = df_all.index.astype(int)
```

```python
# Add min and full
# For Min, if EITHER trigger 1, trigger 2, or trigger 3 is met,
# this is counted as "min" being positive.
# For "ful" the logic is that if EITHER trigger 1 AND trigger 2,
# OR trigger 1 and trigger 3 are met, there is a full activation.
df_all["min"] = "TN"
df_all["full"] = "TN"

for tpfp in ["TP", "FP"]:
    # If there are any TPs / FPs, then min is a TP / FP
    df_all.loc[
        (df_all[["Trigger1", "Trigger2", "Trigger3"]] == tpfp).apply(
            any, axis=1
        ),
        "min",
    ] = tpfp
    # If T1 & T2, or T1 & T3 met, the it's full
    df_all.loc[
        ((df_all["Trigger1"] == tpfp) & (df_all["Trigger2"] == tpfp))
        | ((df_all["Trigger1"] == tpfp) & (df_all["Trigger3"] == tpfp)),
        "full",
    ] = tpfp

# TO DO: lots of gaps in this logic
df_all
```

## Calculate base metrics

```python
df_base = (
    df_all.apply(pd.value_counts)
    .fillna(0)
    .apply(
        lambda x: {
            "far": calc_far(x.TP, x.FP),
            "var": calc_var(x.TP, x.FP),
            "det": calc_det(x.TP, x.FN),
            "mis": calc_mis(x.TP, x.FN),
            "acc": calc_acc(x.TP, x.TN, x.FP, x.FN),
            "atv": calc_atv(x.TP, x.TN, x.FP, x.FN),
            "nTP": x.TP.sum(),
            "nFP": x.FP.sum(),
            "nFN": x.FN.sum(),
        },
        result_type="expand",
    )
    .melt(ignore_index=False)
    .reset_index()
    .rename(columns={"index": "metric", "variable": "trigger"})
    .assign(point="central")
)
df_base
```

## Compute CIs for all trigger moments


```python
# Create a bootstrapped DF
n_bootstrap = 10_000  # 10,000 takes about 2.5 minutes
df_all_bootstrap = pd.DataFrame()
for i in range(n_bootstrap):
    df_new = (
        df_all.sample(frac=1, replace=True, random_state=rng.bit_generator)
        .apply(pd.value_counts)
        .fillna(0)
    )
    # Some realizations are missing certain counts
    for count in ["FN", "FP", "TN", "TP"]:
        if count not in df_new.index:
            df_new.loc[count] = 0
    df_new = (
        df_new.apply(
            lambda x: {
                "far": calc_far(x.TP, x.FP),
                "var": calc_var(x.TP, x.FP),
                "det": calc_det(x.TP, x.FN),
                "mis": calc_mis(x.TP, x.FN),
                "acc": calc_acc(x.TP, x.TN, x.FP, x.FN),
                "atv": calc_atv(x.TP, x.TN, x.FP, x.FN),
                "nTP": x.TP.sum(),
                "nFP": x.FP.sum(),
                "nFN": x.FN.sum(),
            },
            result_type="expand",
        )
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    df_all_bootstrap = df_all_bootstrap.append(df_new, ignore_index=True)
```

```python
# Calculate the quantiles over the bootstrapped df
df_grouped = df_all_bootstrap.groupby("metric")
for ci in [0.8, 0.9, 0.95, 0.99]:
    df_ci = df_base.copy()
    points = {"low_end": (1 - ci) / 2, "high_end": 1 - (1 - ci) / 2}
    for point, ci_val in points.items():
        df = (
            df_grouped.quantile(ci_val)
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={"variable": "trigger"})
        )
        df["point"] = point
        df_ci = df_ci.append(df, ignore_index=True)
    # Save file
    output_filename = f"ner_perf_metrics_table_ci_{ci}_v2.csv"
    df_ci.to_csv(trigger_perf_path / output_filename, index=False)
```

### Example: Aggregate months

There are three funding packages: from Jan-March, from April-Jun, and in August. 
If there is an activation during one of those period, that whole packages is released. 
And the trigger cannot be reached anymore during the other months of that "package". 

We thus want to know the performance over those three "packages". 
In the Niger file, this was already added manually. However, for the futre below an example of how
the aggregation of trigger points can also be computed automatically. 

```python
# load data of when trigger activated for each month
df_all = pd.read_csv(input_path, index_col=0)
```

```python
# change df_all to more machine usable format
df_act = df_all.loc[:, ~df_all.columns.str.contains(".1")]
df_act = df_act.rename(columns={"all-years": "bad-year"})
df_act = df_act.iloc[1:]
df_act.index.rename("year", inplace=True)
df_act.index = df_act.index.astype(int)
```

```python
df_act
```

```python
# 1 indicates activation or observed bad year (i.e. wanted to activate)
# 0 no activation/no bad year observed
df_act_num = (
    df_act.replace("bad", True).replace("yes", True).replace(np.nan, False)
)
```

```python
df_act_num.head()
```

```python
periods=[["Jan","Feb","Mar"],["Apr","May","June"],["August"]]
```

```python
save_file=False
metric_list=["far","var","det","mis","acc","atv"]
col_obs="bad-year"
col_act="activation"
for confidence_interval in [0.8,0.9,0.95]:
    print((1-confidence_interval)/2)
    print(1-((1-confidence_interval)/2))
    df_trig_met=pd.DataFrame(columns=["trigger","metric","point","value"])
    for period in periods: 
        period_str="".join(period)
        #reshape to dataframe with one column per indicator
        df_sel=df_act_num[["bad-year"]+periods[0]]
        #take the max over the period
        #i.e. if activated in one month, that period is activated
        df_sel["activation"]=df_sel[periods[0]].max(axis=1)

        df_sel["TP"]=np.where((df_sel[col_obs])&(df_sel[col_act]),1,0)
        df_sel["FP"]=np.where((~df_sel[col_obs])&(df_sel[col_act]),1,0)
        df_sel["TN"]=np.where((~df_sel[col_obs])&(~df_sel[col_act]),1,0)
        df_sel["FN"]=np.where((df_sel[col_obs])&(~df_sel[col_act]),1,0)
        #get the bootstrap results
        df_bootstrap = bootstrap_resample(df_sel,n_bootstrap=10000)
        for metric in metric_list: 
            df_trig_met=df_trig_met.append({"metric":metric,"point":"central","value":df_bootstrap[metric].median(),"trigger":period_str},ignore_index=True)
            df_trig_met=df_trig_met.append({"metric":metric,"point":"low_end","value":df_bootstrap[metric].quantile((1-confidence_interval)/2),"trigger":period_str},ignore_index=True)
            df_trig_met=df_trig_met.append({"metric":metric,"point":"high_end","value":df_bootstrap[metric].quantile(1-((1-confidence_interval)/2)),"trigger":period_str},ignore_index=True)
    if save_file: 
        df_trig_met["dummy"]=1
        output_filename=f"perf_metrics_table_per_package_ci_{confidence_interval}.csv"
        df_trig_met.to_csv(trigger_perf_path/output_filename,index=False)
```

#### Sanity check values

```python
df_sel
```

```python
df_bootstrap
```

```python
# Plot distribution
df_bootstrap.hist();
plt.show()
```

```python

```
