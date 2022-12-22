### Confidence interval trigger

This notebook tests the confidence we have in the performance metrics of the Niger trigger in different months.

More explanation about the idea of bootstrapping can be found [here](https://docs.google.com/presentation/d/1MyyZXg1roeAwmImNUdDtOtuUCVeXBt3jhZGtybxVWYk/edit#slide=id.g102d7723583_0_22).

We also check what happens if we use a sample population that considers an alternative
definition of droubt, and how that changes the CIs.

## Load libraries and define function

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
from pathlib import Path
import os
import sys
import math

import pandas as pd
import numpy as np
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
N_SAMPLE = 28  # For the BSRS
country_data_exploration_dir = (
    Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
)
trigger_perf_path = country_data_exploration_dir / "trigger_performance"
input_path = trigger_perf_path / "historical_activations_trigger_v1.csv"
input_path_multi = trigger_perf_path / "historical_activ_multi_bad_years.csv"
input_path_update = (
    trigger_perf_path / "historical_activations_2023_methodology.csv"
)
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
# change df_all to more machine usable format
df_all = df_all.loc[:, df_all.columns.str.endswith(".1")]
df_all = df_all.iloc[1:]
df_all.columns = df_all.columns.str[:-2]
df_all.index.rename("year", inplace=True)
df_all.index = df_all.index.astype(int)
```

```python
# Also read in the file with alternative definition
df_all_alt = pd.read_csv(
    input_path_multi, usecols=[25, 29, 31, 33], skiprows=1, nrows=28
)
df_all_alt.columns = ["year", "Trigger3", "Trigger2", "Trigger1"]

df_all_alt.index = df_all_alt["year"]
df_all_alt.index.rename("year", inplace=True)
df_all_alt.index = df_all_alt.index.astype(int)
df_all_alt = df_all_alt[["Trigger1", "Trigger2", "Trigger3"]]
```

```python
# Finally, read in the update
df_all_update = pd.read_csv(
    input_path_update, index_col=0, skiprows=1, skipfooter=4
)

# change df_all to more machine usable format
df_all_update = df_all_update.loc[:, df_all_update.columns.str.endswith(".1")]
df_all_update = df_all_update.loc[
    :, df_all_update.columns.str.startswith("Trigger")
]
df_all_update = df_all_update.iloc[1:]
df_all_update = df_all_update.rename(index={"2021*": "2021"})
df_all_update.columns = df_all_update.columns.str[:-2]
df_all_update.index.rename("year", inplace=True)
df_all_update.index = df_all_update.index.astype(int)
```

```python
# Add min and full
def add_min_and_full(df):
    # For Min, if EITHER trigger 1, trigger 2, or trigger 3 is met,
    # this is counted as "min" being positive.
    # For "ful" the logic is that if EITHER trigger 1 AND trigger 2,
    # OR trigger 1 and trigger 3 are met, there is a full activation.
    df["framework-min"] = "TN"
    df["framework-full"] = "TN"
    df["framework-p1"] = "TN"
    df["framework-p2"] = "TN"

    df["bad_year"] = df["Trigger1"].isin(["TP", "FN"])

    df["min_activation"] = (
        df["Trigger1"].isin(["TP", "FP"])
        | df["Trigger2"].isin(["TP", "FP"])
        | df["Trigger3"].isin(["TP", "FP"])
    )
    df["full_activation"] = (
        df["Trigger1"].isin(["TP", "FP"]) & df["Trigger2"].isin(["TP", "FP"])
    ) | (
        (df["Trigger1"].isin(["TP", "FP"]) & df["Trigger3"].isin(["TP", "FP"]))
    )
    df["p1_activation"] = (
        df["Trigger1"].isin(["TP", "FP"])
        & df["Trigger2"].isin(["TN", "FN"])
        & df["Trigger3"].isin(["TN", "FN"])
    )
    df["p2_activation"] = df["Trigger1"].isin(["TN", "FN"]) & (
        df["Trigger2"].isin(["TP", "FP"]) | df["Trigger3"].isin(["TP", "FP"])
    )

    for activation_type in ["min", "full", "p1", "p2"]:
        df.loc[
            df["bad_year"] & df[f"{activation_type}_activation"],
            f"framework-{activation_type}",
        ] = "TP"
        df.loc[
            df["bad_year"] & ~df[f"{activation_type}_activation"],
            f"framework-{activation_type}",
        ] = "FN"
        df.loc[
            ~df["bad_year"] & df[f"{activation_type}_activation"],
            f"framework-{activation_type}",
        ] = "FP"

    df = df.drop(
        [
            "bad_year",
            "min_activation",
            "full_activation",
            "p1_activation",
            "p2_activation",
        ],
        axis=1,
    )
    return df
```

```python
df_all = add_min_and_full(df_all)
df_all_alt = add_min_and_full(df_all_alt)
df_all_update = add_min_and_full(df_all_update)
```

## Calculate base metrics

```python
def calc_df_base(df):
    return (
        df.apply(pd.value_counts)
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


df_base = calc_df_base(df_all)
df_base_alt = calc_df_base(df_all_alt)
df_base_update = calc_df_base(df_all_update)
```

## Bootstrap

```python
n_bootstrap = 10_000  # 10,000 takes about 5 minutes


def get_df_bootstrap(df, n_bootstrap=1_000):
    # Create a bootstrapped DF
    df_all_bootstrap = pd.DataFrame()
    for i in range(n_bootstrap):
        df_new = (
            df.sample(n=N_SAMPLE, replace=True, random_state=rng.bit_generator)
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
    return df_all_bootstrap


df_all_bootstrap = get_df_bootstrap(df_all, n_bootstrap)
df_all_bootstrap_alt = get_df_bootstrap(
    pd.concat([df_all, df_all_alt], ignore_index=True), n_bootstrap
)
df_all_bootstrap_update = get_df_bootstrap(df_all_update, n_bootstrap)
```

## Compute CIs



In the loop below there is an extra piece of code 
that shouldn't be used generally.
It's employing the 
[rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))
in the case when we don't have any FNs, i.e. for `Trigger1` and `framework-min`.

For the detection rate (DR), we can calculate the CI as follows:
1. Using the rule of 3, we know our sample could have up to 3/n = 3/28 = 10% FNs (to 95% confidence)
2. Therefore we just calculate the DR with this as a bound. We use the rate of TPs in 
   the sample, which happens to be 9/28 = 0.32 for both `Trigger1` and
   `framework-min`, and then use 0.1 for FNs from the step above. 
   And end up with 75% as a lower bound.

A symmetric argument can be made for the MR (upper bound 25%). 

Of course this doesn't take into account that there is also an 
uncertainty around the rate of TPs.



We also want the 68% confidence interval. Generalizing the rule of three:

$(1-p)^n = (1 - CI)$

$p = 1 - (1 - CI)^{1/n}$A

So for n=28 and a CI of 68%, we have up to 4% FNs, and get a lower bound for the CI of $ 0.32 / [0.32 + 0.04] = 0.89%$

```python
replacement_metrics = {0.68: 0.89, 0.95: 0.75}

# Calculate the quantiles over the bootstrapped df
def calc_ci(
    df_bootstrap, df_base, replace_fn_metrics=None, save_filename_suffix=None
):
    df_grouped = df_bootstrap.groupby("metric")
    for ci in [0.68, 0.95]:
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
        # Special case for trigger1 mis and det
        if replace_fn_metrics:
            df_ci.loc[
                (df_ci.metric == "det")
                & (df_ci.trigger.isin(["Trigger1", "framework-min"]))
                & (df_ci.point == "low_end"),
                "value",
            ] = replacement_metrics[ci]
            df_ci.loc[
                (df_ci.metric == "mis")
                & (df_ci.trigger.isin(["Trigger1", "framework-min"]))
                & (df_ci.point == "high_end"),
                "value",
            ] = (
                1 - replacement_metrics[ci]
            )
        # Save file
        output_filename = f"ner_perf_metrics_table_ci_{ci}"
        if save_filename_suffix:
            output_filename += "_" + save_filename_suffix
        output_filename += ".csv"
        df_ci.to_csv(trigger_perf_path / output_filename, index=False)


calc_ci(df_all_bootstrap, df_base, replace_fn_metrics=True)
# Note that in the next case we're using df_base_alt,
# which only uses the alternative data sample.
# Really the base should be some combination of
# both df_base and df_base_alt.
calc_ci(df_all_bootstrap_alt, df_base_alt, save_filename_suffix="alt")
calc_ci(df_all_bootstrap_update, df_base_update, save_filename_suffix="update")
```

## Explore CIs

```python
df_ci = (
    pd.read_csv(trigger_perf_path / "ner_perf_metrics_table_ci_0.95.csv")
    .assign(type="normal")
    .append(
        pd.read_csv(
            trigger_perf_path / "ner_perf_metrics_table_ci_0.95_alt.csv"
        ).assign(type="alt"),
        ignore_index=True,
    )
)
```

```python
fig, ax = plt.subplots(figsize=(8, 14))
df_ci_2 = df_ci[~df_ci["metric"].isin(["nTP", "nFP", "nFN", "nTN"])]
df_ci_2 = df_ci_2[
    df_ci_2["trigger"].isin(["Trigger1", "Trigger2", "Trigger3"])
]


for i, (_, group) in enumerate(df_ci_2.groupby(["metric", "trigger"])):
    ax.fill_betweenx(
        [i, i + 0.25],
        x1=group.loc[
            (group.type == "normal") & (group.point == "low_end"), "value"
        ],
        x2=group.loc[
            (group.type == "normal") & (group.point == "high_end"), "value"
        ],
        fc="blue",
    )
    ax.fill_betweenx(
        [i + 0.25, i + 0.5],
        x1=group.loc[
            (group.type == "alt") & (group.point == "low_end"), "value"
        ],
        x2=group.loc[
            (group.type == "alt") & (group.point == "high_end"), "value"
        ],
        fc="red",
    )
    ax.text(0, i, group.metric.iloc[0], size="x-small")
    ax.text(1, i, group.trigger.iloc[0], size="x-small")
```
