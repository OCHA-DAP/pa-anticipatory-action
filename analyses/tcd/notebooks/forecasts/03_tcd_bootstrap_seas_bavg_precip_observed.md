### Confidence interval relation below avg observed precipitation and historical drought
This notebook tests the confidence we have in the relation between below average observed precipitation and historical drought. 

We use CHIRPS as datasource. 

As can be seen according to our methodology, there is a high chance of the relation being random as a large percentage of the bootstrapped examples have a higher precision/recall than the original data. While we should be careful to attach too much value to this as the list of historical drought is not very well validated, this is a warning sign.

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import pandas as pd
import numpy as np
import math
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

from pathlib import Path
import os
import sys

from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
config = Config()
iso3 = "tcd"
country_data_exploration_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
)
country_data_processed_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3
)
```

```python
def compute_confusionmatrix_column(
    df,
    subplot_col,
    target_col,
    predict_col,
    ylabel,
    xlabel,
    colp_num=3,
    title=None,
    adjust_top=None,
):
    # number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df[subplot_col].unique())
    if num_plots == 1:
        colp_num = 1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig = plt.figure(figsize=(15, 8))
    for i, m in enumerate(
        df.sort_values(by=subplot_col)[subplot_col].unique()
    ):
        ax = fig.add_subplot(rows, colp_num, i + 1)
        y_target = df.loc[df[subplot_col] == m, target_col]
        y_predicted = df.loc[df[subplot_col] == m, predict_col]
        cm = confusion_matrix(y_target=y_target, y_predicted=y_predicted)

        plot_confusion_matrix(
            conf_mat=cm,
            show_absolute=True,
            show_normed=True,
            axis=ax,
            class_names=["No", "Yes"],
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{subplot_col}={m}")
    if title is not None:
        fig.suptitle(title)
    if adjust_top is not None:
        plt.subplots_adjust(top=adjust_top)
    return fig
```

```python
def bootstrap_resample(df, n_bootstrap=1_000):
    df_results = pd.DataFrame(columns=["precision", "recall"])
    for i in range(n_bootstrap):
        df_rs = df.sample(
            frac=1, replace=True, random_state=rng.bit_generator
        ).sum()
        df_results = df_results.append(
            {
                "precision": calc_precision(df_rs.TP, df_rs.FP),
                "recall": calc_recall(df_rs.TP, df_rs.FN),
                "F1": calc_f1(df_rs.TP, df_rs.FP, df_rs.FN),
                "nTP": df_rs.TP.sum(),
                "nFP": df_rs.FP.sum(),
                "nFN": df_rs.FN.sum(),
            },
            ignore_index=True,
        )
    df_results = df_results.dropna()
    return df_results


def calc_precision(TP, FP):
    return TP / (TP + FP)


def calc_recall(TP, FN):
    return TP / (TP + FN)


def calc_f1(TP, FP, FN):
    return TP / (TP + 0.5 * (FP + FN))
```

```python
# load bavg precip observed data
# this data has been produced in `tcd_seas_bavg_precip_observed.md`
df_all = pd.read_csv(
    country_data_exploration_dir / "chirps" / "tcd_perc_aoi_bavg.csv"
)
```

```python
# ground truth
col_drought = "drought"
# "predictor"
col_ind = "rp5"
```

```python
# JAS=July-August-September season, JJA=June-July-August
fig = compute_confusionmatrix_column(
    df_all,
    "season",
    col_drought,
    col_ind,
    "drought year in framework",
    f"percentage bavg above 5 year return period ",
    title=f"Correspondence of 1 in 5 year observed below average precipitation and reported drought years",
    adjust_top=1.2,
)
```

```python
df_all["TP"] = np.where((df_all[col_drought]) & (df_all[col_ind]), 1, 0)
df_all["FP"] = np.where((~df_all[col_drought]) & (df_all[col_ind]), 1, 0)
df_all["TN"] = np.where((~df_all[col_drought]) & (~df_all[col_ind]), 1, 0)
df_all["FN"] = np.where((df_all[col_drought]) & (~df_all[col_ind]), 1, 0)
```

```python
# either base on both seasons (JJA and JAS) or one of the two
# df=df_all.copy()
# Not 100% sure if it makes sense to group the seasons together, so can e.g. have 2FP but think it does
# df=df_all.groupby("year").sum()
# We are currently proposing to only use the JAS season for the trigger, so think that
# should be our focus (though might be changed later)
df = df_all[df_all.season == "JAS"]
```

```python
# Actual metrics
df_sum = df.sum()
precision = calc_precision(df_sum.TP, df_sum.FP)
recall = calc_recall(df_sum.TP, df_sum.FN)

print(f"Precision: {precision: 0.2f}")
print(f"Recall: {recall: 0.2f}")
print(f"Number of TP: {df.TP.sum()}")
```

```python
# Bootstrap resampling
# 10,000 is better but 1,000 is faster
df_results = bootstrap_resample(df)
```

```python
# Check that mean is close to what's expected
df_results.mean()
```

```python
# Plot distribution
df_results.hist();
```

```python
df_results.quantile(1 - 0.95)
```

```python
# check how many of the samples have higher performance metrics than the original
# goal is to test if the correlation we see in the data is by chance
# as rule of thumb, if 2.5% of the bootstrapped samples has a higher performance,
# there is a high chance the correlation is due to chance
perc_better_prec = round(
    df_results.loc[df_results.precision > precision, "precision"].count()
    / df_results["precision"].count()
    * 100,
    2,
)
perc_better_rec = round(
    df_results.loc[df_results.recall > recall, "recall"].count()
    / df_results["recall"].count()
    * 100,
    2,
)
print(
    f"{perc_better_prec}% of the bootstrapped samples has better precision than the original (={round(precision,2)})"
)
print(
    f"{perc_better_rec}% of the bootstrapped samples has better recall than the original (={round(recall,2)})"
)
```

### Years since 2000


Above we looked at all years since 1981. For other data sources we have only compared the years since 2000 due to data availability. We therefore repeat the above analysis for years since 2000 to be able to compare the results to other data sources. 

As can be seen the precision and recall is even worse and again around 40% of the bootstrapped samples have a higher precision/recall than the original, indicating a high chance the already weak relation is due to chance. 

```python
df_2000 = df_all[df_all.year >= 2000].copy()
```

Instead of using the years with the percentage of below average precip being above the 5 year return period, we simply select the worst 5 years since 2000, such that we have as many "positives" in the precipitation data as in the  drought impact data

```python
precip_worst_years = (
    df_2000.sort_values("perc_bavg", ascending=False)
    .drop_duplicates("year")
    .head(5)
    .year
)
```

```python
df_2000["precip_worst"] = np.where(
    df_2000.year.isin(precip_worst_years), True, False
)
```

```python
fig = compute_confusionmatrix_column(
    df_2000,
    "season",
    "drought",
    "precip_worst",
    "drought year in framework",
    f"percentage bavg above 1 in 4 year return period since 2000 ",
    title=f"Correspondence of 5 worst drought and bel avg percipitation since 2000",
    adjust_top=1.2,
)
```

```python
def comp_bootstrap_better(df, col_drought, col_ind):
    df["TP"] = np.where((df[col_drought]) & (df[col_ind]), 1, 0)
    df["FP"] = np.where((~df[col_drought]) & (df[col_ind]), 1, 0)
    df["TN"] = np.where((~df[col_drought]) & (~df[col_ind]), 1, 0)
    df["FN"] = np.where((df[col_drought]) & (~df[col_ind]), 1, 0)

    # Actual metrics
    df_sum = df.sum()
    precision = calc_precision(df_sum.TP, df_sum.FP)
    recall = calc_recall(df_sum.TP, df_sum.FN)

    # Bootstrap resampling
    df_results = bootstrap_resample(df)
    # check how many of the samples have higher performance metrics than the original
    perc_better_prec = round(
        df_results.loc[df_results.precision > precision, "precision"].count()
        / df_results["precision"].count()
        * 100,
        2,
    )
    perc_better_rec = round(
        df_results.loc[df_results.recall > recall, "recall"].count()
        / df_results["recall"].count()
        * 100,
        2,
    )
    print(
        f"{perc_better_prec}% of the bootstrapped samples has better "
        f"precision than the original (={round(precision,2)})"
    )
    print(
        f"{perc_better_rec}% of the bootstrapped samples has better "
        f"recall than the original (={round(recall,2)})"
    )
```

```python
df_2000_jas = df_2000[df_2000.season == "JAS"].copy()
```

```python
comp_bootstrap_better(df_2000_jas, "drought", "precip_worst")
```

### Compared to Biomasse


One of the other data sources we explored for the trigger is Biomasse. We repeat the bootstrapping that we did above but compared to the 5 worst years of Biomasse at the end of the season. 

Again we can see that there is a weak precision and recall. 

Instead of a binary classification we can also compare the continuous values of the percentage of the region with below average precipitation and the biomasse anomaly. We see a negative correlation as expected, though the correlation is not very strong (~-0.6) 

```python
df_bio = pd.read_csv(
    country_data_processed_dir / "biomasse" / "biomasse_tcd_ADM2_dekad_10.csv"
)
```

```python
# dekad 33 is the last dekad of november
df_2000 = df_2000.merge(df_bio[df_bio.dekad == 33], on="year", how="left")
```

```python
df_bio_worst_years = (
    df_2000.sort_values("biomasse_anomaly", ascending=True).head(5).year
)
```

```python
df_2000["biomasse_worst"] = np.where(
    df_2000.year.isin(df_bio_worst_years), True, False
)
```

```python
fig = compute_confusionmatrix_column(
    df_2000,
    "season",
    "biomasse_worst",
    "precip_worst",
    "biomass 1 in 4 worst anomaly",
    f"percentage bavg above 1 in 4 year return period since 2000 ",
    title=f"Correspondence of 5 worst drought and bel avg percipitation since 2000",
    adjust_top=1.2,
)
```

```python
df_2000_jas = df_2000[df_2000.season == "JAS"].copy()
```

```python
comp_bootstrap_better(df_2000_jas, "biomasse_worst", "precip_worst")
```

```python
df_2000_jas[["biomasse_anomaly", "perc_bavg"]].corr()
```

```python
sns.pairplot(df_2000_jas[["biomasse_anomaly", "perc_bavg"]])
```

```python

```
