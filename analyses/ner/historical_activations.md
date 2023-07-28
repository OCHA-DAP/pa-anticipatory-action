---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: pa-anticipatory-action
    language: python
    name: pa-anticipatory-action
---

# Historical activations

```python
%load_ext jupyter_black
```

```python
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.offline as pyo

pyo.init_notebook_mode()
```

```python
AA_DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
EXP_DIR = AA_DATA_DIR / "public/exploration/ner"

COLORS = px.colors.qualitative.Pastel
```

## Downloading data
1. Go to maproom: https://iridl.ldeo.columbia.edu/fbfmaproom2/niger
2. Copy + paste values into sheet: https://docs.google.com/spreadsheets/d/1y9a4eNtMNJR6XQ0xSO-0W-Rr80fS2NFobDkc9SULhxU/edit#gid=0
3. Download sheet as .csv into EXP_DIR


## Processing data year-by-year

```python
# BASELINE = "enacts-precip-jas"
BASELINE = "bad-year-v2"

filepath = EXP_DIR / "iri/ner_trigger_comparison - Sheet1.csv"
df = pd.read_csv(filepath, header=None, index_col=0)
df = df.T

# combine multi-level headers
header_names = ["obsv/pred", "type", "mon", "threshold", "package"]
df["headers"] = df.apply(
    lambda row: "|".join([f"{x}:{row[x]}" for x in header_names]), axis=1
)

df = df.T
df.columns = df.loc["headers"]
df.index = pd.to_numeric(df.index, errors="coerce")
df = df[df.index > 0]
df.index = df.index.astype(int)
df.index.name = "Year"
df = df.replace(["1", "yes"], True).replace(["0", "no"], False)


packages = [1, 2]
thresholds = [20, 35]
pred_months = range(1, 7)
base_col = next(col for col in df.columns if BASELINE in col)

for t in thresholds:
    t_cols = [col for col in df.columns if f"threshold:{t}" in col]
    pred_cols = [col for col in t_cols if "obsv/pred:Predictive" in col]
    obsv_cols = [col for col in t_cols if "obsv/pred:Observational" in col]

    # pred months by package
    for p in packages:
        cols = [col for col in pred_cols if f"package:{p}" in col]
        out_col = f"pred|threshold:{t}|package:{p}"
        df[out_col] = df[cols].any(axis=1)
        # with obsv
        if p == 2:
            cols += obsv_cols
        df[f"pred_obsv|threshold:{t}|package:{p}"] = df[cols].any(axis=1)
    # either package
    out_col = f"pred|threshold:{t}|package:either"
    df[out_col] = df[pred_cols].any(axis=1)
    # with obsv
    cols = [out_col, *obsv_cols]
    df[f"pred_obsv|threshold:{t}|package:either"] = df[cols].any(axis=1)

    # all pred months by package
    for p in packages:
        cols = [col for col in pred_cols if f"package:{p}" in col]
        out_col = f"all_pred|threshold:{t}|package:{p}"
        df[out_col] = df[cols].all(axis=1)
        # with obsv
        cols = [out_col, *obsv_cols] if p == 2 else [out_col]
        df[f"all_pred_obsv|threshold:{t}|package:{p}"] = df[cols].any(axis=1)
    # either package
    out_col = f"all_pred|threshold:{t}|package:either"
    cols = [f"all_pred|threshold:{t}|package:{p}" for p in packages]
    df[out_col] = df[cols].any(axis=1)
    # with obsv
    cols = [out_col, *obsv_cols]
    df[f"all_pred_obsv|threshold:{t}|package:either"] = df[cols].any(axis=1)

    # consecutive predictive months - month by month
    consec_cols = []
    for j in range(len(pred_months) - 1):
        m1, m2 = pred_months[j], pred_months[j + 1]
        cols = [
            col
            for col in pred_cols
            if (f"mon:{m1}" in col) or (f"mon:{m2}" in col)
        ]
        p = 1 if j < 2 else 2
        col_name = f"month:{m1}and{m2}|threshold:{t}|package:{p}"
        df[col_name] = df[cols].all(axis=1)
        consec_cols.append(col_name)
    # consecutive predictive months by package
    for p in packages:
        cols = [col for col in consec_cols if f"package:{p}" in col]
        out_col = f"consec_pred|threshold:{t}|package:{p}"
        df[out_col] = df[cols].any(axis=1)
        # with obsv
        if p == 2:
            cols += obsv_cols
        df[f"consec_pred_obsv|threshold:{t}|package:{p}"] = df[cols].any(
            axis=1
        )
    # consec pred
    out_col = f"consec_pred|threshold:{t}|package:either"
    df[out_col] = df[consec_cols].any(axis=1)
    # with obsv
    cols = [out_col, *obsv_cols]
    df[f"consec_pred_obsv|threshold:{t}|package:either"] = df[cols].any(axis=1)

df.to_csv(EXP_DIR / "historical_triggers.csv")
```

## Checking specific triggers

```python
cols = [
    col
    for col in df.columns
    if (("consec" in col) or ("Observational" in col) or ("bad-years" in col))
    and ("threshold:35" in col)
]
cols.append(base_col)
df_disp = df[cols]
display(
    df_disp.T.style.applymap(lambda x: "background-color : red" if x else "")
)
```

## Calculating confusion matrix and plotting comparison
https://en.wikipedia.org/wiki/Confusion_matrix

```python
# set cutoff year to exclude weird years (1991-1997)
for CUTOFF_YEAR in [1991, 1998]:
    df_cutoff = df[df.index >= CUTOFF_YEAR]

    df_agg = pd.DataFrame(
        index=df_cutoff.columns,
        columns=[
            "TP",
            "FN",
            "FP",
            "TN",
            "TPR",
            "FNR",
            "activation_prob",
            "return_period",
            "corr",
        ],
    )

    # calculate basic confusion matrix outputs
    for col in df_cutoff.columns:
        df_conf = pd.crosstab(
            df_cutoff[base_col], df_cutoff[col], dropna=False
        )
        if True in df_conf.columns:
            FP, TP = df_conf[True]
        else:
            FP, TP = 0, 0
        if False in df_conf.columns:
            TN, FN = df_conf[False]
        else:
            TN, FN = 0, 0
        TPR = TP / (TP + FN)
        FNR = 1 - TPR
        total = FP + TP + TN + FN
        prevalence = (TP + FP) / total
        if prevalence > 0:
            return_period = 1 / prevalence
        else:
            return_period = np.inf
        corr = df_cutoff[base_col].corr(df_cutoff[col])
        df_agg.loc[col] = {
            "TP": TP,
            "FN": FN,
            "FP": FP,
            "TN": TN,
            "TPR": TPR,
            "FNR": FNR,
            "activation_prob": prevalence,
            "return_period": return_period,
            "corr": corr,
        }

    df_agg = df_agg.astype(float)

    # calculate additional confusion matrix outputs
    df_agg["P"] = df_agg["TP"] + df_agg["FN"]
    df_agg["N"] = df_agg["FP"] + df_agg["TN"]
    df_agg["PP"] = df_agg["TP"] + df_agg["FP"]
    df_agg["PN"] = df_agg["FN"] + df_agg["TN"]
    df_agg["TPR"] = df_agg["TP"] / df_agg["P"]
    df_agg["FNR"] = 1 - df_agg["TPR"]
    df_agg["TNR"] = df_agg["TN"] / df_agg["N"]
    df_agg["FPR"] = 1 - df_agg["TNR"]

    df_agg["PPV"] = df_agg["TP"] / df_agg["PP"]
    df_agg["FDR"] = 1 - df_agg["PPV"]
    df_agg["NPV"] = df_agg["TN"] / df_agg["PN"]
    df_agg["FOR"] = 1 - df_agg["NPV"]

    display(df_agg["activation_prob"])

    # I think MMC is just the same as the correlation
    df_agg["MMC"] = np.sqrt(
        df_agg[["TPR", "TNR", "PPV", "NPV"]].prod(axis=1)
    ) - np.sqrt(df_agg[["FNR", "FPR", "FOR", "FDR"]].prod(axis=1))

    df_agg.to_csv(
        EXP_DIR
        / f"historical_triggers_summary_since{CUTOFF_YEAR}_against_{BASELINE}.csv"
    )

    plot_packages = ["either", "1", "2"]
    #     plot_packages = ["either"]

    # plot by package
    for p in plot_packages:
        plot_cols = [
            f"pred_obsv|threshold:35|package:{p}",
            f"consec_pred_obsv|threshold:35|package:{p}",
            f"consec_pred|threshold:35|package:{p}",
            #             f"all_pred_obsv|threshold:35|package:{p}",
            #             f"consec_pred|threshold:35|package:{p}",
            f"pred_obsv|threshold:20|package:{p}",
            f"consec_pred_obsv|threshold:20|package:{p}",
            f"consec_pred|threshold:20|package:{p}",
            #             f"all_pred_obsv|threshold:20|package:{p}",
            #             f"consec_pred|threshold:20|package:{p}",
        ]

        x_axes = [
            [
                *["Seuil: 35%"] * int(len(plot_cols) / 2),
                *["Seuil: 20%"] * int(len(plot_cols) / 2),
            ],
            [
                "N'importe<br>quel mois",
                "2 mois<br>consec.",
                "N'importe<br>quel mois<br>SANS obsv"
                #                 "Tous les 3<br>mois",
                #                 "2 mois<br>consec.<br>sans<br>observationel",
            ]
            * 2,
        ]

        df_plot = df_agg.loc[plot_cols]

        p = "l'un ou l'autre" if p == "either" else p
        fig = go.Figure()
        fig.update_layout(
            template="simple_white",
            title_text=f"Activations depuis {CUTOFF_YEAR} da la fenêtre: {p}<br>"
            f"<sup>comparé avec {BASELINE} comme baseline</sup>",
        )
        fig.add_trace(
            go.Bar(
                x=x_axes,
                y=df_plot["TPR"],
                name="Taux de détection / TPR",
                text=df_plot["TPR"].round(2),
                textposition="auto",
                marker_color=COLORS[0],
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_axes,
                y=df_plot["activation_prob"],
                name="Taux d'activation",
                text=df_plot["activation_prob"].round(2),
                textposition="auto",
                marker_color=COLORS[1],
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_axes,
                y=df_plot["corr"],
                name=f"Corrélation avec {BASELINE}",
                text=df_plot["corr"].round(2),
                textposition="auto",
                marker_color=COLORS[2],
            )
        )
        fig.update_legends(yanchor="top", y=0.99, xanchor="right", x=0.99)

        fig.show()

    fig = px.scatter(
        df_agg[~(df_agg.index == base_col)].reset_index(),
        y="TPR",
        x="activation_prob",
        hover_name="headers",
    )
    fig.update_layout(
        template="simple_white",
        title_text=f"Compromis entre taux de détection et taux d'activation (depuis {CUTOFF_YEAR})<br>",
    )
    fig.update_yaxes(title_text="Taux de détection")
    fig.update_xaxes(title_text="Taux d'activation")
    fig.show()
```

```python

```
