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
#arghh jupyter_black is giving an error I don't understand
# %load_ext jupyter_black
```

```python
import pandas as pd
import numpy as np

from pathlib import Path
import os 
import sys

# from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
config=Config()
iso3="ner"
country_data_exploration_dir = Path(config.DATA_DIR)/config.PRIVATE_DIR/"exploration"/iso3
input_path=country_data_exploration_dir/"trigger_performance"/"historical_activations_trigger_v1.csv"
output_path=country_data_exploration_dir/"trigger_performance"/"perf_metrics_table_for_template.csv"
```

```python
def bootstrap_resample(df, n_bootstrap=1_000):
    df_results = pd.DataFrame()#columns=['precision', 'recall'])
    for i in range(n_bootstrap):
        df_rs = df.sample(frac=1, replace=True, 
                        random_state=rng.bit_generator).sum()
        df_results = df_results.append({
          "far": calc_far(df_rs.TP,df_rs.FP),
          "var": calc_var(df_rs.TP,df_rs.FP),
          "det": calc_det(df_rs.TP,df_rs.FN),
            "mis": calc_mis(df_rs.TP,df_rs.FN),
            "acc": calc_acc(df_rs.TP,df_rs.TN,df_rs.FP,df_rs.FN),
          'nTP': df_rs.TP.sum(),
          'nFP': df_rs.FP.sum(),
          'nFN': df_rs.FN.sum()
      }, ignore_index=True)
    df_results = df_results.dropna()
    return df_results


def calc_far(TP,FP):
    return FP/(TP+FP)

def calc_var(TP,FP):
    return TP/(TP+FP)

def calc_det(TP,FN):
    return TP/(TP+FN)

def calc_mis(TP,FN):
    return TP/(TP+FN)

def calc_acc(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)
```

## Load data

```python
#load data of when trigger activated for each month
df_all=pd.read_csv(input_path,index_col=0)
```

```python
df_all.head()
```

```python
#change df_all to more machine usable format
df_all=df_all.loc[:,df_all.columns.str.contains(".1")]
df_all=df_all.iloc[1:]
df_all.columns=df_all.columns.str[:-2]
df_all.index.rename("year",inplace=True)
df_all.index=df_all.index.astype(int)
```

## Bootstrap for one trigger moment
We do the bootstrap for one trigger moment. 
Here we walk through the metrics and plot them, to understand the process. 
In the next section we do this more effectively and loop through all the trigger moments. 

```python
#select one trigger moment
month="June"
#reshape to dataframe with one column per indicator
df=df_all[month].to_frame()
for metric in ["FP","TP","TN","FN"]:
    df[metric]=np.where(df[month]==metric,1,0)
```

```python
#counts of category occurrences
df.drop("June",axis=1).sum()
```

```python
# Actual metrics
df_sum = df.sum()
var=calc_var(df_sum.TP,df_sum.FP)
det=calc_det(df_sum.TP,df_sum.FN)
print(f"var: {var}")
print(f"det: {det}")
print(f"FAR: {calc_far(df_sum.TP,df_sum.FP)}")
print(f"mis: {calc_mis(df_sum.TP,df_sum.FN)}")
print(f"acc: {calc_acc(df_sum.TP,df_sum.TN,df_sum.FP,df_sum.FN)}")
```

```python
# Bootstrap resampling
# 10,000 is better but 1,000 is faster
df_results = bootstrap_resample(df)#,n_bootstrap=10000)
```

```python
# Plot distribution
df_results.hist();
plt.show()
```

```python
#5% percentile
df_results.quantile(0.05)
```

```python
#95% percentile
df_results.quantile(0.95)
```

### bonus: percentage higher performance
we used this for BGD and TCD, so just leaving that here as comparison, but not gonna thinker about it for now

```python
#check how many of the samples have higher performance metrics than the original
#goal is to test if the correlation we see in the data is by chance
#as rule of thumb, if 2.5% of the bootstrapped samples has a higher performance,
#there is a high chance the correlation is due to chance
perc_better_var=round(df_results.loc[
    df_results["var"] > var,'var'].count()/df_results[
    'var'].count()*100,2)
perc_better_det=round(df_results.loc[
    df_results["det"] > det,'det'].count()/df_results[
    'det'].count()*100,2)
print(f"{perc_better_var}% of the bootstrapped samples has better VAR than the original (={round(var,2)})")
print(f"{perc_better_det}% of the bootstrapped samples has better DET than the original (={round(det,2)})")
```

## Compute for all trigger moments
Now that we have seen how it works for one trigger moment, we compute the metrics for all trigger moments
and write them to a dataframe. 

```python
df_trig_met=pd.DataFrame(columns=["trigger","metric","point","value"])
metric_list=["far","var","det","mis","acc"]
for trigger_month in df_all.columns: 
    #reshape to dataframe with one column per indicator
    df=df_all[trigger_month].to_frame()
    for metric in ["FP","TP","TN","FN"]:
        df[metric]=np.where(df[trigger_month]==metric,1,0)
    #get the bootstrap results
    df_results = bootstrap_resample(df,n_bootstrap=10000)
    for metric in metric_list: 
        df_trig_met=df_trig_met.append({"metric":metric,"point":"central","value":df_results[metric].median(),"trigger":trigger_month},ignore_index=True)
        df_trig_met=df_trig_met.append({"metric":metric,"point":"low_end","value":df_results[metric].quantile(0.05),"trigger":trigger_month},ignore_index=True)
        df_trig_met=df_trig_met.append({"metric":metric,"point":"high_end","value":df_results[metric].quantile(0.95),"trigger":trigger_month},ignore_index=True)
```

```python
df_trig_met
```

```python
df_trig_met.to_csv(output_path,index=False)
```
