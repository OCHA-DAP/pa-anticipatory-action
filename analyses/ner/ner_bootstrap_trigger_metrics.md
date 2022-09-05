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
trigger_perf_path=country_data_exploration_dir/"trigger_performance"
input_path=trigger_perf_path/"historical_activations_trigger_v1.csv"
```

```python
def bootstrap_resample(df, n_bootstrap=1_000):
    df_bootstrap = pd.DataFrame()
    for i in range(n_bootstrap):
        df_rs = df.sample(frac=1, replace=True, 
                        random_state=rng.bit_generator).sum()
        df_bootstrap = df_bootstrap.append({
          "far": calc_far(df_rs.TP,df_rs.FP),
          "var": calc_var(df_rs.TP,df_rs.FP),
          "det": calc_det(df_rs.TP,df_rs.FN),
            "mis": calc_mis(df_rs.TP,df_rs.FN),
            "acc": calc_acc(df_rs.TP,df_rs.TN,df_rs.FP,df_rs.FN),
            "atv": calc_atv(df_rs.TP,df_rs.TN,df_rs.FP,df_rs.FN),
          'nTP': df_rs.TP.sum(),
          'nFP': df_rs.FP.sum(),
          'nFN': df_rs.FN.sum()
      }, ignore_index=True)
    df_bootstrap = df_bootstrap.dropna()
    return df_bootstrap

def original_performance_metrics(df):
    df_bootstrap = pd.DataFrame()
    df_bootstrap = df_bootstrap.append({
      "far": calc_far(df.TP,df.FP),
      "var": calc_var(df.TP,df.FP),
      "det": calc_det(df.TP,df.FN),
        "mis": calc_mis(df.TP,df.FN),
        "acc": calc_acc(df.TP,df.TN,df.FP,df.FN),
        "atv": calc_atv(df.TP,df.TN,df.FP,df.FN),
      }, ignore_index=True)
    df_bootstrap = df_bootstrap.dropna()
    return df_bootstrap

def calc_far(TP,FP):
    return FP/(TP+FP)

def calc_var(TP,FP):
    return TP/(TP+FP)

def calc_det(TP,FN):
    return TP/(TP+FN)

def calc_mis(TP,FN):
    return FN/(TP+FN)

def calc_acc(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)

def calc_atv(TP,TN,FP,FN):
    return (TP+FP)/(TP+TN+FP+FN)
```

## Load data

```python
#load data of when trigger activated for each month
df_all=pd.read_csv(input_path,index_col=0)
```

```python
#change df_all to more machine usable format
df_all=df_all.loc[:,df_all.columns.str.endswith(".1")]
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
print(f"atv: {calc_atv(df_sum.TP,df_sum.TN,df_sum.FP,df_sum.FN)}")
```

```python
# Bootstrap resampling
# 10,000 is better but 1,000 is faster
df_bootstrap = bootstrap_resample(df)
```

```python
# Plot distribution
df_bootstrap.hist();
plt.show()
```

```python
#5% percentile
df_bootstrap.quantile(0.05)
```

```python
#95% percentile
df_bootstrap.quantile(0.95)
```

### bonus: percentage higher performance
we used this for BGD and TCD, so just leaving that here as comparison, but not gonna thinker about it for now

```python
#check how many of the samples have higher performance metrics than the original
#goal is to test if the correlation we see in the data is by chance
#as rule of thumb, if 2.5% of the bootstrapped samples has a higher performance,
#there is a high chance the correlation is due to chance
perc_better_var=round(df_bootstrap.loc[
    df_bootstrap["var"] > var,'var'].count()/df_bootstrap[
    'var'].count()*100,2)
perc_better_det=round(df_bootstrap.loc[
    df_bootstrap["det"] > det,'det'].count()/df_bootstrap[
    'det'].count()*100,2)
print(f"{perc_better_var}% of the bootstrapped samples has better VAR than the original (={round(var,2)})")
print(f"{perc_better_det}% of the bootstrapped samples has better DET than the original (={round(det,2)})")
```

## Compute for all trigger moments
Now that we have seen how it works for one trigger moment, we compute the metrics for all trigger moments
and write them to a dataframe. 

We also have two trigger moments named "min" and "ful". 
"min" refers to that any of the funding packages is released. 
"ful" refers to that all funding is released, i.e. all packages are released. 
This doesn't apply for all triggers, but for Niger it does. 

It thus means that if EITHER trigger 1, trigger 2, or trigger 3 is met, this is counted as "min" being positive. 

For "ful" the logic is that if EITHER trigger 1 AND trigger 2, OR trigger 1 and trigger 3 are met, there is a full activation. 

For "min" and "ful" only the "atv" are reported

```python
def comp_error_types(df,col_obs,col_act):
    df.loc[:,"TP"]=np.where((df[col_obs])&(df[col_act]),1,0)
    df.loc[:,"FP"]=np.where((~df[col_obs])&(df[col_act]),1,0)
    df.loc[:,"TN"]=np.where((~df[col_obs])&(~df[col_act]),1,0)
    df.loc[:,"FN"]=np.where((df[col_obs])&(~df[col_act]),1,0)
    return df
```

```python
def comp_bootstrap_bounds(df,metric_list, confidence_interval, trigger_name, n_bootstrap,metric_name_dict=None):
    df_bootstrap_bounds=pd.DataFrame(columns=["trigger","metric","point","value"])
    #get the bootstrap results
    df_bootstrap = bootstrap_resample(df,n_bootstrap=n_bootstrap)
    df_orig_perf_metrics=original_performance_metrics(df.sum())
    for metric in metric_list:
        if metric_name_dict is None:
            metric_name=metric
        else:
            metric_name=metric_name_dict[metric]
        df_bootstrap_bounds=df_bootstrap_bounds.append({"metric":metric_name,"point":"central","value":df_orig_perf_metrics[metric].values[0],"trigger":trigger_name},ignore_index=True)
        df_bootstrap_bounds=df_bootstrap_bounds.append({"metric":metric_name,"point":"low_end","value":df_bootstrap[metric].quantile((1-confidence_interval)/2),"trigger":trigger_name},ignore_index=True)
        df_bootstrap_bounds=df_bootstrap_bounds.append({"metric":metric_name,"point":"high_end","value":df_bootstrap[metric].quantile(1-((1-confidence_interval)/2)),"trigger":trigger_name},ignore_index=True)
    return df_bootstrap_bounds
```

```python
save_file=True
n_bootstrap=10000
metric_list=["far","var","det","mis","acc","atv"]
#column names over which all metrics should be computed
trigger_name_points=['August', 'June', 'May', 'Apr', 'Mar', 'Feb', 'Jan', 'Trigger3', 'Trigger2', 'Trigger1']
#column names that if one of them triggers the "min" of the "Full framework" is reached
periods_min=["Trigger1","Trigger2","Trigger3"]
#column names that if both within one list trigger the "full" of the "Full framework" is reached
#there is an OR between the lists
periods_max=[["Trigger1","Trigger2"],["Trigger1","Trigger3"]]
for confidence_interval in [0.8,0.9,0.95,0.99]:
    print((1-confidence_interval)/2)
    print(1-((1-confidence_interval)/2))
    df_trig_met=pd.DataFrame(columns=["trigger","metric","point","value"])
    for trigger_name in trigger_name_points: 
        #reshape to dataframe with one column per indicator
        df=df_all[trigger_name].to_frame()
        for metric in ["FP","TP","TN","FN"]:
            df.loc[:,metric]=np.where(df[trigger_name]==metric,1,0)
        #get the bootstrap results 
        df_bootstrap_bounds = comp_bootstrap_bounds(df=df,metric_list=metric_list,confidence_interval=confidence_interval,trigger_name=trigger_name,n_bootstrap=n_bootstrap)
        df_trig_met=pd.concat([df_trig_met,df_bootstrap_bounds])
    
    if periods_min is not None:
        df_min=df_act_num[["bad-year"]+periods_min]
        df_min.loc[:,"activation"]=df_act_num[periods_min].max(axis=1)
        df_min=comp_error_types(df_min,col_obs="bad-year",col_act="activation")
        df_bootstrap_bounds = comp_bootstrap_bounds(
            df=df_min,metric_list=["atv"],confidence_interval=confidence_interval,
            trigger_name="framework",n_bootstrap=n_bootstrap,metric_name_dict={"atv":"min"})
        df_trig_met=pd.concat([df_trig_met,df_bootstrap_bounds])
    
    if periods_max is not None:
        df_max=df_act_num[["bad-year"]+[trig for trig_com in periods_max for trig in trig_com]]
        df_max.loc[:,"activation"]=np.where((df_act_num[periods_max[0]].all(axis=1))|(df_act_num[periods_max[1]].all(axis=1)),True,False)
        df_max=comp_error_types(df_max,col_obs="bad-year",col_act="activation")
        df_bootstrap_bounds = comp_bootstrap_bounds(
            df=df_max,metric_list=["atv"],confidence_interval=confidence_interval,
            trigger_name="framework",n_bootstrap=n_bootstrap,metric_name_dict={"atv":"ful"})
        df_trig_met=pd.concat([df_trig_met,df_bootstrap_bounds])
    
    if save_file: 
        df_trig_met["dummy"]=1
        output_filename=f"ner_perf_metrics_table_ci_{confidence_interval}.csv"
        df_trig_met.to_csv(trigger_perf_path/output_filename,index=False)
```

```python
df_trig_met
```

### Example: Aggregate months

There are three funding packages: from Jan-March, from April-Jun, and in August. 
If there is an activation during one of those period, that whole packages is released. 
And the trigger cannot be reached anymore during the other months of that "package". 

We thus want to know the performance over those three "packages". 
In the Niger file, this was already added manually. However, for the futre below an example of how
the aggregation of trigger points can also be computed automatically. 

```python
#load data of when trigger activated for each month
df_all=pd.read_csv(input_path,index_col=0)
```

```python
#change df_all to more machine usable format
df_act=df_all.loc[:,~df_all.columns.str.contains(".1")]
df_act=df_act.rename(columns={"all-years":"bad-year"})
df_act=df_act.iloc[1:]
df_act.index.rename("year",inplace=True)
df_act.index=df_act.index.astype(int)
```

```python
#1 indicates activation or observed bad year (i.e. wanted to activate)
#0 no activation/no bad year observed
df_act_num=df_act.replace("bad",True).replace("yes",True).replace(np.nan,False)
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
