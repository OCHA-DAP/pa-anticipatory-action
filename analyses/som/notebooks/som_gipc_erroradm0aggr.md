## Disreperancy Global IPC aggregation methods
For some dates different results are gained when calculating the numbers on admin0 and admin1 level in different way. Since the numbers are directly reported in the excel sheet on admin0 level we can use those. Simeltaneously we could sum all the numbers reported on admin2 level and this should give the same results. However, they don't match. This notebook does a short exploration of where the differences occur

```python
import pandas as pd
```

```python
admin_level=0
country="somalia"
```

```python
df=pd.read_excel("../Data/GlobalIPC/somalia_globalipc_newcolumnnames.xlsx",index_col=0)
```

```python
df.loc[:,"date"] = pd.to_datetime(df.loc[:,"date"])
```

```python
df = df[(df["date"].notnull()) & (df[f"ADMIN{admin_level}"].notnull())]
```

```python
df.head()
```

```python
df_agg = df[df["ADMIN0"].str.lower().str.fullmatch(country.lower())].groupby(["ADMIN0","date"],as_index=False).sum()
df_precalc = df[df["ADMIN0"].str.lower().str.match(f"{country.lower()}:")]
df_adm1agg = df[~df["ADMIN0"].str.lower().str.contains(f"{country.lower()}")].groupby("date",as_index=False).sum()
```

```python
df_comb=df_agg.merge(df_precalc,on="date",suffixes=("_agg","_prec")).merge(df_adm1agg.rename(columns={"pop_CS":"pop_CS_adm1"}),on='date')
```

```python
df_comb["pop_CS_diff_aggprec"]=df_comb["pop_CS_agg"]-df_comb["pop_CS_prec"]
df_comb["pop_CS_diff_aggadm1"]=df_comb["pop_CS_agg"]-df_comb["pop_CS_adm1"]
df_comb["pop_CS_diff_precadm1"]=df_comb["pop_CS_prec"]-df_comb["pop_CS_adm1"]
```

```python
df_comb[["date","pop_CS_agg","pop_CS_prec","pop_CS_adm1","pop_CS_diff_aggprec","pop_CS_diff_aggadm1","pop_CS_diff_precadm1"]]
```

### Conclusions on ADM0 level:
For the three methods there is large disperancy on 2017-01, 2017-07, 2018-01, and 2020-09   
On 2020-09 there is a disreperancy between df_adm1agg and the other two. This is due to wrong summing in the raw data of the Woqooyi Galbeed region   
On 2017-07 the population of Mudug is larger in the admin1 numbers given in the sheet compared to summing the admin2 regions within Mudug   
On 2017-01 and 2018-01 magically extra population is added to the national total that isn't present in the sum of the admin1 and admin2's 


### ADMIN1

```python
df_adm1_precalc = df[~df["ADMIN0"].str.lower().str.contains(f"{country.lower()}")]
df_adm1_precalc=df_adm1_precalc.drop("ADMIN1",axis=1)
df_adm1_precalc=df_adm1_precalc.rename(columns={"ADMIN0":"ADMIN1"})
```

```python
admin_level=2
```

```python
df_adm2 = df[(df["date"].notnull()) & (df[f"ADMIN{admin_level}"].notnull())]
```

```python
df_adm1_agg = df_adm2.groupby(["date", "ADMIN1"], dropna=False, as_index=False).sum()
```

```python
df_adm1_agg.head()
```

```python
df_adm1_precalc.equals(df_adm1_agg)
```

```python
df_adm1_precalc.shape
```

```python
df_adm1_agg.shape
```

```python
df_adm1_comb=df_adm1_agg.merge(df_adm1_precalc,on=["date","ADMIN1"],suffixes=("_agg","_prec"))
```

```python
df_adm1_comb["pop_CS_diff"]=df_adm1_comb["pop_CS_agg"]-df_adm1_comb["pop_CS_prec"]
```

```python
df_adm1_comb[df_adm1_comb["date"]=="2017-07-01"][["ADMIN1","pop_CS_agg","pop_CS_prec","pop_CS_diff"]]
```

```python
df_adm1_comb[df_adm1_comb["date"]=="2020-09-01"][["ADMIN1","pop_CS_agg","pop_CS_prec","pop_CS_diff"]]
```

```python
df_adm1_comb[df_adm1_comb["ADMIN1"]=="Woqooyi Galbeed"][["ADMIN1","pop_CS_agg","pop_CS_prec","pop_CS_diff"]]
```

```python
for i in range(1,6):
    df_adm1_comb[f"CS_{i}_diff"]=df_adm1_comb[f"CS_{i}_agg"]-df_adm1_comb[f"CS_{i}_prec"]
```

```python
df_adm1_comb[df_adm1_comb["ADMIN1"]=="Woqooyi Galbeed"][["date"]+["pop_CS_agg","pop_CS_prec","pop_CS_diff"]+[f"CS_{i}_agg" for i in range (1,6)]+[f"CS_{i}_prec" for i in range (1,6)]]
```

```python
df_adm1_comb[df_adm1_comb["ADMIN1"]=="Woqooyi Galbeed"][["date"]+["pop_CS_agg","pop_CS_prec","pop_CS_diff"]+[f"CS_{i}_diff" for i in range (1,6)]]
```

Reported population vs summed population over 5 phases

```python
df_adm1_comb["pop_CS_sum"]=df_adm1_comb[[f"CS_{i}_agg" for i in range(1,6)]].sum(axis=1)
```

```python
df_adm1_comb["pop_CS_sum_diff"]=df_adm1_comb["pop_CS_agg"]-df_adm1_comb["pop_CS_sum"]
```

```python
df_adm1_comb[df_adm1_comb["ADMIN1"]=="Woqooyi Galbeed"][["date","pop_CS_agg","pop_CS_prec","pop_CS_sum","pop_CS_sum_diff"]]
```

```python

```

### ADMIN 2

```python
df=pd.read_excel("../Data/GlobalIPC/somalia_globalipc_newcolumnnames.xlsx",index_col=0)
```

```python
admin_level=2
```

```python
df_adm2 = df[(df["date"].notnull()) & (df[f"ADMIN{admin_level}"].notnull())]
```

```python
df_adm2["pop_CS_sum"]=df_adm2[[f"CS_{i}" for i in range(1,6)]].sum(axis=1)
df_adm2["pop_CS_sum_diff"]=df_adm2["pop_CS"]-df_adm2["pop_CS_sum"]
```

```python
df_adm2[["date","ADMIN1","ADMIN2","pop_CS","pop_CS_sum","pop_CS_sum_diff"]].sort_values(by="pop_CS_sum_diff")
```

```python
df_adm2[df_adm2["ADMIN1"]=="Woqooyi Galbeed"][["date","ADMIN1","ADMIN2","pop_CS","pop_CS_sum","pop_CS_sum_diff"]]
```

```python
df_adm2[(df_adm2["ADMIN1"]=="Woqooyi Galbeed") & (df_adm2["date"]=="2020-01-01")][["ADMIN1","ADMIN2","pop_CS","pop_CS_sum","pop_CS_sum_diff"]]
```

```python
df_adm2[(df_adm2["ADMIN1"]=="Woqooyi Galbeed") & (df_adm2["date"]=="2020-01-01")].pop_CS.sum()
```

```python
df_adm2["CS_sum_agg"]=df_adm1_comb[[f"CS_{i}_agg" for i in range(1,6)]].sum(axis=1)
df_adm2["CS_sum_prec"]=df_adm1_comb[[f"CS_{i}_prec" for i in range(1,6)]].sum(axis=1)
for i in range(1,6):
    df_adm1_comb[f"perc_CS_{i}_agg"]=df_adm1_comb[f"CS_{i}_agg"]/df_adm1_comb[f"CS_sum_agg"]*100
    df_adm1_comb[f"perc_CS_{i}_prec"]=df_adm1_comb[f"CS_{i}_prec"]/df_adm1_comb[f"CS_sum_prec"]*100
    df_adm1_comb[f"perc_CS_{i}_diff"]=df_adm1_comb[f"perc_CS_{i}_agg"]-df_adm1_comb[f"perc_CS_{i}_prec"]
```

```python

```

### percentual differences?

```python
df_adm1_comb["CS_sum_agg"]=df_adm1_comb[[f"CS_{i}_agg" for i in range(1,6)]].sum(axis=1)
df_adm1_comb["CS_sum_prec"]=df_adm1_comb[[f"CS_{i}_prec" for i in range(1,6)]].sum(axis=1)
for i in range(1,6):
    df_adm1_comb[f"perc_CS_{i}_agg"]=df_adm1_comb[f"CS_{i}_agg"]/df_adm1_comb[f"CS_sum_agg"]*100
    df_adm1_comb[f"perc_CS_{i}_prec"]=df_adm1_comb[f"CS_{i}_prec"]/df_adm1_comb[f"CS_sum_prec"]*100
    df_adm1_comb[f"perc_CS_{i}_diff"]=df_adm1_comb[f"perc_CS_{i}_agg"]-df_adm1_comb[f"perc_CS_{i}_prec"]

df_adm1_comb[f"perc_CS_3p_agg"]=df_adm1_comb[[f"CS_{i}_agg" for i in range(3,6)]].sum(axis=1)/df_adm1_comb[f"CS_sum_agg"]*100
df_adm1_comb[f"perc_CS_3p_prec"]=df_adm1_comb[[f"CS_{i}_prec" for i in range(3,6)]].sum(axis=1)/df_adm1_comb[f"CS_sum_prec"]*100
df_adm1_comb[f"perc_CS_3p_diff"]=df_adm1_comb[f"perc_CS_3p_agg"]-df_adm1_comb[f"perc_CS_3p_prec"]
```

```python
df_adm1_comb
```

```python
df_adm1_comb[["date","ADMIN1","pop_CS_agg","perc_CS_1_agg","perc_CS_1_prec","perc_CS_1_diff"]].sort_values(by="perc_CS_1_diff")
```

```python
df_adm1_comb[["date","ADMIN1","pop_CS_agg","perc_CS_3p_agg","perc_CS_3p_prec","perc_CS_3p_diff"]].sort_values(by="perc_CS_3p_diff")
```

```python
df_adm1_comb[df_adm1_comb["date"]=="2020-09-01"][["ADMIN1","pop_CS_agg","perc_CS_1_agg","perc_CS_1_prec"]]
```

```python

```
