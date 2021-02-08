#!/usr/bin/env python
# coding: utf-8

# ### Trigger mechanism for Somalia 
# 
# IPC trigger design as endorsed early 2020 (not clearly documented but was endorsed as using ML1 forecasts):
# 
# - The projected national population in Phase 3 and above exceed 20%, AND 
# - (The national population in Phase 3 is projected to increase by 5 percentage points, OR 
# - The projected national population in Phase 4 or above is 2.5%)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


country="Somalia"


# load world-pop-weighted fewsnet data
# note CS_99 denotes missing values
df_fadm=pd.read_csv(f"../Data/FewsNetWorldPop/somalia_fewsnet_worldpop_admin1.csv",index_col=0)
adm1c="ADMIN1" 
admc="ADMIN1" #"admin1Name"


# extract year and month from date
df_fadm["date"] = pd.to_datetime(df_fadm["date"])
df_fadm["year"] = df_fadm["date"].dt.year
df_fadm["month"] = df_fadm["date"].dt.month


df_fadm


# list column names
df_fadm.columns    


# ### National trigger
# 

# compute national totals
df_ntl = df_fadm.drop(['year', 'month'], axis=1).groupby(by='date', axis=0, as_index = False).sum()
df_ntl.head(10)


def add_percentages(df):
    # calculate percentage of population per period and phase
    for period in ["CS", "ML1", "ML2"]:
        # IPC phases goes up to 5, so define range up to 6
        for i in range(1, 6):
            c = f"{period}_{i}"
            df[f"perc_{c}"] = df[c] / df[f"pop_{period}"] * 100
        # get pop and perc in IPC3+ and IPC2-
        # 3p = IPC phase 3 or higher, 4p = IPC phase 4 or higher
        df[f"{period}_3p"] = df[[f"{period}_{i}" for i in range(3, 6)]].sum(axis=1)
        df[f"perc_{period}_3p"] = df[f"{period}_3p"] / df[f"pop_{period}"] * 100
        df[f"{period}_4p"] = df[[f"{period}_{i}" for i in range(4, 6)]].sum(axis=1)
        df[f"perc_{period}_4p"] = df[f"{period}_4p"] / df[f"pop_{period}"] * 100
    df["perc_inc_ML2_3p"] = df["perc_ML2_3p"] - df["perc_CS_3p"]
    df["perc_inc_ML1_3p"] = df["perc_ML1_3p"] - df["perc_CS_3p"]
    return df


df_ntl=add_percentages(df_ntl)
df_ntl.head()


#Trigger definition:
#The projected population in phase 3 and above exceed 20%, AND
#(The projected population in phase 3 is projected to increase by a further 5%, OR
#The projected population in phase 4 or above is 2.5%)

def get_national_abs_criterion(row, period, phase, threshold):
    """
    Return 1 if percentage of population in row for period in phase "phase" or higher, equals or larger than. 
    Threshold should NOT be a decimal (ie 5 for 5 percent, not .05) 
    """
    # range till 6 cause 5 is max phase
    cols = [f"perc_{period}_{l}" for l in range(phase, 6)]
    if np.isnan(row[f"pop_{period}"]):
        return np.nan
    if round(row[cols].sum()) >= threshold:
        return 1
    else:
        return 0
    
def get_national_increase_criterion(row, period, phase, threshold):
    """
    Return 1 if for row percentage in >="phase" projected at Period minus percentage currently (CS) in >="phase" is expected to be larger than Threshold
    For Global IPC the population analysed in ML2 is sometimes different than in CS. That is why we work directly with percentages and not anymore with (pop period phase+ - pop CS phase+) / pop CS
    Threshold should NOT be a decimal (ie 5 for 5 percent, not .05) 
    """
    # range till 6 cause 5 is max phase
    cols__ml = [f"perc_{period}_{l}" for l in range(phase, 6)]
    cols__cs = [f"perc_CS_{l}" for l in range(phase, 6)]
    if row[["pop_CS", f"pop_{period}"]].isnull().values.any():
        return np.nan
    if row[cols__ml].sum() == 0:
        return 0
    if round(row[cols__ml].sum() - row[cols__cs].sum()) >= threshold:
        return 1
    else:
        return 0    


# apply criteria. Returns 1 if criterion is met.

df_ntl["natl_criterion_ML1_3_20"] = df_ntl.apply(lambda x: get_national_abs_criterion(x,"ML1",3,20),axis=1)
df_ntl["natl_criterion_ML1_3_5in"] = df_ntl.apply(lambda x: get_national_increase_criterion(x,"ML1",3,5),axis=1)
df_ntl["natl_criterion_ML1_4_2half"] = df_ntl.apply(lambda x: get_national_abs_criterion(x,"ML1",4,2.5),axis=1)

df_ntl["natl_criterion_ML2_3_20"] = df_ntl.apply(lambda x: get_national_abs_criterion(x,"ML2",3,20),axis=1)
df_ntl["natl_criterion_ML2_3_5in"] = df_ntl.apply(lambda x: get_national_increase_criterion(x,"ML2",3,5),axis=1)
df_ntl["natl_criterion_ML2_4_2half"] = df_ntl.apply(lambda x: get_national_abs_criterion(x,"ML2",4,2.5),axis=1)


# determine whether national trigger is met

df_ntl['national_trigger_ML1'] =  np.where((df_ntl['natl_criterion_ML1_3_20']==1) & ((df_ntl['natl_criterion_ML1_3_5in'] )==1 | (df_ntl['natl_criterion_ML1_4_2half'] == 1)), 1, 0)
df_ntl['national_trigger_ML2'] =  np.where((df_ntl['natl_criterion_ML2_3_20']==1) & ((df_ntl['natl_criterion_ML2_3_5in'] )==1 | (df_ntl['natl_criterion_ML2_4_2half'] == 1)), 1, 0)


# extract year / month per row

df_ntl["date"] = pd.to_datetime(df_ntl["date"])
df_ntl["year"] = df_ntl["date"].dt.year
df_ntl["month"] = df_ntl["date"].dt.month
df_ntl['adm0c'] = country


# list years / months during which national trigger would have been met

national_activations_ML1 = df_ntl.loc[(df_ntl["national_trigger_ML1"] == 1)]
national_activations_ML1['period'] = 'ML1'
national_activations_ML1['adm0c'] = country

national_activations_ML2 = df_ntl.loc[(df_ntl["national_trigger_ML2"] == 1)]
national_activations_ML2['period'] = 'ML2'
national_activations_ML2['adm0c'] = country

activation_frames = [national_activations_ML1, national_activations_ML2]
national_activations = pd.concat(activation_frames)

display(national_activations_ML1.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))
display(national_activations_ML2.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML2_3p','perc_ML2_4p'].agg(lambda x: list(x)))


# create dictionary of past activations by ML1

dict_natl_activ={}
dict_natl_activ["past_ML1"]={"df": national_activations_ML1,
                  "trig_cols":["CS_3p","ML1_3p","ML1_4p"],
                  "desc":"At least 20% of ADMIN1 population in IPC3+ at ML1 AND (increase by 5 percentage points in ADMIN1 pop. projected in IPC3+ compared to current state OR At least 2.5% of ADMIN1 population projected at IPC4+ by ML1)"}

dict_natl_activ["past_ML2"]={"df": national_activations_ML2,
                  "trig_cols":["CS_3p","ML2_3p","ML2_4p"],
                  "desc":"At least 20% of ADMIN1 population in IPC3+ at ML2 AND (increase by 5 percentage points in ADMIN1 pop. projected in IPC3+ compared to current state OR At least 2.5% of ADMIN1 population projected at IPC4+ by ML2)"}

dict_natl_activ


# function to plot years during which national trigger would have been met (regardless of month or number of activations)

def plot_natl_trig(dict_natl_trigger, adm0c="admin0Name", shape_path="../Data/Shapefiles/som_adm_undp_shp/Som_Admbnda_Adm0_UNDP.shp"):
    gdf = gpd.read_file(shape_path)

    count = 1
    f, ax = plt.subplots(figsize=(12,12))
    for d in range(2009,2021):
        ax2 = plt.subplot(4, 4, count)
        gdf.plot(ax=ax2, color='#DDDDDD', edgecolor='#BBBBBB')
        regions = dict_natl_trigger['adm0c'].loc[dict_natl_trigger['year']==d]
        if len(regions) > 0:
            gdf.loc[gdf[adm0c].isin(regions)].plot(ax=ax2, color='red')
        plt.title(f"Past triggers {d}")
        count+=1
        ax2.axis("off")
    plt.show()
    


# plot past activations of national trigger
print("ML1 projections")
plot_natl_trig(dict_natl_activ["past_ML1"]["df"])

print("ML2 projections")
plot_natl_trig(dict_natl_activ["past_ML2"]["df"])


# define function to plot percentage of population in IPC3+ or IPC4+

def plot_aff_dates(df_d,df_trig,col,shape_path="../Data/Shapefiles/som_adm_undp_shp/Som_Admbnda_Adm0_UNDP.shp",title=None):
    
    num_dates=len(df_trig.date.unique())
    colp_num=2
    rows=num_dates // colp_num
    rows+=num_dates % colp_num
    position = range(1, num_dates + 1)

    gdf = gpd.read_file(shape_path)
    df_geo=gdf[["admin0Name","geometry"]].merge(df_d,left_on="admin0Name",right_on="adm0c",how="left")
    
    colors = len(df_geo[col].unique())
    cmap = 'Blues'
    figsize = (16, 10)
    scheme = "natural_breaks" #equal_interval
    fig = plt.figure(1,figsize=(16,6*rows))
    
    for i,c in enumerate(df_trig.date.unique()):
        ax = fig.add_subplot(rows,colp_num,position[i])
        df_date=df_geo[df_geo.date==c]
        if df_date[col].isnull().values.all():
            print(f"No not-NaN values for {c}")
        elif df_date[col].isnull().values.any():
            df_geo[df_geo.date==c].plot(col, ax=ax,cmap=cmap, figsize=figsize, k = colors,  legend=True,scheme=scheme,missing_kwds={"color": "lightgrey", "edgecolor": "red",
   "hatch": "///",
    "label": "Missing values"})
        else:
            df_geo[df_geo.date==c].plot(col, ax=ax,cmap=cmap, figsize=figsize, k = colors,  legend=True,scheme=scheme)
        gdf.boundary.plot(linewidth=0.2,ax=ax)

        ax.axis("off")
        
        plt.title(pd.DatetimeIndex([c])[0].to_period('M'))
    if title:
        fig.suptitle(title,fontsize=14, y=0.92)
    plt.show()


plot_aff_dates(df_ntl, 
               dict_natl_activ["past_ML1"]["df"],"perc_ML1_3p",
               title="Percentage of population projected in IPC3+ in ML1 for the dates the trigger is met")


# plot percentage of country in IPC3+ or IPC4+ regardless of trigger status (met or not)

df_ntl['adm0c'] = country

plot_aff_dates(df_ntl, 
               dict_natl_activ["past_ML1"]["df"],"perc_ML1_3p",
               title="Percentage of population projected in IPC3+ in ML1 for the dates the trigger is met")

plot_aff_dates(df_ntl, 
               dict_natl_activ["past_ML1"]["df"],"perc_ML1_4p",
               title="Percentage of population projected in IPC4+ in ML1 for the dates the trigger is met")

plot_aff_dates(df_ntl, 
               dict_natl_activ["past_ML2"]["df"],"perc_ML2_3p",
               title="Percentage of population projected in IPC3+ in ML2 for the dates the trigger is met")

plot_aff_dates(df_ntl, 
               dict_natl_activ["past_ML2"]["df"],"perc_ML2_4",
               title="Percentage of population projected in IPC4+ in ML2 for the dates the trigger is met")


# print percentage of country in IPC3+ or IPC4+ regardless of trigger status

df_ntl[['date', 'perc_ML1_3p', 'perc_ML1_4p']]


# ### Subnational Trigger

# - The projected regional population in Phase 3 and above exceed 20%, AND 
# (The regional population in Phase 3 is projected to increase by 5 percentage points OR 
# The projected regional population in Phase 4 or above is 2.5%)
# 
# ML1 or ML2 projections not specified. Looking at both.

# regions that have been or were forecasted to be IPC 5
print("CS 5", df_fadm.CS_5.unique())
print("ML1 5", df_fadm.ML1_5.unique())


def add_columns(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # calculate percentage of population per analysis period and level
    for period in ["CS", "ML1", "ML2"]:
        # IPC level goes up to 5, so define range up to 6
        for i in range(1, 6):
            c = f"{period}_{i}"
            df[f"perc_{c}"] = df[c] / df[f"pop_{period}"] * 100
        # get pop and perc in IPC3+ and IPC2-
        # 3p = IPC level 3 or higher, 2m = IPC level 2 or lower
        df[f"{period}_3p"] = df[[f"{period}_{i}" for i in range(3, 6)]].sum(axis=1)
        df[f"perc_{period}_3p"] = df[f"{period}_3p"] / df[f"pop_{period}"] * 100
        df[f"{period}_4p"] = df[[f"{period}_{i}" for i in range(4, 6)]].sum(axis=1)
        df[f"perc_{period}_4p"] = df[f"{period}_4p"] / df[f"pop_{period}"] * 100
        df[f"{period}_2m"] = df[[f"{period}_{i}" for i in range(1, 3)]].sum(axis=1)
        df[f"perc_{period}_2m"] = df[f"{period}_2m"] / df[f"pop_{period}"] * 100
    df["perc_inc_ML2_3p"] = df["perc_ML2_3p"] - df["perc_CS_3p"]
    df["perc_inc_ML1_3p"] = df["perc_ML1_3p"] - df["perc_CS_3p"]
    return df


df_fadm=add_columns(df_fadm)
df_fadm.head()


def get_trigger(row, period, phase, threshold):
    """
    Return 1 if percentage of population in row for period in phase "phase" or higher, equals or larger than. 
    Threshold should NOT be a decimal (ie 5 for 5 percent, not .05) 
    """
    # range till 6 cause 5 is max phase
    cols = [f"perc_{period}_{l}" for l in range(phase, 6)]
    if np.isnan(row[f"pop_{period}"]):
        return np.nan
#    if round(row[cols].sum()/row[f"pop_{period}"]*100) >= threshold:
    if round(row[cols].sum()) >= threshold:
        return 1
    else:
        return 0


#def get_trigger_increase_rel(row, phase, threshold):
#    """
#    Return 1 if population in row for >="phase" at ML1 is expected to be larger than (current (CS) population in >=phase) * (1+(/100))
#    """
#    # range till 6 cause 5 is max phase
#    cols_ml1 = [f"ML1_{l}" for l in range(phase, 6)]
#    cols_cs = [f"CS_{l}" for l in range(phase, 6)]
#    if row[["pop_CS", "pop_ML1"]].isnull().values.any():
#        return np.nan
#    elif row[cols_ml1].sum() == 0:
#        return 0
#    elif row[cols_ml1].sum() > 0 and row[cols_cs].sum() == 0:
#       return 1
#    elif round((row[cols_ml1].sum() - row[cols_cs].sum())/row[cols_cs].sum() * 100) >= threshold:
#        return 1
#    else:
#        return 0
    
def get_trigger_increase(row, period, phase, threshold):
    """
    Return 1 if for row percentage in >="phase" at period minus percentage in >="phase" currently (CS) is expected to be larger than threshold
    For Global IPC the population analysed in ML2 is sometimes different than in CS. That is why we work directly with percentages and not anymore with (pop period phase+ - pop CS phase+) / pop CS
    Threshold should NOT be a decimal (ie 5 for 5 percent, not .05) 
    """
    # range till 6 cause 5 is max phase
    cols__ml = [f"perc_{period}_{l}" for l in range(phase, 6)]
    cols__cs = [f"perc_CS_{l}" for l in range(phase, 6)]
    if row[["pop_CS", f"pop_{period}"]].isnull().values.any():
        return np.nan
    if row[cols__ml].sum() == 0:
        return 0
    if round(row[cols__ml].sum() - row[cols__cs].sum()) >= threshold:
        return 1
    else:
        return 0


#display most recent numbers
df_fadm.loc[df_fadm.date==df_fadm.date.max(),["date",
                                              "year",
                                              "month",
                                              "ADMIN1",
                                              "perc_CS_3p",
                                              "perc_CS_4p",
                                              "perc_ML1_3p",
                                              "perc_ML1_4p",
                                              "perc_ML2_3p",
                                              "perc_ML2_4p"]]


#Column value for row will be 1 if threshold is met and 0 if it isnt
#The projected population in phase 3 and above exceed 20%, AND 
#(The projected population in phase 3 is projected to increase by a further 5%, OR
#The projected population in phase 4 or above is 2.5%)

df_fadm["trigger_ML1_3_20"]=df_fadm.apply(lambda x: get_trigger(x,"ML1",3,20),axis=1)
df_fadm["trigger_ML1_3_5ir"]=df_fadm.apply(lambda x: get_trigger_increase(x,"ML1",3,5),axis=1)
df_fadm["trigger_ML1_4_2half"]=df_fadm.apply(lambda x: get_trigger(x,"ML1",4,2.5),axis=1)

df_fadm["trigger_ML2_3_20"]=df_fadm.apply(lambda x: get_trigger(x,"ML2",3,20),axis=1)
df_fadm["trigger_ML2_3_5ir"]=df_fadm.apply(lambda x: get_trigger_increase(x,"ML2",3,5),axis=1)
df_fadm["trigger_ML2_4_2half"]=df_fadm.apply(lambda x: get_trigger(x,"ML2",4,2.5),axis=1)


#analyse endorsed trigger applied at subnational level
subnatl_activations_ML1=df_fadm.loc[(df_fadm["trigger_ML1_3_20"]==1) & ((df_fadm["trigger_ML1_3_5ir"]==1) | (df_fadm["trigger_ML1_4_2half"]==1))]
subnatl_activations_ML2=df_fadm.loc[(df_fadm["trigger_ML2_3_20"]==1) & ((df_fadm["trigger_ML2_3_5ir"]==1) | (df_fadm["trigger_ML2_4_2half"]==1))]

display(subnatl_activations_ML1.groupby(['year', 'month'], as_index=False)[admc,'perc_CS_3p','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))
display(subnatl_activations_ML2.groupby(['year', 'month'], as_index=False)[admc,'perc_CS_3p','perc_ML2_3p','perc_ML2_4p'].agg(lambda x: list(x)))


# full list of regions activated per month/year

pd.set_option('display.max_colwidth', -1) # remove max column width for ADMIN1 to print fully
subnatl_activations_ML1.groupby(['year', 'month'], as_index=False)[admc].agg(lambda x: list(x))


# list activations per region

display(subnatl_activations_ML1.groupby(['ADMIN1'], as_index=False)[['date']].agg(lambda x: list(x)))
display(subnatl_activations_ML1.groupby(['ADMIN1'], as_index=False)[['date']].agg(lambda x: len(x)))

pd.reset_option('display.max_colwidth') # re-instate max column width


# create dict with activations based on ML1 and ML2
dict_subnatl_activ={}


dict_subnatl_activ["subnatl_ML1"]={"df": subnatl_activations_ML1,
                                   "trig_cols":["CS_3p","ML1_3p","ML1_4p"],
                                   "desc":"At least 20% of ADMIN1 population in IPC3+ at ML1 AND (increase by 5 percentage points in ADMIN1 pop. projected in IPC3+ compared to current state OR at least 2.5% of ADMIN1 population projected at IPC4+ by ML1)"}

dict_subnatl_activ["subnatl_ML2"]={"df": subnatl_activations_ML2,
                                   "trig_cols":["CS_3p","ML1_3p","ML1_4p"],
                                   "desc":"At least 20% of ADMIN1 population in IPC3+ at ML2 AND (increase by 5 percentage points in ADMIN1 pop. projected in IPC3+ compared to current state OR at least 2.5% of ADMIN1 population projected at IPC4+ by ML2)"}

dict_subnatl_activ


# ### FewsNet, plotting characteristics of the trigger

def plot_regions_trig(df_trig,adm0c="admin0Name",adm1c="admin1Name",shape_path="../Data/Shapefiles/som_adm_undp_shp/Som_Admbnda_Adm1_UNDP.shp"):
    gdf = gpd.read_file(shape_path)

    count = 1
    f, ax = plt.subplots(figsize=(12,12))
    for d in range(2009,2021):
        ax2 = plt.subplot(4, 4, count)
        gdf.plot(ax=ax2, color='#DDDDDD', edgecolor='#BBBBBB')
        regions = df_trig['ADMIN1'].loc[df_trig['year']==d]
        if len(regions) > 0:
            gdf.loc[gdf[adm1c].isin(regions)].plot(ax=ax2, color='red')
        plt.title(f"Regions triggered {d}")
        count+=1
        ax2.axis("off")
    plt.show()


plot_regions_trig(dict_subnatl_activ["subnatl_ML1"]["df"])


def plot_aff_dates(df_d,df_trig,col,shape_path="../Data/Shapefiles/som_adm_undp_shp/Som_Admbnda_Adm1_UNDP.shp",title=None,predef_bins=None):
    """
    Plot the values of "col" for the dates present in df_trig
    If giving predef_bins then the data will be colored according to the bins, else a different colour will be assigned to each unique value in the data for each date
    df_d: DataFrame containing all the data of all regions
    df_trig: DataFrame containing the dates for which plots should be shown (generally those dates that the trigger is met)
    col: string with column to plot
    shape_path: relative path to the admin1 shapefile
    title: string with title of whole figure (so not the subplots)
    predef_bins: list with bin values
    """
    
    num_plots=len(df_trig.date.unique())
    colp_num=2
    rows=num_plots // colp_num
    rows+=num_plots % colp_num
    position = range(1, num_plots + 1)

    gdf = gpd.read_file(shape_path)
    df_geo=gdf[["admin1Name","geometry"]].merge(df_d,left_on="admin1Name",right_on="ADMIN1",how="left")
    
    cmap = 'YlOrRd' #'Blues'
    if predef_bins is not None:
        scheme = None 
        norm2 = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    else:
        scheme="natural_breaks"
        norm2=None 
    
    figsize = (16, 10)
    fig = plt.figure(1,figsize=(16,6*rows))
    
    for i,c in enumerate(df_trig.date.unique()):
        ax = fig.add_subplot(rows,colp_num,position[i])
        
        if predef_bins is None:
            colors = len(df_geo[col].unique())
        else:
            colors=None
        
        df_date=df_geo[df_geo.date==c]
        if df_date[col].isnull().values.all():
            print(f"No not-NaN values for {c}")
        elif df_date[col].isnull().values.any():
            df_date.plot(col, ax=ax,cmap=cmap, figsize=figsize, k = colors, norm=norm2, legend=True,scheme=scheme,missing_kwds={"color": "lightgrey", "edgecolor": "red",
   "hatch": "///",
    "label": "Missing values"})
        else:
            df_date.plot(col, ax=ax,cmap=cmap, figsize=figsize, k = colors, norm=norm2, legend=True,scheme=scheme)
        df_geo.boundary.plot(linewidth=0.2,ax=ax)

        ax.axis("off")
        
        
        if predef_bins is None and not df_date[col].isnull().values.all():
            leg = ax.get_legend()

            for lbl in leg.get_texts():
                label_text = lbl.get_text()
                upper = label_text.split(",")[-1].rstrip(']')

                try:
                    new_text = f'{float(upper):,.2f}'
                except:
                    new_text=upper
                lbl.set_text(new_text)
        
        plt.title(pd.DatetimeIndex([c])[0].to_period('M'))
    if title:
        fig.suptitle(title,fontsize=14, y=0.92)
    plt.show()


#end value is not included, so set one higher than max value of last bin
bins=np.arange(0,101,10)
print(bins)


plot_aff_dates(df_fadm, 
               dict_subnatl_activ["subnatl_ML1"]["df"],
               "perc_ML1_3p",
               title="Percentage of population projected in IPC3+ in ML1 for the dates the trigger is met",
               predef_bins=bins)

plot_aff_dates(df_fadm, 
               dict_subnatl_activ["subnatl_ML1"]["df"],
               "perc_ML1_4p",
               title="Percentage of population projected in IPC4+ in ML1 for the dates the trigger is met",
               predef_bins=bins)

plot_aff_dates(df_fadm, 
               dict_subnatl_activ["subnatl_ML2"]["df"],
               "perc_ML2_3p",
               title="Percentage of population projected in IPC3+ in ML2 for the dates the trigger is met",
               predef_bins=bins)

plot_aff_dates(df_fadm, 
               dict_subnatl_activ["subnatl_ML2"]["df"],
               "perc_ML2_4p",
               title="Percentage of population projected in IPC4+ in ML2 for the dates the trigger is met",
               predef_bins=bins)


# ## Trigger analysis Global IPC data
# Besides FewsNet, there is also Global IPC data for Somalia available from 2017 onwards. During the analysis we realized that the aggregated subnational figures don't equal the national figures. Hence, the analysis directly on the national data and the aggregated national data is shown. Moreover, the trigger on subnational level is analyzed

df_gadm=pd.read_csv(f"../Data/GlobalIPCProcessed/{country}_globalipc_admin1.csv",index_col=0)


glob_adm1c="ADMIN1"


df_gadm=add_columns(df_gadm)


df_gadm.head(n=3)


#Column value for row will be 1 if threshold is met and 0 if it isnt
#The projected population in phase 3 and above exceed 20%, AND
#The projected population in phase 3 is projected to increase by a further 5%, OR
#The projected population in phase 4 or above is 2.5%

df_gadm["trigger_ML1_3_20"]=df_gadm.apply(lambda x: get_trigger(x,"ML1",3,20),axis=1)
df_gadm["trigger_ML1_3_5ir"]=df_gadm.apply(lambda x: get_trigger_increase(x,"ML1",3,5),axis=1)
df_gadm["trigger_ML1_4_2half"]=df_gadm.apply(lambda x: get_trigger(x,"ML1",4,2.5),axis=1)

df_gadm["trigger_ML2_3_20"]=df_gadm.apply(lambda x: get_trigger(x,"ML2",3,20),axis=1)
df_gadm["trigger_ML2_3_5ir"]=df_gadm.apply(lambda x: get_trigger_increase(x,"ML2",3,5),axis=1)
df_gadm["trigger_ML2_4_2half"]=df_gadm.apply(lambda x: get_trigger(x,"ML2",4,2.5),axis=1)


#analyse endorsed trigger applied at subnational level
glob_subnatl_activations_ML1=df_gadm.loc[(df_gadm["trigger_ML1_3_20"]==1) & ((df_gadm["trigger_ML1_3_5ir"]==1) | (df_gadm["trigger_ML1_4_2half"]==1))]
glob_subnatl_activations_ML2=df_gadm.loc[(df_gadm["trigger_ML2_3_20"]==1) & ((df_gadm["trigger_ML2_3_5ir"]==1) | (df_gadm["trigger_ML2_4_2half"]==1))]

print("ML1 below")
display(glob_subnatl_activations_ML1.groupby(['year', 'month'], as_index=False)[admc,'perc_CS_3p','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))
print("ML2 below")
display(glob_subnatl_activations_ML2.groupby(['year', 'month'], as_index=False)[admc,'perc_CS_3p','perc_ML2_3p','perc_ML2_4p'].agg(lambda x: list(x)))


# ### Global IPC on national level by aggregating subnational data

df_gntl=df_gadm.groupby("date",as_index=False).sum()


df_gntl=add_percentages(df_gntl)


df_gntl.head(n=3)


# apply criteria. Returns 1 if criterion is met.

df_gntl["natl_criterion_ML1_3_20"] = df_gntl.apply(lambda x: get_national_abs_criterion(x,"ML1",3,20),axis=1)
df_gntl["natl_criterion_ML1_3_5in"] = df_gntl.apply(lambda x: get_national_increase_criterion(x,"ML1",3,5),axis=1)
df_gntl["natl_criterion_ML1_4_2half"] = df_gntl.apply(lambda x: get_national_abs_criterion(x,"ML1",4,2.5),axis=1)

df_gntl["natl_criterion_ML2_3_20"] = df_gntl.apply(lambda x: get_national_abs_criterion(x,"ML2",3,20),axis=1)
df_gntl["natl_criterion_ML2_3_5in"] = df_gntl.apply(lambda x: get_national_increase_criterion(x,"ML2",3,5),axis=1)
df_gntl["natl_criterion_ML2_4_2half"] = df_gntl.apply(lambda x: get_national_abs_criterion(x,"ML2",4,2.5),axis=1)


# determine whether national trigger is met

df_gntl['national_trigger_ML1'] =  np.where((df_gntl['natl_criterion_ML1_3_20']==1) & ((df_gntl['natl_criterion_ML1_3_5in']==1 ) | (df_gntl['natl_criterion_ML1_4_2half'] == 1)), 1, 0)
df_gntl['national_trigger_ML2'] =  np.where((df_gntl['natl_criterion_ML2_3_20']==1) & ((df_gntl['natl_criterion_ML2_3_5in']==1 ) | (df_gntl['natl_criterion_ML2_4_2half'] == 1)), 1, 0)


# extract year / month per row

df_gntl["date"] = pd.to_datetime(df_gntl["date"])
df_gntl["year"] = df_gntl["date"].dt.year
df_gntl["month"] = df_gntl["date"].dt.month


# list years / months during which national trigger would have been met

national_activations_ML1 = df_gntl.loc[(df_gntl["national_trigger_ML1"] == 1)]
national_activations_ML1['period'] = 'ML1'
national_activations_ML1['adm0c'] = country

national_activations_ML2 = df_gntl.loc[(df_gntl["national_trigger_ML2"] == 1)]
national_activations_ML2['period'] = 'ML2'
national_activations_ML2['adm0c'] = country

activation_frames = [national_activations_ML1, national_activations_ML2]
national_activations = pd.concat(activation_frames)

display(national_activations_ML1.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))
display(national_activations_ML2.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML2_3p','perc_ML2_4p'].agg(lambda x: list(x)))


# GlobalIPC on national level, by directly using the reported national numbers

df_gntl_noagg=pd.read_excel(f"../Data/GlobalIPC/{country.lower()}_globalipc_newcolumnnames.xlsx",index_col=0)


df_gntl_noagg.head(n=2)


#names of population totals have been changed in globalipc_newcolumnnames, so map them to old names to make code work
df_gntl_noagg.rename(columns={"reported_pop_CS":"pop_CS","reported_pop_ML1":"pop_ML1","reported_pop_ML2":"pop_ML2"},inplace=True)


#remove rows with nanvalues as date
df_gntl_noagg = df_gntl_noagg[(df_gntl_noagg["date"].notnull()) & (df_gntl_noagg[f"ADMIN0"].notnull())]


#rows with country and then a colon, indicate the numbers on national level
df_gntl_noagg=df_gntl_noagg[df_gntl_noagg['ADMIN0'].str.contains(f"{country}:")]


df_gntl_noagg = add_percentages(df_gntl_noagg)


df_gntl_noagg.head(n=3)


# apply criteria. Returns 1 if criterion is met.

df_gntl_noagg["natl_criterion_ML1_3_20"] = df_gntl_noagg.apply(lambda x: get_national_abs_criterion(x,"ML1",3,20),axis=1)
df_gntl_noagg["natl_criterion_ML1_3_5in"] = df_gntl_noagg.apply(lambda x: get_national_increase_criterion(x,"ML1",3,5),axis=1)
df_gntl_noagg["natl_criterion_ML1_4_2half"] = df_gntl_noagg.apply(lambda x: get_national_abs_criterion(x,"ML1",4,2.5),axis=1)

df_gntl_noagg["natl_criterion_ML2_3_20"] = df_gntl_noagg.apply(lambda x: get_national_abs_criterion(x,"ML2",3,20),axis=1)
df_gntl_noagg["natl_criterion_ML2_3_5in"] = df_gntl_noagg.apply(lambda x: get_national_increase_criterion(x,"ML2",3,5),axis=1)
df_gntl_noagg["natl_criterion_ML2_4_2half"] = df_gntl_noagg.apply(lambda x: get_national_abs_criterion(x,"ML2",4,2.5),axis=1)


# determine whether national trigger is met
df_gntl_noagg['national_trigger_ML1'] =  np.where((df_gntl_noagg['natl_criterion_ML1_3_20']==1) & ((df_gntl_noagg['natl_criterion_ML1_3_5in']==1 ) | (df_gntl_noagg['natl_criterion_ML1_4_2half'] == 1)), 1, 0)
df_gntl_noagg['national_trigger_ML2'] =  np.where((df_gntl_noagg['natl_criterion_ML2_3_20']==1) & ((df_gntl_noagg['natl_criterion_ML2_3_5in']==1 ) | (df_gntl_noagg['natl_criterion_ML2_4_2half'] == 1)), 1, 0)


# extract year / month per row

df_gntl_noagg["date"] = pd.to_datetime(df_gntl_noagg["date"])
df_gntl_noagg["year"] = df_gntl_noagg["date"].dt.year
df_gntl_noagg["month"] = df_gntl_noagg["date"].dt.month


# When using directly national level data, the trigger is also met in July 2017. 

# list years / months during which national trigger would have been met

national_activations_ML1_noagg = df_gntl_noagg.loc[(df_gntl_noagg["national_trigger_ML1"] == 1)]
national_activations_ML1_noagg['period'] = 'ML1'
national_activations_ML1_noagg['adm0c'] = country

national_activations_ML2_noagg = df_gntl_noagg.loc[(df_gntl_noagg["national_trigger_ML2"] == 1)]
national_activations_ML2_noagg['period'] = 'ML2'
national_activations_ML2_noagg['adm0c'] = country

activation_frames_noagg = [national_activations_ML1_noagg, national_activations_ML2_noagg]
national_activations_noagg = pd.concat(activation_frames_noagg)

display(national_activations_ML1_noagg.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))
display(national_activations_ML2_noagg.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML2_3p','perc_ML2_4p'].agg(lambda x: list(x)))


# Analysis of trigger defined in national notebook
# In the past we have also defined the trigger as being 
# - The projected national population in Phase 3 and above exceed 20%, AND  
# - (The national population in Phase 3 is projected to increase relatively by 5%, OR The projected national population in Phase 4 or above is 2.5%)   
# Here we analyze this trigger on national level from the Global IPC data, both from aggregated subnational data and directly from the national data
# 

def get_national_relative_increase_criterion(row, period, phase, threshold):
    """
    Return 1 if for row percentage in >="phase" projected at Period minus percentage currently (CS) in >="phase" is expected to be larger than Threshold
    For Global IPC the population analysed in ML2 is sometimes different than in CS. That is why we work directly with percentages and not anymore with (pop period phase+ - pop CS phase+) / pop CS
    Threshold should NOT be a decimal (ie 5 for 5 percent, not .05) 
    """
    # range till 6 cause 5 is max phase
    cols__ml = [f"{period}_{l}" for l in range(phase, 6)]
    cols__cs = [f"CS_{l}" for l in range(phase, 6)]
    if row[["pop_CS", f"pop_{period}"]].isnull().values.any():
        return np.nan
    elif row[cols__ml].sum() == 0:
        return 0
    elif row[cols__ml].sum() > 0 and row[cols__cs].sum() == 0:
        return 1
    elif round((row[cols__ml].sum() - row[cols__cs].sum())/row[cols__cs].sum() * 100) >= threshold:
        return 1
    else:
        return 0


# determine whether national trigger is met
df_gntl["natl_criterion_ML1_3_5relin"] = df_gntl.apply(lambda x: get_national_relative_increase_criterion(x,"ML1",3,5),axis=1)
df_gntl['national_trigger_springanalysis'] =  np.where((df_gntl['natl_criterion_ML1_3_20']==1) & ((df_gntl['natl_criterion_ML1_3_5relin']==1) | (df_gntl['natl_criterion_ML1_4_2half'] == 1)), 1, 0)
# df_gntl['national_trigger_ML2'] =  np.where((df_gntl['natl_criterion_ML2_3_20'] & df_gntl['natl_criterion_ML2_3_5in'] ) | (df_gntl['natl_criterion_ML2_4_2half'] == 1), 1, 0)


# list years / months during which national trigger would have been met

national_activations_ML1_springanalysis = df_gntl.loc[(df_gntl["national_trigger_springanalysis"] == 1)]
national_activations_ML1_springanalysis['period'] = 'ML1'
national_activations_ML1_springanalysis['adm0c'] = country


display(national_activations_ML1_springanalysis.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))


# determine whether national trigger is met
df_gntl_noagg["natl_criterion_ML1_3_5relin"] = df_gntl_noagg.apply(lambda x: get_national_relative_increase_criterion(x,"ML1",3,5),axis=1)
df_gntl_noagg['national_trigger_springanalysis'] =  np.where((df_gntl_noagg['natl_criterion_ML1_3_20']==1) & ((df_gntl_noagg['natl_criterion_ML1_3_5relin']==1) | (df_gntl_noagg['natl_criterion_ML1_4_2half'] == 1)), 1, 0)


# list years / months during which national trigger would have been met

national_activations_ML1_springanalysis = df_gntl_noagg.loc[(df_gntl_noagg["national_trigger_springanalysis"] == 1)]
national_activations_ML1_springanalysis['period'] = 'ML1'
national_activations_ML1_springanalysis['adm0c'] = country


display(national_activations_ML1_springanalysis.round(2).groupby(['year', 'month'], as_index=False)['period','perc_CS_3p','perc_CS_4','perc_ML1_3p','perc_ML1_4p'].agg(lambda x: list(x)))




