import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

Water_threshold=19.5+0.85
ndays_threshold_ffwc=3
# Water_threshold=19.5
# from https://docs.google.com/spreadsheets/d/1J5B9pktZYnlBwAtb8n907A8P6xFVlCCd/edit#gid=1706380269 
FFWC_RL_LOG_FILENAME='Forecast Log Sheet - 2020.xlsx - FFWC.csv'
# from Hassan
FFWC_RL_HIS_FILENAME='2020-06-07 Water level data Bahadurabad Upper danger level.xlsx'
FFWC_RL_FOLDER='FFWC_DATA'

# from https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical?tab=overview 
Discharge_threshold=100000
ndays_threshold_glofas=2
GLOFAS_DS_FILENAME='{}.csv'
GLOFAS_DS_FOLDER='GLOFAS_data'


def get_glofas_df():
    glofas_df=pd.DataFrame(columns=['dis24'])
    for year in range(1979,2021):
        glofas_fn=GLOFAS_DS_FILENAME.format(year)
        glofas_df=glofas_df.append(pd.read_csv('{}/{}/{}'.format(DIR_PATH,GLOFAS_DS_FOLDER,glofas_fn),
                                                index_col=0))
    glofas_df.index=pd.to_datetime(glofas_df.index,format='%Y-%m-%d')
    return glofas_df

# def get_ffwc_log_df():
#     ffwc_rl_name='{}/{}/{}'.format(DIR_PATH,FFWC_RL_FOLDER,FFWC_RL_LOG_FILENAME)
#     ffwc_df=pd.read_csv(ffwc_rl_name,index_col=0,skiprows=0,header=1)
#     ffwc_df.index=pd.to_datetime(ffwc_df.index,format='%d/%m/%Y %H:%M')
#     ffwc_df=ffwc_df['Observed Water Level Today']
#     ffwc_df.dropna(inplace=True)
#     ffwc_df = ffwc_df.resample('D').mean()
#     return ffwc_df

def get_ffwc_his_df():
    ffwc_rl_name='{}/{}/{}'.format(DIR_PATH,FFWC_RL_FOLDER,FFWC_RL_HIS_FILENAME)
    ffwc_df=pd.read_excel(ffwc_rl_name,index_col=0,header=0)
    ffwc_df.index=pd.to_datetime(ffwc_df.index,format='%d/%m/%y')
    # ffwc_df=ffwc_df['WL']
    ffwc_df.dropna(inplace=True)
    return ffwc_df

def calculate_activations(days_above,ndays_threshold):
    activations=pd.DataFrame(columns=['start_date','end_date'])
    start_group=True
    for i,day in enumerate(days_above):
        if start_group:
            start_date=day
            start_group=False
        try:
            next_day=days_above[i+1]
        except:
            activations=activations.append({'start_date':start_date,'end_date':day},ignore_index=True)
            continue
        if(next_day-day)==timedelta(days=1):
            continue
        activations=activations.append({'start_date':start_date,'end_date':day},ignore_index=True)
        start_group=True
    activations['ndays'] = (activations['end_date'] - activations['start_date']).dt.days +1
    activations = activations[activations['ndays']>=ndays_threshold]
    return activations

glofas_df=get_glofas_df()
# ffwc_log_df=get_ffwc_log_df()
ffwc_his_df=get_ffwc_his_df()

fig, (ax1,ax1_t) = plt.subplots(figsize=[15,7],nrows=2, sharex=True)
# draw GLOFAS
glofas_df=glofas_df.loc[min(ffwc_his_df.index):max(ffwc_his_df.index),:]

glofas_df['dis24'].plot(label='GLOFAS water discharge',ax=ax1,c='green')
ax1.axhline(y=Discharge_threshold,c='green',ls='--',label='GLOFAS discharge threshold')
ax1.legend(loc='best')


def bar_color(df,threshold,color1,color2):
    return np.where(df.values>threshold,color1,color2).T

ffwc_his_df['WL'].plot(ax=ax1_t,legend=False,label='',alpha=0)
ax1_t.scatter(ffwc_his_df.index,ffwc_his_df['WL'],
            color=bar_color(ffwc_his_df['WL'],Water_threshold,'r','b'),
            label='FFWC Water Depth - Historical data')
# ffwc_his_df['WL'].plot(c='blue',label='FFWC Water Depth - Historical data',ax=ax1_t)
ax1_t.axhline(y=Water_threshold,c='blue',ls='--',label='FFWC water level threshold')
ax1_t.legend(loc='best')

# calculate activations 
# GLOFAS
GLOFAS_activations=calculate_activations(glofas_df[glofas_df['dis24']>=Discharge_threshold].index,ndays_threshold_glofas)
for iactivation,( _,activation) in enumerate(GLOFAS_activations.iterrows()):
    #print(activation)
    mean_days=(activation['start_date'])
    ax1.annotate("",
                # '{} d'.format(activation['ndays']),color='green',
                xy=(mean_days,Discharge_threshold),
                xytext=(mean_days,Discharge_threshold+20000),
                arrowprops=dict(facecolor='green', shrink=0.05))
    ax1.axvspan(activation['start_date']-timedelta(days=0.5),
                activation['end_date']+timedelta(days=0.5),facecolor='green', alpha=0.5)    
# FFWC
FFWC_activations=calculate_activations(ffwc_his_df[ffwc_his_df['WL']>=Water_threshold].index,ndays_threshold_ffwc)
for iactivation,( _,activation) in enumerate(FFWC_activations.iterrows()):
    mean_days=(activation['start_date'])
    ax1_t.annotate("",
                # '{} d'.format(activation['ndays']),color='green',
                xy=(mean_days,Water_threshold+0.5),
                xytext=(mean_days,Water_threshold+1),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    ax1_t.axvspan(activation['start_date']-timedelta(days=0.5),
                activation['end_date']+timedelta(days=0.5),facecolor='blue', alpha=0.5)

# plt.tight_layout()
plt.show()