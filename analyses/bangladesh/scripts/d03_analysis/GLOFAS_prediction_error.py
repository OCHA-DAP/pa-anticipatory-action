import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# from https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical?tab=overview 
GLOFAS_DS_FILENAME='{}.csv'
GLOFAS_DS_ENSEMBLE_FILENAME='10daysleadtime_19972018_allensemble.xls'
GLOFAS_DS_FOLDER='GLOFAS_data'

def get_glofas_df():
    glofas_df=pd.DataFrame(columns=['dis24'])
    for year in range(1979,2021):
        glofas_fn=GLOFAS_DS_FILENAME.format(year)
        glofas_df=glofas_df.append(pd.read_csv('{}/{}/{}'.format(DIR_PATH,GLOFAS_DS_FOLDER,glofas_fn),
                                                index_col=0))
    glofas_df.index=pd.to_datetime(glofas_df.index)
    return glofas_df

def get_glofas_ensemble_df():
    glofas_ens_df=pd.read_excel('{}/{}/{}'.format(DIR_PATH,GLOFAS_DS_FOLDER,GLOFAS_DS_ENSEMBLE_FILENAME),
                                    index_col=0)
    glofas_ens_df.index=pd.to_datetime(glofas_ens_df.index,format='%Y-%m-%d')
    # shift 10 days
    glofas_ens_df.index=glofas_ens_df.index+timedelta(days=10)
    glofas_ens_df['dis24_avg']=glofas_ens_df.mean(axis=1)
    glofas_ens_df = glofas_ens_df.resample('D').mean().interpolate(method='linear')
    glofas_ens_df=glofas_ens_df['dis24_avg']
    # print(glofas_ens_df)
    return glofas_ens_df

glofas_df=get_glofas_df()
glofas_ens_df=get_glofas_ensemble_df()
all_projections=pd.merge(glofas_ens_df,glofas_df, left_index=True,right_index=True,how='left')

fig1,(ax1, ax2) = plt.subplots(2,figsize=[15,7],sharex=True)
# ax1=plt.subplot(211)
# draw GLOFAS
all_projections['dis24'].plot(label='GLOFAS water discharge - reanalysis',ax=ax1,c='green')
all_projections['dis24_avg'].plot(label='GLOFAS water discharge - projection (10 days shifted)',ax=ax1,c='blue')
ax1.legend(loc='best')

all_projections['rel_diff']=(all_projections['dis24_avg']-all_projections['dis24'])/all_projections['dis24']
# print(all_projections.index.month)
all_projections=all_projections[all_projections.index.month.isin([6,7,8,9])]

def bar_color(df,color1,color2):
    return np.where(df.values>0,color1,color2).T

all_projections['rel_diff'].plot(ax=ax2,legend=False,label='',alpha=0)
# all_projections['rel_diff'].plot.bar(color=bar_color(all_projections['rel_diff'],'g','r'),
# all_projections['rel_diff'].plot.bar(color=bar_color(all_projections['rel_diff'],'g','r'),
                                    #  label='Relative error - Projection',ax=ax2)
ax2.scatter(all_projections.index,all_projections['rel_diff'],
            color=bar_color(all_projections['rel_diff'],'g','r'),
            label='Relative error - Projection')
ax2.axhline(y=all_projections['rel_diff'].mean(),c='green',ls='--',
                label='Average prediction error between June and September')
print(all_projections['rel_diff'].mean())

# ax2.set
ax2.legend(loc='best')

plt.show()