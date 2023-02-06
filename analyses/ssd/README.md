# CERF Anticipatory Action Pilot in South Sudan

 [![Generic badge](https://img.shields.io/badge/STATUS-ON%20HOLD-%23F2645A)](https://shields.io/) 

 Initial exploration of data was done throughout 2021. An in-country mission was
 done in February 2022, with additional analytical work done in early
 2022 to support an early action allocation.

 Last updated: Jan 2023

 ## Background information

 The pilot on Anticipatory Action in South Sudan is on floods. 
 The analytical work explored the potentials of a trigger framework on floods.
 
 ## Overview of analysis

 The initial analysis explored the monthly precipiation in ADMIN1 regions in South Sudan. 
 It then made a first attempt to link this to the extent of flooded area. 
 An overview of the analysis can be found [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/ssd/docs/ssd_doc_corr_floodscan_monthlyprecip.html), 
 which is produced from the files in `docs`. 

 Additional analysis explored the performance of using the [Global Flood Awareness System (GloFAS)](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/develop/analyses/ssd/flood_trigger/ssd_glofas_vs_floodscan.md), seasonal forecasts and
 [upstream water levels](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/develop/analyses/ssd/flood_trigger/ssd_dahiti.md). Due to
 inability to predict the most severe years of flooding using the sources above, it was decided that an anticipatory action trigger was not feasible.

The final analysis outputs and results used in deciding on an early action allocation were summarized in [this blog post](https://centre.humdata.org/flood-risks-for-south-sudans-2022-rainy-season/).

 ## Data description

The analysis uses [CHIRPS](https://www.chc.ucsb.edu/data/chirps) as source of observational data. 
This data is publicly available. It is updated every month but has a daily frequency.

Upstream water levels were sourced from the [Database for Hydrological Time Series of Inland Waters (DAHITI)](https://dahiti.dgfi.tum.de/en/).

To determine the flood extent, we use data by [FloodScan](https://www.aer.com/weather-risk-management/floodscan-near-real-time-and-historical-flood-mapping/). This data is not openly available. 

Flood modeling was explored using model outputs from [GloFAS](https://www.globalfloods.eu).

 We also make use of the administrative boundaries from the 
 Common Operational Datasets (CODs) on HDX. 

 ## Reproducing this analysis
Since the Floodscan data is only available privately, data is not fully reproducible without those sources.
However, with access to Floodscan data, all scripts in `flood_trigger` should be available.