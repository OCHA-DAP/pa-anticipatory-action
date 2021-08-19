# CERF Anticipatory Action Pilot in South Sudan

 [![Generic badge](https://img.shields.io/badge/STATUS-ON%20HOLD-%23F2645A)](https://shields.io/) 

 Initial exploration of data done in April 2021. 
 Currently the analytical work is on hold.  

 Last updated: Aug 2021

 ## Background information
 The pilot on Anticipatory Action in South Sudan is on floods. 
 The analytical work is in an exploration phase. 
 
 ## Overview of analysis
 The analysis explores the monthly precipiation in ADMIN1 regions in South Sudan. 
 It then makes a first attempt to link this to the extent of flooded area. 
 An overview of the analysis can be found [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/ssd/docs/ssd_doc_corr_floodscan_monthlyprecip.html), 
 which is produced from the files in `docs`. 

 ## Data description

The analysis uses [CHIRPS](https://www.chc.ucsb.edu/data/chirps) as source of observational data. 
This data is publicly available. It is updated every month but has a daily frequency.  

To determine the flood extent, we use data by [FloodScan](https://www.aer.com/weather-risk-management/floodscan-near-real-time-and-historical-flood-mapping/). This data is not openly available. 

 We also make use of the administrative boundaries from the 
 Common Operational Datasets (CODs) on HDX. 

 ## Reproducing this analysis
 Make sure you have the processed CHIRPS data and floodscan data. 
 The floodscan data can be computed with `get_floodscan_data.py`, 
 but uses data that is not openly accessible. 