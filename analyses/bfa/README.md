# Burkina Faso Anticipatory Action

[![Generic badge](https://img.shields.io/badge/STATUS-UNDER%20DEVELOPMENT-%23007CE0)](https://shields.io/)

Ongoing work since April 2021. Will not be monitored before March 2022. 

Last updated: Aug 2021

## Background information
The pilot on Anticipatory Action in Burkina Faso is on seasonal drought. 
To determine the likelihood of such a drought, we make use of precipitation as meteorological indicator. 
More specifically we look at the probability of below-average precipitation according to the seasonal forecast
by the International Research Institute for Climate and Society (IRI).  The pilot covers four ADMIN1 regions, namely Boucle du Mouhoun, Nord, Centre-Nord, and Sahel.
 
## Overview of analysis
We first explored data on humanitarian risk and vulnerabilities to help define the geographic scope and shock for the pilot. This exploration is summarized in [this document](https://ocha-dap.github.io/pa-anticipatory-action/analyses/bfa/notebooks/bfa_risk_overview.html), the code for which can be found in  `docs/bfa_risk_overview.Rmd` and `drought_trigger/01_data_summary.R`. 

Thereafter we analyzed observed and forecasted precipitation patterns. This code can be found in the `.md` files in the `drought_trigger/` directory. 

## Data description

The forecast source for the trigger is 
[IRI's seasonal forecast](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html). 
This forecast is issued on the 15th of each month and has 1 to 4 months leadtime. 
The data can be downloaded after the creation of an account. 

To validate the forecast, we use observational precipitation data by [CHIRPS](https://www.chc.ucsb.edu/data/chirps). 
This data is publicly available and is updated every month. 

We also make use of the administrative boundaries from the 
Common Operational Datasets (CODs) on HDX. 

## Reproducing this analysis
Make sure you have the processed CHIRPS and IRI data which can be
computed with `src/bfa/get_chirps_data.py` and 
`src/indicators/drought/iri_rainfallforecast.py`
