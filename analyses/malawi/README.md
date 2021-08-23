# CERF Anticipatory Action framework in Malawi

Two types of shocks to anticipate to are explored:
1) dry spells, for which all the analysis can be found in `dryspells_trigger`
2) flooding, for which the analyses can be found in `flooding_trigger`

## Dry spells

[![Generic badge](https://img.shields.io/badge/STATUS-UNDER%20REVISION-%23CCCCCC)](https://shields.io/)

A trigger has been proposed and is currently under revision by the larger Anticipatory Action team. 

Last update: Aug 2021

### Background information
In Malawi the pilot is focussed on two shocks, of which one is dry spells. For the trigger, a dry spell is defined as
a period of 14 consecutive days with no more than 2 millimiters of cumulative precipitation. 

The first step of the trigger development was to create a historical dataset of dry spells. 
Thereafter the correlation of dry spells with several **observed** indicators was explored. 
Based on this exploration the correlation of forecasts and dry spells was explored. 

Based on these results it was chosen to use ECMWF's seasonal forecast, 
which forecast total precipitation per month. 

### Overview of analysis 
The historical dataset is described [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/docs/mwi_historical_dry_spells_description.html).
An overview on the work of linking historical dry spells to impact indicators can be found [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/docs/mwi_impact_summary.html) 
Most of the code in this directory is aimed to understand the correlation of dry spells and 
observed and forecasted meteorological indicators is described in this [summary document](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/docs/mwi_dry_spells_indicator_analyses.html).  

The scripts and notebooks feeding into these written summaries can be found in `dryspells_trigger` 
where the files are numbered based on the flow of the summary document.  

### Data description
As source for observational data, [CHIRPS](https://www.chc.ucsb.edu/data/chirps) was used. 
This data is publicly available. It is updated every month but has a daily frequency.
The final forecasting source that is used for the trigger is ECMWF's seasonal forecast. 
This data is openly available through [Copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-monthly-single-levels?tab=overview). 
It is updated every month and provides monthly precipitation for 0 to 5 months ahead. 
Other data sources included ENSO, CHIRPSGEFS and ARC2. References to these can be found in the summary document. 

### Reproducing this analysis
Make sure to have the CHIRPS and ECMWF data clipped to Malawi, for which scripts can be found in `src/malawi`

## Flooding

[![Generic badge](https://img.shields.io/badge/STATUS-ON%20HOLD-%23007CE0)](https://shields.io/)

Analytical work for a flooding trigger is currently on hold while the work on dry spells is being prioritized. Last updated: August 2021.

The code in the ```flooding_trigger``` directory contains work to explore the use of GloFAS streamflow forecasts to predict upcoming flooding along the Shire River in two districts in Malawi: Chikwawa and Nsanje. 

### Input data

Code to download and process GloFAS data for Malawi can be found in the ```src/malawi``` directory, from within which you should run:  

```
python get_glofas_data.py
```

In this script, the paramaters for the station locations and desired forecast leadtimes have been set. Station locations are pulled from the appropriate reporting points as defined on the [GloFAS website](https://www.globalfloods.eu/glofas-forecasting/) (login required).

Various historical flood event datasets have also been used to validate the forecast performance of GloFAS. The only publicly available data can be downloaded from the [EM-DAT disasters database](https://public.emdat.be/) (login also required). 

### Overview of analysis

The main components of this analysis are all contained within the numbered ```.md``` files in the ```flooding_trigger``` directory. These notebooks might not be fully reproducible as many require private data as inputs. The analysis includes the following key components: 

- Cleaning up, exploring, and validating various datasets of historical flood events.
- Calculating metrics of forecast skill and performance for the GloFAS stations of interest.
- Investigating the correlation between streamflow at the stations to determine if just a single station can be monitored for both districts.
- Calculating performance metrics (precision, recall) to evaluate how well GloFAS streamflow (historical modelled and historical forecasted) can predict flood events at various forecast leadtimes and streamflow thresholds. Streamflow thresholds are defined by return periods (eg. 1 in 3-year event). 

Key parameters for the analysis; such as return period threshold, forecast leadtime, GloFAS station of interest; are defined as global variables within each ```.md``` file.