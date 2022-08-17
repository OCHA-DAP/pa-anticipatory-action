# Malawi Anticipatory Action

Two types of shocks to anticipate to were explored:

1) dry spells, for which all the analysis can be found in `dryspells_trigger`
2) flooding, for which the analyses can be found in `flooding_trigger`

## Dry spells

[![Generic badge](https://img.shields.io/badge/STATUS-IMPLEMENTED-%231EBFB3)](https://shields.io/)

A [framework on dry spells](https://reliefweb.int/report/malawi/anticipatory-action-framework-malawi-dry-spells-2021-2022) is in place and is currently being monitored.
 The work on flooding has been halted.

Last update: Aug 2022

### Background information

In Malawi the pilot is on dry spells. For the trigger, a dry spell is defined as
a period of 14 consecutive days with no more than 2 millimiters of cumulative precipitation. 

The first step of the trigger development was to create a historical dataset of dry spells. 
Thereafter the correlation of dry spells with several **observed** indicators was explored. 
Based on this exploration the correlation of forecasts and dry spells was explored. 

Based on these results it was chosen to have a two stage trigger: one before the season using
ECMWF's long-range forecast, and the second at the end of the season using observational rainfall
data by ARC2. 

The predictive trigger makes use of ECMWF’s probabilistic long-range
forecast which predicts monthly total rainfall. Its threshold is met when
the forecast predicts a 50% probability of ≤ 210 millimeters across the
Southern region.

The observational trigger makes use of ARC2's observational rainfall data. Its threshold is met 
when 14 days with no more than 2 mm of cumulative precipitation are observed. 

See [the framework](https://reliefweb.int/report/malawi/anticipatory-action-framework-malawi-dry-spells-2021-2022) for more information on the trigger, funding, and activities. 

### Overview of analysis

The historical dataset is described [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/mwi/docs/mwi_historical_dry_spells_description.html).
An overview on the work of linking historical dry spells to impact indicators can be found [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/mwi/docs/mwi_impact_summary.html) 
Most of the code in this directory is aimed to understand the correlation of dry spells and 
observed and forecasted meteorological indicators. A summary of this work is described in this [document](https://ocha-dap.github.io/pa-anticipatory-action/analyses/mwi/docs/mwi_dry_spells_indicator_analyses.html).  

The scripts and notebooks feeding into these written summaries can be found in the `dryspells_trigger` directory, where the files are numbered based on the flow of the summary documents.  

### Data description

As source for the historical analysis of observational data, [CHIRPS](https://www.chc.ucsb.edu/data/chirps) was used. 
This data is publicly available. It is updated every month but has a daily frequency.
For the monitoring of the observational trigger, ARC2 was used as this is updated much more frequently, with a lag of around 2 days. 
For the predictive trigger, [ECMWF's long-range forecast](https://www.ecmwf.int/en/forecasts/documentation-and-support/long-range) is used. It is updated every month and provides monthly precipitation for 0 to 5 months ahead. 
The resolution of their forecast is 0.4 degrees and this forecast is published on the 5th of the month. 
We have access to this data, but it is not openly available data. However, this is the data used for the monitoring of the trigger.
The same data is openly available at a lower resolution (1 degree) and with a week delay through [Copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-monthly-single-levels?tab=overview). 
Many of the earlier explorations were done with this openly available data. 

Other data sources that were explored are ENSO and CHIRPSGEFS. References to these can be found in the summary document. 

### Reproducing this analysis

The code to compute the predictive trigger can be found in `src/mwi/compute_trigger_ecmwf.py`. 
This script computes the trigger for the original as well as lower resolution forecasts. Note that the script automatically downloads the lower resolutoin forecasts. To download the original forecasts, a script has to be run 
in the aa-toolbox [aa-toolbox repository](https://github.com/OCHA-DAP/pa-aa-toolbox/tree/feature/ecmwf-seas-realtime). More details on this are provided in the docstring of the `src/mwi/compute_trigger_ecmwf.py` script. 

The code to compute the observational trigger can be found in `analyses/mwi/dryspells_trigger/15_mwi_obs_trigger_monitoring.md`

Some notebooks in `analyses/mwi/dryspells_trigger` require you to first download the relevant data,
which can be done by running the scripts in `src/mwi`.

## Flooding

[![Generic badge](https://img.shields.io/badge/STATUS-ON%20HOLD-%23F2645A)](https://shields.io/)

Analytical work for a flooding trigger is currently on hold. Last updated: August 2022.

The code in the ```flooding_trigger``` directory contains work to explore the use of GloFAS streamflow forecasts to predict upcoming flooding along the Shire River in two districts in Malawi: Chikwawa and Nsanje. 

### Overview of analysis

The main components of this analysis are all contained within the numbered ```.md``` files in the ```flooding_trigger``` directory. These notebooks might not be fully reproducible as many require private data as inputs. The analysis includes the following key components: 

- Cleaning up, exploring, and validating various datasets of historical flood events.
- Calculating metrics of forecast skill and performance for the GloFAS stations of interest.
- Investigating the correlation between streamflow at the stations to determine if just a single station can be monitored for both districts.
- Calculating performance metrics (precision, recall) to evaluate how well GloFAS streamflow (historical modelled and historical forecasted) can predict flood events at various forecast leadtimes and streamflow thresholds. Streamflow thresholds are defined by return periods (eg. 1 in 3-year event). 

Key parameters for the analysis; such as return period threshold, forecast leadtime, GloFAS station of interest; are defined as global variables within each ```.md``` file.

### Data description

Code to download and process GloFAS data for Malawi can be found in the ```src/mwi``` directory, from within which you should run:  

```
python get_glofas_data.py
```

In this script, the paramaters for the station locations and desired forecast leadtimes have been set. Station locations are pulled from the appropriate reporting points as defined on the [GloFAS website](https://www.globalfloods.eu/glofas-forecasting/) (login required).

Various historical flood event datasets have also been used to validate the forecast performance of GloFAS. The only publicly available data can be downloaded from the [EM-DAT disasters database](https://public.emdat.be/) (login also required). 