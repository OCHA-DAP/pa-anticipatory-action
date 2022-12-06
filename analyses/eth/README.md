# Ethiopia Anticipatory Action

[![Generic badge](https://img.shields.io/badge/STATUS-IMPLEMENTED-%231EBFB3)](https://shields.io/)

The framework has been activated in December 2020 and February 2021.
No changes to the analytical work have been made after December 2020.
Last updated: August 2022

## Background information

The pilot on Anticipatory Action in Ethiopia is focused on drought-related food insecurity.
The full framework and other documents can be found [here](https://centre.humdata.org/anticipatory-action/).

The trigger is based on a combination of food insecurity and precipitation forecasts.

The trigger is evaluated per admin1 region, and is met if any of the admin1 regions meets
both the food insecurity and drought criterium. These criteria are as follows:

**Food Insecurity**

At least 20% population of one or more ADMIN1 regions be projected at IPC4+

OR

At least 30% of ADMIN1 population be projected at IPC3+ AND with an increase by 5 percentage points compared to current state

**Drought**

At least 50% probability of below average rainfall from at least two seasonal rainfall forecasts

OR

Drought named as a driver of the deterioration of the situation in food security by FewsNet or Global IPC

More background on the how and why of this trigger can be found in the
[technical](https://docs.google.com/document/d/1aYM3Bii2Eu7oSdjiR-M6Mfbz5zjnJLVyf6AS_yI4J3s/edit?usp=sharing)
and [methodological](https://docs.google.com/document/d/1yGNgp-jHm_uWwJJ4hJnbFMjBcl6-y6-kP2rsc29eggI/edit?usp=sharing) note.

## Overview of analysis

The framework makes use of IPC analysis provided by FewsNet and IPC global as well as seasonal precipitation forecasts provided by National Meteorological Agency (NMA) Ethiopia, ICPAC, IRI, and NMME.

For the food insecurity part, the `eth/analyses` folder contains notebooks to analyze historical IPC levels from both sources.
This analysis was used to determine the criteria.
The `src` folder contains the scripts that retrieve the raw data and process it.
The scripts in `src` compute the percentage of population in each IPC phase and whether the food insecurity criterium is met.
The data produced by these scripts are used in the notebooks.

For the drought part, several rainfall forecasts are compared in `04_eth_rainfall_aggregate.ipynb`. This data is download in `src`.
For the current trigger, the rainfall forecasts are evaluated based on the raw figures published
by the different organizations and thus the code in this repo is currently not used for evaluation.  

## Data description

The Ethiopian pilot uses two sets of data, namely IPC classifications and rainfall forecasts.

1. IPC classification:

- FewsNet publishes thrice a year their classification containing current situation, near-term projections and medium-term projections. This data is given at livelihood level, and the scripts in this repository aggregate that to admin2,1,and 0 level.
- GlobalIPC publishes 1 to 2 times a year and uses a combination of livelihood zones and admin2 regions for reporting
Both sources are open data. It was chosen to use IPC as main source.

2. Seasonal rainfall forecasts:
We make use of 4 providers: NMA, ICPAC, IRI, and NMME.
We chose these because they all publish tercile forecasts and are either a regional organization or have a good reputation.

Not all forecasts are published in the same format:

- [NMA](http://www.ethiomet.gov.et/other_forecasts/seasonal_forecast) only publishes its data as PNG
- [ICPAC](https://www.icpac.net/seasonal-forecast/) publishes pngs on their website, and raster files on their FTP server, to which you can ask access.
- [IRI](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/) publish their data as raster files, but you need to create an account to get access.
- [NMME](https://www.cpc.ncep.noaa.gov/products/international/nmme/probabilistic_seasonal/africa_nmme_prec_3catprb_FebIC_Mar2021-May2021.png)'s data is openly available via their FTP.

## Reproducing this analysis

From the directory `src/indicators/food_insecurity`, run `process_fewsnet_subnatpop.py`
and `process_globalipc.py` to retrieve and compute the FewsNet and IPC global data respectively.  
After that run `src/ethiopia/eth_foodinsec_trigger.py` to compute the metrics of the trigger.
Several parameters from the `config.yml` are used in these scripts so make sure these are set correctly.

Not all code is working anymore, since changes to the general codebase have
been implemented after endorsement and monitoring of this trigger.
Since large changes to a potential revision of this trigger are expected,
it was decided to not invest in keeping the code up-to-date for now
