# CERF Anticipatory Action framework in Malawi

Two types of shocks to anticipate to are explored:
1) dry spells, for which all the analysis can be found in `dryspells_trigger`
2) flooding, for which the analyses can be found in `flooding_trigger`

## Dry spells


## Flooding

[![Generic badge](https://img.shields.io/badge/STATUS-ON_HOLD-blue.svg)](https://shields.io/)

The code in the ```flooding_trigger``` directory contains work to explore the use of GloFAS streamflow forecasts to predict upcoming flooding along the Shire River in two districts in Malawi: Chikwawa and Nsanje. 

### Input data

Code to download and process GloFAS data for Malawi can be found in the ```src/malawi``` directory, 

```
get_glofas_data.py
```

within which paramaters for the station locations and desired forecast leadtimes have been set. Station locations are pulled from the appropriate reporting points as defined on the [GloFAS website](https://www.globalfloods.eu/glofas-forecasting/) (login required).

Various historical flood event datasets have also been used to validate the forecast performance of GloFAS. The only publicly available data can be downloaded from the [EM-DAT disasters database](https://public.emdat.be/) (login also required). 

### Overview of analysis

The main components of this analysis are all contained within the numbered ```.md``` files in the ```flooding_trigger``` directory. These notebooks might not be fully reproducible as many require private data as inputs. The analysis includes the following key components: 

- Cleaning up, exploring, and validating various datasets of historical flood events.
- Calculating metrics of forecast skill and performance for the GloFAS stations of interest.
- Investigating the correlation between streamflow at the stations to determine if just a single station can be monitored for both districts.
- Calculating performance metrics (precision, recall) to evaluate how well GloFAS streamflow (historical modelled and historical forecasted) can predict flood events at various forecast leadtimes and streamflow thresholds. Streamflow thresholds are defined by return periods (eg. 1 in 3-year event). 

Key parameters for the analysis; such as return period threshold, forecast leadtime, GloFAS station of interest; are defined as global variables within each ```.md``` file. 