# Nepal Anticipatory Action

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

Endorsed in July 2021, readiness trigger activated August 2021.

Last updated: September 2021

## Background information

The pilot on Anticipatory Action in Nepal is focused on monsoon flooding,
particularly in the Saptakoshi Watershed (located in the east)
and the Karnali, West Rapti and Babi basins (located in the west). 
The trigger is based on [GlofAS](https://www.globalfloods.eu/) 
river discharge forecasts at Chatara and Chisapani stations, combined with 
a [government](http://www.hydrology.gov.np/#/?_k=etlhwb) bulletin
and water level measurements. 

To mitigate risk, the trigger has two components, readiness and action.
The readiness trigger is reached when the 7-day GloFAS river discharge forecast
has a 70% probability or higher of exceeding the 1-in-2-year return period value.

The action trigger is somewhat more complex: either the 3-day GloFAS river 
discharge forecast has a 70% probability or higher of exceeding the 1-in-2-year
return period, or the government measured water level reaches or 
exceeds the danger leve. Furthermore, in either case the government-issued
flood forecast bulletin must predict flood warnings for the targeted 
river basins.

## Overview of analysis

The analysis primarily consists of analyzing the GloFAS data, in particular,
comparing the forecast and reanalysis to past events and water level,
and quantifying the forecast skill. The correlation of water level
and river discharge across various stations is also explored. 

## Data description

All [GloFAS](https://www.globalfloods.eu/) was downloaded from the
[Climate Data Store](https://cds.climate.copernicus.eu/#!/home),
see the section on [reproducing the analysis](#reproducing-this-analysis)

The data from the RCO has been provided to us privately and unfortunately is
not public at this time. 

## Directory structure 

The content within this repository is structured as follows: 
```
├── *.md                       <- Analysis notebooks
├── nple_parameters.py         <- Shared parameters
└── README.md                  <- Description of this project
```

## Reproducing this analysis 

Prepare your setup according to the 
[top-level README](https://github.com/OCHA-DAP/pa-anticipatory-action#getting-started)
(install requirements, create data directory environment variable).
Run `python src/npl/get_glofas_data.py` and to download and process the
GloFAS data. 
