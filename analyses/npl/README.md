# Nepal Anticipatory Action

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

Endorsed in July 2021, readiness trigger activated August 2021.

Last updated: September 2021

## Background information

## Overview of analysis

Processing FFWC and GLOFAS forecasting data, 
used to trigger this pilot's anticipatory action. 

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
Run `python src/nepal/get_glofas_data.py` and to download and process the
GloFAS data. 
