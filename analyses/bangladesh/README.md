# CERF Anticipatory Action framework in Bangladesh

## Background

Building on the work by individual agencies, such as the IFRC, WFP, FAO and NGOs, the pilot will methodologically combine three components, scaling-up anticipatory humanitarian action in Bangladesh:

i) A robust forecasting embedded in a clear decision-making process (the model).
The pilot will use a government run and endorsed early warning system, combining a 10-day probabilistic flood forecast for operational readiness and a 5-day deterministic flood forecast for activation of anticipatory action. This trigger has been successfully used by IFRC and WFP in the past.
  
ii) Pre-agreed, coordinated, multi-sectoral actions that can fundamentally alter the trajectory of the crisis (the action plan).
Given the short lead times of the forecasts, cash is a major component of the pilot. Bringing together the reach of WFP and IFRC (through the BDRCS), up to 70,000 households could receive $53 each about 5 days ahead of a flood.
In addition, FAO would preposition animal fodder and vaccinate livestock against waterborne diseases. UNFPA would distribute dignity and hygiene kits, and communication material preventing sexual and gender-based violence.
In the future, other actions could be integrated, including better early warning systems or prepositioning of essential medicine. Unfortunately, due to COVID-19, these options were deemed unrealistic for the upcoming monsoon season.
An inter-agency lens would be applied to determining the selection criteria for beneficiaries to ensure the most vulnerable people benefit from the anticipatory action.

iii) Pre-arranged finance (the money).
CERF set aside around $5 million [pending final plan and confirmation] for anticipatory action for floods in
Bangladesh. This funding will become available immediately once the defined trigger is reached to active the actions described above
In addition, the pilot seeks to amplify and coordinate similar anticipatory action pilots at the agency scale, including from WFP, IFRC, and others.

## Analysis

The analysis within this repository contains two components. 

1. Processing FFWC and GLOFAS forecasting data, used to trigger this pilot's anticipatory action. 
2. Calculating historical estimates of flooding extent over time in five high priority districts (Bogra, Gaibandha, Jamalpur, Kurigram, Sirajganj). This work is largely based on an analysis of Sentinel-1 SAR imagery in Google Earth Engine, accessible [here](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). This imagery processing methodology is adapted from [guidance from the UN-SPIDER Knowledge Portal](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step).

## Structure of this repository 

The content within this repository is structured as follows: 

```
├── data
│   ├── processed             <- Data that has been processed or transformed in some way.  
│   └── raw                   <- Original, immutable data. 
│
├── notebooks                 <- Jupyter notebooks that contain a walkthrough of data analysis steps. 
│
├── scripts                   <- Scripts to perform data processing and analysis.    
│   ├── d01_data              <- Scripts to load in datasets. 
│   ├── d02_processing        <- Scripts to perform basic cleaning and preprocessing on data.
│   ├── d03_analysis          <- Scripts to conduct more in-depth analysis of data
│   └── d04_visualization     <- Scripts to create visualizations. 
│
├── Generate_flood_frac.py    <- Top level script to generate data.
├── Generate_interpolated.py  <- Description of this project.
├── README.md                 <- Description of this project.
├── config.yml                <- Global variables.
└── environment.yml           <- Contains dependencies to set up a conda environment. 

```

Larger raw and processed data files are currently not included within this repository. As described below, the historical analysis of flood extent can be reproduced using shapefiles generated from a Google Earth Engine script. Note that this requires a Google Earth Engine account. 

## Getting started 

Set up and activate a Python environment in Anaconda using the ```environment.yml``` file provided: 

```
conda env create -f environment.yml
conda activate bang_floods
```

### To reproduce the historical analysis of flood evolution:

1. Generate shapefiles that delineate flood extent over time using [this Google Earth Engine Script](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). Within the script, the following parameters can be changed: 1) start and end dates of a pre-flood period, 2) start and end dates of a flood period, and 3) shapefile to delineate geographic area of interest.

2. Edit the ```config.yml``` file to include the location of the directory where your output shapefiles are stored (```shp_dir```) and the location where you want your output data to go (```data_dir```).

3. Run a Python script from the terminal in the repository root directory to generate .CSV files that include the flood fraction over time within admin areas. This script can be run by aggregating to ```ADM4```, ```ADM3```, and ```ADM2``` levels. For example: 

```
python Generate_flood_frac.py ADM4 
```

4. Run another Python script to generate the interpolated estimates of flood extent over time. These estimates are created by fitting polynomial and Gaussian functions to the original estimates derived from Sentinel-1 SAR imagery (calculated in the above Google Earth Engine script). This script requires the output from the previous script as input. 

```
python Generate_interpolated.py ADM4
```

