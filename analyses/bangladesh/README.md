# CERF Anticipatory Action framework in Bangladesh

##  Table of Contents
1. [Background Information](#background-information)
2. [Overview of analysis](#overview-of-analysis)
3. [Data description](#data-description)
4. [Directory structure](#directory-structure)
5. [Reproducing this analysis](#reproducing-this-analysis)

## Background information

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

## Overview of analysis

The analysis within this repository contains two components. 

1. **Trigger analysis**: Processing FFWC and GLOFAS forecasting data, used to trigger this pilot's anticipatory action. 
2. **Pilot evaluation**: Calculating historical estimates of flooding extent over time in five high priority districts (Bogra, Gaibandha, Jamalpur, Kurigram, Sirajganj). This work is largely based on an analysis of Sentinel-1 SAR imagery in Google Earth Engine, accessible [here](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). This imagery processing methodology is adapted from [guidance from the UN-SPIDER Knowledge Portal](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step). This work also includes technical support provided to the Centre for Disaster Protection in processing and analysing the geospatial dimensions of survey data. The results of the historical flood analysis are summarized in [this slide deck](https://docs.google.com/presentation/d/1D5tj83Q63L-9lI343t0tcHwpxQ14XkGQK6PobUcXb6g/edit#slide=id.p3).

## Data description

All input and output files are stored [on Google Drive](https://drive.google.com/drive/folders/16TR6uta4XgMhpuBVHJdH4WM529TkK_hF?usp=sharing). 

Input data includes: 
- Shapefiles of Admin 0 --> Admin 4 units in Bangladesh, originally accessed from [HDX](https://data.humdata.org/dataset/administrative-boundaries-of-bangladesh-as-of-2015). ```Adm_Shp```
- Shapefile of mauzas (sub Admin 4) in Bangladesh, provided by a local contact. ```Adm_Shp/mauza``` 
- Informant reports of flood extent over time in selected unions (Admin 4), provided by the Centre for Disaster Protection. ```CDP_Informant``` 
- Selected household locations from CDP survey results. ```CDP_Survey``` 
- [GLOFAS](https://www.globalfloods.eu/) river discharge measurements from selected stations along the Jamuna river. ```GLOFAS_Data``` 
- [FFWC](http://www.ffwc.gov.bd/) water level measurements from selected stations along the Jamuna river. ```FFWC_Data```
- [GSW](https://global-surface-water.appspot.com/download) Seasonality 2019 raster files covering Bangladesh. ```GSW_Data``` 

Output data includes:
- Output shapefiles from Google Earth Engine script. ```GEE_Output```
- Flood extent results by Admin 4 regions over time, derived from ```GEE_Output``` files. ```FE_Results```
- Interpolated flood extent results from Gaussian and polynomial fitting. ```FE_Results```
- Permanent water extent shapefiles. ```GSW_Data```
- Geolocated household locations with distance to water. ```CDP_Survey```

## Directory structure 

The content within this repository is structured as follows: 

```

├── notebooks                 <- Jupyter notebooks that contain a walkthrough of data analysis steps. 
│
├── pilot_evaluation          <- Scripts related to the trigger evaluation
│   ├── d01_data              <- Scripts to load in datasets. 
│   ├── d02_processing        <- Scripts to perform basic cleaning and preprocessing on data.
│   ├── d03_analysis          <- Scripts to conduct more in-depth analysis of data
│   ├── d04_visualization     <- Scripts to create visualizations. 
│   ├── config.yml                <- Global variables.
│   ├── Generate_flood_frac.py <- Top level script to generate flood extent from GEE output shapefiles.
│   └── Generate_interpolated.py <- Top level script to generate interpolated flood extent estimates. 
│
├── results                   <- Summary documentation and final outputs of analyses. 
│   ├── figures               <- Output figures to summarize results. 
│   └── write_up              <- Summary documentation addressing methodology and results.  
│
├── trigger_analysis          <- Scripts related to the trigger analysis
│   ├── GetGLOFAS_Data.py     <- Download GloFAS data from CDS
│   ├── GLOFAS_prediction_error.py <- Compare GLoFAS forecast with observations
│   ├── HistoricalValidation_triggers.py <- Plot GloFAS forecast against FFWC triggers  
│   ├── station_comparison.py <- calculates the time offset between different stations 
│   └── utils.py              <- methods shared by the various trigger scripts 
│
└── README.md                 <- Description of this project.
```

Larger raw and processed data files are currently not included within this repository. As described below, the historical analysis of flood extent can be reproduced using shapefiles generated from a Google Earth Engine script. Note that this requires a Google Earth Engine account. 

## Reproducing this analysis 

#### Setting up the Python environment

Set up and activate a Python environment in Anaconda using the ```environment.yml``` file provided: 

```
conda env create -f environment.yml
conda activate bang_floods
```

#### Historical flooding analysis

1. Generate shapefiles that delineate flood extent over time using [this Google Earth Engine Script](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). Within the script, the following parameters can be changed: 1) start and end dates of a pre-flood period, 2) start and end dates of a flood period, and 3) shapefile to delineate geographic area of interest.

2. Edit the ```config.yml``` file to include the location of the directory where your output shapefiles are stored (```gee_dir```), the admin areas are stored (```adm_dir```) and the location where you want your output data to go (```data_dir```).

3. Run a Python script from the terminal in the repository root directory to generate .CSV files that include the flood fraction over time within admin areas. This script can be run by aggregating to ```ADM4```, ```ADM3```, and ```ADM2``` levels, or to the mauza level (with ```MAUZ```). For example: 

```
python Generate_flood_frac.py ADM4 
```

4. Run another Python script to generate the interpolated estimates of flood extent over time. These estimates are created by fitting polynomial and Gaussian functions to the original estimates derived from Sentinel-1 SAR imagery (calculated in the above Google Earth Engine script). This script requires the output from the previous script as input. 

```
python Generate_interpolated.py ADM4
```

5. Visualise the results through basic choropleth mapping, using the same input aggregation level as the previous steps.

```
python Make_plots.py ADM4
```

