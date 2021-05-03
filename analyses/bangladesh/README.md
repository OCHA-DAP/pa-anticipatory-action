# CERF Anticipatory Action framework in Bangladesh

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
2. **Pilot evaluation**: Calculating historical estimates of flooding extent over time in five high priority districts (Bogra, Gaibandha, Jamalpur, Kurigram, Sirajganj). This work is largely based on an analysis of Sentinel-1 SAR imagery in Google Earth Engine, accessible [here](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). This imagery processing methodology is adapted from [guidance from the UN-SPIDER Knowledge Portal](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step). The results of the historical flood analysis are summarized in [here](https://ocha-dap.github.io/pa-anticipatory-action/analyses/bangladesh/validation/summary_flooding.html).

## Data description

All input and output files are stored in a private Google Drive folder. We're working to make part of this publicly available.

## Directory structure 

The content within this repository is structured as follows: 

```
├── pilot_evaluation           <- Material related to the pilot evaluation
│   ├── notebooks              <- Notebooks with 'one-off' analyses
│   ├── scripts                <- Reusable scripts
│   └── config.yml             <- Parameters to configure the analysis
│
├── trigger_development       <- Scripts related to the trigger analysis
│
└── README.md                 <- Description of this project
```

## Reproducing this analysis 

#### Historical flooding analysis

1. Generate shapefiles that delineate flood extent over time using [this Google Earth Engine Script](https://code.earthengine.google.com/0fe2c1f3b2cf8ef6fe9aa81382b00191). Within the script, the following parameters can be changed: 1) start and end dates of a pre-flood period, 2) start and end dates of a flood period, and 3) shapefile to delineate geographic area of interest. Within the GEE editor you will be required to manually download the output shapefiles. 

2. Edit the ```config.yml``` file to include the location of the directory where your output shapefiles are stored (```gee_dir```), the admin areas are stored (```adm_dir```) and the location where you want your output data to go (```data_dir```). Also edit the ```adm``` level according to the scale of aggregation that you want to perform. This script can be run by aggregating to ```ADM4```, ```ADM3```, and ```ADM2``` levels, or to the mauza level (with ```MAUZ```).

3. Run the ```generate_flood_frac.py``` script from the terminal in the repository root directory to generate .CSV files that include the flood fraction over time within admin areas. For example: 

```
python analyses/bangladesh/pilot_evaluation/scripts/generate_flood_frac.py 
```

4. Run the ```generate_interpolated.py``` script to generate the interpolated estimates of flood extent over time. These estimates are created by fitting polynomial and Gaussian functions to the original estimates derived from Sentinel-1 SAR imagery (calculated in the above Google Earth Engine script). This script requires the output from the previous script as input. 

```
python analyses/bangladesh/pilot_evaluation/scripts/generate_interpolated.py
```

5. Visualise the results through basic choropleth mapping, using the same input aggregation level as the previous steps.

```
python analyses/bangladesh/pilot_evaluation/scripts/make_plots.py
```

