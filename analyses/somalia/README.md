# Somalia Anticipatory Action

[![Generic badge](https://img.shields.io/badge/STATUS-IMPLEMENTED-%231EBFB3)](https://shields.io/)

Has been activated in June 2020 and at the end of 2020. 
Last updated: Aug 2021
## Background information
The pilot on Anticipatory Action in Somalia is on food insecurity. 
The trigger uses IPC as indicator from two organizations, namely FewsNet and IPC global. 
The trigger is evaluated at the country level. 
The monitoring of the trigger is done by FSNAU, so not by OCHA.  

The trigger is as follows: 

The projected national population in Phase 3 and above exceed 20%, 

AND 

(The national population in Phase 3 is projected to increase by 5 percentage points, OR 
The projected national population in Phase 4 or above is 2.5%)

The activation of the trigger is described [here](https://www.unocha.org/story/un-central-emergency-response-fund-supports-anticipatory-action-ethiopia-and-somalia)
and [here](https://www.unocha.org/story/un-humanitarian-chief-release-140m-cerf-funds-anticipatory-action-projects)


## Overview of analysis
The framework makes use of IPC analysis provided by FewsNet and IPC global.
This folder contains notebooks to analyze historical IPC levels from both sources. 
This analysis was used to determine the criteria. 
In the `src` folder complementing scripts can be found. 
These compute the percentage of population in each IPC phase and whether the food insecurity criterium is met.

## Data description

We make use of two sources of IPC classifications, by FewsNet and IPC global.  
Both sources are open data. 

- FewsNet publishes thrice a year their classification containing current situation, near-term projections and medium-term projections. 
This data is shared at livelihood level, and the scripts in this repository aggregate that to admin2,1,and 0 level. 
- IPC global publishes 1 to 2 times a year and uses a combination of livelihood zones and admin2 regions for reporting their classifications. 
IPC global has been sharing data in Somalia since 2017. 

To compute the metrics we also make use of two Common Operational Datasets (CODs) from HDX
namely the administrative boundaries and population statistics 

## Directory structure
```

├── notebooks                 <- Jupyter notebooks that contain a walkthrough of data analysis steps. 
└── README.md                 <- Description of this project.

```

## Reproducing this analysis
From the directory `src/indicators/food_insecurity`, run `process_fewsnet_subnatpop.py` 
and `process_globalipc.py` to retrieve and compute the FewsNet and IPC global data respectively.  
After that run `src/somalia/som_foodinsec_trigger.py` to compute the metrics of the trigger. 
Several parameters from the `config.yml` are used in these scripts so make sure these are set correctly. 