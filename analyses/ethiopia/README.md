# [COUNTRY NAME] Anticipatory Action

## Background information

Provide a basic overview of the context of anticipatory action in this country. Link to the GDrive Trigger Card document for greater context and details.  

## Overview of analysis

To pilot Anticipatory Action for drought in Ethiopia, OCHA and partners designed a framework which includes a combination of food security and climatic triggers so as to activate the framework in response to drought-related food insecurity.
As of January 2021, the framework makes use of IPC analysis provided by FewsNet and Global IPC as well as seasonal precipitation forecasts provided by ICPAC, IRI, NMME, and ECMWF.
This folder contains the definition and analysis of this drigger, making use of the scripts found in the `indicators` folder

## Data description

The Ethiopian pilot uses two sets of data, namely IPC classifications and rainfall forecasts. 

1. *IPC classification*: 
- FewsNet publishes thrice a year their classification containing current situation, near-term projections and medium-term projections. This data is given at livelihood level, and the scripts in this repository aggregate that to admin2,1,and 0 level. 
- GlobalIPC publishes 1 to 2 times a year and uses a combination of livelihood zones and admin2 regions for reporting
Both sources are open data. It was chosen to use IPC

2. *Seasonal rainfall forecasts*:


- Where does the data come from? Are there any licensing or usage restrictions?
- How can the data be accessed?
- Why were these datasets selected?
- Are there any limitations with these datasets that one should be aware of when running the analysis and interpreting results?

## Directory structure

Modify the structure outlined below as needed. 

```

├── notebooks                 <- Jupyter notebooks that contain a walkthrough of data analysis steps. 
│
├── results                   <- Results from analysis which may include model outputs, figures, reports.  
|
├── scripts                   <- Scripts to perform generalized data processing and analysis steps. These scripts may be applied in the notebooks.    
|
├── config.yml                <- config file to specify country specific variables   
|
└── README.md                 <- Description of this project.

```

## Reproducing this analysis

Include guidance to:
- Reproduce the required computational environment 
- Run top-level scripts (if needed)
- Configure paramerers