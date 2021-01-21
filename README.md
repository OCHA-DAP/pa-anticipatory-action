# OCHA Anticipatory Action Framework
**Note: the code presented in this repository is work in progress. If you have questions or run into any issues, please contact us at centrehumdata@un.org.**

An OCHA anticipatory action framework (AAF) comprises three pre-agreed pillars: 
financing, activities, and a trigger. 
The trigger defines an exceptional shock (cause and impact) to which the framework should respond. 
The framework activates when projections indicate that the shock is likely to realise. 
Funds are then rapidly disbursed and interventions are implemented. 

The trigger is based on data and enables the automated activation of the framework. 
This repository contains the triggers that are part of the several AAF pilots. 
While the triggers are adjusted to the local context, most of the data processing is transferable to other countries.

The AAF pilots implemented as of January 2021 are:
- Bangladesh, flood
- Somalia, food insecurity
- Ethiopia, drought-related food insecurity
- Malawi and Chad, exploratory phase

Some data sources are privately shared by partners, but most of the data is openly available. 

By making our code and learnings open-source, we hope to encourage everyone to together reach more impactful anticipatory action. 
We are always happy to discuss ideas, so don't hesitate to contact us. 

## Getting started
Create a virtual environment and install the requirements with 
   ``` bash
   pip install -r requirements.txt
   ```
If an error occurs you might have to install spatialindex, on MacOS this can be done with `brew install spatialindex`

The indicators and analyses folders contain more specific information on the data, processing and how to run the scripts. 


## Repository structure
```
├── indicators              <- generalized code to retrieve and process data, compute indices, etc.
|    ├── drought            <- contains indicator related scripts
|    |    ├── data          <- this data is not saved on Github, but folder will be created when data is downloaded
|    |    └── README        <- indicator specific information
|    |
|    ├── flooding
|    |
|    ├── food_insecurity
|    |
|    └── cholera
|
├── analyses <- analyses at the country level 
|    ├── country_template   <- contains standardized country directory structure
|    |    ├── notebooks     <- Jupyter notebooks that contain a walkthrough of data analysis steps. 
|    |    |
|    |    ├── results       <- Results from analysis which may include model outputs, figures, reports.  
|    |    |
|    |    ├── scripts       <- Scripts to perform generalized data processing and analysis steps. These scripts might refer to the indicators folders
|    |    |
|    |    ├── data          <- Can include raw data as well as processed data
|    |    |
|    |    ├── config.yml    <- config file to specify country specific variables   
|    |    |
|    |    └── README.md     <- details about the project, instructions to reproduce the analysis
|    ├── bangladesh
|    |
|    ├── chad
|    |
|    ├── ethiopia
|    |
|    ├── malawi
|    |
|    └── somalia
|
├── utils_general           <-- All-purpose code that is generalizable across indicators and/or countries
|
├── requirements.txt
|
├── README
|
└── LICENSE
```