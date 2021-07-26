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

The AA frameworks that are implemented and/or currently under development are displayed in the map below

![image](https://drive.google.com/uc?export=view&id=1gjpXeH9rL_7uhZylNwsoOYw_6YHnSopy)

Three of these frameworks have had at least one activation as of June 2021:
- [Bangladesh, flood](https://centre.humdata.org/anticipatory-action-in-bangladesh-before-peak-monsoon-flooding/)
- [Ethiopia, drought-related food insecurity](https://centre.humdata.org/predicting-drought-related-food-insecurity-in-ethiopia/)
- [Somalia, food insecurity](https://www.unocha.org/story/un-central-emergency-response-fund-supports-anticipatory-action-ethiopia-and-somalia)

The other countries with frameworks under development as of June 2021 are:
- Malawi, dry spells & floods
- Nepal, floods
- Philippines, storms
- Burkina Faso, drought
- South Sudan, floods
- Chad, floods
- Niger, drought
- Madagascar, infectious diseases

Some data sources are privately shared by partners, but most of the data is openly available. 

By making our code and learnings open-source, we hope to encourage everyone to together reach more impactful anticipatory action. 
We are always happy to discuss ideas, so don't hesitate to contact us. 

## Examples

The pages below hold documented examples of some of the analyses that are held in this repository. Many of these pages are components of larger, potentially still in progress, pieces of work for an AA pilot. 

#### Malawi
- [Identifying historical dry spells in Malawi](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/notebooks/historical_dry_spells_description.html)
- [Baseline overview of factors relating to dry spells in Malawi](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/notebooks/mwi_impact_summary.html)
- [Forecasting dry spells](https://ocha-dap.github.io/pa-anticipatory-action/analyses/malawi/notebooks/mwi_technical_background_pilot.html)

#### Bangladesh
- [Analysis of satellite imagery to identify past flooding](https://ocha-dap.github.io/pa-anticipatory-action/analyses/bgd/validation/summary_flooding.html)

#### Burkina Faso
- [Baseline overview of risk and vulnerability](https://ocha-dap.github.io/pa-anticipatory-action/analyses/bfa/notebooks/bfa_risk_overview.html)

## Getting started

Create a virtual environment and install the requirements with 
   ``` bash
   pip install -r requirements.txt
   ```
If an error occurs you might have to install spatialindex, on MacOS this can be done with `brew install spatialindex`

The indicators and analyses folders contain more specific information on the data, processing and how to run the scripts. 

## Repository structure
```

├── analyses           <- analyses at the country level 
|
├── dashboard          <- RShiny dashboard to show status of potential upcoming pilot activations
|
├── src                <- data collection and processing scripts, generalized on a per-country or per-indicator level 
|
├── requirements.txt
├── README
└── LICENSE
```

## Data directory

Sync [this](https://drive.google.com/drive/u/3/folders/1RVpnCUpxHQ-jokV_27xLRqOs6qR_8mqQ)
directory from Google drive to your local machine. Create an environment variable called
`AA_DATA_DIR` that points to this directory.

The structure of the directory is as follows:
```
├── AA_DATA_DIR 
     ├── public
     |    ├── exploration <-- data used for notebook analyses, same structure as in raw
     |    ├── processed <-- data taken from raw and processed by scripts in src, 
     |    |                 same substructure as raw
     |    └── raw <-- raw data
     |         ├── [all country iso3s]
     |         |    ├── glofas <-- example data source name
     |         |    └── etc. (all country data sources)
     |         └── glb <-- all global-level data
     |              ├── chirps <-- data source name
     |              └── etc (all global data sources)
     └── private <-- same substructure as public
```
The naming conventions for the data source directories are available in 
[this spreadsheet](https://docs.google.com/spreadsheets/d/155buqH6hcox2IG54NSRkdIjiLcPDmrs6JcjwjdFFA8g/edit?usp=sharing)

#### Development
All code is formatted according to [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) guidelines. 
The repo is set-up to use [pre-commit](https://github.com/pre-commit/pre-commit). So please run `pre-commit install` the first time you are editing. 
Thereafter all commits will be checked against black and flake8 guidelines