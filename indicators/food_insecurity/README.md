This folder includes scripts to process IPC data by FewsNet and GlobalIPC. Moreover, the scripts compute whether a predefined trigger is met.

### Computation for existing country
1. Run `process_fewsnet.py [Country ISO code]` or `process_fewsnet_worldpop.py [Country ISO code]` this will return two csv's with the IPC phases of the FewsNet data for  for the current situation (CS), projections up to four months ahead (ML1) and projections up to 8 months ahead (ML2). One IPC phase is assigned per admin2 together with the population, per admin1 the population per IPC phase is returned, based on the admin2 results.  
2. Run `process_globalipc.py [Country ISO code]` this will return two csv's with the IPC phases of the GlobalIPC data per admin2 and admin1. For each spatial level the population per IPC phase is returned. 
3. Run `IPC_computetrigger.py[Country ISO code]` this will return a csv with processed columns, including if defined triggers are met. The FewsNet and GlobalIPC data are combined in this script, if they are both present
3. Do further analysis. The jupyter notebooks in `ethiopia/` can guide as examples

### Adding a new country
##### General
1. Download the shapefiles of the country, one on admin2 and one on admin1 level. Place the files in `country_name/Data` and set the specific path in the `config.yml`. 
Generally shapefiles can be found on the [Humanitarian Data Exchange](data.humdata.org)), here it is good practice to go to the data grid of the country, for which the url is `https://data.humdata.org/group/[COUNTRY ISO3 CODE]` and then look at Geography&Infrastructure --> administrative boundaries. FewsNet also [provides administrative boundaries](https://fews.net/fews-data/334). 
2. Add the country-specific variables to `config.yml`

3a. If using `process_fewsnet.py`, download regional population data for one year and place it in `country_name/Data`. Often available on the [Humanitarian Data Exchange](data.humdata.org) through the data grid (`https://data.humdata.org/group/[COUNTRY ISO3 CODE]`) under Population & Socio-economy --> Baseline population
3b. If using `process_fewsnet_worldpop.py`, download WorldPop's raster population data for all years to be included and save them in `country_name/Data/WorldPop`. The script expects to find the 1km, UNAdjusted files there.

##### FewsNet
1. Download [all FewsNet IPC classifications](https://fews.net/fews-data/333) that covers the country of interest and place it in `Data/FewsNetRaw`. 
Check if FewsNet publishes regional classifications that include your country of interest and/or country specific classifications. 
Both should be included and will be automatically masked to the country shapefile by the code.
Don't change the folder and file names since they are assumed to have the same naming as how they are published on the FewsNet website.
##### GlobalIPC
1. Download the excel with country IPC classifications from [the IPC Global tracking tool](http://www.ipcinfo.org/ipc-country-analysis/population-tracking-tool/en/) and save it to `country_name/Data`.
2. Change column names to be compatible with `process_globalipc.py`. An example can be found in `ethiopia/Data/GlobalIPC_newcolumnnames.xlsx`