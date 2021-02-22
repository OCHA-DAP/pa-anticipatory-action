### Drought indicators
This folder includes scripts to retrieve and process probabilistic seasonal rainfall forecasts from IRI, ICPAC and NMME. 
For each of the three sources, data can be downloaded automatically. 
For IRI and ICPAC a private key is needed, which should be saved as an environment variable with the name `IRI_AUTH` and `GAPI_AUTH` respectively. 

The scripts are not country specific. To retrieve the relevant forecasts for the country of interest, a separate script should be constructed in the `analyses/[country]/scripts` folder. 
An example for the different processing possibilities is given in `analyses/ethiopia/scripts/eth_seasonalrainfallforecasts.py`

The code in this folder is still in early stages and will continue to be updated.  