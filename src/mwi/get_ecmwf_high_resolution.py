"""
Get and process the high resolution data from ECMWF directly.

ECMWF distributes the recent data to our AWS bucket, the historical data is
available from their API.
To access the ECMWF API, you need an authorized account.
More information on the ECMWF API and how to initialize the usage,
can be found
`here <https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api>`_

THe recent data from the AWS bucket currently needs to manually copied to our
Google Drive. Do this by accessing the bucket (credentials in our BitWarden)
and selecting the files starting with T4L. Copy these to `raw/public/ecmwf` in
the AA_DATA_DIR folder.

This script uses the ecmwf branch of the aa-toolbox repo. Since this is in
experimental state, it has to be pip installed with the following command.

Note that once the code in the toolbox is finalized, toolbox can be installed
as default in this repository and this script could be incorporated in the
trigger script.
"""

# pip install -e git+https://github.com/OCHA-DAP/pa-aa-toolbox.git@feature/
# ecmwf-seas-realtime#egg=aa-toolbox

from aatoolbox.datasources.ecmwf.api_seas5_monthly import EcmwfApi
from aatoolbox.datasources.ecmwf.combine_seas5_monthly import Ecmwf
from aatoolbox.pipeline import Pipeline

iso3 = "mwi"
pipeline_mwi = Pipeline(iso3)
# loading from config such that it matches the realtime coordinates
mwi_geobb = pipeline_mwi.load_geoboundingbox(
    from_codab=False, from_config=True
)
ecmwf_api = EcmwfApi(iso3=iso3, geo_bounding_box=mwi_geobb)
# min_date to limit the amount of data being downloaded. But if you want to
# do historical analysis you can set this to None (the default) to download all
# data
ecmwf_api.download(min_date="2021-01-01")
ecmwf_api.process()

pipeline_mwi.load_ecmwf_realtime(process=True)

ecmwf = Ecmwf(iso3=iso3)
# note need to improve process_sources=True, so process them separately first
# and combine them after so that can set process_sources=False
ecmwf.process(process_sources=False)
print(ecmwf.load())
