"""Code to download and process WRSI data

The WRSI data is made publicly available through
the USGS website. The data is dekadal and each
dekadal dataset is shared through zip files.
https://edcftp.cr.usgs.gov/project/fews/dekadal/

These include two separate zones for West Africa:

- WA: the largest region, covering from the coast at
    5 degrees to around 15 degrees latitude, ranging
    from the western coast of Senegal to the western
    border of Sudan. This is specifically for croplands:
    https://earlywarning.usgs.gov/fews/product/56
- W1: smaller, ranges from around 11 degrees latitude
    to around 17, and from the eastern border of Chad
    to Senegal. This is specifically for rangelands:
    https://earlywarning.usgs.gov/fews/product/57

Both are published using EPSG:7764. There is overlap between the
two datasets where they don't agree, even though both are
measuring the WRSI with respect to millet, due to their
calculations for rangeland and cropland. Since most
countries in the Sahel overlap with both datasets, both
should be considered and we should get clarification from
experts on the differences between the two, how they relate,
and how best to use them.
"""

import os
from typing import Literal, get_args

_base_url = "https://edcftp.cr.usgs.gov/project/fews/dekadal/africa_west/"

_raw_zip_filename = "{year}{dekad}{region}.zip"

_raw_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "raw", "general", "wrsi"
)


def _get_raw_path(year, dekad, region):
    return os.path.join(
        _raw_path,
        _raw_zip_filename.format(year=year, dekad=dekad, region=region),
    )


_processed_filename = "biomasse_{admin}_dekad_{start_dekad}.csv"

_processed_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "processed", "general", "biomasse"
)


def _get_processed_path(admin, start_dekad):
    return os.path.join(
        _processed_path,
        _processed_filename.format(admin=admin, start_dekad=start_dekad),
    )


AdminArgument = Literal["ADM0", "ADM1", "ADM2"]
_valid_admin = get_args(AdminArgument)
