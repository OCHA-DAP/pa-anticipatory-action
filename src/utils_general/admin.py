import os
from pathlib import Path


def _get_shapefile_base_dir(country_iso3):
    return (
        Path(os.environ["AA_DATA_DIR"])
        / "public"
        / "raw"
        / country_iso3
        / "cod_ab"
    )


def get_shapefile(country_iso3, base_dir, base_zip, base_shapefile):
    return (
        _get_shapefile_base_dir(country_iso3)
        / base_dir
        / f"{base_zip}!{base_shapefile}"
    )
