import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import rioxarray

from src.utils_general.raster_manipulation import compute_raster_statistics

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
PUBLIC_DATA_DIR = "public"
RAW_DATA_DIR = "raw"
PROCESSED_DATA_DIR = "processed"
ARC2_DIR = "arc2"


class ARC2:
    def __init__(
        self, country_iso3: str, date_min: str, date_max: str, range_x, range_y
    ):
        self.country_iso3 = country_iso3
        self.date_min = date_min
        self.date_max = date_max
        self.range_x = range_x
        self.range_y = range_y

    def _download(self):

        """
        Download data from IRI servers and save raw .nc file on
        gdrive at the location returned from get_raw_filepath
        """

        # TODO: Configure url with input parameters
        url = """
        https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/
        .Africa/.DAILY/.ARC2/.daily/.est_prcp/T/
        %281%20Jan%202021%29%2830%20Mar%202021%29RANGEEDGES/
        X/%2832E%29%2836E%29RANGEEDGES/
        Y/%2820S%29%285S%29RANGEEDGES/
        data.nc
        """

        raw_filepath = self._get_raw_filepath(
            self.country_iso3, self.date_min, self.date_max
        )

        if os.path.exists(raw_filepath):
            os.remove(raw_filepath)

        cookies = {
            "__dlauth_id": os.getenv("IRI_AUTH"),
        }

        # TODO: Add in logging to let user know which file is being downloaded
        response = requests.get(url, cookies=cookies, verify=False)

        Path(raw_filepath.parent).mkdir(parents=True, exist_ok=True)

        with open(raw_filepath, "wb") as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)
        return

    def _get_raw_filepath(
        self, country_iso3: str, date_min: str, date_max: str
    ) -> Path:
        directory = (
            DATA_DIR / PUBLIC_DATA_DIR / RAW_DATA_DIR / country_iso3 / ARC2_DIR
        )
        filename = f"arc2_daily_precip_{country_iso3}_{date_min}_{date_max}.nc"
        return directory / Path(filename)

    def _get_processed_filepath(
        self,
        country_iso3: str,
        date_min: str,
        date_max: str,
        agg_method: str = "centroid",
    ) -> Path:
        directory = (
            DATA_DIR
            / PUBLIC_DATA_DIR
            / PROCESSED_DATA_DIR
            / country_iso3
            / ARC2_DIR
        )
        filename = (
            f"arc2_{agg_method}_long_{country_iso3}_{date_min}_{date_max}.csv"
        )
        return directory / Path(filename)

    def _get_monitoring_filepath(
        self, country_iso3: str, date_max: str
    ) -> Path:
        directory = (
            DATA_DIR
            / PUBLIC_DATA_DIR
            / PROCESSED_DATA_DIR
            / country_iso3
            / ARC2_DIR
            / "monitoring"
        )
        filename = f"{date_max}_results.txt"
        return directory / Path(filename)

    def _write_to_monitoring_file(self, dry_spells=None):
        monitoring_file = self._get_monitoring_filepath(
            self.country_iso3, self.date_max
        )
        """
        Write a simple output of the number
        of dry spells observed in the last 14 days.
        """
        result = ""
        with open(monitoring_file, "w") as f:
            if dry_spells is not None:
                result += "No dry spells identified in the last 14 days."
                f.write(result)
            else:
                f.write(
                    f"Dry spells identified in \
                    {len(dry_spells)} admin regions:\n{dry_spells}"
                )
        f.close()
        return

    def process_data(
        self,
        crs,
        clip_bounds=None,
        bound_col: str = None,
        all_touched: bool = False,
    ):

        """
        Get mean aggregation by admin boundary for the downloaded arc2 data.
        Outputs a csv with daily aggregated statistics.
        """

        if all_touched:
            agg_method = "touching"
        else:
            agg_method = "centroid"

        processed_filepath = self._get_processed_filepath(
            self.country_iso3, self.date_min, self.date_max, agg_method
        )

        raw_filepath = self._get_raw_filepath(
            self.country_iso3, self.date_min, self.date_max
        )

        da = (
            rioxarray.open_rasterio(raw_filepath, masked=True)
            .squeeze()
            .rio.write_crs(f"EPSG:{crs}")
        )
        gdf = gpd.read_file(clip_bounds)

        df_zonal_stats = compute_raster_statistics(
            gdf=gdf,
            bound_col=bound_col,
            raster_array=da,
            all_touched=all_touched,
            stats_list=["mean"],
        )

        Path(processed_filepath.parent).mkdir(parents=True, exist_ok=True)
        df_zonal_stats.to_csv(processed_filepath, index=False)
        return df_zonal_stats

    def identify_dry_spells(
        self,
        rolling_window: int = 14,
        rainfall_mm: int = 2,
        agg_method: str = "centroid",
    ):
        """
        Read the processed data and check if any dry spells occurred
        in that time period. Processed data should cover >= 14 days of
        precipitation. We're assuming that this data is from during
        the rainy season.
        """

        processed_file = self._get_processed_filepath(
            self.country_iso3, self.date_min, self.date_max, agg_method
        )

        print(processed_file)

        # TODO:
        # 1. Read in processed data
        df = pd.read_csv(processed_file)

        # 2. Check that it covers the min days needed to define a dry spell

        # 3. Calculate the rolling sum
        adm_col = df.columns[2]
        precip_col = df.columns[1]
        grouped = df.groupby(adm_col)[precip_col].rolling(rolling_window).sum()

        # 4. Identify dry spells based on rolling sum (rainfall_mm)
        print(grouped)

        # 5. Notify if any admin areas are in a dry spell
        return
