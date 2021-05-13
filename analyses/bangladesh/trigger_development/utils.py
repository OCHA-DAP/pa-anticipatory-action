from pathlib import Path
import os

import pandas as pd


GLOFAS_DS_FOLDER = Path(os.environ['AA_DATA_DIR']) / 'exploration/bangladesh/GLOFAS_Data'
print(GLOFAS_DS_FOLDER)


def get_glofas_df(glofas_dir: Path = GLOFAS_DS_FOLDER,
                  district_list: list = None,
                  year_min: int = 1979,
                  year_max: int = 2021) -> pd.DataFrame:
    glofas_df = pd.DataFrame(columns=district_list)
    for year in range(year_min, year_max):
        glofas_filename = Path(f'{year}.csv')
        glofas_df = glofas_df.append(
            pd.read_csv(glofas_dir / glofas_filename,
                        index_col=0))
    glofas_df.index = pd.to_datetime(glofas_df.index, format='%Y-%m-%d')
    return glofas_df
