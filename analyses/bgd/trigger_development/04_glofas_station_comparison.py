from utils import utils

import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
import pandas as pd

STATION_LIST = [
    "Noonkhawa",
    "Chilmari",
    "Bahadurabad",
    "Sariakandi",
    "Kazipur",
    "Serajganj",
    "Aricha",
]

glofas_df = utils.get_glofas_df()

# Get x axis

day_range = (glofas_df.index[-1] - glofas_df.index[0]).days
x = np.arange(-day_range, day_range + 1)

offset_df = pd.DataFrame(index=STATION_LIST, columns=STATION_LIST)
offset_df.index.name = "station"
interp_offset = 10

for station1 in STATION_LIST:
    for station2 in STATION_LIST:
        y = correlate(
            glofas_df[f"dis24_{station1}"], glofas_df[f"dis24_{station2}"]
        )
        imax = np.argmax(y)
        offset_crude = x[imax]
        # Do interpolation
        x_sub = x[imax - interp_offset : imax + interp_offset + 1] * 24
        y_sub = y[imax - interp_offset : imax + interp_offset + 1]
        y_interp_func = interp1d(x_sub, y_sub, kind="cubic")
        x_hours = np.arange(x_sub[0], x_sub[-1] + 1)
        y_hours = y_interp_func(x_hours)
        offset = x_hours[np.argmax(y_hours)]
        offset_df.at[station1, station2] = offset
print(offset_df)
offset_df.to_csv("offset.csv")
