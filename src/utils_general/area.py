from collections import namedtuple
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np

Station = namedtuple("Station", "lon lat")


class Area:
    def __init__(self, north: float, south: float, east: float, west: float):
        self.north = north
        self.south = south
        self.east = east
        self.west = west

    def list_for_api(
        self,
        round_val: float = None,
        offset_val: float = None,
    ) -> List[float]:
        """
        List the coordinates in the order that they're needed for the
        API :param do_not_round: Don't round to the format x.y5, which
        is required for the API. Only left as a parameter for now to be
        able to replicate older datasets -- should generally not be
        toggled :return: List of coordinates in the correct order for
        the API (north, west, south, east)
        """
        if round_val is None and offset_val is None:
            return [self.north, self.west, self.south, self.east]
        # Round North and East up, South and West down (to maximize area)
        north = self._round_coord(
            coord=self.north,
            direction="up",
            round_val=round_val,
            offset_val=offset_val,
        )
        east = self._round_coord(
            coord=self.east,
            direction="up",
            round_val=round_val,
            offset_val=offset_val,
        )
        south = self._round_coord(
            coord=self.south,
            direction="down",
            round_val=round_val,
            offset_val=offset_val,
        )
        west = self._round_coord(
            coord=self.west,
            direction="down",
            round_val=round_val,
            offset_val=offset_val,
        )
        return [north, west, south, east]

    @staticmethod
    def _round_coord(
        coord: float, direction: str, round_val, offset_val
    ) -> float:
        """
        Rounding coordinates
        Used for GloFAS in the CDS API, to the format x.y5
        :param coord: The coordinate to round
        :param direction: Round up or down
        :return: Coordinate rounded to x.y5
        """
        if direction == "up":
            function = np.ceil
            offset_factor = 1
        elif direction == "down":
            function = np.floor
            offset_factor = -1
        return (
            function(coord / round_val) * round_val
            + offset_factor * offset_val
        )


class AreaFromStations(Area):
    def __init__(self, stations: Dict[str, Station], buffer: float = 0.2):
        """
        Args: stations: dictionary of form {station_name: Station]
            buffer: degrees above / below maximum lat / lon from
            stations to include.
             Used to query GloFAS from CDS
             Returns: list with format [N, W, S, E]
        """
        lon_list = [station.lon for station in stations.values()]
        lat_list = [station.lat for station in stations.values()]
        super().__init__(
            north=max(lat_list) + buffer,
            south=min(lat_list) - buffer,
            east=max(lon_list) + buffer,
            west=min(lon_list) - buffer,
        )


class AreaFromShape(Area):
    def __init__(self, shape: Union[gpd.GeoSeries, gpd.GeoDataFrame]):
        # total_bounds is of form (minx, miny, maxx, maxy)
        super().__init__(
            north=shape.total_bounds[3],
            south=shape.total_bounds[1],
            east=shape.total_bounds[2],
            west=shape.total_bounds[0],
        )
