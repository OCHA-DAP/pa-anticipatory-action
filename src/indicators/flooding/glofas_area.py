from collections import namedtuple
from typing import Dict

from shapely.geometry import Polygon

Station = namedtuple("Station", "lon lat")


class Area:
    def __init__(self, north, south, east, west):
        self.north = north
        self.south = south
        self.east = east
        self.west = west

    def list_for_api(self):
        return [self.north, self.west, self.south, self.east]


class AreaFromStations(Area):
    def __init__(self, stations: Dict[str, Station], buffer: float = 0.5):
        """
        Args:
            stations: dictionary of form {station_name: Station]
            buffer: degrees above / below maximum lat / lon from stations to include in GloFAS query
        Returns:
            list with format [N, W, S, E]
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
    def __init__(self, shape: Polygon):
        # bounds is of form (minx, miny, maxx, maxy)
        super().__init__(
            north=shape.bounds[3],
            south=shape.bounds[1],
            east=shape.bounds[2],
            west=shape.bounds[0]
        )
