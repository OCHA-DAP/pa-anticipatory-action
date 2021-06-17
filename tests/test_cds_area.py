from shapely.geometry import Polygon

from src.indicators.flooding.cds.area import Station, AreaFromStations, AreaFromShape


FAKE_STATIONS = {
    "station_north": Station(lon=0, lat=1),
    "station_south": Station(lon=0, lat=-2),
    "station_east": Station(lon=3, lat=0),
    "station_west": Station(lon=-4, lat=0),
}


def test_get_area_from_stations():
    area = AreaFromStations(FAKE_STATIONS, buffer=0.1)
    assert area.north == 1.1
    assert area.south == -2.1
    assert area.east == 3.1
    assert area.west == -4.1


def test_get_list_for_api():
    area = AreaFromStations(FAKE_STATIONS, buffer=0)
    assert area.list_for_api() == [1, -4, -2, 3]


def test_get_area_from_shape():
    n, s, e, w = (1, -2, 3, -4)
    shape = Polygon([(e, n), (e, s), (w, s), (w, n)])
    area = AreaFromShape(shape)
    assert area.north == n
    assert area.south == s
    assert area.east == e
    assert area.west == w
