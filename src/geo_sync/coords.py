


"""Coordinate helpers for GCJ-02 / WGS84 handling."""

from __future__ import annotations

import math
from typing import Tuple


PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323


def out_of_china(lon: float, lat: float) -> bool:
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(lon: float, lat: float) -> float:
    ret = (
        -100.0
        + 2.0 * lon
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lon * lat
        + 0.2 * math.sqrt(abs(lon))
    )
    ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    ret = (
        300.0
        + lon
        + 2.0 * lat
        + 0.1 * lon * lon
        + 0.1 * lon * lat
        + 0.1 * math.sqrt(abs(lon))
    )
    ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * PI) + 40.0 * math.sin(lon / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * PI) + 300.0 * math.sin(lon / 30.0 * PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon: float, lat: float) -> Tuple[float, float]:
    if out_of_china(lon, lat):
        return lon, lat

    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / (((A * (1 - EE)) / (magic * sqrtmagic)) * PI)
    dlon = (dlon * 180.0) / ((A / sqrtmagic * math.cos(radlat)) * PI)
    mg_lat = lat + dlat
    mg_lon = lon + dlon
    return mg_lon, mg_lat


def gcj02_to_wgs84_approx(lon: float, lat: float, *, iterations: int = 6) -> Tuple[float, float]:
    """
    Approximate inverse transform from GCJ-02 to WGS84.

    This uses iterative refinement, so the output should be treated as an
    approximation suitable for GIS interoperability, not a survey-grade truth.
    """
    if out_of_china(lon, lat):
        return lon, lat

    wgs_lon = lon
    wgs_lat = lat
    for _ in range(max(1, int(iterations))):
        proj_lon, proj_lat = wgs84_to_gcj02(wgs_lon, wgs_lat)
        wgs_lon -= proj_lon - lon
        wgs_lat -= proj_lat - lat
    return wgs_lon, wgs_lat
