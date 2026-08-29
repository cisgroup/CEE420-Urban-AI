import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
P01_DATA = REPO / "P01" / "data"


@pytest.fixture(scope="session")
def ground_truth():
    return json.loads((P01_DATA / "ground_truth.json").read_text())


@pytest.fixture(scope="session")
def boundary():
    return gpd.read_file(P01_DATA / "princeton_boundary.geojson")


@pytest.fixture(scope="session")
def buildings():
    return gpd.read_file(P01_DATA / "princeton_buildings.geojson")


@pytest.fixture(scope="session")
def food():
    return gpd.read_file(P01_DATA / "princeton_food.geojson")


@pytest.fixture(scope="session")
def mystery():
    return gpd.read_file(P01_DATA / "mystery_boundary.geojson")


@pytest.fixture(scope="session")
def landmarks():
    df = pd.read_csv(P01_DATA / "landmarks.csv")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
