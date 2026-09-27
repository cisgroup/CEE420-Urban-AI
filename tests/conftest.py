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


# Precept 02, the town's own data. Layers are read once per session.
P02_DATA = REPO / "P02" / "data"


@pytest.fixture(scope="session")
def p02():
    return json.loads((P02_DATA / "ground_truth.json").read_text())["p02"]


@pytest.fixture(scope="session")
def municipal():
    """A dict of every layer in the municipal GeoPackage, in the CRS it was served in."""
    path = P02_DATA / "princeton_municipal.gpkg"
    return {name: gpd.read_file(path, layer=name) for name in gpd.list_layers(path)["name"]}


@pytest.fixture(scope="session")
def njgin_parcels():
    return gpd.read_file(P02_DATA / "njgin_parcels.gpkg", layer="parcels")


@pytest.fixture(scope="session")
def grid():
    return gpd.read_file(P02_DATA / "princeton_grid_250m.gpkg", layer="grid")


@pytest.fixture(scope="session")
def census():
    path = P02_DATA / "census_princeton.gpkg"
    return {name: gpd.read_file(path, layer=name) for name in gpd.list_layers(path)["name"]}


# Precept 03, the imagery around the site and the flight. Rasters are opened once per session.
P03_DATA = REPO / "P03" / "data"


@pytest.fixture(scope="session")
def p03():
    return json.loads((P03_DATA / "ground_truth.json").read_text())["p03"]


@pytest.fixture(scope="session")
def site03():
    """A dict of every layer in the site GeoPackage, in the CRS it was written in."""
    path = P03_DATA / "site.gpkg"
    return {name: gpd.read_file(path, layer=name) for name in gpd.list_layers(path)["name"]}


@pytest.fixture(scope="session")
def rasters03():
    """Every GeoTIFF the kit ships, as an open rasterio dataset keyed by file name."""
    import rasterio

    opened = {path.name: rasterio.open(path) for path in sorted(P03_DATA.glob("*.tif"))}
    yield opened
    for src in opened.values():
        src.close()


# Precept 04, the town scene and the labels of the first classifier. Rasters are opened once per session.
P04_DATA = REPO / "P04" / "data"


@pytest.fixture(scope="session")
def p04():
    return json.loads((P04_DATA / "ground_truth.json").read_text())["p04"]


@pytest.fixture(scope="session")
def rasters04():
    """Every GeoTIFF the kit ships, as an open rasterio dataset keyed by file name."""
    import rasterio

    opened = {path.name: rasterio.open(path) for path in sorted(P04_DATA.glob("*.tif"))}
    yield opened
    for src in opened.values():
        src.close()
