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


# Precept 03b, the flight's photographs through the software's stages.
P03B_KIT = REPO / "P03b"
P03B_DATA = P03B_KIT / "data"


@pytest.fixture(scope="session")
def sfm():
    """The kit's toolbox, loaded from its file so that no sys.path entry leaks into other tests."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("sfm_tools_for_tests", P03B_KIT / "sfm_tools.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def p03b():
    return json.loads((P03B_DATA / "ground_truth.json").read_text())["p03b"]


@pytest.fixture(scope="session")
def choices03b():
    return json.loads((P03B_DATA / "flight.json").read_text())["p03b"]


# Precept 05, the lecture's footprint model on one test block, and one street image through a published segmenter.
P05_KIT = REPO / "P05"
P05_DATA = P05_KIT / "data"


@pytest.fixture(scope="session")
def seg():
    """The kit's toolbox, loaded from its file so that no sys.path entry leaks into other tests."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("seg_tools_for_tests", P05_KIT / "seg_tools.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def p05():
    return json.loads((P05_DATA / "ground_truth.json").read_text())["p05"]


@pytest.fixture(scope="session")
def scene05():
    return json.loads((P05_DATA / "scene.json").read_text())


# Precept 06, the zero-shot models of Lecture 06 on the test blocks, and the town's satellite embeddings.
P06_KIT = REPO / "P06"
P06_DATA = P06_KIT / "data"


@pytest.fixture(scope="session")
def fm():
    """The kit's toolbox, loaded from its file so that no sys.path entry leaks into other tests."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("fm_tools_for_tests", P06_KIT / "fm_tools.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def p06():
    return json.loads((P06_DATA / "ground_truth.json").read_text())["p06"]


@pytest.fixture(scope="session")
def scene06():
    return json.loads((P06_DATA / "scene.json").read_text())
