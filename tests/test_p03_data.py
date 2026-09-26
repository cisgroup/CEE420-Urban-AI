"""Every published P03 reference number is recomputed from the shipped files.

The kit's rasters arrive in three coordinate systems and two units on purpose, so these
tests hold the pipeline to its promises: every raster carries its provenance, every
vector layer its metadata, nothing that identifies a person ships, and the lessons the
notebooks are built on still hold on the data as published.
"""

import json
import math
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio
from rasterio.features import geometry_mask

from conftest import P03_DATA

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator", re.I)
US_FOOT_M = 0.3048006096
SENSOR_FILES = {"landsat_9": "landsat_20250914_site.tif", "sentinel_2": "s2_20250822_campus_10m.tif",
                "naip_2022": "naip_2022_site.tif", "naip_2023": "naip_2023_site.tif", "nj_2020": "nj2020_site_1ft.tif"}


def cell_m(src):
    unit, factor = src.crs.linear_units_factor
    return src.res[0] * factor


def test_every_sensor_file_reads_as_recorded(rasters03, p03):
    for key, name in SENSOR_FILES.items():
        src, ref = rasters03[name], p03["sensors"][key]
        assert src.crs.to_string() == ref["crs"], f"{name} is in {src.crs}"
        assert round(src.res[0], 4) == ref["res"] and src.dtypes[0] == ref["dtype"] and src.count == ref["bands"]
        assert (src.width, src.height) == (ref["width"], ref["height"])
    crs = {p03["sensors"][k]["crs"] for k in SENSOR_FILES}
    assert len(crs) == 3, "the three coordinate systems are the lesson of 03_01"
    assert rasters03["nj2020_site_1ft.tif"].crs.linear_units == "US survey foot", "the State's file must stay in feet"


def test_every_raster_carries_its_provenance(rasters03):
    for name, src in rasters03.items():
        tags = src.tags()
        assert {"source", "licence", "retrieved"} <= set(tags), f"{name} lacks provenance tags: {sorted(tags)}"
    assert "Copernicus" in rasters03["s2_20250822_campus_10m.tif"].tags()["licence"], "the Sentinel credit is a licence condition"


def test_every_layer_carries_its_provenance_and_the_osm_credit():
    path = P03_DATA / "site.gpkg"
    names = set(gpd.list_layers(path)["name"])
    assert {"site", "places", "section", "buildings_site"} <= names
    for name in names:
        meta = pyogrio.read_info(path, layer=name).get("layer_metadata") or {}
        assert {"source", "licence", "retrieved"} <= set(meta), f"site.gpkg:{name} lacks provenance: {meta}"
        assert "OpenStreetMap" in meta.get("attribution", ""), f"site.gpkg:{name} does not credit OpenStreetMap"


def test_the_site_and_its_rectangle_reproduce(site03, p03):
    field = site03["site"].to_crs("EPSG:26918").geometry.iloc[0]
    corners = np.array(field.minimum_rotated_rectangle.exterior.coords)[:4]
    sides = np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)
    s = p03["site"]
    assert round(field.area / 1e4, 2) == s["area_ha"]
    assert round(sides.max(), 1) == s["rect_length_m"] and round(sides.min(), 1) == s["rect_width_m"]
    assert len(site03["buildings_site"]) == s["buildings_within_400m"]
    assert s["datum"]["pyproj_default_shift_m"] == 0.0 and 1.0 < s["datum"]["itrf2014_plan_shift_m"] < 1.6, "the datum lesson of 03_01"
    assert abs(np.hypot(s["datum"]["itrf2014_shift_e_m"], s["datum"]["itrf2014_shift_n_m"]) - s["datum"]["itrf2014_plan_shift_m"]) < 0.02, "the datum's east and north make its magnitude"


def test_pixels_across_the_field(rasters03, p03):
    for key, name in SENSOR_FILES.items():
        assert round(p03["site"]["rect_length_m"] / cell_m(rasters03[name]), 1) == pytest.approx(p03["sensors"][key]["pixels_across_field"], abs=0.2)
    assert p03["sensors"]["nj_2020"]["res"] == 1.0 and p03["sensors"]["nj_2020"]["res_m"] == round(US_FOOT_M, 4), "one foot is the 513 trap in raster form"


def test_ndvi_at_the_five_places(rasters03, site03, p03):
    places = site03["places"].to_crs("EPSG:32618")
    for name, key in (("s2_20250822_campus_10m.tif", "places_20250822"), ("s2_20250310_campus_10m.tif", "places_20250310")):
        readings = list(rasters03[name].sample([(g.x, g.y) for g in places.geometry]))
        for place, (b, g, r, n) in zip(places["name"], readings):
            assert round(float((n - r) / (n + r)), 3) == pytest.approx(p03["ndvi"][key][place], abs=0.002), f"{place} on {name}"
    aug, mar = p03["ndvi"]["places_20250822"], p03["ndvi"]["places_20250310"]
    assert aug["Institute Woods"] > 0.8 > 0.5 > mar["Institute Woods"], "the leaf-off lesson is gone"
    assert aug["Lake Carnegie"] < 0 and mar["Lake Carnegie"] < 0 and abs(aug["Nassau Street"]) < 0.25


def test_the_year_of_three_pixels(p03):
    year = pd.read_csv(P03_DATA / "ndvi_year.csv")
    y = p03["ndvi"]["year"]
    assert year["item"].nunique() == y["passes"] and year["date"].nunique() == y["dates"] and int(year["clear"].sum()) == y["clear_readings"]
    clear = year[year["clear"]]
    for place, ref in y["places"].items():
        g = clear[clear["place"] == place]
        above = g[g["ndvi"] >= 0.5].sort_values("date")
        assert len(g) == ref["clear_readings"] and str(above["date"].iloc[0]) == ref["first_above_0_5"] and str(above["date"].iloc[-1]) == ref["last_above_0_5"]
    woods = y["places"]["Institute Woods"]
    assert woods["first_above_0_5"] > "2025-04-01" and woods["last_above_0_5"] < "2025-11-15", "the deciduous season is the lesson"


def test_the_camera_and_the_plan_are_consistent(p03):
    c, pl = p03["camera"], p03["plan"]
    gsd_m = c["pixel_um"] * 1e-6 * c["altitude_m"] / (c["focal_mm"] * 1e-3)
    assert round(gsd_m * 100, 2) == c["gsd_cm"]["80"]
    across, along = c["width_px"] * gsd_m, c["height_px"] * gsd_m
    assert [round(across, 1), round(along, 1)] == c["footprint_m"]
    for forward, side in ((80, 70), (60, 30)):
        o = pl[f"overlap_{forward}_{side}"]
        assert o["base_m"] == pytest.approx(along * (1 - forward / 100), abs=0.01)
        assert o["spacing_m"] == pytest.approx(across * (1 - side / 100), abs=0.01)
        assert o["per_km2"] == pytest.approx(1e6 / (o["base_m"] * o["spacing_m"]), rel=0.01)
        assert o["n_photos"] == o["n_strips"] * (o["n_per_strip"] + 2), "one extra photograph beyond each end of each strip"
        assert o["n_strips"] == math.ceil(pl["rect_width_m"] / o["spacing_m"]) + 1
        assert o["n_per_strip"] == math.ceil(pl["rect_length_m"] / o["base_m"]) + 1
    assert pl["overlap_80_70"]["n_photos"] > 2.5 * pl["overlap_60_30"]["n_photos"], "the two densities must differ by the factor the lecture quotes"
    assert pl["chosen"]["forward_pct"] >= 80 and pl["chosen"]["interval_s"] >= c["min_interval_s"]
    assert p03["target_cm"] == round(10 * gsd_m * 100)


def test_the_ground_under_the_field(rasters03, site03, p03):
    src = rasters03["dem_3dep_1m_site.tif"]
    z = src.read(1)
    field = site03["site"].to_crs(src.crs).geometry.iloc[0]
    inside = ~geometry_mask([field], out_shape=z.shape, transform=src.transform)
    t = p03["terrain"]
    assert src.tags()["units"] == t["units"] == "metres above NAVD88"
    assert round(float(np.median(z[inside])), 2) == t["site_ground_m"]
    assert round(float(z[inside].max() - z[inside].min()), 2) == t["relief_over_site_m"] < 5, "a playing field is flat"
    assert round(float(z.min()), 2) == t["window_min_m"] and round(float(z.max()), 2) == t["window_max_m"]
    assert -40 < t["geoid18_n_m"] < -25, "GEOID18 in New Jersey is about minus thirty three metres"
    assert "date" in t["vintage"], "the missing date of the lidar is part of the lesson"
    section = site03["section"]
    assert int(section["azimuth_deg"].iloc[0]) == t["section_azimuth_deg"] and float(section["length_m"].iloc[0]) == t["section_length_m"]


def test_heights_from_shadows(p03):
    s = p03["shadows"]
    assert s["item"].startswith("nj_m_") and 45 < s["sun_elevation_deg"] < 60
    assert abs(s["sun_elevation_deg"] - s["sun_elevation_nominal_deg"]) > 5, "the nominal time stamp must not be the solved one, or the lesson is gone"
    guyot = next(b for b in s["buildings"] if b["name"] == "Guyot Hall")
    assert 10 < guyot["height_m"] < 20 and guyot["rays"] >= 5
    assert round(guyot["shadow_m"] * math.tan(math.radians(s["sun_elevation_deg"])), 1) == guyot["height_m"]


def test_nothing_that_identifies_a_person_ships():
    for gpkg in sorted(P03_DATA.glob("*.gpkg")):
        for name in gpd.list_layers(gpkg)["name"]:
            bad = [c for c in gpd.read_file(gpkg, layer=name, rows=1).columns if PII.search(c)]
            assert not bad, f"{gpkg.name}:{name} carries {bad}"
    for csv in sorted(P03_DATA.glob("*.csv")):
        bad = [c for c in pd.read_csv(csv, nrows=1).columns if PII.search(c)]
        assert not bad, f"{csv.name} carries {bad}"
    if (P03_DATA / "flight.json").exists():
        flight = json.loads((P03_DATA / "flight.json").read_text())
        assert not {"operator", "operator_certificate", "approval_reference"} & set(flight), "the operator's details never ship"
        assert flight.get("photos_checked_for_people") is True


def test_published_p03_ground_truth_carries_no_answers():
    published = json.loads((P03_DATA / "ground_truth.json").read_text())
    leaked = [k for k in ("p03_answers", "p02_answers", "duel", "stretch", "mystery") if k in published]
    assert not leaked, f"published P03 ground truth leaks {leaked}"
    assert set(published) == {"generated", "p03"}


@pytest.mark.parametrize("name", ["princeton_boundary.geojson", "site.gpkg", "ndvi_year.csv"])
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries its own copies."""
    assert (P03_DATA / name).exists()


def test_the_flight_table_matches_its_summary(p03):
    """03_05 reads the flight as a table; every number the ground truth quotes about it is recomputed here."""
    if not (P03_DATA / "photos.csv").exists():
        pytest.skip("no flight published yet")
    photos = pd.read_csv(P03_DATA / "photos.csv")
    flight = json.loads((P03_DATA / "flight.json").read_text())
    f = p03["flown"]
    assert len(photos) == f["n_photos"] == flight["n_photos"] and flight["label"] == f["label"]
    assert round(float(photos["seconds_from_previous"].median()), 2) == f["median_interval_s"]
    assert round(float(photos["rel_alt_m"].median()), 1) == f["median_relative_alt_m"]
    assert photos["camera_model"].nunique() == 1 and str(photos["camera_model"].iloc[0]) == f["camera_model"]
    assert f["coverage_inside_site"]["min"] >= 3, "part of the field is seen by fewer than three photographs; the flight cannot be modelled there"
    assert 0 < f["forward_overlap_pct"] < 100 and abs(f["sun_elevation_deg_start"] - f["sun_elevation_deg_end"]) < 15
    for name in flight["samples"]:
        path = P03_DATA / name["file"]
        assert path.exists() and path.stat().st_size < 1_500_000, f"{name['file']} is missing or too large to ship"
    from PIL import Image
    exif = Image.open(P03_DATA / "photo_01.jpg").getexif()
    assert 42033 not in exif.get_ifd(0x8769) and 37500 not in exif.get_ifd(0x8769), "a sample photograph still carries a serial number or the maker note"


def test_the_products_match_their_summary(p03, rasters03, site03):
    """03_06 to 03_08 read the products; the numbers the ground truth quotes about them are recomputed here."""
    if "ortho" not in p03:
        pytest.skip("no products published yet")
    o, src = p03["ortho"], rasters03["ortho_10cm.tif"]
    assert src.crs.to_string() == o["crs"] and (src.width, src.height) == (o["width_shipped"], o["height_shipped"]) and src.count == o["bands"]
    mask = src.read_masks(1) > 0
    site_o = site03["site"].to_crs(src.crs).geometry.iloc[0]
    inside = ~geometry_mask([site_o], out_shape=mask.shape, transform=src.transform)
    assert round(float(mask[inside].mean()), 3) == o["valid_share_of_site"] > 0.95, "the model must cover the field"
    assert o["pixels_full"] == o["width_full"] * o["height_full"] > 30_000_000, "the lecture promised tens of millions of pixels"
    assert src.overviews(1) and src.block_shapes[0][0] >= 256, "the shipped orthomosaic must be tiled with overviews, the lesson of 03_06"
    h = p03["heights"]
    assert h["geoid18_n_m"] == p03["terrain"]["geoid18_n_m"]
    assert abs(h["vertical_shift_m"] - h["geoid18_n_m"] - h["datum_height_shift_m"] - h["receiver_height_error_m"]) < 0.01, "the vertical gap is the geoid plus the datum plus the receiver"
    assert h["vertical_shift_mad_m"] < 1.0, "the drone's surface and the lidar disagree on the lawn by more than their precision"
    assert h["dtm_minus_3dep_under_trees_median_m"] > h["dtm_minus_3dep_on_lawn_median_m"], "the software's terrain under trees should float above the lidar's"
    c = p03["control"]
    for ref in c["usable_references"]:
        assert c["references"][ref]["self_test"]["passed"] and not ref.startswith("ortho_"), "only independent references that pass the self test enter the consensus"
    assert abs(c["consensus_e_m"] - c["datum_e_m"] - c["receiver_e_m"]) < 0.02 and abs(c["consensus_n_m"] - c["datum_n_m"] - c["receiver_n_m"]) < 0.02, "the consensus is the datum plus the receiver"
    assert c["consensus_e_m"] is not None and np.hypot(c["consensus_e_m"], c["consensus_n_m"]) < 10, "an offset of more than ten metres means the check failed, not the flight"
    with rasterio.open(P03_DATA / "dsm_25cm.tif") as dsm:
        assert dsm.nodata == h["nodata"] and dsm.dtypes[0] == "float32" and dsm.compression is not None and dsm.compression.name != "jpeg", "heights are never JPEG compressed"
