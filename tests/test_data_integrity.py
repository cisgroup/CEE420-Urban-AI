"""Every published reference number is recomputed from the shipped files."""

import json
from pathlib import Path

METERS = "EPSG:26918"
NJ_FEET = "EPSG:3424"


def test_boundary_matches_ground_truth(boundary, ground_truth):
    area = boundary.to_crs(METERS).geometry.area.iloc[0] / 1e6
    assert round(area, 2) == ground_truth["boundary"]["area_km2"]
    assert [round(v, 4) for v in boundary.total_bounds] == ground_truth["boundary"]["total_bounds"]


def test_layer_counts(buildings, food, ground_truth):
    assert len(buildings) == ground_truth["counts"]["buildings"]
    assert len(food) == ground_truth["counts"]["food"]


def test_duel_answer(food, landmarks, ground_truth):
    e225 = landmarks[landmarks.id == "e225"].to_crs(METERS).geometry.iloc[0]
    radius = ground_truth["walk_radius_m"]
    count = int(food.to_crs(METERS).within(e225.buffer(radius)).sum())
    assert count == ground_truth["duel"]["food_within_400m_of_e225"]


def test_stretch_answers(buildings, landmarks, ground_truth):
    dinky = landmarks[landmarks.id == "dinky"].to_crs(METERS).geometry.iloc[0]
    near = int(buildings.to_crs(METERS).intersects(dinky.buffer(400)).sum())
    assert near == ground_truth["stretch"]["buildings_within_400m_of_dinky"]
    largest = buildings.to_crs(NJ_FEET).geometry.area.max()
    assert round(largest) == ground_truth["stretch"]["largest_building_ft2"]


def test_mystery_fails_the_three_checks(mystery, ground_truth):
    # The whole point of the file: it must be busted by exactly the checks students type.
    bounds = mystery.total_bounds
    real = ground_truth["boundary"]["total_bounds"]
    assert abs(bounds[0] - real[0]) > 1, "impostor longitude suspiciously close to the real Princeton"
    assert abs(bounds[1] - real[1]) > 1, "impostor latitude suspiciously close to the real Princeton"
    area = mystery.to_crs(METERS).geometry.area.iloc[0] / 1e6
    assert abs(area - ground_truth["boundary"]["area_km2"]) > 5


def test_mystery_carries_no_spoilers(mystery):
    leaky = [c for c in mystery.columns if c.lower() not in {"id", "geometry"}]
    assert not leaky, f"mystery file leaks identifying attributes: {leaky}"


def test_osm_files_carry_attribution():
    data = Path(__file__).resolve().parents[1] / "P01" / "data"
    for name in ("princeton_buildings.geojson", "princeton_food.geojson"):
        gj = json.loads((data / name).read_text())
        assert "OpenStreetMap" in gj.get("attribution", ""), f"{name} missing ODbL attribution"


def test_landmarks_inside_boundary(landmarks, boundary):
    assert landmarks.within(boundary.geometry.iloc[0]).all()
