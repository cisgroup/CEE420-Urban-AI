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


def test_published_ground_truth_carries_no_answers(ground_truth):
    # The Duel answer, the stretch answers and the impostor's numbers stay instructor-side.
    # They are asserted in data/scripts/verify_answers.py, which never ships to students.
    leaked = [k for k in ("mystery", "duel", "stretch") if k in ground_truth]
    assert not leaked, f"published ground truth leaks answers: {leaked}"


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


def test_notebook_reference_numbers_match_the_data(ground_truth):
    """Numbers quoted in the notebook prose must not drift from the shipped data.

    The pipeline's promise is that no published number is hand-typed. The mystery-file
    markdown and the break-glass comment quote the boundary area and bounds, so they
    are the one place that promise could silently break.
    """
    import json as _json

    nb = _json.loads((Path(__file__).resolve().parents[1] / "P01" / "01-first-map.ipynb").read_text())
    prose = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")

    area = str(ground_truth["boundary"]["area_km2"])
    assert area in prose, f"notebook prose no longer quotes the real area {area}"
    for value in ground_truth["boundary"]["total_bounds"][:2]:
        assert str(value) in prose, f"notebook prose no longer quotes the bound {value}"
