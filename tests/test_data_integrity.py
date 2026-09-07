"""Every published reference number is recomputed from the shipped files."""

import json
import json as _json
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


def test_notebook_area_claims_match_the_shipped_data(ground_truth):
    """An area quoted in prose must be the shipped one, and the reference must be read.

    The pipeline's promise is that no published number is hand-typed. The first version
    of this test enforced that promise backwards, by requiring the prose to quote 47.66,
    which forbade the better fix of printing it from ground_truth.json.

    So it is enforced two ways here. Any prose that claims an area in square kilometres
    must claim the shipped one, which is the number that would actually go stale after a
    data rebuild. And at least one notebook must read the reference file, so a student
    has something to check their own answer against.
    """
    import re as _re

    claim = _re.compile(r"(\d+\.\d+)\s*(?:square kilometres|square kilometers|km2)")
    area = ground_truth["boundary"]["area_km2"]
    acceptable = {f"{area:.{places}f}" for places in (1, 2, 3)} | {str(area)}

    root = Path(__file__).resolve().parents[1]
    reads_the_file = False
    for notebook in sorted(root.glob("*/*.ipynb")):
        cells = _json.loads(notebook.read_text())["cells"]
        prose = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        reads_the_file |= "ground_truth.json" in code

        # A notebook that loads the impostor is comparing two towns on purpose, so a
        # second, different area is the whole point of it.
        teaches_the_impostor = "mystery_boundary.geojson" in code
        for match in claim.finditer(prose):
            quoted = match.group(1)
            if quoted in acceptable:
                continue
            assert teaches_the_impostor, (
                f"{notebook.name} claims an area of {quoted} square kilometres, but the "
                f"shipped boundary is {area}. Print it from data/ground_truth.json."
            )

    assert reads_the_file, (
        "no notebook reads data/ground_truth.json, so a student has nothing to check "
        "their own numbers against"
    )


def test_f00_tables_match_the_shipped_geojson(food, buildings):
    """The foundations series teaches on the same data, with the geometry taken off.

    Both are written by data/scripts/05_publish.py, so a rebuild that changed one and
    not the other would silently teach different numbers in F00 than in P01.
    """
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    food_table = pd.read_csv(root / "F00" / "data" / "food_table.csv")
    buildings_table = pd.read_csv(root / "F00" / "data" / "buildings_table.csv")

    assert len(food_table) == len(food)
    assert len(buildings_table) == len(buildings)
    assert food_table["amenity"].value_counts().to_dict() == food["amenity"].value_counts().to_dict()
    assert list(food_table.columns) == ["name", "amenity", "lat", "lon"]
