"""The P07 toolbox pieces the kit added for the revision of 2026-10-31: the toy built from its edge list, the edge
betweenness on its own, the next train of a timetable, and the own-trace lane's readers, which have to take whatever
a tracker exports on the day of the hunt.

The trace tests write small synthetic files in the formats loggers use: GPX with UTC times, CSV with local times and
no zone, the date and the time in two columns, hemisphere letters in their own column or after the number, an
upper-case suffix, a first fix at (0, 0) or blank before the receiver has its position, and a diary written by hand.
"""

import numpy as np
import pandas as pd
import pytest

TOY_EDGES = [("A", "B", 2), ("A", "C", 3), ("B", "C", 2), ("B", "D", 3), ("C", "E", 2), ("D", "E", 2), ("D", "F", 3), ("E", "G", 4), ("F", "G", 2)]
TOY_LAYOUT = {"A": (0, 0), "B": (200, 0), "C": (225, 198), "D": (430, 192), "E": (333, 367), "F": (729, 170), "G": (733, 370)}
NASSAU = (40.34873, -74.65933)


@pytest.fixture(scope="module")
def toy(mob):
    return mob.graph_from_edges(TOY_EDGES, TOY_LAYOUT, unit=100.0)


def test_the_toy_from_its_edges_gives_the_lectures_table(mob, toy):
    r = mob.search(toy, "A")
    assert [r["dist"][n] for n in "ABCDEFG"] == [0, 200, 300, 500, 500, 800, 900]
    assert toy.nodes["C"]["x"] == 225.0 and toy["E"]["G"]["length"] == 400.0


def test_the_edge_betweenness_alone_equals_the_centralitys(mob, toy):
    _, edges = mob.centrality(toy)
    alone = mob.edge_betweenness(toy)
    pd.testing.assert_frame_equal(edges.reset_index(drop=True), alone.reset_index(drop=True))


def test_the_next_train_waits_and_rides(mob):
    r = mob.next_departure([10, 30, 50], [0, 14, 15, 40], access_min=16, ride_min=5)
    assert r["train"].tolist()[:3] == [30, 30, 50] and np.isnan(r["train"].iloc[3])
    assert r["wait"].tolist()[:3] == [14, 0, 19] and r["total"].tolist()[:3] == [35, 21, 40]


def _gpx(path, rows):
    pts = "".join(f'<trkpt lat="{la}" lon="{lo}"><ele>40</ele><time>{t}</time></trkpt>' for la, lo, t in rows)
    path.write_text(f'<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>{pts}</trkseg></trk></gpx>')
    return path


def _walk(n=600, step_deg=0.00002):
    """A straight walk north from Nassau Hall, one fix a second, about 2.2 m a step."""
    return [(NASSAU[0] + i * step_deg, NASSAU[1]) for i in range(n)]


def test_a_gpx_in_utc_lands_on_the_local_clock(mob, tmp_path):
    rows = [(la, lo, f"2026-11-04T19:{i // 60:02d}:{i % 60:02d}Z") for i, (la, lo) in enumerate(_walk(120))]
    fx = mob.fixes_from_trace(mob.read_trace(_gpx(tmp_path / "t.gpx", rows)))
    assert len(fx) == 120 and mob.clock(fx["t_s"].iloc[0]) == "14:00"          # 19:00 UTC is 14:00 in New York on 4 November


def test_a_csv_without_a_zone_is_read_as_the_local_clock(mob, tmp_path):
    w = _walk(60)
    pd.DataFrame({"time": [f"2026-11-04 14:05:{i:02d}" for i in range(60)], "lat": [a for a, _ in w], "lon": [b for _, b in w]}).to_csv(tmp_path / "t.csv", index=False)
    fx = mob.fixes_from_trace(mob.read_trace(tmp_path / "t.csv"))
    assert mob.clock(fx["t_s"].iloc[0]) == "14:05" and len(fx) == 60


def test_a_logger_csv_with_split_date_time_and_hemisphere_columns(mob, tmp_path):
    w = _walk(60)
    df = pd.DataFrame({"INDEX": range(60), "UTC DATE": ["2026/11/04"] * 60, "UTC TIME": [f"19:30:{i:02d}" for i in range(60)],
                       "LATITUDE": [a for a, _ in w], "N/S": ["N"] * 60, "LONGITUDE": [abs(b) for _, b in w], "E/W": ["W"] * 60, "HEIGHT": [40] * 60})
    df.to_csv(tmp_path / "LOG0001.CSV", index=False)
    path = mob.first_trace(tmp_path)
    assert path is not None and path.name == "LOG0001.CSV"
    fx = mob.fixes_from_trace(mob.read_trace(path))
    assert len(fx) == 60 and fx["lon"].lt(0).all() and mob.clock(fx["t_s"].iloc[0]) == "14:30"


def test_hemisphere_letters_after_the_number(mob, tmp_path):
    w = _walk(30)
    pd.DataFrame({"Date Time": [f"2026-11-04 15:00:{i:02d}" for i in range(30)], "Latitude N/S": [f"{a:.6f}N" for a, _ in w],
                  "Longitude E/W": [f"{abs(b):.6f} W" for _, b in w]}).to_csv(tmp_path / "t.csv", index=False)
    tr = mob.read_trace(tmp_path / "t.csv")
    assert np.allclose(tr["lon"], [b for _, b in w], atol=1e-6) and np.allclose(tr["lat"], [a for a, _ in w], atol=1e-6)


def test_a_first_fix_at_zero_or_blank_does_not_defeat_the_trim(mob, tmp_path):
    w = _walk(600)
    rows = [{"time": "2026-11-04 14:00:00", "lat": 0.0, "lon": 0.0}, {"time": "2026-11-04 14:00:01", "lat": np.nan, "lon": np.nan}]
    rows += [{"time": f"2026-11-04 14:{(i + 2) // 60:02d}:{(i + 2) % 60:02d}", "lat": a, "lon": b} for i, (a, b) in enumerate(w)]
    pd.DataFrame(rows).to_csv(tmp_path / "t.csv", index=False)
    tr = mob.read_trace(tmp_path / "t.csv")
    kept, dropped = mob.clean_fixes(tr)
    assert dropped == 2 and len(kept) == 600
    fx = mob.fixes_from_trace(tr)
    trimmed, head, tail = mob.trim_ends(fx, 200.0)
    assert 80 <= head <= 100 and 80 <= tail <= 100 and len(trimmed) == len(fx) - head - tail


def test_a_handwritten_diary(mob, tmp_path):
    pd.DataFrame({"Stop": ["Nassau Hall", "Firestone", "", "Dinky"], "Start": ["2:05 pm", "14.30", "", "15:10"],
                  "End": ["14:20", "2:45pm", "", "3:20 PM"]}).to_csv(tmp_path / "diary.csv", index=False)
    d = mob.read_diary(tmp_path / "diary.csv")
    assert d["poi_name"].tolist() == ["Nassau Hall", "Firestone", "Dinky"]
    assert d["start"].tolist() == [50700.0, 52200.0, 54600.0] and d["duration_min"].tolist() == [15.0, 15.0, 10.0]
    assert mob.first_trace(tmp_path) is None                     # the diary is never taken for a trace
