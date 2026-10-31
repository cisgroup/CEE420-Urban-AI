"""The six privacy tools of Lecture 07's fourth chapter do what their docstrings say, on cases small enough to check by hand.

The tools live in the privacy section of P07/mob_tools.py. Every test builds its own tiny data, so nothing here reads a published file: the
toy's seven people of the lecture (4, 5 and 4 of 7 alone), a four-user table for the unicity test, a commuting table
with weights, a key space of 260 keys, a power law whose halving factor is known in closed form, and fixes on both
sides of a local midnight.
"""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd
import pytest

from conftest import P07_KIT

TOOLS = ("example_uniqueness", "anonymity_sets", "distinct_per_cell", "hash_keys", "halving_factors", "recording_days")


def _load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def priv():
    """The six tools of the toolbox's privacy section."""
    module = _load(P07_KIT / "mob_tools.py", "mob_tools_for_privacy_tests")
    assert all(hasattr(module, t) for t in TOOLS), "mob_tools lacks a privacy tool"
    return module


def test_every_tool_is_exported(priv):
    assert set(TOOLS) <= set(priv.__all__)


# --------------------------------------------------------------------------- #
# the toy's seven people, the lecture's hand count
# --------------------------------------------------------------------------- #

PEOPLE = [("A", "D"), ("A", "D"), ("B", "D"), ("C", "G"), ("A", "B"), ("D", "E"), ("F", "D")]
ZONE = {"A": "W", "B": "W", "C": "W", "D": "M", "E": "M", "F": "E", "G": "E"}


def test_the_toys_seven_people_are_four_five_and_four_alone(priv):
    homes = priv.anonymity_sets([h for h, _ in PEOPLE])
    pairs = priv.anonymity_sets(PEOPLE)
    zones = priv.anonymity_sets([(ZONE[h], ZONE[w]) for h, w in PEOPLE])
    alone = [round(a["alone_people"] * a["people"]) for a in (homes, pairs, zones)]
    assert alone == [4, 5, 4]
    assert list(homes["sizes"]) == [3, 3, 1, 1, 3, 1, 1]
    assert pairs["median"] == 1 and zones["median"] == 1


def test_anonymity_sets_count_people_through_weights(priv):
    # three block pairs with 1, 1 and 4 workers: two pairs of three are alone, two workers of six
    a = priv.anonymity_sets(["p", "q", "r"], weights=[1, 1, 4])
    assert a["keys"] == 3 and a["people"] == 6
    assert a["alone_keys"] == pytest.approx(2 / 3)
    assert a["alone_people"] == pytest.approx(2 / 6)
    assert a["median"] == 4                    # the middle worker of six sits in the pair of four
    # a key repeated across records adds its weights
    b = priv.anonymity_sets(["p", "p", "q"], weights=[1, 2, 1])
    assert list(b["sizes"]) == [3, 3, 1]


# --------------------------------------------------------------------------- #
# one person's unicity test
# --------------------------------------------------------------------------- #

def _four_users():
    # user 0 visits (1, 9), (2, 10), (3, 11), (4, 12); user 1 shares the first two, user 2 the first, user 3 the third
    rows = [(0, 1, 9), (0, 2, 10), (0, 3, 11), (0, 4, 12),
            (1, 1, 9), (1, 2, 10), (1, 7, 8), (1, 8, 8),
            (2, 1, 9), (2, 5, 13), (2, 6, 14), (2, 9, 15),
            (3, 3, 11), (3, 5, 9), (3, 6, 9), (3, 7, 9)]
    u, c, h = (np.array(col) for col in zip(*rows))
    return u, c, h


def test_example_uniqueness_writes_out_the_test(priv):
    u, c, h = _four_users()
    ex = priv.example_uniqueness(u, c, h, who=0, p=4, seed=1)
    assert ex["n_points"] == 4 and sorted(ex["points"]) == [(1, 9), (2, 10), (3, 11), (4, 12)]
    # the most shared point first: (1, 9) by users 1 and 2, then (2, 10) by 1 and (3, 11) by 3, then (4, 12) by nobody
    assert ex["sharers"] == [2, 1, 1, 0]
    assert ex["points"][0] == (1, 9)
    assert ex["together"][0] == 2 and ex["together"][-1] == 0
    assert ex["unique"] is True


def test_example_uniqueness_with_two_points_is_not_unique_for_a_shared_pair(priv):
    u, c, h = _four_users()
    # user 1 has four points; (1, 9) and (2, 10) are shared with user 0, so whenever the draw holds both they are not unique
    found = [priv.example_uniqueness(u, c, h, who=1, p=2, seed=s) for s in range(40)]
    both = [ex for ex in found if sorted(ex["points"]) == [(1, 9), (2, 10)]]
    assert both and all(ex["together"][-1] == 1 and not ex["unique"] for ex in both)


def test_example_uniqueness_picks_a_median_user_by_default(priv):
    u, c, h = _four_users()
    ex = priv.example_uniqueness(u, c, h, p=2, seed=7)
    assert ex["n_points"] == 4                 # every user has four points, so the median is four
    assert ex["users"] == 4


# --------------------------------------------------------------------------- #
# distinct individuals per cell
# --------------------------------------------------------------------------- #

def test_distinct_per_cell_counts_individuals_not_fixes(priv):
    ids = ["a", "a", "a", "b", "b", "c", "c"]
    cell = [0, 0, 1, 1, 1, -1, 3]
    counts = priv.distinct_per_cell(ids, cell, n_cells=5)
    assert counts.tolist() == [1, 2, 0, 1, 0]
    assert priv.distinct_per_cell(ids, cell).tolist() == [1, 2, 0, 1]


# --------------------------------------------------------------------------- #
# the key space, hashed and looked up
# --------------------------------------------------------------------------- #

def test_hash_keys_enumerates_and_recovers(priv):
    r = priv.hash_keys(["9A"], example="7K")
    assert r["patterns"] == {"9A": 260} and r["keys"] == 260
    digest = hashlib.md5(b"7K").hexdigest()
    assert r["example_digest"] == digest and r["found"] == {digest: "7K"}
    assert r["seconds"] >= 0 and r["per_second"] > 0


def test_hash_keys_finds_released_digests_and_other_algorithms(priv):
    released = [hashlib.sha1(k).hexdigest() for k in (b"12", b"99")]
    r = priv.hash_keys(["99"], algo="sha1", released=released)
    assert r["keys"] == 100
    assert sorted(r["found"].values()) == ["12", "99"]


# --------------------------------------------------------------------------- #
# halving factors of a uniqueness curve
# --------------------------------------------------------------------------- #

def test_halving_factors_of_a_power_law_are_constant(priv):
    beta = 0.129                               # de Montjoye's 0.157 - 0.007 p at four points
    sides = np.array([250.0, 500.0, 1000.0, 2000.0])
    shares = (sides ** 2) ** (-beta)           # a power of the cell's area
    rows = priv.halving_factors(sides, shares, dims=2)
    assert [r["halvings"] for r in rows] == pytest.approx([2.0, 2.0, 2.0])
    assert [r["factor"] for r in rows] == pytest.approx([2 ** -beta] * 3)
    hours = priv.halving_factors([1, 2, 4], [0.9, 0.9 * 2 ** -beta, 0.9 * 4 ** -beta], dims=1)
    assert [r["drop_pct"] for r in hours] == pytest.approx([100 * (1 - 2 ** -beta)] * 2)


# --------------------------------------------------------------------------- #
# recording days in local time
# --------------------------------------------------------------------------- #

def test_recording_days_follow_the_local_clock(priv):
    # in July Princeton keeps UTC minus four: 03:30 UTC on 2 July is 23:30 on 1 July, 04:10 UTC is ten past midnight
    times = ["2023-07-02T03:30:00Z", "2023-07-02T04:10:00Z", "2023-07-02T15:00:00Z"]
    who, hour, labels = priv.recording_days(times)
    assert labels == ["2023-07-01", "2023-07-02"]
    assert who.tolist() == [0, 1, 1] and hour.tolist() == [23, 0, 11]
    # in December it keeps UTC minus five
    who, hour, _ = priv.recording_days(["2026-12-02T04:30:00Z"])
    assert hour.tolist() == [23]


def test_recording_days_become_uploader_days_with_codes(priv):
    times = ["2023-05-01T14:00:00Z", "2023-05-01T15:00:00Z", "2023-05-01T16:00:00Z", "2023-05-02T14:00:00Z"]
    who, _, labels = priv.recording_days(times, ids=[7, 8, 7, -1])
    assert labels == ["-1:2023-05-02", "7:2023-05-01", "8:2023-05-01"]
    assert who.tolist() == [1, 2, 1, 0]
