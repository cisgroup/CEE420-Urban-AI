"""The three tools for flows into one destination do what their documentation says.

The gravity model fitted to one column of the matrix recovers the constant and the decay it was drawn with, matches
the observed total, gives an origin without residents no weight, and finds the residents' exponent near one when the
offset form is right. The radiation model for one column is the column of the toolbox's full radiation matrix,
counts the circle the way its definition says, and applies the finite-size correction. The rate by band sums what it
should. The tests build their own small data, so they run without the course's caches.
"""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import P07_KIT

NAMES = ("gravity_fit_destination", "radiation_to", "rate_by_band")


def _load(path, name):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def flows(mob):
    """The three tools of the toolbox's flows section."""
    assert all(hasattr(mob, n) for n in NAMES), "mob_tools lacks a flows tool"
    return mob


@pytest.fixture(scope="module")
def column():
    """A column drawn from the gravity model with a power decay: 400 origins, K 3, gamma 1.8, residents 500 to 5,000."""
    rng = np.random.default_rng(20261102)
    m = rng.integers(500, 5000, 400).astype(float)
    d = rng.uniform(1, 60, 400)
    T = rng.poisson(3.0 * m * d ** -1.8)
    return m, d, T


def test_the_tools_are_exported(flows):
    assert set(NAMES) <= set(getattr(flows, "__all__", NAMES))


def test_the_power_fit_recovers_its_decay_and_matches_the_total(flows, column):
    m, d, T = column
    fit, pred = flows.gravity_fit_destination(T, m, d, decay="power")
    assert fit["decay"] == "power" and fit["a"] == 1.0 and fit["n"] == len(m)
    assert abs(fit["gamma"] - 1.8) < 0.05 and abs(math.log(fit["K"]) - math.log(3.0)) < 0.15
    assert pred.sum() == pytest.approx(T.sum(), rel=1e-6), "a Poisson fit with a constant matches the observed total"
    assert fit["deviance"] > 0 and np.all(pred >= 0)


def test_the_exponential_fit_recovers_its_rate_and_halving_distance(flows):
    rng = np.random.default_rng(7)
    m = rng.integers(500, 5000, 500).astype(float)
    d = rng.uniform(0, 60, 500)
    T = rng.poisson(0.1 * m * np.exp(-0.08 * d))
    fit, pred = flows.gravity_fit_destination(T, m, d, decay="exp")
    assert abs(fit["beta"] - 0.08) < 0.005
    assert fit["half_km"] == pytest.approx(math.log(2) / fit["beta"])
    assert pred.sum() == pytest.approx(T.sum(), rel=1e-6)


def test_the_free_exponent_is_near_one_when_the_offset_is_right(flows, column):
    m, d, T = column
    free, _ = flows.gravity_fit_destination(T, m, d, decay="power", free_exponent=True)
    assert abs(free["a"] - 1.0) < 0.1 and abs(free["gamma"] - 1.8) < 0.08


def test_the_free_exponent_finds_a_wrong_offset(flows):
    rng = np.random.default_rng(11)
    m = rng.integers(200, 20000, 600).astype(float)
    d = rng.uniform(1, 40, 600)
    T = rng.poisson(5.0 * m ** 0.8 * d ** -1.5)
    free, _ = flows.gravity_fit_destination(T, m, d, decay="power", free_exponent=True)
    assert abs(free["a"] - 0.8) < 0.05


def test_an_origin_without_residents_gets_no_weight_and_no_flow(flows, column):
    m, d, T = column
    m0, T0 = np.r_[m, 0.0], np.r_[T, 0]
    fit0, pred0 = flows.gravity_fit_destination(T0, m0, np.r_[d, 2.0], decay="power")
    fit, _ = flows.gravity_fit_destination(T, m, d, decay="power")
    assert pred0[-1] == 0 and fit0["n"] == fit["n"]
    assert fit0["gamma"] == pytest.approx(fit["gamma"], rel=1e-6)


def test_an_unknown_decay_is_refused(flows, column):
    m, d, T = column
    with pytest.raises(ValueError):
        flows.gravity_fit_destination(T, m, d, decay="gaussian")


@pytest.mark.parametrize("finite_size", [True, False])
def test_radiation_to_is_a_column_of_the_full_radiation_matrix(flows, mob, finite_size):
    """The toy's three zones W, M and E with their residents, placed without ties, and twelve random zones."""
    cases = [(np.array([5.0, 1.0, 1.0]), np.array([[0.0, 0.0], [300.0, 40.0], [590.0, -30.0]]))]
    rng = np.random.default_rng(3)
    cases.append((rng.integers(100, 3000, 12).astype(float), rng.uniform(0, 10_000, (12, 2))))
    for pop, xy in cases:
        outflow = 0.6 * pop
        R = mob.radiation(pop, xy, outflow, finite_size=finite_size)
        for dest in range(len(pop)):
            keep = np.arange(len(pop)) != dest
            T, _ = flows.radiation_to(pop[keep], xy[keep], xy[dest], pop[dest], outflow[keep], finite_size=finite_size, total=pop.sum())
            assert np.allclose(T, R[keep, dest]), f"destination {dest} of {len(pop)} zones"


def test_radiation_to_counts_the_circle_by_its_definition(flows):
    """Four origins on a line east of the destination at 0: an origin's circle reaches back to the destination, so it
    holds the origins closer to it than the destination is, never the origin itself."""
    xy = np.array([[1000.0, 0.0], [2000.0, 0.0], [2500.0, 0.0], [6000.0, 0.0]])
    pop = np.array([10.0, 20.0, 30.0, 40.0])
    T, s = flows.radiation_to(pop, xy, (0.0, 0.0), 100.0, pop, finite_size=False)
    assert s.tolist() == [0.0, 40.0, 30.0, 60.0]
    assert T[0] == pytest.approx(10 * 10 * 100 / ((10 + 0) * (10 + 100 + 0)))
    assert T[3] == pytest.approx(40 * 40 * 100 / ((40 + 60) * (40 + 100 + 60)))


def test_the_finite_size_correction_divides_the_outflow(flows):
    xy = np.array([[1000.0, 0.0], [0.0, 3000.0]])
    pop, out = np.array([50.0, 150.0]), np.array([40.0, 100.0])
    plain, _ = flows.radiation_to(pop, xy, (0.0, 0.0), 200.0, out, finite_size=False)
    corrected, _ = flows.radiation_to(pop, xy, (0.0, 0.0), 200.0, out, finite_size=True)
    assert np.allclose(corrected, plain / (1 - pop / 400.0))


def test_the_rate_by_band_sums_what_it_should(flows):
    d = np.array([1.0, 4.0, 6.0, 12.0, 30.0])
    T = np.array([10.0, 30.0, 20.0, 5.0, 1.0])
    m = np.array([100.0, 300.0, 400.0, 500.0, 1000.0])
    b = flows.rate_by_band(d, T, m, [0, 5, 10, 20, math.inf])
    assert isinstance(b, pd.DataFrame) and list(b["origins"]) == [2, 1, 1, 1]
    assert list(b["flow"]) == [40.0, 20.0, 5.0, 1.0] and list(b["residents"]) == [400.0, 400.0, 500.0, 1000.0]
    assert b["rate"].tolist() == pytest.approx([100.0, 50.0, 10.0, 1.0])
    assert b.loc[0, "mean_km"] == pytest.approx((1 * 100 + 4 * 300) / 400)
    assert b["flow"].sum() == T.sum()
