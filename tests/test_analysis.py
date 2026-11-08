"""Checks on the data and on the analysis maths."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as A  # noqa: E402

D = A.load()


def test_dataset_shape_and_failure_count():
    assert len(D) == 10_000
    assert D["failure"].sum() == 339


def test_engineered_features():
    r = D.iloc[0]
    assert r.temp_gap_K == pytest.approx(r.process_K - r.air_K)
    assert r.power_W == pytest.approx(r.torque_Nm * r.rpm * 2 * np.pi / 60)
    assert r.strain_minNm == pytest.approx(r.wear_min * r.torque_Nm)


def test_heat_failure_rule_holds_for_every_case():
    hdf = D[D.HDF == 1]
    assert (hdf.temp_gap_K < 8.6 + 1e-9).all() and (hdf.rpm < 1380).all()


def test_pareto_adds_up():
    p, _ = A.pareto(D)
    assert p["count"].sum() == D[list(A.MODES)].sum().sum()
    assert p["cum_share"].iloc[-1] == pytest.approx(1.0)


def test_control_limits_use_moving_range():
    s = A.spc(D)
    x = D["torque_Nm"].values
    sigma = np.abs(np.diff(x)).mean() / 1.128
    assert s["torque_ucl"] == pytest.approx(x.mean() + 3 * sigma)
    assert s["torque_lcl"] == pytest.approx(x.mean() - 3 * sigma)


def test_fmea_rpn_is_s_times_o_times_d():
    f = A.fmea(D)
    assert (f["RPN"] == f["S"] * f["O"] * f["D"]).all()


def test_engineered_features_improve_the_tree():
    r = A.predict(D).set_index("model")
    raw, eng = r.loc["Decision tree (depth 5), raw sensors"], r.loc["Decision tree (depth 5), + engineered"]
    assert eng["f1"] > raw["f1"]
    assert eng["false_alarms"] < raw["false_alarms"]


def test_tool_policy_monotonic():
    p, _ = A.tool_policy(D)
    assert p["wear_failures_at_or_past_limit"].is_monotonic_decreasing


def test_special_cause_is_the_hot_period():
    sc = A.special_cause(D)
    c = sc["conditions"]
    assert c.loc["air_K", "flagged period"] > c.loc["air_K", "rest"] + 2
    assert sc["hdf_share_in_period"] > 0.9
