"""Developer validation mode: data integrity, units, error arithmetic, repeatability, the
separation of real-participant records from software tests, and operator-only access.

Every scan and record in this module is SYNTHETIC test data, created in memory or in a pytest
temp directory. Nothing here is, or is written as, a real-world measurement.
"""

import json
import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from glovesize import validation_records as vr
from glovesize.api import app
from glovesize.config import (
    CalibrationValidation,
    MeasurementCalibration,
    clear_caches,
    config_dir,
    load_measurement_calibration,
)
from glovesize.operator_access import access_mode
from glovesize.pipeline import process_scan

from .conftest import make_frames


@pytest.fixture
def enabled(monkeypatch, tmp_path, no_depth_config):
    monkeypatch.setenv("GLOVESIZE_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("GLOVESIZE_VALIDATION_PATH", str(tmp_path / "records.jsonl"))
    monkeypatch.delenv("GLOVESIZE_OPERATOR_TOKEN", raising=False)
    clear_caches()
    yield tmp_path / "records.jsonl"
    clear_caches()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def body(readings=((187.0, "mm"), (187.5, "mm")), frames=None, origin="software_test", consent=False, **over):
    b = {
        "data_origin": origin,
        "consent_confirmed": consent,
        "participant_code": "VAL-0001",
        "reference": {
            "protocol": "HL-1",
            "hand_length": [{"value": v, "unit": u} for v, u in readings],
            "tool": "ruler",
            "measured_by": "operator",
        },
        "trial": {"hand": "right", "condition": "standard"},
        "conditions": {"device": "pytest (synthetic frames)", "lighting": "indoor_bright", "distance_cm": 35},
        "scan": {"frames": frames or make_frames(), "reference": {"object_id": "us_quarter", "confirmed": True}},
    }
    b.update(over)
    return b


# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #


def test_disabled_by_default(operator_client):
    assert operator_client.post("/api/validation/scans", json=body()).status_code == 403
    assert operator_client.get("/api/validation/records").status_code == 403


def test_enabled_mode_is_localhost_only(client, operator_client, enabled):
    assert operator_client.get("/api/validation/records").status_code == 200
    # Remote client address.
    assert client.get("/api/validation/records").status_code == 403
    remote = TestClient(app, base_url="http://127.0.0.1:8000", client=("203.0.113.9", 4000))
    assert remote.get("/api/validation/records").status_code == 403
    # DNS rebinding: local socket but a foreign Host name.
    assert operator_client.get("/api/validation/records", headers={"host": "evil.example"}).status_code == 403
    # Behind a reverse proxy every client looks local, so forwarded requests are refused.
    assert (
        operator_client.get("/api/validation/records", headers={"x-forwarded-for": "198.51.100.4"}).status_code == 403
    )
    # [::1] with a port is local.
    assert operator_client.get("/api/validation/records", headers={"host": "[::1]:8080"}).status_code == 200


def test_state_changes_need_the_operator_header(operator_client, enabled):
    r = operator_client.post("/api/validation/scans", json=body(), headers={"x-glovesize-operator": "0"})
    assert r.status_code == 403
    # Refused before the body is parsed: an invalid body still gets 403, not a schema error.
    r = operator_client.post("/api/validation/scans", json={}, headers={"x-glovesize-operator": ""})
    assert r.status_code == 403


def test_token_mode(monkeypatch, client, operator_client, enabled):
    token = "t" * 40
    monkeypatch.setenv("GLOVESIZE_OPERATOR_TOKEN", token)
    remote = TestClient(app, base_url="https://lab.example", client=("203.0.113.9", 4000))
    assert remote.get("/api/validation/records").status_code == 401
    assert operator_client.get("/api/validation/records").status_code == 401  # local is not enough
    assert remote.get("/api/validation/records", headers={"authorization": "Bearer wrong"}).status_code == 401
    ok = remote.get("/api/validation/records", headers={"authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    assert client.get("/api/config").json()["operator_access"] == "token"
    monkeypatch.setenv("GLOVESIZE_OPERATOR_TOKEN", "short")
    with pytest.raises(ValueError, match="at least 32"):
        access_mode()


def test_research_endpoints_are_operator_only(monkeypatch, tmp_path, client, operator_client):
    monkeypatch.setenv("GLOVESIZE_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("GLOVESIZE_RESEARCH_PATH", str(tmp_path / "research.jsonl"))
    clear_caches()
    assert client.post("/api/research/records", json={}).status_code == 403
    assert client.delete("/api/research/records/P-7f3a9c").status_code == 403
    assert operator_client.delete("/api/research/records/P-7f3a9c").json() == {"deleted": 0}
    clear_caches()


# --------------------------------------------------------------------------- #
# Records: references, units, server-computed errors, origin
# --------------------------------------------------------------------------- #


def test_record_errors_computed_by_server(operator_client, enabled):
    r = operator_client.post("/api/validation/scans", json=body(readings=((187.0, "mm"), (187.5, "mm"))))
    assert r.status_code == 201
    rec = r.json()["record"]
    assert rec["data_origin"] == "software_test" and rec["quality_passed"] is True
    rs = rec["reference_summary"]
    assert (rs["hand_length_mm"], rs["reliability"], rs["repeated"]) == (187.25, "agreed", True)
    hl = next(c for c in rec["comparisons"] if c["key"] == "hand_length_mm")
    # The synthetic hand is 185 mm: the reference is the mean of the two agreed readings.
    assert hl["reference_mm"] == 187.25
    assert hl["estimate_mm"] == pytest.approx(185, abs=1.0)
    assert hl["signed_error_mm"] == pytest.approx(hl["estimate_mm"] - 187.25, abs=0.01)
    assert hl["abs_error_mm"] == pytest.approx(abs(hl["signed_error_mm"]), abs=0.01)
    assert hl["pct_error"] == pytest.approx(100 * hl["signed_error_mm"] / 187.25, abs=0.01)
    assert rec["calibration_status"] == "uncalibrated" and rec["calibration_validated"] is False
    cap = rec["capture"]
    assert cap["coin_accepted_frames"] == 24 and cap["px_per_mm"] > 0 and cap["landmark_motion_px"] > 0.5


@pytest.mark.parametrize(
    ("readings", "expected_mm"),
    [(((7.25, "in"), (7.25, "in")), 184.15), (((18.4, "cm"), (18.45, "cm")), 184.25), (((184, "mm"),), 184.0)],
)
def test_units_are_explicit_and_converted(readings, expected_mm):
    ref = vr.ReferenceInput.model_validate(body(readings=readings)["reference"])
    assert vr.summarize_reference(ref, 2.0).hand_length_mm == pytest.approx(expected_mm, abs=0.01)


def test_wrong_unit_is_rejected_not_converted(operator_client, enabled):
    r = operator_client.post("/api/validation/scans", json=body(readings=((7.25, "mm"),)))
    assert r.status_code == 422 and "check the value and the unit" in r.text
    r = operator_client.post("/api/validation/scans", json=body(readings=((184, "cm"),)))
    assert r.status_code == 422
    b = body()
    b["reference"]["hand_length"][0]["unit"] = "inch"
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422


@pytest.mark.parametrize(
    ("readings", "value", "reliability"),
    [
        ((184.0,), 184.0, "single_reading"),
        ((184.0, 186.0), 185.0, "agreed"),
        ((184.0, 187.0), 185.5, "discrepant"),
        ((184.0, 189.0, 188.0), 188.0, "agreed"),  # third reading resolves it: median
        ((180.0, 184.0, 188.0), 184.0, "discrepant"),
    ],
)
def test_reference_repeat_rule(readings, value, reliability):
    ref = vr.ReferenceInput.model_validate(body(readings=[(v, "mm") for v in readings])["reference"])
    s = vr.summarize_reference(ref, 2.0)
    assert (s.hand_length_mm, s.reliability) == (value, reliability)


def test_estimates_cannot_be_submitted_directly(operator_client, enabled):
    b = body()
    b["estimate_mm"] = 185.0
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422
    b = body()
    b["reference"]["hand_length_mm"] = 185.0  # old single-number format
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422


def test_real_participant_needs_consent_and_no_identifiers(operator_client, enabled):
    assert operator_client.post("/api/validation/scans", json=body(origin="real_participant")).status_code == 422
    b = body(origin="real_participant", consent=True)
    b["participant_code"] = "jane@example.com"
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422
    for field in ("name", "email", "phone"):
        assert operator_client.post("/api/validation/scans", json=body(**{field: "x"})).status_code == 422


def test_still_image_input_is_flagged_and_excluded(operator_client, enabled):
    rec = operator_client.post(
        "/api/validation/scans", json=body(origin="real_participant", consent=True, frames=make_frames(noise_px=0.0))
    ).json()["record"]
    assert rec["integrity_flags"] == ["static_frames"]
    s = operator_client.get("/api/validation/records").json()["summary"]
    assert s["real_world"]["scans"]["total"] == 0 and s["excluded"]["integrity_flag"] == 1


def test_software_tests_never_enter_statistics(operator_client, enabled):
    for _ in range(3):
        assert operator_client.post("/api/validation/scans", json=body()).status_code == 201
    data = operator_client.get("/api/validation/records").json()
    s = data["summary"]
    assert len(data["records"]) == 3
    assert s["excluded"]["software_test"] == 3
    assert s["real_world"]["scans"]["total"] == 0
    assert s["real_world"]["accuracy"]["by_scan"] == {"n": 0}
    assert "no real-world accuracy" in s["real_world"]["accuracy"]["sample_note"]


def test_rejected_scans_are_counted_not_scored(operator_client, enabled):
    real = {"origin": "real_participant", "consent": True}
    operator_client.post("/api/validation/scans", json=body(**real))
    rej = operator_client.post("/api/validation/scans", json=body(frames=make_frames(n=4), **real)).json()["record"]
    assert rej["quality_passed"] is False and rej["scan_rejection_reasons"] == ["insufficient_frames"]
    assert rej["comparisons"] == []
    sc = operator_client.get("/api/validation/records").json()["summary"]["real_world"]["scans"]
    assert (sc["total"], sc["accepted"], sc["rejected"], sc["rejected_pct"]) == (2, 1, 1, 50.0)
    assert sc["rejection_reasons"] == {"insufficient_frames": 1}


def test_calibration_file_is_not_changed_by_validation(operator_client, enabled):
    before = (config_dir() / "measurement_calibration.json").read_bytes()
    for _ in range(5):
        operator_client.post("/api/validation/scans", json=body(origin="real_participant", consent=True))
    assert (config_dir() / "measurement_calibration.json").read_bytes() == before
    s = operator_client.get("/api/validation/records").json()["summary"]
    assert s["calibration"] == {"status": "uncalibrated", "validated": False}


def test_unreadable_lines_are_reported_and_kept(operator_client, enabled):
    rec = operator_client.post("/api/validation/scans", json=body()).json()["record"]
    with enabled.open("a", encoding="utf-8") as f:
        f.write('{"broken": true}\n')
    s = operator_client.get("/api/validation/records").json()["summary"]
    assert s["excluded"]["unreadable_lines"] == 1
    assert operator_client.delete(f"/api/validation/records/{rec['id']}").json() == {"deleted": rec["id"]}
    assert enabled.read_text(encoding="utf-8") == '{"broken": true}\n'
    assert operator_client.delete(f"/api/validation/records/{rec['id']}").status_code == 404


def test_csv_export(operator_client, enabled):
    b = body(readings=((7.25, "in"), (7.3, "in")))
    b["conditions"]["notes"] = "=HYPERLINK(1)"
    rec = operator_client.post("/api/validation/scans", json=b).json()["record"]
    r = operator_client.get("/api/validation/records.csv")
    assert r.headers["content-type"].startswith("text/csv")
    rows = r.text.splitlines()
    header = rows[0].split(",")
    for col in ("readings_as_entered", "readings_mm", "reference_mm", "estimate_mm", "uncertainty_1sd_mm",
                "signed_error_mm", "abs_error_mm", "pct_error", "data_origin", "hand", "condition"):  # fmt: skip
        assert col in header
    assert rec["id"] in r.text and "7.25 in; 7.3 in" in r.text and "184.15; 185.42" in r.text
    assert "'=HYPERLINK(1)" in r.text


# --------------------------------------------------------------------------- #
# Statistics on constructed records (exact arithmetic)
# --------------------------------------------------------------------------- #


@pytest.fixture
def template(no_depth_config):
    req = vr.ValidationScanRequest.model_validate(body(origin="real_participant", consent=True))
    return vr.build_record(req, process_scan(req.scan))


def make_record(template, code="P-000001", hand="right", condition="standard", estimate=185.0, readings=(185.0, 185.0),
                accepted=True, origin="real_participant", flags=(), fingerprint=None, distance_cm=35.0,
                coin_distance_mm=None):  # fmt: skip
    d = template.model_dump(mode="json")
    d["conditions"]["distance_cm"] = distance_cm
    if coin_distance_mm is not None:
        d["capture"]["estimated_camera_distance_mm"] = coin_distance_mm
    d.update(id=str(uuid.uuid4()), participant_code=code, data_origin=origin, trial={"hand": hand, "condition": condition},
             quality_passed=accepted, scan_status="accepted" if accepted else "rejected", integrity_flags=list(flags))  # fmt: skip
    d["reference"]["hand_length"] = [{"value": v, "unit": "mm"} for v in readings]
    rs = vr.summarize_reference(vr.ReferenceInput.model_validate(d["reference"]), 2.0)
    d["reference_summary"] = rs.model_dump()
    if accepted:
        err = estimate - rs.hand_length_mm
        d["comparisons"] = [{"key": "hand_length_mm", "reference_mm": rs.hand_length_mm, "estimate_mm": estimate,
                             "uncertainty_mm": 2.0, "status": "valid", "abs_error_mm": round(abs(err), 2),
                             "pct_error": round(100 * err / rs.hand_length_mm, 2), "signed_error_mm": round(err, 2),
                             "within_95": abs(err) <= vr.Z95 * 2.0}]  # fmt: skip
        d["scan_rejection_reasons"] = []
    else:
        d["comparisons"], d["scan_rejection_reasons"] = [], ["coin_not_found"]
    if fingerprint:
        d["config_fingerprint"] = fingerprint
    return vr.ValidationRecord.model_validate(d)


def test_describe_errors_exact():
    d = vr.describe_errors([2.0, -1.0, 3.0, -4.0])
    assert d["n"] == 4
    assert d["mae_mm"] == pytest.approx(2.5)
    assert d["rmse_mm"] == pytest.approx(7.5**0.5)
    assert d["mean_signed_error_mm"] == pytest.approx(0.0)
    assert d["median_abs_error_mm"] == pytest.approx(2.5)
    assert d["max_abs_error_mm"] == 4.0
    assert d["errors_mm"] == [-4.0, -1.0, 2.0, 3.0]
    counts = dict(zip(d["histogram"]["bin_edges_mm"], d["histogram"]["counts"], strict=False))
    assert (counts[-4.0], counts[-2.0], counts[2.0], sum(d["histogram"]["counts"])) == (1, 1, 2, 4)
    h = vr.describe_errors([25.0, -30.0])["histogram"]
    assert (h["below"], h["above"], sum(h["counts"])) == (1, 1, 0)
    assert vr.describe_errors([])["n"] == 0


def test_summary_accuracy_by_scan_and_by_hand(template, enabled):
    rs = [
        make_record(template, code="P-AAAAAA", estimate=187.0),  # +2
        make_record(template, code="P-AAAAAA", estimate=189.0),  # +4 (same hand)
        make_record(template, code="P-BBBBBB", estimate=184.0),  # -1
        make_record(template, code="P-CCCCCC", estimate=188.0),  # +3
        make_record(template, code="P-DDDDDD", estimate=181.0),  # -4
        # Not scored: single reading, discrepant readings, non-standard condition, rejected.
        make_record(template, code="P-EEEEEE", estimate=200.0, readings=(185.0,)),
        make_record(template, code="P-FFFFFF", estimate=200.0, readings=(180.0, 190.0)),
        make_record(template, code="P-AAAAAA", estimate=150.0, condition="distance_far"),
        make_record(template, code="P-GGGGGG", accepted=False),
        # Excluded entirely: software test, other configuration, integrity flag.
        make_record(template, code="P-HHHHHH", estimate=100.0, origin="software_test"),
        make_record(template, code="P-IIIIII", estimate=100.0, fingerprint="000000000000"),
        make_record(template, code="P-JJJJJJ", estimate=100.0, flags=["static_frames"]),
    ]
    s = vr.summarize(rs)
    rw = s["real_world"]
    assert s["excluded"] == {"software_test": 1, "other_configuration": 1, "integrity_flag": 1, "unreadable_lines": 0}
    assert rw["scans"]["total"] == 9 and rw["scans"]["rejected"] == 1
    assert rw["references"] == {"agreed": 7, "single_reading": 1, "discrepant": 1}
    by_scan = rw["accuracy"]["by_scan"]
    assert by_scan["errors_mm"] == [-4.0, -1.0, 2.0, 3.0, 4.0]
    assert by_scan["mae_mm"] == pytest.approx(14 / 5)
    assert by_scan["mean_signed_error_mm"] == pytest.approx(4 / 5)
    assert by_scan["rmse_mm"] == pytest.approx((46 / 5) ** 0.5)
    assert by_scan["median_abs_error_mm"] == pytest.approx(3.0)
    # One value per hand: P-AAAAAA's two standard scans average to +3.
    assert rw["accuracy"]["n_hands"] == 4
    assert rw["accuracy"]["by_hand"]["errors_mm"] == [-4.0, -1.0, 3.0, 3.0]
    assert rw["accuracy"]["within_reported_95_interval"] == {"count": 3, "of": 5}
    assert "Only 4 hand(s)" in rw["accuracy"]["sample_note"]
    assert rw["conditions"]["distance_far"]["mean_signed_error_mm"] == pytest.approx(-35.0)


def test_reference_mismatch_between_records_is_flagged(template, enabled):
    a = make_record(template, code="P-AAAAAA", readings=(185.0, 185.0))
    b = make_record(template, code="P-AAAAAA", readings=(190.0, 190.0))
    other_hand = make_record(template, code="P-AAAAAA", hand="left", readings=(180.0, 180.0))
    s = vr.summarize([a, b, other_hand])
    assert s["record_flags"] == {a.id: ["reference_mismatch"], b.id: ["reference_mismatch"]}
    assert s["excluded"]["integrity_flag"] == 2 and s["real_world"]["scans"]["total"] == 1


def test_repeatability_and_condition_shifts(template, enabled):
    rs = [
        make_record(template, code="P-AAAAAA", estimate=185.0),
        make_record(template, code="P-AAAAAA", estimate=187.0),
        make_record(template, code="P-AAAAAA", estimate=190.0, condition="distance_far"),
        make_record(template, code="P-BBBBBB", estimate=180.0),
        make_record(template, code="P-BBBBBB", estimate=181.0),
        make_record(template, code="P-BBBBBB", estimate=182.0),
        make_record(template, code="P-BBBBBB", estimate=181.5, condition="lighting_changed"),
    ]
    rep = vr.summarize(rs)["real_world"]["repeatability"]
    # Pooled within-hand SD: sum of squares 2 + 2 over 1 + 2 degrees of freedom.
    assert rep["hands_with_repeats"] == 2
    assert rep["within_hand_sd_mm"] == pytest.approx((4 / 3) ** 0.5)
    assert rep["repeatability_coefficient_mm"] == pytest.approx(2.77 * (4 / 3) ** 0.5)
    far = rep["condition_shifts"]["distance_far"]
    assert (far["n_hands"], far["mean_shift_mm"], far["exceeds_repeatability"]) == (1, pytest.approx(4.0), True)
    light = rep["condition_shifts"]["lighting_changed"]
    assert (light["mean_shift_mm"], light["exceeds_repeatability"]) == (pytest.approx(0.5), False)
    assert "not accuracy" in rep["note"]


# --------------------------------------------------------------------------- #
# Calibration can only become "validated" through explicit human sign-off
# --------------------------------------------------------------------------- #


def test_shipped_calibration_is_uncalibrated_and_unvalidated():
    clear_caches()
    cal = load_measurement_calibration()
    assert (cal.status, cal.validation, cal.validated) == ("uncalibrated", None, False)


def test_validation_evidence_rules():
    base = load_measurement_calibration().model_dump(mode="json")
    evidence = {"real_world": True, "held_out_participants": 60, "acceptance_criteria_met": True,
                "report": "held-out-001", "approved_by": "QA lead", "approved_at": "2026-01-01"}  # fmt: skip
    with pytest.raises(ValidationError):  # uncalibrated cannot carry evidence
        MeasurementCalibration.model_validate({**base, "validation": evidence})
    for bad in (
        {"held_out_participants": 12},
        {"real_world": False},
        {"acceptance_criteria_met": False},
        {"approved_by": ""},
    ):
        with pytest.raises(ValidationError):
            CalibrationValidation.model_validate({**evidence, **bad})
    ok = MeasurementCalibration.model_validate({**base, "status": "calibrated", "validation": evidence})
    assert ok.validated and ok.validation.approved_at == date(2026, 1, 1)
    assert not MeasurementCalibration.model_validate({**base, "status": "calibrated"}).validated


def test_fit_refuses_to_write_the_live_config(monkeypatch, capsys):
    from validation import fit_calibration

    out = config_dir() / "measurement_calibration.json"
    before = out.read_bytes()
    monkeypatch.setattr("sys.argv", ["fit_calibration", "p.csv", "s.csv", "--out", str(out)])
    with pytest.raises(SystemExit):
        fit_calibration.main()
    assert "refusing to write into the live config" in capsys.readouterr().err
    assert out.read_bytes() == before
    assert json.loads(before)["status"] == "uncalibrated"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_keeps_real_and_synthetic_apart(template, enabled):
    from scripts.validation_report import real_section

    rs = [
        make_record(template, code="P-AAAAAA", estimate=187.0),
        make_record(template, code="P-AAAAAA", estimate=186.0),
        make_record(template, code="P-SOFTWR", estimate=100.0, origin="software_test"),
    ]
    text = "\n".join(real_section(rs, 0))
    assert "P-AAAAAA" in text and "P-SOFTWR" not in text
    assert "| Scan | 2 | +1.50 | 1.50 |" in text
    assert "Excluded from all statistics: software_test 1" in text
    assert "Only 1 hand(s)" in text and "not accuracy" in text
    # Per-participant row and distance bands (both scans 35 cm; one hand, so not interpretable).
    assert "| P-AAAAAA | right | 185.0 (agreed) | 2 | 0 | 2 | +1.50 | 1.50 | +1.0 to +2.0 | 0, n/a |" in text
    assert "| 30-40 cm | 2 | 0 | 2 | 1 | +1.50 | 1.50 | no (too few hands) |" in text
    assert "### By camera distance" in text and "Coin-implied distance" in text


def test_report_with_no_real_records_says_so(monkeypatch, tmp_path, enabled):
    from scripts import validation_report

    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["validation_report", "--out", str(out), "--no-pytest", "--no-simulation"])
    validation_report.main()
    text = out.read_text(encoding="utf-8")
    assert "no real-world result can be reported" in text
    assert text.index("## 1. Real physical") < text.index("## 2. Synthetic") < text.index("## 3. Simulated")
    assert "unvalidated prototype" in text


# --------------------------------------------------------------------------- #
# Per-frame numeric data and configuration versions
# --------------------------------------------------------------------------- #


def test_frames_and_config_snapshot_are_stored(operator_client, enabled):
    b = body()
    rec = operator_client.post("/api/validation/scans", json=b).json()["record"]
    assert rec["frames_stored"] is True
    data = vr.load_frames(rec["id"])
    assert data.record_id == rec["id"] and data.config_fingerprint == rec["config_fingerprint"]
    submitted = vr.ValidationScanRequest.model_validate(b).scan
    assert data.scan == submitted
    stored = (enabled.parent / "frames" / f"{rec['id']}.json").read_text(encoding="utf-8")
    assert "image" not in stored.replace('"image_width"', "").replace('"image_height"', "")
    # The fingerprint resolves to the exact configuration that measured the scan.
    snap = vr.load_config_snapshot(rec["config_fingerprint"])
    assert snap.measurement["depth_model"]["enabled"] is False  # this test runs the no-depth config
    assert snap.calibration["status"] == "uncalibrated"
    # The record list stays light: frames are not sent to the browser.
    assert "frames" not in operator_client.get("/api/validation/records").json()["records"][0]


def test_stored_frames_reproduce_the_estimate(operator_client, enabled):
    rec = operator_client.post("/api/validation/scans", json=body()).json()["record"]
    again = vr.remeasure(rec["id"])
    hl = next(m for m in again.measurements if m.key == "hand_length_mm")
    assert hl.value_mm == next(c for c in rec["comparisons"] if c["key"] == "hand_length_mm")["estimate_mm"]


def test_config_versions_are_kept_apart(monkeypatch, operator_client, enabled, cfg):
    import glovesize.pipeline as pipeline

    first = operator_client.post("/api/validation/scans", json=body()).json()["record"]["config_fingerprint"]
    monkeypatch.setattr(pipeline, "load_measurement_config", lambda: cfg)  # depth model on
    second = operator_client.post("/api/validation/scans", json=body()).json()["record"]["config_fingerprint"]
    assert first != second
    assert {p.stem for p in (enabled.parent / "configs").iterdir()} == {first, second}
    assert vr.load_config_snapshot(second).measurement["depth_model"]["enabled"] is True


def test_altered_config_snapshot_is_detected(operator_client, enabled):
    fp = operator_client.post("/api/validation/scans", json=body()).json()["record"]["config_fingerprint"]
    path = enabled.parent / "configs" / f"{fp}.json"
    text = path.read_text(encoding="utf-8")
    assert '"max_tilt_deg": 14.0' in text
    path.write_text(text.replace('"max_tilt_deg": 14.0', '"max_tilt_deg": 20.0'), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its fingerprint"):
        vr.load_config_snapshot(fp)


def test_frame_storage_can_be_switched_off(monkeypatch, operator_client, enabled):
    from glovesize.config import AppConfig, load_app_config

    off = AppConfig.model_validate({**load_app_config().model_dump(), "validation_store_frame_data": False})
    monkeypatch.setattr(vr, "load_app_config", lambda: off)
    rec = operator_client.post("/api/validation/scans", json=body()).json()["record"]
    assert rec["frames_stored"] is False and vr.load_frames(rec["id"]) is None
    with pytest.raises(FileNotFoundError):
        vr.remeasure(rec["id"])


def test_deleting_a_record_deletes_its_frames(operator_client, enabled):
    rec = operator_client.post("/api/validation/scans", json=body()).json()["record"]
    path = enabled.parent / "frames" / f"{rec['id']}.json"
    assert path.exists()
    operator_client.delete(f"/api/validation/records/{rec['id']}")
    assert not path.exists()


@pytest.mark.parametrize("field", ["image", "image_data", "jpeg", "pixels"])
def test_images_cannot_be_submitted(operator_client, enabled, field):
    b = body()
    b["scan"]["frames"][0][field] = "data:image/jpeg;base64,AAAA"
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422
    b = body()
    b["scan"][field] = "AAAA"
    assert operator_client.post("/api/validation/scans", json=b).status_code == 422
    assert not enabled.exists() and not (enabled.parent / "frames").exists()


# --------------------------------------------------------------------------- #
# Per-participant and distance breakdowns (constructed records)
# --------------------------------------------------------------------------- #


def test_per_participant_rows(template, enabled):
    rs = [
        make_record(template, code="P-AAAAAA", estimate=187.0),
        make_record(template, code="P-AAAAAA", estimate=189.0),
        make_record(template, code="P-AAAAAA", estimate=190.0, condition="distance_far"),
        make_record(template, code="P-AAAAAA", accepted=False),
        make_record(template, code="P-AAAAAA", hand="left", estimate=180.0, readings=(182.0, 182.0)),
        make_record(template, code="P-BBBBBB", estimate=200.0, readings=(184.0, 190.0)),  # discrepant: not scored
    ]
    rows = vr.summarize(rs)["real_world"]["by_participant"]
    keys = [(r["participant_code"], r["hand"]) for r in rows]
    assert keys == [("P-AAAAAA", "left"), ("P-AAAAAA", "right"), ("P-BBBBBB", "right")]
    right = rows[1]
    assert (right["scans"], right["rejected"], right["standard_scored"]) == (4, 1, 2)
    assert right["standard_mean_signed_error_mm"] == pytest.approx(3.0)
    assert right["standard_mae_mm"] == pytest.approx(3.0)
    assert right["standard_error_range_mm"] == [2.0, 4.0]
    assert right["other_conditions_scored"] == 1
    assert right["other_conditions_mean_signed_error_mm"] == pytest.approx(5.0)
    assert rows[0]["standard_mean_signed_error_mm"] == pytest.approx(-2.0)
    bbb = rows[2]
    assert (bbb["reference_reliability"], bbb["standard_scored"], bbb["standard_mae_mm"]) == ("discrepant", 0, None)


def test_distance_breakdown_counts_each_hand_once(template, enabled):
    rs = [
        make_record(template, code="P-AAAAAA", estimate=187.0, distance_cm=35),  # +2
        make_record(template, code="P-AAAAAA", estimate=189.0, distance_cm=38),  # +4, same hand and band
        make_record(template, code="P-BBBBBB", estimate=183.0, distance_cm=36),  # -2
        make_record(template, code="P-CCCCCC", estimate=186.0, distance_cm=39.9),  # +1
        make_record(template, code="P-AAAAAA", estimate=191.0, distance_cm=55, condition="distance_far"),  # +6
        make_record(template, code="P-BBBBBB", accepted=False, distance_cm=27, condition="distance_near"),
        make_record(template, code="P-CCCCCC", estimate=185.0, distance_cm=None),
    ]
    d = vr.summarize(rs)["real_world"]["by_distance"]
    rows = {r["band"]: r for r in d["operator_measured"]["rows"]}
    assert list(rows) == ["under 30 cm", "30-40 cm", "40-50 cm", "50 cm and over"]
    mid = rows["30-40 cm"]
    assert (mid["scans"], mid["scans_scored"], mid["hands_scored"], mid["sufficient"]) == (4, 4, 3, True)
    # Hand means +3, -2, +1.
    assert mid["mean_signed_error_mm"] == pytest.approx(2 / 3)
    assert mid["mae_mm"] == pytest.approx(2.0)
    far = rows["50 cm and over"]
    assert (far["hands_scored"], far["mean_signed_error_mm"], far["sufficient"]) == (1, pytest.approx(6.0), False)
    assert (rows["under 30 cm"]["rejected"], rows["under 30 cm"]["hands_scored"]) == (1, 0)
    assert rows["40-50 cm"]["scans"] == 0
    assert d["operator_measured"]["without_distance"] == 1


def test_coin_implied_distance_bands(template, enabled):
    rs = [
        make_record(template, code="P-AAAAAA", estimate=187.0, coin_distance_mm=(300.0, 480.0)),  # ~37.9 cm
        make_record(template, code="P-BBBBBB", estimate=181.0, coin_distance_mm=(450.0, 700.0)),  # ~56.1 cm
        make_record(template, code="P-CCCCCC", accepted=False),
    ]
    rs[2].capture.estimated_camera_distance_mm = None
    coin = vr.summarize(rs)["real_world"]["by_distance"]["coin_implied"]
    rows = {r["band"]: r for r in coin["rows"]}
    assert rows["30-40 cm"]["mean_signed_error_mm"] == pytest.approx(2.0)
    assert rows["50 cm and over"]["mean_signed_error_mm"] == pytest.approx(-4.0)
    assert coin["without_distance"] == 1 and "assumed" in coin["note"]
