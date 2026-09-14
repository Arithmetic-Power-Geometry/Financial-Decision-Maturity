import json
from pathlib import Path


def test_preregistered_protocol_is_frozen_and_strict():
    p = Path(__file__).resolve().parents[1] / "config" / "pre_registered_ieee.json"
    d = json.loads(p.read_text())
    assert d["split"] == {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.20,
        "order": "chronological",
    }
    assert d["high_confidence_threshold"] == 0.90
    assert d["bootstrap"]["replicates"] == 1000
    assert set(d["required_baselines"]) == {
        "confidence_uncertainty",
        "trajectory_model",
    }
    assert d["primary_endpoints"] == ["auroc", "top10_precision"]
    assert "both have lower bound > 0" in d["success_rule"]
