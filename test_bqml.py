import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    os.environ["BQML_STORE_PATH"] = path
    # Import after setting env var so the module picks up the temp store path.
    import importlib
    import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c
    if os.path.exists(path):
        os.remove(path)


def row(id_, entity, event_time, pred_time, version, split, features=None):
    return {
        "id": id_,
        "entity": entity,
        "eventTime": event_time,
        "predictionTime": pred_time,
        "version": version,
        "split": split,
        "features": features or {},
    }


def feat(value, available_at):
    return {"value": value, "availableAt": available_at}


def test_unknown_phase_returns_400(client):
    resp = client.post("/bqml", json={"phase": "bogus"})
    assert resp.status_code == 400
    assert resp.json() == {"error": "INVALID_INPUT"}


def test_select_dedup_feature_eligibility_and_tie_break(client):
    body = {
        "phase": "select",
        "runId": "run-1",
        "forbiddenFeatures": ["banned"],
        "numTrialsLimit": 10,
        "rows": [
            row(
                "b",
                "e1",
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                1,
                "TRAIN",
                {"f1": feat("1", "2024-01-01T00:00:00Z")},
            ),
            row(
                "a",
                "e1",
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                2,
                "TRAIN",
                {"f1": feat("1", "2024-01-01T00:00:00Z")},
            ),
            row(
                "c",
                "e2",
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                1,
                "EVAL",
                {"f1": feat("2", "2024-01-01T00:00:00Z"), "banned": feat("x", "2024-01-01T00:00:00Z")},
            ),
        ],
        "trials": [
            {"trialId": 9, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 4, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 1, "status": "FAILED", "evalMetric": 0.99},
        ],
    }
    resp = client.post("/bqml", json=body)
    assert resp.status_code == 200
    data = resp.json()
    # Two "e1" rows dedup to version 2 (id "a").
    assert data["trainRowIds"] == ["a"]
    assert data["evalRowIds"] == ["c"]
    # "banned" is forbidden, "f1" appears in every retained row and is available in time.
    assert data["featureNames"] == ["f1"]
    # Equal metrics for trial 9 and 4 -> smallest trialId (4) wins.
    assert data["selectedTrialId"] == 4
    assert data["reasonCodes"] == []
    assert len(data["datasetDigest"]) == 64


def test_select_replay_and_conflict(client):
    body = {
        "phase": "select",
        "runId": "run-2",
        "forbiddenFeatures": [],
        "numTrialsLimit": 5,
        "rows": [row("a", "e1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 1, "TRAIN")],
        "trials": [{"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.5}],
    }
    first = client.post("/bqml", json=body)
    assert first.status_code == 200
    replay = client.post("/bqml", json=body)
    assert replay.status_code == 200
    assert replay.json() == first.json()

    changed = dict(body)
    changed["numTrialsLimit"] = 6
    conflict = client.post("/bqml", json=changed)
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "RUN_ID_CONFLICT"}


def test_select_no_successful_trial_and_trial_limit(client):
    body = {
        "phase": "select",
        "runId": "run-3",
        "forbiddenFeatures": [],
        "numTrialsLimit": 1,
        "rows": [row("a", "e1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 1, "TRAIN")],
        "trials": [
            {"trialId": 1, "status": "FAILED", "evalMetric": 0.5},
            {"trialId": 2, "status": "FAILED", "evalMetric": 0.6},
        ],
    }
    resp = client.post("/bqml", json=body)
    data = resp.json()
    assert data["selectedTrialId"] is None
    assert set(data["reasonCodes"]) == {"TRIAL_LIMIT_EXCEEDED", "NO_SUCCESSFUL_TRIAL"}
    assert data["datasetDigest"] is not None  # dataset still computed


def test_evaluate_admit_and_reject(client):
    select_body = {
        "phase": "select",
        "runId": "run-4",
        "forbiddenFeatures": [],
        "numTrialsLimit": 5,
        "rows": [row("a", "e1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 1, "TRAIN")],
        "trials": [{"trialId": 7, "status": "SUCCEEDED", "evalMetric": 0.5}],
    }
    sel = client.post("/bqml", json=select_body).json()

    good_eval = {
        "phase": "evaluate",
        "runId": "run-4",
        "selectedTrialId": sel["selectedTrialId"],
        "datasetDigest": sel["datasetDigest"],
        "metricFloor": 0.5,
        "requiredSlices": {"critical": 0.5},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 0, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "other"},
        ],
        "bytesProcessed": 100,
        "maxBytes": 200,
    }
    resp = client.post("/bqml", json=good_eval).json()
    assert resp["decision"] == "admit"
    assert resp["criticalSlicePass"] is True
    assert abs(resp["testMetric"] - (2 / 3)) < 1e-9

    bad_lineage = dict(good_eval)
    bad_lineage["datasetDigest"] = "0" * 64
    resp2 = client.post("/bqml", json=bad_lineage).json()
    assert resp2["decision"] == "reject"
    assert resp2["criticalSlicePass"] is False
    assert "INVALID_LINEAGE" in resp2["reasonCodes"]

    missing_slice = dict(good_eval)
    missing_slice["requiredSlices"] = {"nope": 0.5}
    resp3 = client.post("/bqml", json=missing_slice).json()
    assert "MISSING_SLICE:nope" in resp3["reasonCodes"]
    assert resp3["decision"] == "reject"
    assert resp3["criticalSlicePass"] is False