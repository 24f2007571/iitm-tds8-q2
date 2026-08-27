"""
Leakage-safe BigQuery ML experiment gate.

POST /bqml supports two phases:
  - "select":   dedupe rows, compute an eligible feature set, pick the best
                trial, and freeze a dataset digest. Never touches final-test
                rows.
  - "evaluate": admit/reject a frozen (selectedTrialId, datasetDigest) pair
                against held-out test rows, subject to lineage, metric-floor,
                slice-floor and byte-budget gates.

State from successful/attempted "select" calls is persisted to a small JSON
file (keyed by runId) so that:
  - an identical replay of a select call returns the exact same response,
  - reusing a runId with a *different* selection payload is rejected with
    HTTP 409 {"error": "RUN_ID_CONFLICT"},
  - a later "evaluate" call can verify lineage against the frozen selection.
"""

import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Leakage-Safe BQML Gate")

# ---------------------------------------------------------------------------
# Persistence (file-backed so state survives process restarts)
# ---------------------------------------------------------------------------

STORE_PATH = os.environ.get(
    "BQML_STORE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bqml_store.json")
)
_lock = threading.Lock()


def _load_store() -> Dict[str, Any]:
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(store: Dict[str, Any]) -> None:
    tmp_path = STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f)
    os.replace(tmp_path, STORE_PATH)


def _get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _load_store().get(run_id)


def _put_run(run_id: str, record: Dict[str, Any]) -> None:
    with _lock:
        store = _load_store()
        store[run_id] = record
        _save_store(store)


# ---------------------------------------------------------------------------
# Timestamp handling: strict YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
# ---------------------------------------------------------------------------

_TS_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:\.(?P<frac>\d{1,3}))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)


def parse_instant(s: Any) -> Optional[datetime]:
    """Parse a strict ISO-8601 instant into an aware UTC datetime, or None."""
    if not isinstance(s, str):
        return None
    m = _TS_RE.match(s)
    if not m:
        return None

    year = int(m.group("year"))
    month = int(m.group("month"))
    day = int(m.group("day"))
    hour = int(m.group("hour"))
    minute = int(m.group("minute"))
    second = int(m.group("second"))
    frac = m.group("frac") or "0"
    micros = int(frac.ljust(6, "0")[:6])
    tz = m.group("tz")

    if not (1 <= month <= 12):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None

    if tz == "Z":
        offset = timedelta(0)
    else:
        sign = 1 if tz[0] == "+" else -1
        oh = int(tz[1:3])
        om = int(tz[4:6])
        if oh > 23 or om > 59:
            return None
        offset = sign * timedelta(hours=oh, minutes=om)

    try:
        wall_clock = datetime(year, month, day, hour, minute, second, micros, tzinfo=timezone.utc)
    except ValueError:
        return None
    # wall_clock holds the literal Y/M/D/H/M/S values tagged as UTC; the real
    # UTC instant is wall_clock minus the stated offset.
    return wall_clock - offset


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def is_safe_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and abs(v) <= 2**53 - 1


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def utf8_sort(items: List[str]) -> List[str]:
    return sorted(items, key=lambda s: s.encode("utf-8"))


def utf8_lt(a: str, b: str) -> bool:
    return a.encode("utf-8") < b.encode("utf-8")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dataset_digest(train_ids: List[str], eval_ids: List[str], feature_names: List[str]) -> str:
    payload = (
        '{"trainRowIds":' + json.dumps(train_ids, separators=(",", ":"), ensure_ascii=False)
        + ',"evalRowIds":' + json.dumps(eval_ids, separators=(",", ":"), ensure_ascii=False)
        + ',"featureNames":' + json.dumps(feature_names, separators=(",", ":"), ensure_ascii=False)
        + "}"
    )
    return sha256_hex(payload)


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Phase 1: "select" — schema validation
# ---------------------------------------------------------------------------

def validate_select(body: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not isinstance(body, dict):
        return False, None

    run_id = body.get("runId")
    if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
        return False, None

    forbidden = body.get("forbiddenFeatures")
    if not isinstance(forbidden, list) or not all(isinstance(x, str) for x in forbidden):
        return False, None

    num_limit = body.get("numTrialsLimit")
    if not is_safe_int(num_limit) or num_limit < 1:
        return False, None

    rows = body.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        return False, None

    trials = body.get("trials")
    if not isinstance(trials, list):
        return False, None

    parsed_rows = []
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            return False, None
        rid = row.get("id")
        entity = row.get("entity")
        event_time = row.get("eventTime")
        pred_time = row.get("predictionTime")
        version = row.get("version")
        split = row.get("split")
        features = row.get("features")

        if not isinstance(rid, str) or rid == "":
            return False, None
        if rid in seen_ids:
            return False, None
        seen_ids.add(rid)

        if not isinstance(entity, str):
            return False, None

        event_dt = parse_instant(event_time)
        if event_dt is None:
            return False, None

        pred_dt = parse_instant(pred_time)
        if pred_dt is None:
            return False, None

        if not is_safe_int(version) or version < 0:
            return False, None

        if split not in ("TRAIN", "EVAL"):
            return False, None

        if not isinstance(features, dict):
            return False, None

        parsed_features = {}
        for fname, fval in features.items():
            if not isinstance(fname, str):
                return False, None
            if not isinstance(fval, dict) or "value" not in fval:
                return False, None
            avail_dt = parse_instant(fval.get("availableAt"))
            if avail_dt is None:
                return False, None
            parsed_features[fname] = {"availableAt": avail_dt}

        parsed_rows.append(
            {
                "id": rid,
                "entity": entity,
                "eventDt": event_dt,
                "predictionDt": pred_dt,
                "version": version,
                "split": split,
                "features": parsed_features,
            }
        )

    parsed_trials = []
    seen_trial_ids = set()
    for t in trials:
        if not isinstance(t, dict):
            return False, None
        tid = t.get("trialId")
        status = t.get("status")
        metric = t.get("evalMetric")
        if not is_safe_int(tid) or tid < 0:
            return False, None
        if tid in seen_trial_ids:
            return False, None
        seen_trial_ids.add(tid)
        if status not in ("SUCCEEDED", "FAILED"):
            return False, None
        if not is_number(metric):
            return False, None
        parsed_trials.append({"trialId": tid, "status": status, "evalMetric": float(metric)})

    return True, {
        "runId": run_id,
        "forbiddenFeatures": set(forbidden),
        "numTrialsLimit": num_limit,
        "rows": parsed_rows,
        "trials": parsed_trials,
    }


def compute_select(parsed: Dict[str, Any]) -> Dict[str, Any]:
    rows = parsed["rows"]
    forbidden = parsed["forbiddenFeatures"]
    trials = parsed["trials"]
    num_limit = parsed["numTrialsLimit"]

    # Deduplicate by [entity, UTC(eventTime)]: highest version wins, then
    # smallest UTF-8-byte ID.
    groups: Dict[Tuple[str, datetime], Dict[str, Any]] = {}
    for row in rows:
        key = (row["entity"], row["eventDt"])
        current = groups.get(key)
        if current is None:
            groups[key] = row
        elif row["version"] > current["version"]:
            groups[key] = row
        elif row["version"] == current["version"] and utf8_lt(row["id"], current["id"]):
            groups[key] = row

    retained = list(groups.values())

    # A feature is eligible only if it appears in every retained row, is not
    # forbidden, and its availableAt <= that row's predictionTime everywhere.
    if retained:
        common = set(retained[0]["features"].keys())
        for r in retained[1:]:
            common &= set(r["features"].keys())
    else:
        common = set()

    eligible_features = []
    for fname in common:
        if fname in forbidden:
            continue
        ok = True
        for r in retained:
            feat = r["features"][fname]
            if feat["availableAt"] > r["predictionDt"]:
                ok = False
                break
        if ok:
            eligible_features.append(fname)

    feature_names = utf8_sort(eligible_features)
    train_ids = utf8_sort([r["id"] for r in retained if r["split"] == "TRAIN"])
    eval_ids = utf8_sort([r["id"] for r in retained if r["split"] == "EVAL"])

    codes = []
    if len(trials) > num_limit:
        codes.append("TRIAL_LIMIT_EXCEEDED")

    eligible_trials = [t for t in trials if t["status"] == "SUCCEEDED" and math.isfinite(t["evalMetric"])]
    if not eligible_trials:
        codes.append("NO_SUCCESSFUL_TRIAL")

    selected_trial_id = None
    if not codes:
        best = eligible_trials[0]
        for t in eligible_trials[1:]:
            if t["evalMetric"] > best["evalMetric"] or (
                t["evalMetric"] == best["evalMetric"] and t["trialId"] < best["trialId"]
            ):
                best = t
        selected_trial_id = best["trialId"]

    digest = dataset_digest(train_ids, eval_ids, feature_names)

    return {
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": utf8_sort(list(set(codes))),
    }


def handle_select(body: Dict[str, Any]) -> JSONResponse:
    run_id = body.get("runId")
    has_usable_run_id = isinstance(run_id, str) and 1 <= len(run_id) <= 128

    if has_usable_run_id:
        existing = _get_run(run_id)
        if existing is not None and existing.get("phase") == "select":
            if existing.get("request") == body:
                return JSONResponse(status_code=200, content=existing["response"])
            return JSONResponse(status_code=409, content={"error": "RUN_ID_CONFLICT"})

    ok, parsed = validate_select(body)

    if not ok:
        response = {
            "runId": body.get("runId"),
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }
        if has_usable_run_id:
            _put_run(run_id, {"phase": "select", "request": body, "response": response})
        return JSONResponse(status_code=200, content=response)

    result = compute_select(parsed)
    response = {
        "runId": run_id,
        "selectedTrialId": result["selectedTrialId"],
        "trainRowIds": result["trainRowIds"],
        "evalRowIds": result["evalRowIds"],
        "featureNames": result["featureNames"],
        "datasetDigest": result["datasetDigest"],
        "reasonCodes": result["reasonCodes"],
    }
    _put_run(run_id, {"phase": "select", "request": body, "response": response})
    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------------------------
# Phase 2: "evaluate"
# ---------------------------------------------------------------------------

def validate_evaluate(body: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    run_id = body.get("runId")
    if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
        return False, None

    selected_trial_id = body.get("selectedTrialId")
    if not is_safe_int(selected_trial_id) or selected_trial_id < 0:
        return False, None

    digest = body.get("datasetDigest")
    if not isinstance(digest, str) or not HEX64_RE.match(digest):
        return False, None

    metric_floor = body.get("metricFloor")
    if not is_number(metric_floor) or not math.isfinite(metric_floor) or not (0 <= metric_floor <= 1):
        return False, None

    required_slices = body.get("requiredSlices")
    if not isinstance(required_slices, dict):
        return False, None
    parsed_slices = {}
    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return False, None
        if not is_number(floor) or not math.isfinite(floor) or not (0 <= floor <= 1):
            return False, None
        parsed_slices[name] = float(floor)

    rows = body.get("rows")
    if not isinstance(rows, list):
        return False, None

    bytes_processed = body.get("bytesProcessed")
    if not is_safe_int(bytes_processed) or bytes_processed < 0:
        return False, None

    max_bytes = body.get("maxBytes")
    if not is_safe_int(max_bytes) or max_bytes < 0:
        return False, None

    return True, {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": digest,
        "metricFloor": float(metric_floor),
        "requiredSlices": parsed_slices,
        "rows": rows,
        "bytesProcessed": bytes_processed,
        "maxBytes": max_bytes,
    }


def validate_test_rows(rows: List[Any]) -> Optional[List[Dict[str, Any]]]:
    if len(rows) == 0:
        return None
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        label = row.get("label")
        prediction = row.get("prediction")
        slice_name = row.get("slice")
        if not is_safe_int(label) or label not in (0, 1):
            return None
        if not is_safe_int(prediction) or prediction not in (0, 1):
            return None
        if not isinstance(slice_name, str) or slice_name == "":
            return None
        parsed.append({"label": label, "prediction": prediction, "slice": slice_name})
    return parsed


def handle_evaluate(body: Dict[str, Any]) -> JSONResponse:
    ok, parsed = validate_evaluate(body)
    if not ok:
        raw_bytes = body.get("bytesProcessed")
        if not (is_safe_int(raw_bytes) and raw_bytes >= 0):
            raw_bytes = 0
        response = {
            "runId": body.get("runId"),
            "selectedTrialId": body.get("selectedTrialId"),
            "datasetDigest": body.get("datasetDigest"),
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": raw_bytes,
            "reasonCodes": ["INVALID_INPUT"],
        }
        return JSONResponse(status_code=200, content=response)

    run_id = parsed["runId"]
    selected_trial_id = parsed["selectedTrialId"]
    digest = parsed["datasetDigest"]
    metric_floor = parsed["metricFloor"]
    required_slices = parsed["requiredSlices"]
    bytes_processed = parsed["bytesProcessed"]
    max_bytes = parsed["maxBytes"]

    codes = set()

    stored = _get_run(run_id)
    lineage_ok = (
        stored is not None
        and stored.get("phase") == "select"
        and stored["response"].get("selectedTrialId") is not None
        and stored["response"].get("selectedTrialId") == selected_trial_id
        and stored["response"].get("datasetDigest") == digest
    )
    if not lineage_ok:
        codes.add("INVALID_LINEAGE")

    valid_rows = validate_test_rows(parsed["rows"])
    rows_ok = valid_rows is not None
    if not rows_ok:
        codes.add("INVALID_TEST_ROW")

    test_metric = None
    if rows_ok:
        total = len(valid_rows)
        correct = sum(1 for r in valid_rows if r["label"] == r["prediction"])
        aggregate = round(correct / total, 12)
        test_metric = aggregate

        if aggregate < metric_floor:
            codes.add("AGGREGATE_FLOOR")

        for slice_name, floor in required_slices.items():
            slice_rows = [r for r in valid_rows if r["slice"] == slice_name]
            if not slice_rows:
                codes.add(f"MISSING_SLICE:{slice_name}")
                continue
            s_total = len(slice_rows)
            s_correct = sum(1 for r in slice_rows if r["label"] == r["prediction"])
            s_acc = round(s_correct / s_total, 12)
            if s_acc < floor:
                codes.add(f"SLICE_FLOOR:{slice_name}")

    byte_ok = bytes_processed <= max_bytes
    if not byte_ok:
        codes.add("BYTE_LIMIT")

    critical_slice_pass = not (
        ("INVALID_LINEAGE" in codes)
        or ("INVALID_TEST_ROW" in codes)
        or any(c.startswith("MISSING_SLICE:") for c in codes)
        or any(c.startswith("SLICE_FLOOR:") for c in codes)
    )

    admit = (
        lineage_ok
        and rows_ok
        and "AGGREGATE_FLOOR" not in codes
        and not any(c.startswith("MISSING_SLICE:") for c in codes)
        and not any(c.startswith("SLICE_FLOOR:") for c in codes)
        and byte_ok
    )

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": "admit" if admit else "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": utf8_sort(list(codes)),
    }
    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/bqml")
async def bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict) or body.get("phase") not in ("select", "evaluate"):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if body["phase"] == "select":
        return handle_select(body)
    return handle_evaluate(body)