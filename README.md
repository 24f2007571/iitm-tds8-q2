# Leakage-Safe BQML Experiment Gate

A single `POST /bqml` endpoint that enforces a **two-phase gate** so that
model *selection* never sees final-test rows, and the final-test *admission*
decision can only run against a dataset/trial that was frozen during
selection.

## How selection stays separate from final-test admission

- **Phase `select`** only ever receives `TRAIN`/`EVAL` rows. It computes a
  deterministic dataset (dedup + point-in-time-safe feature eligibility) and
  a trial, then freezes both as a `datasetDigest` — a SHA-256 hash of the
  exact `{trainRowIds, evalRowIds, featureNames}` triple.
- The response for a given `runId` is **persisted** (to a local JSON file,
  so it survives restarts). Replaying the identical select payload returns
  the identical response; reusing the same `runId` with a *different*
  payload is rejected with `409 RUN_ID_CONFLICT` — so a run's dataset can't
  be silently redefined after the fact.
- **Phase `evaluate`** never recomputes the dataset. It only accepts a
  `(selectedTrialId, datasetDigest)` pair and checks it **exactly** matches
  a stored, successful `select` response for that `runId` (`INVALID_LINEAGE`
  otherwise). Only then are held-out test rows scored. This is what
  prevents "final-test admission" from being reachable except through a
  frozen, previously-completed selection.

## Endpoint contract

See the assignment spec for full field-level rules. Summary:

- `phase: "select"` — dedupes rows by `[entity, UTC(eventTime)]` (highest
  `version`, then UTF-8-byte-smallest `id`), computes eligible features
  (present in every retained row, not forbidden, `availableAt <=
  predictionTime` everywhere), and selects the best `SUCCEEDED` trial with a
  finite `evalMetric` (max metric, tie-break smallest `trialId`). Any of
  `INVALID_INPUT` / `TRIAL_LIMIT_EXCEEDED` / `NO_SUCCESSFUL_TRIAL` forces
  `selectedTrialId: null`; only `INVALID_INPUT` also forces
  `datasetDigest: null` (and empty row/feature arrays).
- `phase: "evaluate"` — verifies lineage against the stored selection, scores
  binary label/prediction rows (aggregate + per-required-slice accuracy,
  rounded to 12 decimals), and admits only if lineage + all rows are valid,
  aggregate and slice floors are met, every required slice is present, and
  `bytesProcessed <= maxBytes`.
- Unknown/missing `phase` → `HTTP 400 {"error": "INVALID_INPUT"}`.
- Reused `runId` with a changed select payload → `HTTP 409 {"error":
  "RUN_ID_CONFLICT"}`.
- Everything else is `HTTP 200`, including business-rule failures — those
  are reported via `reasonCodes`, not HTTP status.

## Files

- `app.py` — the FastAPI service (single file, no external state besides
  the JSON store).
- `tests/test_bqml.py` — pytest smoke tests covering dedup, feature
  eligibility, the trial tie-break example from the spec, replay/conflict
  behavior, and the evaluate admit/reject/lineage/slice gates.
- `requirements.txt` — pinned dependencies.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/bqml \
  -H "Content-Type: application/json" \
  -d '{"phase":"select","runId":"r1","forbiddenFeatures":[],"numTrialsLimit":10,
       "rows":[{"id":"a","entity":"e1","eventTime":"2024-01-01T00:00:00Z",
                "predictionTime":"2024-01-02T00:00:00Z","version":1,"split":"TRAIN",
                "features":{"f1":{"value":"1","availableAt":"2024-01-01T00:00:00Z"}}}],
       "trials":[{"trialId":1,"status":"SUCCEEDED","evalMetric":0.9}]}'
```

Run tests:

```bash
pytest tests/ -v
```

## Deploying to Render (same pattern as your other graded FastAPI services)

1. Push this folder to a GitHub repo.
2. New **Web Service** on Render, pointing at that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. The JSON store defaults to a file next to `app.py`; on Render's ephemeral
   filesystem this persists for the life of the running instance (fine for
   grading within one deploy), or set `BQML_STORE_PATH` to a persistent disk
   path if you attach one.

## Notes / assumptions worth knowing if you need to defend a design choice

- `INVALID_LINEAGE` and byte-budget (`BYTE_LIMIT`) checks are computed
  independently of row validity — only `AGGREGATE_FLOOR` / `MISSING_SLICE` /
  `SLICE_FLOOR` are skipped when rows are empty/invalid (per spec: "skip
  aggregate and required-slice checks" is scoped to row validity only).
- `criticalSlicePass` is deliberately *not* affected by `AGGREGATE_FLOOR` or
  `BYTE_LIMIT` — only by invalid input/lineage/rows and slice-specific
  failures, per spec.
- All reasonCodes are UTF-8-byte sorted and deduplicated before being
  returned.