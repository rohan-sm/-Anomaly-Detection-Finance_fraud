"""API-level tests for the fraud detection service.

These cover the request/response contract, input validation, the feature
contract enforced by ModelService, and one behavioural assertion that the
per-feature explanation actually reflects the transaction that was submitted.
"""

import pytest

RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

NORMAL_TXN = {
    "customer_id": "CUST_00001",
    "amount": 850.0,
    "timestamp": "2025-02-14T14:30:00",
    "hour": 14,
    "distance_from_home": 6.5,
}

FAR_TXN = {
    "customer_id": "CUST_00002",
    "amount": 45.0,
    "timestamp": "2025-02-14T03:10:00",
    "hour": 3,
    "distance_from_home": 980.0,
}


# ---------------- HEALTH ----------------


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "isolation_forest_v1"
    assert body["features"] == 9


def test_health_probe(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---------------- RESPONSE CONTRACT ----------------


def test_predict_returns_full_contract(client):
    response = client.post("/predict", json=NORMAL_TXN)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "fraud_score",
        "fraud_probability",
        "is_fraud",
        "risk_level",
        "contributing_factors",
        "explanation",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["fraud_score"], float)
    assert body["risk_level"] in RISK_LEVELS
    assert body["explanation"]


def test_predict_contributing_factors_shape(client):
    body = client.post("/predict", json=NORMAL_TXN).json()

    factors = body["contributing_factors"]
    assert 1 <= len(factors) <= 3
    for factor in factors:
        assert set(factor) == {"feature", "value", "z_score", "direction"}
        assert factor["direction"] in {"above_normal", "below_normal"}
        assert isinstance(factor["z_score"], (int, float))


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=NORMAL_TXN).json()
    second = client.post("/predict", json=NORMAL_TXN).json()
    assert first == second


def test_risk_level_is_consistent_with_probability(client):
    body = client.post("/predict", json=FAR_TXN).json()

    probability = body["fraud_probability"]
    if probability >= 0.8:
        expected = "HIGH"
    elif probability >= 0.5:
        expected = "MEDIUM"
    else:
        expected = "LOW"

    assert body["risk_level"] == expected
    assert body["is_fraud"] == (probability >= 0.5)


# ---------------- BEHAVIOUR ----------------


def test_normal_local_daytime_transaction_is_low_risk(client):
    body = client.post("/predict", json=NORMAL_TXN).json()
    assert body["risk_level"] == "LOW"
    assert body["is_fraud"] is False


def test_explanation_surfaces_dominant_feature(client):
    """A 980 km transaction should name distance as the top driver.

    This is the assertion that proves the per-feature contribution ranking is
    wired to real feature values rather than a static template.
    """
    body = client.post("/predict", json=FAR_TXN).json()

    top = body["contributing_factors"][0]
    assert top["feature"] == "distance_from_home"
    assert top["z_score"] > 2.0
    assert top["direction"] == "above_normal"
    assert top["feature"] in body["explanation"]


# ---------------- VALIDATION ----------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**NORMAL_TXN, "amount": -100.0}, id="negative_amount"),
        pytest.param({**NORMAL_TXN, "amount": 0}, id="zero_amount"),
        pytest.param({**NORMAL_TXN, "hour": 24}, id="hour_above_range"),
        pytest.param({**NORMAL_TXN, "hour": -1}, id="hour_below_range"),
        pytest.param({**NORMAL_TXN, "distance_from_home": -5.0}, id="negative_distance"),
        pytest.param({**NORMAL_TXN, "timestamp": "not-a-timestamp"}, id="bad_timestamp"),
    ],
)
def test_invalid_payloads_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "missing",
    ["customer_id", "amount", "timestamp", "hour", "distance_from_home"],
)
def test_missing_required_field_is_rejected(client, missing):
    payload = {k: v for k, v in NORMAL_TXN.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/batch_predict", json={"transactions": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    payload = {"transactions": [NORMAL_TXN] * 1001}
    assert client.post("/batch_predict", json=payload).status_code == 422


# ---------------- BATCH ----------------


def test_batch_predict_matches_single_predictions(client):
    """The vectorised batch path must agree with the single-transaction path.

    Both routes share `score_transactions`, so any divergence means the two
    serving paths have drifted apart.

    Each transaction uses a distinct customer on purpose. Rolling features
    (`avg_amount_24h`, `txn_count_1h`, `time_since_last_txn_sec`) are computed
    within the submitted batch, so two rows for the same customer would share a
    window and legitimately score differently from the single-request path.
    """
    transactions = [
        NORMAL_TXN,
        FAR_TXN,
        {**NORMAL_TXN, "customer_id": "CUST_00003", "amount": 12.0, "hour": 23},
    ]

    batch = client.post("/batch_predict", json={"transactions": transactions}).json()
    assert batch["count"] == 3
    assert len(batch["predictions"]) == 3
    assert batch["flagged"] == sum(1 for p in batch["predictions"] if p["is_fraud"])

    for transaction, prediction in zip(transactions, batch["predictions"]):
        single = client.post("/predict", json=transaction).json()
        assert prediction["fraud_score"] == single["fraud_score"]
        assert prediction["fraud_probability"] == single["fraud_probability"]
        assert prediction["risk_level"] == single["risk_level"]


def test_batch_preserves_submission_order(client):
    """Regression: build_features sorts by customer, which must not leak out.

    Submitting CUST_00002 first and CUST_00001 second used to return the
    predictions transposed, because the sort inside build_features permuted the
    rows and the scores were zipped back onto the permuted frame.
    """
    far_first = {**FAR_TXN, "customer_id": "CUST_00002"}
    normal_second = {**NORMAL_TXN, "customer_id": "CUST_00001"}

    response = client.post(
        "/batch_predict", json={"transactions": [far_first, normal_second]}
    )
    predictions = response.json()["predictions"]

    assert predictions[0] == client.post("/predict", json=far_first).json()
    assert predictions[1] == client.post("/predict", json=normal_second).json()


def test_batch_flag_rate_is_consistent(client):
    batch = client.post(
        "/batch_predict", json={"transactions": [NORMAL_TXN, FAR_TXN]}
    ).json()
    assert batch["flag_rate"] == pytest.approx(batch["flagged"] / batch["count"], abs=1e-4)


# ---------------- MODEL SERVICE ----------------


def test_feature_contract_rejects_missing_feature(model_service):
    with pytest.raises(ValueError, match="Missing features"):
        model_service.predict({"amount_dev_log": 0.0})


def test_feature_contract_matches_scaler(model_service, client):
    body = client.get("/").json()
    assert body["features"] == len(model_service.features)
    assert len(model_service.features) == len(model_service.scaler.mean_)


def test_far_transaction_scores_higher_than_routine_one(client):
    """A 980 km, 3 AM transaction must outrank a routine local afternoon one."""
    routine = client.post("/predict", json=NORMAL_TXN).json()
    far = client.post("/predict", json=FAR_TXN).json()
    assert far["fraud_score"] > routine["fraud_score"]
    assert far["fraud_probability"] > routine["fraud_probability"]


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/predict" in paths
    assert "/batch_predict" in paths
