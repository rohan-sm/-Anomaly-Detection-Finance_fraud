import joblib

from src.models.isolation_forest import run_isolation_forest
from src.models.evaluate import ranking_auc
from src.utils.config import MODELS_DIR

# Operating-point band the shipped artifact must stay inside. Widening the
# contamination without retraining is a silent behavioural change, so it is
# pinned here.
MIN_ANOMALY_RATE = 0.005
MAX_ANOMALY_RATE = 0.02


def test_isolation_forest_inference(contract, scaler, features_frame, labels):
    model = joblib.load(MODELS_DIR / "isolation_forest_v1.pkl")
    features = contract["features"]
    threshold = contract["thresholds"]["isolation_forest"]["threshold_value"]

    X_scaled = scaler.transform(features_frame[features])

    scores, flags = run_isolation_forest(model, X_scaled, threshold)

    assert len(scores) == len(features_frame)
    assert len(flags) == len(features_frame)

    anomaly_rate = flags.mean()
    assert MIN_ANOMALY_RATE <= anomaly_rate <= MAX_ANOMALY_RATE, (
        f"anomaly rate {anomaly_rate:.4%} outside "
        f"[{MIN_ANOMALY_RATE:.1%}, {MAX_ANOMALY_RATE:.1%}]"
    )


def test_isolation_forest_ranking_auc(contract, scaler, features_frame, labels):
    """Guard the documented ranking ROC-AUC.

    `decision_function` is lower-is-anomalous, so the score must be negated.
    This pins the number quoted in the model comparison notebook.
    """
    model = joblib.load(MODELS_DIR / "isolation_forest_v1.pkl")
    features = contract["features"]

    scores, _ = run_isolation_forest(
        model,
        scaler.transform(features_frame[features]),
        contract["thresholds"]["isolation_forest"]["threshold_value"],
    )

    auc = ranking_auc(labels, scores)
    assert 0.58 <= auc <= 0.62, f"ranking ROC-AUC drifted to {auc:.4f}"


def test_isolation_forest_beats_random(contract, scaler, features_frame, labels):
    """A model at or below 0.5 AUC is no better than random ranking.

    Guards the core premise: the engineered features actually carry fraud signal.
    """
    model = joblib.load(MODELS_DIR / "isolation_forest_v1.pkl")
    features = contract["features"]

    scores, _ = run_isolation_forest(
        model,
        scaler.transform(features_frame[features]),
        contract["thresholds"]["isolation_forest"]["threshold_value"],
    )

    assert ranking_auc(labels, scores) > 0.5
