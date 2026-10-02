import joblib

from src.models.one_class_svm import run_one_class_svm
from src.utils.config import MODELS_DIR

MIN_ANOMALY_RATE = 0.005
MAX_ANOMALY_RATE = 0.02


def test_one_class_svm_inference(contract, scaler, features_frame):
    model = joblib.load(MODELS_DIR / "one_class_svm_v1.pkl")
    features = contract["features"]
    threshold = contract["thresholds"]["one_class_svm"]["threshold_value"]

    X_scaled = scaler.transform(features_frame[features])

    scores, flags = run_one_class_svm(model, X_scaled, threshold)

    assert len(scores) == len(features_frame)
    assert len(flags) == len(features_frame)

    anomaly_rate = flags.mean()
    assert MIN_ANOMALY_RATE <= anomaly_rate <= MAX_ANOMALY_RATE, (
        f"anomaly rate {anomaly_rate:.4%} outside "
        f"[{MIN_ANOMALY_RATE:.1%}, {MAX_ANOMALY_RATE:.1%}]"
    )


def test_one_class_svm_scores_are_oriented(contract, scaler, features_frame):
    """`decision_function` is lower-is-anomalous for the SVM too.

    Pins the sign convention shared by `src/models/isolation_forest.py` and
    `src/models/one_class_svm.py`; `app/model_loader.py` negates to match.
    """
    model = joblib.load(MODELS_DIR / "one_class_svm_v1.pkl")
    features = contract["features"]

    scores, flags = run_one_class_svm(
        model,
        scaler.transform(features_frame[features]),
        contract["thresholds"]["one_class_svm"]["threshold_value"],
    )

    flagged_scores = scores[flags]
    unflagged_scores = scores[~flags]

    assert flagged_scores.max() < unflagged_scores.min()
