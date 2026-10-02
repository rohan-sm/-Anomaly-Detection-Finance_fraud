"""End-to-end orchestration for the fraud anomaly detection pipeline.

Stages, in dependency order:

    generate  -> data/raw/transactions_raw.csv
    features  -> data/processed/transactions_features.csv
    train     -> models/*.pkl, models/*.json
    evaluate  -> reports/iso_results.csv

Usage
-----
    python run_pipeline.py                      # run every stage
    python run_pipeline.py --stage train        # retrain only
    python run_pipeline.py --stage evaluate     # re-score reports only
    python run_pipeline.py --stage train --contamination 0.05

Training is seeded (RANDOM_SEED), so re-running reproduces the committed
artifacts bit-for-bit.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.models.isolation_forest import (
    CONTAMINATION,
    train_isolation_forest,
    score_transactions,
)
from src.models.evaluate import contamination_sweep, evaluate_flags, ranking_auc

FEATURES_CSV = PROCESSED_DATA_DIR / "transactions_features.csv"
MODEL_PATH = MODELS_DIR / "isolation_forest_v1.pkl"
SCALER_PATH = MODELS_DIR / "standard_scaler_v1.pkl"
FEATURES_PATH = MODELS_DIR / "model_features_v1.json"
THRESHOLDS_PATH = MODELS_DIR / "thresholds_v1.json"

SWEEP_RATES = [0.005, 0.01, 0.02, 0.05, 0.10]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def configure_console_output() -> None:
    """Stop a console encoding error from aborting the run.

    Windows consoles commonly default to cp1252, which cannot encode characters
    like the rupee sign. A single un-encodable character in a log line would
    otherwise kill the pipeline partway through, so un-encodable output is
    replaced rather than raised.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


# ---------------- STAGES ----------------


def stage_generate() -> None:
    from src.data_generation.generate_dataset import main as generate_main

    banner("[1/4] GENERATE  synthetic transaction dataset")
    generate_main()


def stage_features() -> None:
    from src.feature_engineering.preprocess import run_feature_engineering

    banner("[2/4] FEATURES  engineer behavioral features")
    run_feature_engineering()


def _load_features() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    with open(FEATURES_PATH) as f:
        features = json.load(f)["features"]

    df = pd.read_csv(FEATURES_CSV)
    if "is_fraud" not in df.columns:
        raise ValueError(
            f"'is_fraud' label missing from {FEATURES_CSV}. Run --stage features first."
        )

    return df, df["is_fraud"].to_numpy(dtype=int), features


def stage_train(contamination: float = CONTAMINATION) -> dict:
    banner(f"[3/4] TRAIN     Isolation Forest (contamination={contamination})")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, labels, features = _load_features()
    print(f"[INFO] {len(df):,} rows x {len(features)} features | "
          f"fraud rate {labels.mean():.4%} | seed {RANDOM_SEED}")

    started = time.perf_counter()
    model, scaler, scores, threshold = train_isolation_forest(
        features_df=df,
        labels=labels,
        model_path=str(MODEL_PATH),
        scaler_path=str(SCALER_PATH),
        features_path=str(FEATURES_PATH),
        thresholds_path=str(THRESHOLDS_PATH),
        features=features,
        contamination=contamination,
    )
    elapsed = time.perf_counter() - started

    flags = scores < threshold
    metrics = evaluate_flags(labels, flags)
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    auc = ranking_auc(labels, scores)

    print(f"[SUCCESS] Fitted in {elapsed:.2f}s")
    print(f"    threshold ({contamination:.1%} pct of decision_function) = {threshold:.6e}")
    print(f"    flagged {flags.sum():,} / {len(flags):,} rows "
          f"({metrics['anomaly_rate']:.2%})")
    print(f"    precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} "
          f"f1={metrics['f1']:.4f} "
          f"ranking_roc_auc={auc:.4f}")
    print(f"    confusion matrix TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"[SUCCESS] Artifacts written -> {MODELS_DIR}")

    return {"scores": scores, "labels": labels}


def stage_evaluate(scores=None, labels=None) -> pd.DataFrame:
    banner("[4/4] EVALUATE  contamination sensitivity sweep")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if scores is None or labels is None:
        from src.models.isolation_forest import load_isolation_forest
        import joblib

        df, labels, features = _load_features()
        scaler = joblib.load(SCALER_PATH)
        model = load_isolation_forest(str(MODEL_PATH))
        scores = score_transactions(model, scaler.transform(df[features]))

    results = contamination_sweep(
        y_true=labels,
        scores=scores,
        rates=SWEEP_RATES,
        model_name="Isolation Forest",
        higher_is_anomalous=False,
    )

    out_path = REPORTS_DIR / "iso_results.csv"
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\n[SUCCESS] Saved -> {out_path}")

    base_rate = float(np.mean(labels))
    best = results.loc[results["f1"].idxmax()]
    lift = best["precision"] / base_rate
    print(f"\nBase fraud rate           : {base_rate:.4%}")
    print(f"Best F1 operating point   : flag rate {best['anomaly_rate']:.2%} "
          f"-> precision {best['precision']:.4f} ({lift:.2f}x lift over base rate)")
    print("Note: ROC-AUC is constant across sweep points because the model is")
    print("      evaluated as a ranking function; only the operating point moves.")

    return results


# ---------------- ENTRY POINT ----------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fraud anomaly detection pipeline end to end."
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "generate", "features", "train", "evaluate"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=CONTAMINATION,
        help="Target anomaly rate used to fit the model (default: 0.01)",
    )
    args = parser.parse_args()

    configure_console_output()
    started = time.perf_counter()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "features":
        stage_features()
    elif args.stage == "train":
        stage_train(args.contamination)
    elif args.stage == "evaluate":
        stage_evaluate()
    else:
        np.random.seed(RANDOM_SEED)
        stage_generate()
        stage_features()
        result = stage_train(args.contamination)
        stage_evaluate(result["scores"], result["labels"])

    print(f"\nPipeline finished in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
