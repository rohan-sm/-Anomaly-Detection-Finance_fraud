import numpy as np
from tensorflow.keras.models import load_model


def load_autoencoder(model_path: str):
    # Load trained autoencoder model.
    return load_model(model_path)


def compute_reconstruction_error(
    model,
    X_scaled: np.ndarray
) -> np.ndarray:
    """ Compute reconstruction error (MSE per sample).
        Higher error = more anomalous."""
    X_recon = model.predict(X_scaled, verbose=0)
    errors = np.mean(np.square(X_scaled - X_recon), axis=1)
    return errors


def flag_anomalies(
    errors: np.ndarray,
    threshold: float
) -> np.ndarray:
    # Convert reconstruction errors into anomaly flags.
    return errors > threshold


def run_autoencoder(
    model,
    X_scaled: np.ndarray,
    threshold: float
):
   
    # Full autoencoder inference pipeline.
    errors = compute_reconstruction_error(model, X_scaled)
    flags = flag_anomalies(errors, threshold)
    return errors, flags
