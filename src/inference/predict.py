import pickle
import numpy as np

MODEL_PATH = "models/final/model.pkl"
SCALER_PATH = "models/final/scaler.pkl"
ENCODER_PATH = "models/final/label_encoder.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

N_FEATURES = scaler.n_features_in_

def predict(features):
    x = np.array(features)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, got {x.shape[1]}"
        )

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)
    return encoder.inverse_transform(pred)[0]
