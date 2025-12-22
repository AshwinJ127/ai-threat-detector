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

# Fake sample with correct feature count
n_features = scaler.mean_.shape[0]
x = np.random.rand(1, n_features)

x_scaled = scaler.transform(x)
pred = model.predict(x_scaled)
label = encoder.inverse_transform(pred)

print("Model loaded successfully")
print("Prediction:", label[0])
