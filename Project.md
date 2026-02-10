# AI Threat Detector Project Documentation

## Project Overview

The AI Threat Detector is an end-to-end system designed for real-time classification of network traffic as either benign or malicious. It leverages a streaming architecture to process data, apply machine learning models for inference, and visualize the results on a live dashboard.

**Primary Goal:** To provide real-time identification of network threats by analyzing traffic patterns and presenting actionable insights through an interactive dashboard.

## Architecture & Data Flow

The system is built around a decoupled, event-driven architecture using Apache Kafka as its central message broker.

1.  **Data Simulation (Producer)**:
    *   A Python script (`src/streaming/producer.py`) simulates incoming network traffic.
    *   It generates a 70-feature vector representing network packet characteristics, along with associated metadata (e.g., source/destination IPs, protocol).
    *   This simulated data is then published to a Kafka topic named `network-traffic`.

2.  **Streaming Bridge (Consumer)**:
    *   Another Python script (`src/streaming/consumer.py`) acts as a Kafka consumer.
    *   It subscribes to the `network-traffic` topic, continuously retrieving the simulated traffic data.
    *   Upon receiving data, the consumer acts as a bridge, forwarding the network traffic feature vector via an HTTP POST request to the prediction API's `/predict` endpoint.

3.  **Inference Service (FastAPI)**:
    *   A FastAPI application (`src/api/app.py`) serves as the central API for the system.
    *   It exposes a `/predict` endpoint that receives the feature vectors from the Kafka consumer.
    *   Internally, `src/api/app.py` utilizes the prediction logic from `src/inference/predict.py`.
    *   The `predict.py` module loads a pre-trained LightGBM model, a data scaler, and a label encoder (all typically pickled and stored in `models/final/`). It then uses these components to classify the incoming network traffic as benign or malicious.
    *   After classification, the FastAPI application broadcasts the prediction result (along with the original traffic data) over a WebSocket connection.

4.  **Real-time Dashboard (React)**:
    *   A modern web frontend developed with React (`dashboard/`) serves as the user interface.
    *   The dashboard establishes a WebSocket connection to the FastAPI application.
    *   It continuously receives real-time classification results.
    *   The UI displays these results in a live-updating log or table, visually highlighting traffic identified as malicious.

## Technologies Used

*   **Backend**: Python, FastAPI, `kafka-python` (for Kafka integration), LightGBM and Scikit-learn (for machine learning models), PyTorch/Transformers (potentially for experimental deep learning models).
*   **Frontend**: React (with Vite for fast development), Tailwind CSS (for styling).
*   **Infrastructure**: Apache Kafka (as the distributed streaming platform).
*   **Model Management**: `pickle` for serializing and deserializing ML models and preprocessing components.

## How to Use/Run the Project

To get the AI Threat Detector running, you'll need to have Python and Node.js installed, along with a running Apache Kafka instance.

### Prerequisites:

*   **Python 3.x**: Ensure Python and `pip` are installed.
*   **Node.js & npm**: For the frontend dashboard.
*   **Apache Kafka**: A running Kafka broker. You can set it up locally or use a cloud service. Ensure the `network-traffic` topic exists, or configure your Kafka instance to auto-create topics.

### Setup and Running Instructions:

1.  **Clone the Repository**:
    ```bash
    # Assuming you are in your desired parent directory
    git clone <repository_url>
    cd ai-threat-detector
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train a Model (Optional, but recommended for fresh setup)**:
    If you don't have a `model.pkl` in `models/final/`, you'll need to train one.
    ```bash
    # This script will train a LightGBM model and save it to models/final/model.pkl
    python train_lightgbm.py
    ```
    Ensure that `src/inference/predict.py` correctly points to your trained model, scaler, and label encoder paths. These are typically saved alongside the model during training.

4.  **Start the FastAPI Prediction Service**:
    Navigate to the project root and run the FastAPI application using Uvicorn.
    ```bash
    # --reload is useful for development, remove in production
    uvicorn src.api.app:app --reload
    ```
    This will typically start the API on `http://127.0.0.1:8000`.

5.  **Start the Kafka Consumer (Streaming Bridge)**:
    In a new terminal, from the project root:
    ```bash
    python src/streaming/consumer.py
    ```
    This consumer will listen to Kafka and forward data to the FastAPI `/predict` endpoint.

6.  **Start the Kafka Producer (Data Simulator)**:
    In another new terminal, from the project root:
    ```bash
    python src/streaming/producer.py
    ```
    This will start generating simulated network traffic and sending it to Kafka. You should see output indicating data being sent.

7.  **Start the Frontend Dashboard**:
    In a new terminal, navigate to the `dashboard` directory:
    ```bash
    cd dashboard
    npm install # Install frontend dependencies (if you haven't already)
    npm run dev
    ```
    This will start the React development server, usually on `http://localhost:5173` or similar. Open your web browser to this address to view the live dashboard.

You should now see network traffic data being generated, processed, classified, and displayed in real-time on the dashboard.

## File Structure

```
/Users/ashwinjoshi/Projects/ai-threat-detector/
├───.gitattributes
├───.gitignore
├───analyze_preprocess.py
├───combine.py
├───create_holdout_split.py
├───datacheck.py
├───dl_preprocess.py
├───eda.py
├───evaluate_holdout.py
├───finalize_model.py
├───preprocess.py
├───README.md
├───requirements.txt
├───tempCodeRunnerFile.py
├───train_dnn.py
├───train_dnn2.py
├───train_lightgbm.py
├───train_lightgbm2.py
├───train_rf.py
├───train_test.py
├───validate_final_model.py
├───.git/...
├───.zencoder/
│   └───workflows/
├───.zenflow/
│   └───workflows/
├───dashboard/
│   ├───.gitignore
│   ├───eslint.config.js
│   ├───index.html
│   ├───package-lock.json
│   ├───package.json
│   ├───postcss.config.js
│   ├───README.md
│   ├───tailwind.config.js
│   ├───tailwindcss.js
│   ├───vite.config.js
│   ├───node_modules/...
│   ├───public/
│   └───src/
│       ├───App.css
│       ├───App.jsx
│       ├───index.css
│       ├───main.jsx
│       ├───NetSentryDashboard.tsx
│       └───assets/
├───data/
│   └───processed/
├───logs/
├───models/
│   ├───lightgbm_booster_20251112_174402.txt
│   ├───lightgbm_checkpoint_20251112_131404.txt
│   ├───lightgbm_checkpoint_20251112_141803.txt
│   ├───lightgbm_checkpoint_20251112_182516.txt
│   ├───metrics_20251112_183742.json
│   └───final/
├───notebooks/
├───results/
├───src/
│   ├───api/
│   │   ├───app.py
│   │   └───__pycache__/...
│   ├───dashboard/
│   ├───data/
│   ├───inference/
│   │   ├───predict.py
│   │   └───__pycache__/...
│   ├───models/
│   ├───streaming/
│   │   ├───consumer.py
│   │   └───producer.py
│   └───utils/
└───venv/
    ├───bin/...
    ├───include/...
    ├───lib/...
    └───share/...
