import json
import requests
from kafka import KafkaConsumer

TOPIC = "network-traffic"
API_URL = "http://127.0.0.1:8000/predict"

def run_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )

    print(f"Consumer listening on {TOPIC}...")

    for message in consumer:
        payload = message.value
        
        try:
            # Forward to Inference API
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                pred = response.json()
                # UPDATED KEY: 'prediction' instead of 'class_id'
                print(f"[<] Prediction: {pred['prediction']} | Status: {response.status_code}")
            else:
                print(f"[!] API Error: {response.text}")

        except Exception as e:
            print(f"[!] Connection Error: {e}")

if __name__ == "__main__":
    run_consumer()