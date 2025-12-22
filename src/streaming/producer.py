import json
import time
import random
import numpy as np
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "network-traffic"

def get_dummy_packet():
    # 1. Generate the 70 ML Features (The model needs these)
    features = np.random.rand(70).tolist()
    
    # 2. Generate Readable Metadata (The Dashboard needs these)
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
    src_ip = f"192.168.1.{random.randint(2, 254)}"
    dst_ip = f"10.0.0.{random.randint(1, 20)}"
    
    return {
        "features": features,
        "metadata": {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": random.choice(protocols),
            "length": random.randint(64, 1500),
            "timestamp": time.strftime("%H:%M:%S")
        }
    }

def run_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"Producer started. Sending enriched packets to {TOPIC}...")

    try:
        while True:
            packet_data = get_dummy_packet()
            
            # Send the full object (Features + Metadata)
            producer.send(TOPIC, packet_data)
            producer.flush()
            
            print(f"[>] Sent {packet_data['metadata']['protocol']} packet from {packet_data['metadata']['src_ip']}")
            time.sleep(0.5) # Speed up slightly (2 packets per second)
            
    except KeyboardInterrupt:
        producer.close()

if __name__ == "__main__":
    run_producer()