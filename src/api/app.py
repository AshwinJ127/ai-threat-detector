import sys
import os
import json
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.predict import predict

app = FastAPI()

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- UPDATED Data Model ---
class TrafficPayload(BaseModel):
    features: List[float]
    metadata: Dict[str, Any]  # This field captures the IPs and Protocols

# --- Endpoints ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() 
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/predict")
async def predict_traffic(payload: TrafficPayload):
    if len(payload.features) != 70:
        raise HTTPException(status_code=400, detail="Shape mismatch")

    try:
        # 1. Run the ML Prediction
        vec = np.array(payload.features)
        pred_label = int(predict(vec))
        
        # 2. Prepare the Dashboard Update
        # We combine the Prediction result + the Packet Metadata
        response_data = {
            "prediction": pred_label, 
            "status": "processed",
            "packet_info": payload.metadata 
        }
        
        # 3. Send to Dashboard
        await manager.broadcast(response_data)
        
        return response_data

    except Exception as e:
        print(f"Error processing packet: {e}")
        raise HTTPException(status_code=500, detail=str(e))