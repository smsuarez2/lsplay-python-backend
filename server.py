
import base64
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import urllib.request
 
app = Flask(__name__)
CORS(app)
 
# ── Descargar modelo si no existe ──
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Modelo descargado")
 
# ── Configurar MediaPipe ──
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)
 
# ── Estado global ──
state = {
    "hand_visible":  False,
    "detected_sign": None,
    "confidence":    0,
    "sequence":      [],
    "samples":       {},
    "training":      False,
    "train_sign":    None,
    "train_count":   {},
}
 
SEQUENCE_LENGTH = 15
 
def extract_features(landmarks):
    xs = [l.x for l in landmarks]
    ys = [l.y for l in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max_x - min_x or 1
    ry = max_y - min_y or 1
    return [((l.x - min_x) / rx, (l.y - min_y) / ry) for l in landmarks]
 
def sequence_distance(a, b):
    if not a or not b:
        return float('inf')
    n = min(len(a), len(b))
    total = sum(
        sum((x - y) ** 2 for x, y in zip(a[i], b[i])) ** 0.5
        for i in range(n)
    )
    return total / n
 
def classify(sequence, samples):
    if not samples or not sequence:
        return None, 0
    best_label, best_dist = None, float('inf')
    for label, seqs in samples.items():
        for s in seqs:
            d = sequence_distance(sequence, s)
            if d < best_dist:
                best_dist = d
                best_label = label
    conf = max(0, min(100, int((1 - best_dist / 3) * 100)))
    return best_label, conf
 
# ── ENDPOINT PRINCIPAL: recibe frame del navegador ──
@app.route('/api/frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400
 
    # Decodificar imagen base64 del navegador
    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
 
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
 
    # Procesar con MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
 
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        features = extract_features(landmarks)
        state["hand_visible"] = True
        state["sequence"].append(features)
        if len(state["sequence"]) > SEQUENCE_LENGTH:
            state["sequence"].pop(0)
 
        # Guardar muestra si está entrenando
        if state["training"] and state["train_sign"] and len(state["sequence"]) == SEQUENCE_LENGTH:
            label = state["train_sign"]
            if label not in state["samples"]:
                state["samples"][label] = []
            state["samples"][label].append(list(state["sequence"]))
            state["train_count"][label] = state["train_count"].get(label, 0) + 1
            state["sequence"] = []
            print(f"✅ Muestra {state['train_count'][label]} para '{label}'")
 
        # Detectar seña si tiene muestras
        elif not state["training"] and state["samples"] and len(state["sequence"]) == SEQUENCE_LENGTH:
            label, conf = classify(state["sequence"], state["samples"])
            state["detected_sign"] = label
            state["confidence"] = conf
            state["sequence"] = []
    else:
        state["hand_visible"] = False
        state["sequence"] = []
 
    return jsonify({
        "hand_visible":  state["hand_visible"],
        "detected_sign": state["detected_sign"],
        "confidence":    state["confidence"],
        "train_count":   state["train_count"],
        "trained_signs": list(state["samples"].keys()),
    })
 
@app.route('/api/train/start/<sign>', methods=['POST'])
def start_training(sign):
    state["training"]   = True
    state["train_sign"] = sign
    state["sequence"]   = []
    return jsonify({"ok": True, "mensaje": f"Entrenando: {sign}"})
 
@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    state["training"]   = False
    state["train_sign"] = None
    return jsonify({"ok": True})
 
@app.route('/api/samples/clear', methods=['POST'])
def clear_samples():
    state["samples"]     = {}
    state["train_count"] = {}
    state["sequence"]    = []
    return jsonify({"ok": True, "mensaje": "Muestras borradas"})
 
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "hand_visible":  state["hand_visible"],
        "detected_sign": state["detected_sign"],
        "confidence":    state["confidence"],
        "train_count":   state["train_count"],
        "trained_signs": list(state["samples"].keys()),
        "training":      state["training"],
        "train_sign":    state["train_sign"],
    })
 
@app.route('/', methods=['GET'])
def index():
    return jsonify({"mensaje": "✅ LS Play Python Server - puerto 5000"})
 
if __name__ == '__main__':
    print("🐍 LS Play Python Server")
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Iniciando en el puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
 