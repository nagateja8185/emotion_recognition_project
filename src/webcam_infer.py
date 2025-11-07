# src/multimodal_server.py
"""
Simple threaded HTTP server for image/text/multimodal emotion prediction.
Usage:
    python src/multimodal_server.py
Server listens on http://localhost:8000
"""
import os
import json
import base64
import cgi
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

# ---------- CONFIG: update if you moved models ----------------
BASE = r"C:\emotion_detection_project"
MODEL_IMG_PATH = os.path.join(BASE, "models", "image_emotion.h5")
MODEL_TEXT_PIPELINE = os.path.join(BASE, "models", "text_emotion", "pipeline.joblib")
# attempt to read class order from your training folder (alphabetical order -> ImageDataGenerator)
TRAIN_IMAGE_DIR = os.path.join(BASE, "data", "images", "fer2013", "train")
PORT = 8000
# ----------------------------------------------------------------

# Default fallback labels (if detection from folders fails)
FALLBACK_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

def load_image_labels():
    try:
        classes = [d for d in os.listdir(TRAIN_IMAGE_DIR) if os.path.isdir(os.path.join(TRAIN_IMAGE_DIR, d))]
        classes_sorted = sorted(classes)  # ImageDataGenerator uses sorted dir names -> mapping
        if len(classes_sorted) == 0:
            return FALLBACK_LABELS
        return classes_sorted
    except Exception:
        return FALLBACK_LABELS

def preprocess_image_bytes(image_bytes, target_size=(48,48)):
    # decode bytes -> cv2 image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        # fallback: center crop/resize full image
        h, w = gray.shape
        minside = min(h, w)
        cy, cx = h//2, w//2
        half = minside//2
        crop = gray[cy-half:cy+half, cx-half:cx+half]
        if crop.size == 0:
            crop = cv2.resize(gray, target_size)
        else:
            crop = cv2.resize(crop, target_size)
    else:
        # take largest face
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        x,y,w,h = faces[0]
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop, target_size)
    arr = crop.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1,H,W,1)
    return arr

class MMHandler(BaseHTTPRequestHandler):
    server_version = "MMEmotionServer/0.1"

    def _set_headers(self, code=200, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)
        # body optional for OPTIONS
        return

    def do_POST(self):
        try:
            if self.path == "/predict_image":
                self._handle_predict_image()
            elif self.path == "/predict_text":
                self._handle_predict_text()
            elif self.path == "/predict_multimodal":
                self._handle_predict_multimodal()
            else:
                self._set_headers(404)
                self.wfile.write(b'{"error":"not found"}')
        except Exception as e:
            self._set_headers(500)
            payload = {"error": str(e)}
            self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _handle_predict_image(self):
        # Expect multipart form with field 'file' (image blob)
        ctype, pdict = cgi.parse_header(self.headers.get('content-type', ''))
        if ctype.startswith("multipart"):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST'})
            fileitem = form["file"] if "file" in form else None
            if not fileitem:
                self._set_headers(400)
                self.wfile.write(b'{"error":"no file provided"}')
                return
            data = fileitem.file.read()
        else:
            # raw binary body (fallback)
            length = int(self.headers.get('content-length', 0))
            data = self.rfile.read(length)

        arr = preprocess_image_bytes(data)
        probs = self.server.img_model.predict(arr)[0].tolist()
        idx = int(np.argmax(probs))
        label = self.server.img_labels[idx] if idx < len(self.server.img_labels) else "unknown"
        resp = {"label": label, "probs": probs}
        self._set_headers(200)
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def _handle_predict_text(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode('utf-8'))
            text = data.get("text", "")
        except Exception:
            # fallback: try parse body as utf-8 string
            text = body.decode('utf-8')

        if not text:
            self._set_headers(400)
            self.wfile.write(b'{"error":"no text provided"}')
            return

        pred = self.server.text_pipe.predict([text])[0]
        probs = None
        if hasattr(self.server.text_pipe, "predict_proba"):
            try:
                probs = self.server.text_pipe.predict_proba([text])[0].tolist()
                classes = list(self.server.text_pipe.classes_)
            except Exception:
                probs = None
                classes = list(self.server.text_pipe.classes_)
        else:
            classes = list(self.server.text_pipe.classes_)

        resp = {"label": pred, "probs": probs, "classes": classes}
        self._set_headers(200)
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def _handle_predict_multimodal(self):
        # Accept multipart with 'file' (optional) and 'text' (optional)
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST'})
        text = form.getvalue("text", "")
        fileitem = form["file"] if "file" in form else None

        img_probs = None
        text_probs = None
        text_classes = None

        if fileitem:
            data = fileitem.file.read()
            arr = preprocess_image_bytes(data)
            img_probs = np.array(self.server.img_model.predict(arr)[0])

        if text:
            try:
                text_probs = np.array(self.server.text_pipe.predict_proba([text])[0])
                text_classes = list(self.server.text_pipe.classes_)
            except Exception:
                # fallback: one-hot for predicted class
                pred = self.server.text_pipe.predict([text])[0]
                text_classes = list(self.server.text_pipe.classes_)
                onehot = np.zeros(len(text_classes))
                onehot[text_classes.index(pred)] = 1.0
                text_probs = onehot

        # Fusion: need label alignment. We'll try to align by label names if possible.
        if img_probs is not None and text_probs is not None:
            img_classes = self.server.img_labels
            # map text_probs into image label order (if possible)
            mapped_text_probs = np.zeros_like(img_probs, dtype=float)
            for i, lbl in enumerate(img_classes):
                if lbl in text_classes:
                    mapped_text_probs[i] = text_probs[text_classes.index(lbl)]
                else:
                    # if text class missing, leave as 0
                    mapped_text_probs[i] = 0.0
            # if mapped_text_probs sums to 0 (no overlap), fall back to trimmed/averaged by rounding sizes
            if mapped_text_probs.sum() == 0:
                # fallback: if sizes equal assume same order, else pad/truncate
                minlen = min(len(img_probs), len(text_probs))
                mapped_text_probs[:minlen] = text_probs[:minlen]
            fused = (img_probs + mapped_text_probs) / 2.0
            idx = int(np.argmax(fused))
            resp = {
                "label": self.server.img_labels[idx] if idx < len(self.server.img_labels) else "unknown",
                "probs_image": img_probs.tolist(),
                "probs_text_mapped": mapped_text_probs.tolist(),
                "probs_fused": fused.tolist()
            }
        elif img_probs is not None:
            idx = int(np.argmax(img_probs))
            resp = {"label": self.server.img_labels[idx], "probs_image": img_probs.tolist()}
        elif text_probs is not None:
            idx = int(np.argmax(text_probs))
            resp = {"label": text_classes[idx], "probs_text": text_probs.tolist(), "text_classes": text_classes}
        else:
            resp = {"error": "no valid inputs provided"}

        self._set_headers(200)
        self.wfile.write(json.dumps(resp).encode("utf-8"))


def run(port=PORT):
    # Load models
    print("Loading image model from:", MODEL_IMG_PATH)
    img_model = load_model(MODEL_IMG_PATH)
    print("Loading text pipeline from:", MODEL_TEXT_PIPELINE)
    text_pipe = joblib.load(MODEL_TEXT_PIPELINE)
    img_labels = load_image_labels()
    print("Image labels:", img_labels)
    print("Text classes (example):", getattr(text_pipe, "classes_", "n/a"))

    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, MMHandler)
    # attach models to server object so handlers can use them
    httpd.img_model = img_model
    httpd.text_pipe = text_pipe
    httpd.img_labels = img_labels

    print(f"Serving on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server")
        httpd.server_close()

if __name__ == "__main__":
    run()
