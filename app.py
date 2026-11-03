import os
import gc
import cv2
import time
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# Paksa pembersihan memori RAM awal
gc.collect()

app = Flask(__name__)

# Load Model YOLOv8 secara aman (Fallback jika memori terbatas)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
yolo_model = None

try:
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        print("Model YOLOv8 berhasil dimuat.")
    else:
        print("File yolov8n.pt tidak ditemukan di root folder.")
except Exception as e:
    print(f"Gagal memuat YOLO model: {e}")

# Class ID YOLOv8 untuk kendaraan: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def generate_cctv_frames(stream_url):
    """Generator streaming CCTV / Video dengan resize frame hemat RAM"""
    # Menggunakan VideoCapture OpenCV
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = cap.read()
        if not success:
            # Reconnect otomatis jika stream lag / terputus
            time.sleep(1)
            cap.open(stream_url)
            continue

        # Resize frame ke 640x360 agar beban RAM Render tetap stabil (< 512MB)
        frame = cv2.resize(frame, (640, 360))

        # Deteksi objek menggunakan YOLO jika model siap
        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False, imgsz=320)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls in VEHICLE_CLASSES:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            label = f"{yolo_model.names[cls]} {conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception as e:
                pass

        # Encode ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint pemutar stream CCTV aman (HTTPS bypass)"""
    url = request.args.get('url', '')
    if not url:
        # Default CCTV Simpang 4 Rejowinangun Magelang
        url = "https://dev.cctv.magelangkota.go.id/memfs/342540b3-545c-41a4-b3a2-624f7d824cdc_output_0.m3u8"
    return Response(generate_cctv_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/lhr_pkji', methods=['POST'])
def hitung_lhr_pkji():
    """API Perhitungan PKJI 2023 & LOS Lalu Lintas"""
    try:
        data = request.json
        mc = float(data.get('motor', 0))
        lv = float(data.get('mobil', 0))
        hv = float(data.get('truk_bus', 0))
        kapasitas = float(data.get('kapasitas', 2900)) # Kapasitas dasar jalan

        # Ekivalensi Mobil Penumpang (EMP) PKJI
        q_smp = (mc * 0.5) + (lv * 1.0) + (hv * 1.3)
        ds = q_smp / kapasitas if kapasitas > 0 else 0

        # Penentuan Level of Service (LOS)
        if ds <= 0.20: los = "A (Sangat Lancar)"
        elif ds <= 0.45: los = "B (Lancar)"
        elif ds <= 0.75: los = "C (Stabil)"
        elif ds <= 0.85: los = "D (Mendekati Jenuh)"
        elif ds <= 1.00: los = "E (Jenuh/Macet)"
        else: los = "F (Macet Total / Antrean)"

        return jsonify({
            "status": "success",
            "q_smp": round(q_smp, 2),
            "ds": round(ds, 3),
            "los": los
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
