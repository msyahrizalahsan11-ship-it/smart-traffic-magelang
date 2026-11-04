import os
import gc
import cv2
import time
from flask import Flask, render_template, Response, request, jsonify

# Bersihkan sisa memori RAM
gc.collect()

app = Flask(__name__)

# Load YOLO secara opsional / aman
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
yolo_model = None

try:
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        print("Model YOLOv8 berhasil dimuat.")
except Exception as e:
    print(f"Bypass YOLO load: {e}")

VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck

def generate_frames(stream_url):
    """Generator stream yang aman dari timeout OpenCV"""
    # Beri opsi open via backend FFMPEG
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fail_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            fail_count += 1
            time.sleep(0.5)
            # Jika gagal 5x berturut-turut, coba hubungi ulang stream
            if fail_count > 5:
                cap.open(stream_url, cv2.CAP_FFMPEG)
                fail_count = 0
            continue
        
        fail_count = 0
        
        # Resize frame agar hemat RAM Render (512MB limit)
        frame = cv2.resize(frame, (640, 360))

        # Deteksi YOLO jika model tersedia
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
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            except Exception:
                pass

        # Compress ke JPEG hemat resource
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    url = request.args.get('url', '')
    if not url:
        url = "https://dev.cctv.magelangkota.go.id/memfs/342540b3-545c-41a4-b3a2-624f7d824cdc_output_0.m3u8"
    return Response(generate_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint API Pendukung Fitur Dashboard
@app.route('/api/pothole_counts')
def pothole_counts():
    return jsonify({"potholes": 4, "cracks": 2, "status": "Waspada"})

@app.route('/api/parking_counts')
def parking_counts():
    return jsonify({"total_slots": 20, "occupied": 14, "available": 6})

@app.route('/api/traffic_counts')
def traffic_counts():
    return jsonify({"motorcycle": 128, "car": 42, "bus": 3, "truck": 5})

@app.route('/api/lhr_pkji', methods=['POST'])
def hitung_lhr_pkji():
    try:
        data = request.json
        mc = float(data.get('motor', 0))
        lv = float(data.get('mobil', 0))
        hv = float(data.get('truk_bus', 0))
        kapasitas = float(data.get('kapasitas', 2900))

        q_smp = (mc * 0.5) + (lv * 1.0) + (hv * 1.3)
        ds = q_smp / kapasitas if kapasitas > 0 else 0

        if ds <= 0.20: los = "A (Sangat Lancar)"
        elif ds <= 0.45: los = "B (Lancar)"
        elif ds <= 0.75: los = "C (Stabil)"
        elif ds <= 0.85: los = "D (Mendekati Jenuh)"
        elif ds <= 1.00: los = "E (Jenuh/Macet)"
        else: los = "F (Macet Total)"

        return jsonify({"status": "success", "q_smp": round(q_smp, 2), "ds": round(ds, 3), "los": los})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
