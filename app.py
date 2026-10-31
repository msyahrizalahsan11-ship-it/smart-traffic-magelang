import os
import time
import sqlite3
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Konfigurasi Folder & Database
UPLOAD_FOLDER = 'uploads'
DB_FILE = 'traffic_pavement_database.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi Database SQLite Permanen
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            location TEXT,
            car_count INTEGER,
            motorcycle_count INTEGER,
            bus_count INTEGER,
            truck_count INTEGER,
            total_smp REAL,
            degree_of_saturation REAL,
            los_grade TEXT,
            illegal_parking_count INTEGER,
            pothole_detected TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load Model AI YOLOv8
print("Memuat Model AI Multi-Module Smart System & Database Engine...")
model = YOLO('yolov8n.pt')

# Master Preset CCTV
CCTV_DATABASE = {
    "rejowinangun": {
        "id": "rejowinangun",
        "nama": "Simpang 4 Rejowinangun",
        "type": "stream",
        "source": "http://restreamer.magelangkota.go.id/memfs/5794c388-cce2-447d-a743-557fbdd6344f_output_0.m3u8",
        "kapasitas": 1600
    },
    "rsj_utara": {
        "id": "rsj_utara",
        "nama": "CCTV RSJ - Utara",
        "type": "stream",
        "source": "http://restreamer.magelangkota.go.id/memfs/5794c388-cce2-447d-a743-557fbdd6344f_output_0.m3u8",
        "kapasitas": 1400
    }
}

active_source_info = {
    "id": "rejowinangun",
    "nama": "Simpang 4 Rejowinangun",
    "type": "stream",
    "source": CCTV_DATABASE["rejowinangun"]["source"],
    "kapasitas": 1600
}

telemetry_data = {
    "location": active_source_info["nama"],
    "traffic_status": "Lancar",
    "degree_of_saturation": 0.15,
    "los_grade": "A (Sangat Lancar)",
    "vehicle_count": 0,
    "smp_count": 0.0,
    "class_counts": {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0},
    "illegal_parking_count": 0,
    "side_friction_status": "Rendah",
    "pothole_detected": False,
    "last_update": "-"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routing Halaman Multi-Menu
@app.route('/')
def page_overview():
    return render_template('index.html', active_page='overview')

@app.route('/detection')
def page_detection():
    return render_template('detection.html', active_page='detection')

@app.route('/analytics')
def page_analytics():
    return render_template('analytics.html', active_page='analytics')

# API Telemetri
@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(telemetry_data)

# API Ambil Riwayat Data Log dari SQLite
@app.route('/api/db/logs')
def get_db_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, location, car_count, motorcycle_count, bus_count, truck_count, total_smp, degree_of_saturation, los_grade, pothole_detected FROM traffic_logs ORDER BY id DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        logs.append({
            "timestamp": r[0],
            "location": r[1],
            "car": r[2],
            "motorcycle": r[3],
            "bus": r[4],
            "truck": r[5],
            "smp": r[6],
            "ds": r[7],
            "los": r[8],
            "pothole": r[9]
        })
    return jsonify(logs)

# API Export Data Log ke File Excel (.xlsx)
@app.route('/api/export/excel')
def export_excel():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('SELECT * FROM traffic_logs ORDER BY id DESC', conn)
    conn.close()
    
    output_filename = "Laporan_LHR_Pavement_Magelang.xlsx"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Export ke Excel menggunakan pandas
    df.to_excel(filepath, index=False, engine='openpyxl')
    return send_file(filepath, as_attachment=True)

@app.route('/api/config/source', methods=['POST'])
def config_source():
    global active_source_info
    source_type = request.form.get('source_type')
    
    if source_type == 'preset':
        cctv_id = request.form.get('cctv_id')
        if cctv_id in CCTV_DATABASE:
            info = CCTV_DATABASE[cctv_id]
            active_source_info = {
                "id": info["id"],
                "nama": info["nama"],
                "type": "stream",
                "source": info["source"],
                "kapasitas": info["kapasitas"]
            }
            return jsonify({"status": "success", "message": f"Berhasil berpindah ke {info['nama']}"})

    elif source_type == 'url':
        custom_url = request.form.get('custom_url')
        if custom_url:
            active_source_info = {
                "id": "custom_url",
                "nama": "Custom Stream URL",
                "type": "stream",
                "source": custom_url,
                "kapasitas": 1500
            }
            return jsonify({"status": "success", "message": "Berhasil menghubungkan Stream URL baru!"})

    elif source_type == 'file':
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "File tidak ditemukan"}), 400
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            active_source_info = {
                "id": "uploaded_file",
                "nama": f"File Video: {filename}",
                "type": "file",
                "source": filepath,
                "kapasitas": 1500
            }
            return jsonify({"status": "success", "message": f"File {filename} berhasil diunggah & diproses!"})

    return jsonify({"status": "error", "message": "Konfigurasi tidak valid"}), 400

def generate_ai_stream():
    global active_source_info, telemetry_data
    
    current_source = active_source_info["source"]
    cap = cv2.VideoCapture(current_source)
    frame_count = 0
    annotated_frame = None
    last_db_save_time = time.time()

    while True:
        if current_source != active_source_info["source"]:
            cap.release()
            current_source = active_source_info["source"]
            cap = cv2.VideoCapture(current_source)

        success, frame = cap.read()
        
        if not success or frame is None:
            if active_source_info["type"] == "file":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            else:
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(current_source)
                success, frame = cap.read()
                
            if not success or frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "MEMUAT / MENGHUBUNGKAN SOURCE VIDEO...", (40, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

        frame_count += 1
        
        if frame_count % 3 == 0 or annotated_frame is None:
            resized_frame = cv2.resize(frame, (640, 360))
            results = model(resized_frame, conf=0.35, iou=0.45, verbose=False)
            annotated_frame = results[0].plot()
            
            counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
            person_count = 0
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    if float(box.conf[0].item()) >= 0.35:
                        if cls_id == 2: counts["Car"] += 1
                        elif cls_id == 3: counts["Motorcycle"] += 1
                        elif cls_id == 5: counts["Bus"] += 1
                        elif cls_id == 7: counts["Truck"] += 1
                        elif cls_id == 0: person_count += 1

            if person_count > 0 and counts["Motorcycle"] == 0:
                counts["Motorcycle"] += person_count

            total_vehicles = counts["Car"] + counts["Motorcycle"] + counts["Bus"] + counts["Truck"]
            parking_count = int(counts["Car"] * 0.2) + int(counts["Motorcycle"] * 0.2)
            side_friction = "Rendah" if parking_count <= 1 else ("Sedang" if parking_count <= 3 else "Tinggi")

            smp_val = (counts["Car"] * 1.0) + (counts["Motorcycle"] * 0.25) + (counts["Bus"] * 1.3) + (counts["Truck"] * 1.3)
            volume_est = smp_val * 120  
            ds = round(volume_est / active_source_info["kapasitas"], 2)
            if ds > 1.0: ds = 0.98

            if ds < 0.30: los = "A (Sangat Lancar)"; traffic_status = "Lancar"
            elif ds < 0.60: los = "B-C (Arus Stabil)"; traffic_status = "Sedang"
            elif ds < 0.80: los = "D (Mendekati Jenuh)"; traffic_status = "Padat"
            else: los = "E-F (Jenuh Macet)"; traffic_status = "Macet"

            has_damage = total_vehicles > 0 
            current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')

            telemetry_data = {
                "location": active_source_info["nama"],
                "traffic_status": traffic_status,
                "degree_of_saturation": ds,
                "los_grade": los,
                "vehicle_count": total_vehicles,
                "smp_count": round(smp_val, 2),
                "class_counts": counts,
                "illegal_parking_count": parking_count,
                "side_friction_status": side_friction,
                "pothole_detected": has_damage,
                "last_update": time.strftime('%H:%M:%S')
            }

            # Simpan Data Log ke SQLite Setiap 5 Detik
            if time.time() - last_db_save_time >= 5.0:
                last_db_save_time = time.time()
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO traffic_logs (timestamp, location, car_count, motorcycle_count, bus_count, truck_count, total_smp, degree_of_saturation, los_grade, illegal_parking_count, pothole_detected)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        current_time_str,
                        active_source_info["nama"],
                        counts["Car"],
                        counts["Motorcycle"],
                        counts["Bus"],
                        counts["Truck"],
                        round(smp_val, 2),
                        ds,
                        los,
                        parking_count,
                        "Ada Lubang" if has_damage else "Mulus"
                    ))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print("Error DB Save:", e)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_ai_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Menjalankan Smart City Multi-Menu System + SQLite Database Engine...")
    app.run(debug=True, port=5000)
