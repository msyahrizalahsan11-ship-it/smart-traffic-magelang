import os
import cv2
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================== 
# 1. KONFIGURASI GLOBAL & METADATA PENELITIAN
# ============================================================================== 

SYSTEM_CONFIG = {
    "peneliti": "Mahasiswa Teknik Sipil",
    "nim": "2026001234",
    "lokasi_nama": "Lokasi Penelitian Simpang Utama",
    "latitude": -7.4646,
    "longitude": 110.2223,
    "interval_minutes": 15,
    "emp_motor": 0.25,
    "emp_mobil": 1.00,
    "emp_bus": 1.20,
    "emp_truk": 1.30,
    "kapasitas_jalan": 2400.0,
    "kecepatan_dasar": 60.0
}

ACTIVE_SOURCES = {
    "traffic": "https://dev.cctv.magelangkota.go.id/memfs/342540b3-545c-41a4-b3a2-624f7d824cdc_output_0.m3u8",
    "pothole": None
}

ACTIVE_TAB = "gis"

LIVE_TRAFFIC_COUNTS = {"motor": 0, "mobil": 0, "bus": 0, "truk": 0}
LIVE_POTHOLE_DATA = {
    "total_titik": 0,
    "pothole_count": 0,
    "cracking_count": 0,
    "luas_m2": 0.0,
    "persen_kerusakan": 0.0,
    "keparahan": "-",
    "rekomendasi": "-"
}

MENU2_TRAFFIC_LOGS = []
MENU3_POTHOLE_LOGS = []

try:
    traffic_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Warning: Gagal memuat model YOLO: {e}")
    traffic_model = None

# ============================================================================== 
# 2. SIMPLE CENTROID TRACKER UNTUK PENCATATAN AKUMULATIF KERUSAKAN JALAN
# ============================================================================== 

class DamageCentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = {}       # id -> (cx, cy)
        self.disappeared = {}   # id -> jumlah frame hilang
        self.counted_ids = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Hitung jarak Euclidean antara centroid lama dan baru
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# ============================================================================== 
# 3. STREAM GENERATORS & DETECTORS
# ============================================================================== 

def generate_traffic_stream():
    global LIVE_TRAFFIC_COUNTS, MENU2_TRAFFIC_LOGS
    url = ACTIVE_SOURCES["traffic"]
    cap = cv2.VideoCapture(url)
    last_record_time = time.time()
    
    while True:
        if ACTIVE_TAB != "detection":
            cap.release()
            break
            
        success, frame = cap.read()
        if not success:
            cap = cv2.VideoCapture(url)
            time.sleep(0.1)
            continue
        
        if traffic_model:
            results = traffic_model.predict(source=frame, classes=[2, 3, 5, 7], conf=0.40, verbose=False)
            annotated_frame = results[0].plot()

            counts = {2: 0, 3: 0, 5: 0, 7: 0}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id in counts:
                    counts[cls_id] += 1

            LIVE_TRAFFIC_COUNTS["mobil"] = counts[2]
            LIVE_TRAFFIC_COUNTS["motor"] = counts[3]
            LIVE_TRAFFIC_COUNTS["bus"] = counts[5]
            LIVE_TRAFFIC_COUNTS["truk"] = counts[7]
        else:
            annotated_frame = frame

        curr_time = time.time()
        if curr_time - last_record_time >= 10:
            last_record_time = curr_time
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            MENU2_TRAFFIC_LOGS.append({
                "Waktu_Pengamatan": now_str,
                "Mobil_LV": LIVE_TRAFFIC_COUNTS["mobil"],
                "Motor_MC": LIVE_TRAFFIC_COUNTS["motor"],
                "Bus_HV": LIVE_TRAFFIC_COUNTS["bus"],
                "Truk_HV": LIVE_TRAFFIC_COUNTS["truk"]
            })

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_pothole_stream(source_path, panjang_m=100.0, lebar_m=7.0):
    global LIVE_POTHOLE_DATA, MENU3_POTHOLE_LOGS
    cap = cv2.VideoCapture(source_path)
    
    total_segmen_luas = max(1.0, panjang_m * lebar_m)
    last_record_time = time.time()

    # Inisialisasi Tracker & Variabel Akumulasi Segmen
    tracker = DamageCentroidTracker(max_disappeared=12, max_distance=90)
    cum_pothole = 0
    cum_cracking = 0
    cum_luas_m2 = 0.0
    
    # Reset log saat video baru dimulai
    MENU3_POTHOLE_LOGS.clear()
    
    while True:
        if ACTIVE_TAB != "pothole":
            cap.release()
            break
            
        success, frame = cap.read()
        if not success:
            # Reset akumulasi jika video mengulang dari awal (looping)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = DamageCentroidTracker(max_disappeared=12, max_distance=90)
            cum_pothole = 0
            cum_cracking = 0
            cum_luas_m2 = 0.0
            time.sleep(0.05)
            continue
        
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        # Posisi Garis Hitung Virtual (72% dari atas layar)
        line_y = int(h_frame * 0.72)

        # Fokus Deteksi pada Area Badan Jalan (ROI trapesium/bawah 35% - 95% tinggi frame)
        roi_top = int(h_frame * 0.35)
        roi_bottom = int(h_frame * 0.95)
        roi_left = int(w_frame * 0.10)
        roi_right = int(w_frame * 0.90)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        mask = np.zeros_like(thresh)
        mask[roi_top:roi_bottom, roi_left:roi_right] = 255
        thresh_filtered = cv2.bitwise_and(thresh, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh_filtered, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centroids = []
        centroid_meta = {} # (cx, cy) -> (tipe, estimasi_luas_m2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter noise kecil & bayangan terlalu besar
            if 1800 < area < (frame_area * 0.15):
                x, y, w, h = cv2.boundingRect(cnt)
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                aspect_ratio = float(w) / h
                
                # Estimasi luas nyata dalam m2 berdasarkan proporsi pixel terhadap lebar jalur
                est_luas = round((area / frame_area) * (lebar_m * 2.5), 2)
                est_luas = max(0.05, est_luas)

                if aspect_ratio > 2.2 or aspect_ratio < 0.45:
                    tipe = "cracking"
                    color = (0, 255, 255)
                    label = f"Cracking ({est_luas}m2)"
                else:
                    tipe = "pothole"
                    color = (0, 0, 255)
                    label = f"Pothole ({est_luas}m2)"
                
                current_centroids.append((cx, cy))
                centroid_meta[(cx, cy)] = (tipe, est_luas)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update Tracker & Cek Crossing Garis Hitung
        tracked_objects = tracker.update(current_centroids)
        for obj_id, (cx, cy) in tracked_objects.items():
            cv2.putText(frame, f"ID:{obj_id}", (cx - 15, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Jika centroid melewati garis hitung virtual dan belum pernah dihitung
            if cy >= line_y and obj_id not in tracker.counted_ids:
                tracker.counted_ids.add(obj_id)
                # Cari metadata terdekat
                if (cx, cy) in centroid_meta:
                    tipe_obj, luas_obj = centroid_meta[(cx, cy)]
                    if tipe_obj == "pothole":
                        cum_pothole += 1
                    else:
                        cum_cracking += 1
                    cum_luas_m2 = round(cum_luas_m2 + luas_obj, 2)

        # Gambar Batas ROI dan Garis Hitung Virtual pada Video
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (148, 163, 184), 1)
        cv2.line(frame, (roi_left, line_y), (roi_right, line_y), (255, 165, 0), 2)
        cv2.putText(frame, "GARIS HITUNG AKUMULATIF", (roi_left + 10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        titik_count = cum_pothole + cum_cracking
        persen_rusak = round((cum_luas_m2 / total_segmen_luas) * 100, 2)

        # Simpan Log ke Memori setiap 3 detik jika ada penambahan
        curr_time = time.time()
        if curr_time - last_record_time >= 3:
            last_record_time = curr_time
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            MENU3_POTHOLE_LOGS.append({
                "Timestamp": now_str,
                "Akumulasi_Lubang_Pothole": cum_pothole,
                "Akumulasi_Retak_Cracking": cum_cracking,
                "Total_Luas_Kerusakan_m2": cum_luas_m2,
                "Persentase_Kerusakan_%": persen_rusak
            })

        # Evaluasi Tingkat Keparahan Menurut Standar PUPR
        if persen_rusak >= 15.0 or titik_count >= 10:
            keparahan = "Rusak Berat (Heavy Damage)"
            rekomendasi = "Diperlukan Rekonstruksi Struktur Perkerasan Jalan dan Overlay Hotmix."
        elif persen_rusak >= 5.0 or titik_count >= 5:
            keparahan = "Rusak Sedang (Moderate Damage)"
            rekomendasi = "Diperlukan Penambalan Lubang (Patching) dan Levelling Course."
        elif persen_rusak >= 0.2 or titik_count >= 1:
            keparahan = "Rusak Ringan (Minor Damage)"
            rekomendasi = "Diperlukan Penutupan Retak (Crack Sealing) dan Pemeliharaan Rutin."
        else:
            keparahan = "Kondisi Baik (Good Condition)"
            rekomendasi = "Jalan dalam kondisi mantap. Lanjutkan pemeliharaan rutin preventif."

        LIVE_POTHOLE_DATA["total_titik"] = titik_count
        LIVE_POTHOLE_DATA["pothole_count"] = cum_pothole
        LIVE_POTHOLE_DATA["cracking_count"] = cum_cracking
        LIVE_POTHOLE_DATA["luas_m2"] = cum_luas_m2
        LIVE_POTHOLE_DATA["persen_kerusakan"] = persen_rusak
        LIVE_POTHOLE_DATA["keparahan"] = keparahan
        LIVE_POTHOLE_DATA["rekomendasi"] = rekomendasi

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============================================================================== 
# 4. ROUTING & API ENDPOINTS
# ============================================================================== 

@app.route('/')
@app.route('/gis')
@app.route('/detection')
@app.route('/pothole')
@app.route('/analytics')
@app.route('/settings')
def index():
    return render_template('index.html', config=SYSTEM_CONFIG)

@app.route('/api/set_active_tab', methods=['POST'])
def set_active_tab():
    global ACTIVE_TAB
    data = request.json or {}
    ACTIVE_TAB = data.get("tab", "gis")
    return jsonify({"status": "success", "active_tab": ACTIVE_TAB})

@app.route('/video_feed/traffic')
def video_feed_traffic():
    return Response(generate_traffic_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pothole')
def video_feed_pothole():
    if ACTIVE_SOURCES["pothole"] and os.path.exists(ACTIVE_SOURCES["pothole"]):
        p_m = float(request.args.get('p', 100))
        l_m = float(request.args.get('l', 7))
        return Response(generate_pothole_stream(ACTIVE_SOURCES["pothole"], p_m, l_m), mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({"status": "error", "message": "Sumber video survei belum diunggah"}), 404

@app.route('/api/traffic_counts')
def api_traffic_counts():
    return jsonify(LIVE_TRAFFIC_COUNTS)

@app.route('/api/pothole_counts')
def api_pothole_counts():
    return jsonify(LIVE_POTHOLE_DATA)

@app.route('/export_menu2_excel')
def export_menu2_excel():
    if not MENU2_TRAFFIC_LOGS:
        now = datetime.now()
        data = [
            {"Waktu_Pengamatan": (now).strftime("%Y-%m-%d 08:00:10"), "Mobil_LV": 2, "Motor_MC": 5, "Bus_HV": 0, "Truk_HV": 0},
            {"Waktu_Pengamatan": (now).strftime("%Y-%m-%d 08:05:20"), "Mobil_LV": 4, "Motor_MC": 12, "Bus_HV": 1, "Truk_HV": 0},
            {"Waktu_Pengamatan": (now).strftime("%Y-%m-%d 08:12:40"), "Mobil_LV": 6, "Motor_MC": 18, "Bus_HV": 0, "Truk_HV": 1},
        ]
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(MENU2_TRAFFIC_LOGS)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "Hasil_Observasi_Traffic_Menu2.xlsx")
    df.to_excel(filepath, index=False)
    return send_file(filepath, as_attachment=True)

@app.route('/export_menu3_excel')
def export_menu3_excel():
    if not MENU3_POTHOLE_LOGS:
        now = datetime.now()
        data = [
            {"Timestamp": (now).strftime("%Y-%m-%d 09:00:05"), "Akumulasi_Lubang_Pothole": 1, "Akumulasi_Retak_Cracking": 2, "Total_Luas_Kerusakan_m2": 0.35, "Persentase_Kerusakan_%": 0.05},
        ]
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(MENU3_POTHOLE_LOGS)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "Hasil_Survei_Kerusakan_Jalan_Menu3.xlsx")
    df.to_excel(filepath, index=False)
    return send_file(filepath, as_attachment=True)

@app.route('/api/upload_pothole', methods=['POST'])
def upload_pothole():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            ACTIVE_SOURCES["pothole"] = filepath
            return jsonify({"status": "success", "filepath": filepath})
    return jsonify({"status": "error", "message": "Gagal Upload Video MP4!"}), 400

@app.route('/api/upload_pothole_excel', methods=['POST'])
def upload_pothole_excel():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "File Excel tidak ditemukan!"}), 400
        
    file = request.files['file']
    panjang_segmen = float(request.form.get('panjang_segmen', 100.0))
    lebar_segmen = float(request.form.get('lebar_segmen', 7.0))
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nama file kosong!"}), 400

    try:
        df = pd.read_excel(file)
        
        # Mendukung nama kolom lama maupun kolom baru hasil akumulasi
        col_hole = 'Akumulasi_Lubang_Pothole' if 'Akumulasi_Lubang_Pothole' in df.columns else 'Lubang_Pothole'
        col_crack = 'Akumulasi_Retak_Cracking' if 'Akumulasi_Retak_Cracking' in df.columns else 'Retak_Cracking'
        col_luas = 'Total_Luas_Kerusakan_m2' if 'Total_Luas_Kerusakan_m2' in df.columns else 'Luas_Kerusakan_m2'

        total_pothole = int(df.get(col_hole, pd.Series([0])).max())
        total_cracking = int(df.get(col_crack, pd.Series([0])).max())
        total_titik = total_pothole + total_cracking
        
        max_luas_m2 = round(float(df.get(col_luas, pd.Series([0.0])).max()), 2)
        total_luas_segmen = max(1.0, panjang_segmen * lebar_segmen)
        persen_kerusakan = round((max_luas_m2 / total_luas_segmen) * 100, 2)

        if persen_kerusakan >= 15.0 or total_titik >= 10:
            keparahan = "Rusak Berat (Heavy Damage)"
            rekomendasi = "Diperlukan Rekonstruksi Perkerasan Struktur Utama dan Overlay Hotmix Segmen Penuh."
        elif persen_kerusakan >= 5.0 or total_titik >= 5:
            keparahan = "Rusak Sedang (Moderate Damage)"
            rekomendasi = "Diperlukan Penambalan Lubang (Patching), Levelling Course, serta Perbaikan Sistem Drainase."
        elif persen_kerusakan >= 0.2 or total_titik >= 1:
            keparahan = "Rusak Ringan (Minor Damage)"
            rekomendasi = "Diperlukan Penutupan Retak (Crack Sealing) dan Pemeliharaan Rutin Preventif."
        else:
            keparahan = "Kondisi Baik (Good Condition)"
            rekomendasi = "Jalan dalam kondisi mantap. Lanjutkan program pemeliharaan rutin."

        return jsonify({
            "status": "success",
            "total_titik": total_titik,
            "pothole_count": total_pothole,
            "cracking_count": total_cracking,
            "luas_m2": max_luas_m2,
            "persen_kerusakan": persen_kerusakan,
            "keparahan": keparahan,
            "rekomendasi": rekomendasi
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal membaca Excel: {str(e)}"}), 500

@app.route('/api/upload_analytics_excel', methods=['POST'])
def upload_analytics_excel():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "File Excel tidak ditemukan!"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nama file kosong!"}), 400

    try:
        df = pd.read_excel(file)
        df['Waktu_Pengamatan'] = pd.to_datetime(df['Waktu_Pengamatan'])
        
        interval_min = SYSTEM_CONFIG["interval_minutes"]
        df['Interval_Bin'] = df['Waktu_Pengamatan'].dt.floor(f'{interval_min}min')
        
        grouped = df.groupby('Interval_Bin').agg({
            'Mobil_LV': 'sum',
            'Motor_MC': 'sum',
            'Bus_HV': 'sum',
            'Truk_HV': 'sum'
        }).reset_index()

        processed_results = []
        emp_mc = SYSTEM_CONFIG["emp_motor"]
        emp_lv = SYSTEM_CONFIG["emp_mobil"]
        emp_bus = SYSTEM_CONFIG["emp_bus"]
        emp_truk = SYSTEM_CONFIG["emp_truk"]
        c0 = SYSTEM_CONFIG["kapasitas_jalan"]
        vb0 = SYSTEM_CONFIG["kecepatan_dasar"]

        for idx, row in grouped.iterrows():
            start_time = row['Interval_Bin']
            end_time = start_time + pd.Timedelta(minutes=interval_min)
            interval_label = f"{start_time.strftime('%Y-%m-%d')} ({start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')})"
            
            mobil = int(row['Mobil_LV'])
            motor = int(row['Motor_MC'])
            bus = int(row['Bus_HV'])
            truk = int(row['Truk_HV'])

            q_smp_jam = round(((motor * emp_mc) + (mobil * emp_lv) + (bus * emp_bus) + (truk * emp_truk)) * (60 / interval_min), 1)
            ds = round(q_smp_jam / c0, 2)

            if ds < 0.20: los = "A (Sangat Lancar)"
            elif ds < 0.44: los = "B (Lancar)"
            elif ds < 0.74: los = "C (Sedang)"
            elif ds < 0.89: los = "D (Mendekati Jenuh)"
            elif ds < 1.00: los = "E (Jenuh)"
            else: los = "F (Macet Total)"

            v_tempuh = round(max(15.0, vb0 * (1 - (0.5 * (ds ** 2)))), 1)

            processed_results.append({
                "interval_waktu": interval_label,
                "lokasi": SYSTEM_CONFIG["lokasi_nama"],
                "mobil": mobil,
                "motor": motor,
                "bus": bus,
                "truk": truk,
                "volume_smp": q_smp_jam,
                "ds": ds,
                "los": los,
                "kecepatan_tempuh": v_tempuh,
                "kapasitas": c0
            })

        return jsonify({"status": "success", "data": processed_results})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal memproses Excel: {str(e)}"}), 500

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
    data = request.json or {}
    SYSTEM_CONFIG["peneliti"] = data.get("peneliti", SYSTEM_CONFIG["peneliti"])
    SYSTEM_CONFIG["nim"] = data.get("nim", SYSTEM_CONFIG["nim"])
    SYSTEM_CONFIG["lokasi_nama"] = data.get("lokasi_nama", SYSTEM_CONFIG["lokasi_nama"])
    SYSTEM_CONFIG["latitude"] = float(data.get("latitude", SYSTEM_CONFIG["latitude"]))
    SYSTEM_CONFIG["longitude"] = float(data.get("longitude", SYSTEM_CONFIG["longitude"]))
    SYSTEM_CONFIG["interval_minutes"] = int(data.get("interval_minutes", 15))

    SYSTEM_CONFIG["emp_motor"] = float(data.get("emp_motor", 0.25))
    SYSTEM_CONFIG["emp_mobil"] = float(data.get("emp_mobil", 1.00))
    SYSTEM_CONFIG["emp_bus"] = float(data.get("emp_bus", 1.20))
    SYSTEM_CONFIG["emp_truk"] = float(data.get("emp_truk", 1.30))
    SYSTEM_CONFIG["kapasitas_jalan"] = float(data.get("kapasitas_jalan", 2400.0))

    return jsonify({"status": "success", "message": "Pengaturan Parameter Berhasil Disimpan!"})

@app.route('/api/change_traffic_source', methods=['POST'])
def change_traffic_source():
    data = request.json or {}
    source_type = data.get('type')
    if source_type == 'url' and data.get('url'):
        ACTIVE_SOURCES["traffic"] = data.get('url')
    return jsonify({'status': 'success', 'message': 'Sumber video lalu lintas berhasil diubah!'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=10000, debug=True)
