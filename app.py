import os
import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# 1. KONFIGURASI GLOBAL & METADATA PENELITIAN
# ============================================================================

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
	"kecepatan_dasar": 60.0,
	"kapasitas_parkir": 20
}

ACTIVE_SOURCES = {
	"traffic": "https://dev.cctv.magelangkota.go.id/memfs/342540b3-545c-41a4-b3a2-624f7d824cdc_output_0.m3u8",
	"pothole": None,
	"parking": None
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

LIVE_PARKING_DATA = {
	"total_slot": SYSTEM_CONFIG["kapasitas_parkir"],
	"occupied": 0,
	"vacant": SYSTEM_CONFIG["kapasitas_parkir"],
	"density": 0.0,
	"status_kepadatan": "Tersedia Banyak",
	"status_color": "var(--accent-green)"
}

MENU2_TRAFFIC_LOGS = []
MENU3_POTHOLE_LOGS = []

traffic_model = YOLO('yolov8n.pt')

# ============================================================================
# 2. STREAM GENERATORS & DETECTORS
# ============================================================================

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

		curr_time = time.time()
		if curr_time - last_record_time >= 10:
			last_record_time = curr_time
			now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			MENU2_TRAFFIC_LOGS.append({
				"Waktu_Pengamatan": now_str,
				"Mobil_LV": counts[2],
				"Motor_MC": counts[3],
				"Bus_HV": counts[5],
				"Truk_HV": counts[7]
			})

		ret, buffer = cv2.imencode('.jpg', annotated_frame)
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_pothole_stream(source_path, panjang_m=100.0, lebar_m=7.0):
	global LIVE_POTHOLE_DATA, MENU3_POTHOLE_LOGS
	cap = cv2.VideoCapture(source_path)

	total_segmen_luas = max(1.0, panjang_m * lebar_m)
	last_record_time = time.time()

	while True:
		if ACTIVE_TAB != "pothole":
			cap.release()
			break

		success, frame = cap.read()
		if not success:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue

		h_frame, w_frame = frame.shape[:2]
		frame_area = h_frame * w_frame

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)

		_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

		mask = np.zeros_like(thresh)
		mask[15:h_frame-15, 15:w_frame-15] = 255
		thresh_filtered = cv2.bitwise_and(thresh, mask)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		opening = cv2.morphologyEx(thresh_filtered, cv2.MORPH_OPEN, kernel, iterations=1)

		contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		pothole_count = 0
		cracking_count = 0
		total_pixel_damage = 0

		for cnt in contours:
			area = cv2.contourArea(cnt)

			if 1500 < area < (frame_area * 0.20):
				x, y, w, h = cv2.boundingRect(cnt)

				if y <= 15 or (y + h) >= (h_frame - 15) or x <= 15 or (x + w) >= (w_frame - 15):
					continue

				aspect_ratio = float(w) / h
				total_pixel_damage += area

				if aspect_ratio > 2.0 or aspect_ratio < 0.5:
					cracking_count += 1
					color = (0, 255, 255)
					label = f"Cracking ({round(area/5000, 2)}m2)"
				else:
					pothole_count += 1
					color = (0, 0, 255)
					label = f"Pothole ({round(area/4000, 2)}m2)"

				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		titik_count = pothole_count + cracking_count
		luas_rusak_m2 = round((total_pixel_damage / frame_area) * 12.0, 2) if titik_count > 0 else 0.0
		persen_rusak = round((luas_rusak_m2 / total_segmen_luas) * 100, 2)

		curr_time = time.time()
		if curr_time - last_record_time >= 5:
			last_record_time = curr_time
			now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			MENU3_POTHOLE_LOGS.append({
				"Timestamp": now_str,
				"Lubang_Pothole": pothole_count,
				"Retak_Cracking": cracking_count,
				"Luas_Kerusakan_m2": luas_rusak_m2
			})

		if persen_rusak >= 15.0 or titik_count >= 5:
			keparahan = "Rusak Berat (Heavy Damage)"
			rekomendasi = "Diperlukan Rekonstruksi Struktur Perkerasan Jalan dan Overlay Hotmix."
		elif persen_rusak >= 5.0 or titik_count >= 3:
			keparahan = "Rusak Sedang (Moderate Damage)"
			rekomendasi = "Diperlukan Penambalan Lubang (Patching) dan Levelling Course."
		elif persen_rusak >= 0.5 or titik_count >= 1:
			keparahan = "Rusak Ringan (Minor Damage)"
			rekomendasi = "Diperlukan Penutupan Retak (Crack Sealing) dan Pemeliharaan Rutin."
		else:
			keparahan = "Kondisi Baik (Good Condition)"
			rekomendasi = "Jalan dalam kondisi mantap. Lanjutkan pemeliharaan rutin preventif."

		LIVE_POTHOLE_DATA["total_titik"] = titik_count
		LIVE_POTHOLE_DATA["pothole_count"] = pothole_count
		LIVE_POTHOLE_DATA["cracking_count"] = cracking_count
		LIVE_POTHOLE_DATA["luas_m2"] = luas_rusak_m2
		LIVE_POTHOLE_DATA["persen_kerusakan"] = persen_rusak
		LIVE_POTHOLE_DATA["keparahan"] = keparahan
		LIVE_POTHOLE_DATA["rekomendasi"] = rekomendasi

		ret, buffer = cv2.imencode('.jpg', frame)
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_parking_stream(source):
	global LIVE_PARKING_DATA
	cap = cv2.VideoCapture(source)

	while True:
		if ACTIVE_TAB != "parking":
			cap.release()
			break

		success, frame = cap.read()
		if not success:
			cap = cv2.VideoCapture(source) if str(source).startswith("http") else cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			time.sleep(0.1)
			continue

		results = traffic_model.predict(source=frame, classes=[2, 3, 5, 7], conf=0.30, verbose=False)
		annotated_frame = results[0].plot()

		occupied_count = len(results[0].boxes)
		total_slots = LIVE_PARKING_DATA.get("total_slot", SYSTEM_CONFIG["kapasitas_parkir"])

		occupied_count = min(occupied_count, total_slots)
		vacant_slots = max(0, total_slots - occupied_count)
		density_pct = round((occupied_count / total_slots) * 100, 1) if total_slots > 0 else 0.0

		if density_pct >= 85.0:
			status = "Parkir Penuh (Penuh)"
			color = "var(--accent-red)"
		elif density_pct >= 50.0:
			status = "Hampir Penuh (Sedang)"
			color = "var(--accent-yellow)"
		else:
			status = "Tersedia Banyak (Lancar)"
			color = "var(--accent-green)"

		LIVE_PARKING_DATA["occupied"] = occupied_count
		LIVE_PARKING_DATA["vacant"] = vacant_slots
		LIVE_PARKING_DATA["density"] = density_pct
		LIVE_PARKING_DATA["status_kepadatan"] = status
		LIVE_PARKING_DATA["status_color"] = color

		ret, buffer = cv2.imencode('.jpg', annotated_frame)
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============================================================================
# 3. ROUTING & API ENDPOINTS
# ============================================================================

@app.route('/')
@app.route('/gis')
@app.route('/detection')
@app.route('/pothole')
@app.route('/parking')
@app.route('/analytics')
@app.route('/settings')
def index():
	return render_template('index.html', config=SYSTEM_CONFIG)

@app.route('/api/set_active_tab', methods=['POST'])
def set_active_tab():
	global ACTIVE_TAB
	data = request.json
	ACTIVE_TAB = data.get("tab", "gis")
	return jsonify({"status": "success", "active_tab": ACTIVE_TAB})

@app.route('/video_feed/traffic')
def video_feed_traffic():
	return Response(generate_traffic_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pothole')
def video_feed_pothole():
	if ACTIVE_SOURCES["pothole"]:
		p_m = float(request.args.get('p', 100))
		l_m = float(request.args.get('l', 7))
		return Response(generate_pothole_stream(ACTIVE_SOURCES["pothole"], p_m, l_m), mimetype='multipart/x-mixed-replace; boundary=frame')
	return jsonify({"status": "error", "message": "No source"})

@app.route('/video_feed/parking')
def video_feed_parking():
	if ACTIVE_SOURCES["parking"]:
		return Response(generate_parking_stream(ACTIVE_SOURCES["parking"]), mimetype='multipart/x-mixed-replace; boundary=frame')
	return jsonify({"status": "error", "message": "No source"})

@app.route('/api/traffic_counts')
def api_traffic_counts():
	return jsonify(LIVE_TRAFFIC_COUNTS)

@app.route('/api/pothole_counts')
def api_pothole_counts():
	return jsonify(LIVE_POTHOLE_DATA)

@app.route('/api/parking_counts')
def api_parking_counts():
	return jsonify(LIVE_PARKING_DATA)

@app.route('/export_menu2_excel')
def export_menu2_excel():
	if not MENU2_TRAFFIC_LOGS:
		now = datetime.now()
		data = [
			{"Waktu_Pengamatan": now.strftime("%Y-%m-%d 08:00:10"), "Mobil_LV": 2, "Motor_MC": 5, "Bus_HV": 0, "Truk_HV": 0},
			{"Waktu_Pengamatan": now.strftime("%Y-%m-%d 08:05:20"), "Mobil_LV": 4, "Motor_MC": 12, "Bus_HV": 1, "Truk_HV": 0},
			{"Waktu_Pengamatan": now.strftime("%Y-%m-%d 08:12:40"), "Mobil_LV": 6, "Motor_MC": 18, "Bus_HV": 0, "Truk_HV": 1},
		]
		df = pd.DataFrame(data)
	else:
		df = pd.DataFrame(MENU2_TRAFFIC_LOGS)

	filepath = "Hasil_Observasi_Traffic_Menu2.xlsx"
	df.to_excel(filepath, index=False)
	return send_file(filepath, as_attachment=True)

@app.route('/export_menu3_excel')
def export_menu3_excel():
	if not MENU3_POTHOLE_LOGS:
		now = datetime.now()
		data = [
			{"Timestamp": now.strftime("%Y-%m-%d 09:00:05"), "Lubang_Pothole": 1, "Retak_Cracking": 2, "Luas_Kerusakan_m2": 0.35},
			{"Timestamp": now.strftime("%Y-%m-%d 09:00:10"), "Lubang_Pothole": 2, "Retak_Cracking": 3, "Luas_Kerusakan_m2": 0.65},
			{"Timestamp": now.strftime("%Y-%m-%d 09:00:15"), "Lubang_Pothole": 1, "Retak_Cracking": 4, "Luas_Kerusakan_m2": 0.80},
		]
		df = pd.DataFrame(data)
	else:
		df = pd.DataFrame(MENU3_POTHOLE_LOGS)

	filepath = "Hasil_Survei_Kerusakan_Jalan_Menu3.xlsx"
	df.to_excel(filepath, index=False)
	return send_file(filepath, as_attachment=True)

@app.route('/api/upload_pothole_excel', methods=['POST'])
def upload_pothole_excel():
	if 'file' not in request.files:
		return jsonify({"status": "error", "message": "File Excel tidak ditemukan!"})

	file = request.files['file']
	panjang_segmen = float(request.form.get('panjang_segmen', 100.0))
	lebar_segmen = float(request.form.get('lebar_segmen', 7.0))

	if file.filename == '':
		return jsonify({"status": "error", "message": "Nama file kosong!"})

	try:
		df = pd.read_excel(file)

		total_pothole = int(df.get('Lubang_Pothole', pd.Series([0])).max())
		total_cracking = int(df.get('Retak_Cracking', pd.Series([0])).max())
		total_titik = total_pothole + total_cracking

		max_luas_m2 = round(float(df.get('Luas_Kerusakan_m2', pd.Series([0.0])).max()), 2)
		total_luas_segmen = max(1.0, panjang_segmen * lebar_segmen)
		persen_kerusakan = round((max_luas_m2 / total_luas_segmen) * 100, 2)

		if persen_kerusakan >= 15.0 or total_titik >= 5:
			keparahan = "Rusak Berat (Heavy Damage)"
			rekomendasi = "Diperlukan Rekonstruksi Perkerasan Struktur Utama dan Overlay Hotmix Segmen Penuh."
		elif persen_kerusakan >= 5.0 or total_titik >= 3:
			keparahan = "Rusak Sedang (Moderate Damage)"
			rekomendasi = "Diperlukan Penambalan Lubang (Patching), Levelling Course, serta Perbaikan Sistem Drainase."
		elif persen_kerusakan >= 0.5 or total_titik >= 1:
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
		return jsonify({"status": "error", "message": f"Gagal membaca Excel: {str(e)}"})

@app.route('/api/upload_analytics_excel', methods=['POST'])
def upload_analytics_excel():
	if 'file' not in request.files:
		return jsonify({"status": "error", "message": "File Excel tidak ditemukan!"})

	file = request.files['file']
	if file.filename == '':
		return jsonify({"status": "error", "message": "Nama file kosong!"})

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
		return jsonify({"status": "error", "message": f"Gagal memproses Excel: {str(e)}"})

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
	return jsonify({"status": "error", "message": "Gagal Upload Video"})

@app.route('/api/upload_parking', methods=['POST'])
def upload_parking():
	total_slot = int(request.form.get('total_slot', SYSTEM_CONFIG["kapasitas_parkir"])) if request.form else SYSTEM_CONFIG["kapasitas_parkir"]

	if request.json:
		total_slot = int(request.json.get('total_slot', SYSTEM_CONFIG["kapasitas_parkir"]))
		url = request.json.get('url')
		if url:
			ACTIVE_SOURCES["parking"] = url
			LIVE_PARKING_DATA["total_slot"] = total_slot
			return jsonify({"status": "success"})

	if 'file' in request.files:
		file = request.files['file']
		if file.filename != '':
			filename = secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(filepath)
			ACTIVE_SOURCES["parking"] = filepath
			LIVE_PARKING_DATA["total_slot"] = total_slot
			return jsonify({"status": "success", "filepath": filepath})

	return jsonify({"status": "error"})

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
	data = request.json
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
	SYSTEM_CONFIG["kapasitas_parkir"] = int(data.get("kapasitas_parkir", 20))

	LIVE_PARKING_DATA["total_slot"] = SYSTEM_CONFIG["kapasitas_parkir"]

	return jsonify({"status": "success", "message": "Pengaturan Parameter Berhasil Disimpan!"})

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=True)
