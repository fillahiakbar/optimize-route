from flask import Flask, jsonify, request
from google.cloud import firestore
from google.cloud.firestore import SERVER_TIMESTAMP
import google.auth
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import logging
from math import radians, cos, sin, asin, sqrt

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Inisialisasi Firestore
try:
    db = firestore.Client()
    logging.info("Firestore initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Firestore: {e}")
    db = None  # Handle jika koneksi gagal

# Fungsi untuk menghitung jarak Haversine antara dua titik
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak antara dua titik di bumi menggunakan rumus Haversine.
    Hasilnya dalam meter.
    """
    # Konversi derajat ke radian
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # Radius bumi dalam meter (6371000)
    r = 6371000
    return c * r

# Endpoint 1: Home (status service)
@app.route("/")
def home():
    return jsonify({"status": "Service is active"}), 200

# Endpoint 2: Tambah Lokasi (POST)
@app.route("/add_locations", methods=["POST"])
def add_locations():
    try:
        # Pastikan Firestore diinisialisasi
        if db is None:
            return jsonify({"status": "error", "message": "Firestore is not initialized."}), 500

        data = request.json
        locations = data.get("locations", [])
        
        # Validasi input
        if not locations or not isinstance(locations, list):
            return jsonify({"status": "error", "message": "Invalid input format"}), 400

        # Validasi format lokasi
        for location in locations:
            if not isinstance(location, dict) or "latitude" not in location or "longitude" not in location:
                return jsonify({"status": "error", "message": "Each location must have 'latitude' and 'longitude'"}), 400
            
            lat = location["latitude"]
            lng = location["longitude"]

            # Optional: Validasi tipe data latitude dan longitude
            if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
                return jsonify({"status": "error", "message": "'latitude' and 'longitude' must be numbers"}), 400

        # Simpan semua lokasi dalam satu dokumen
        doc_ref = db.collection("locations").add({
            "locations": locations,
            "created_at": SERVER_TIMESTAMP
        })

        return jsonify({"status": "success", "message": "Locations added successfully"}), 201
    except Exception as e:
        logging.error(f"Error adding locations: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Endpoint 3: Ambil Lokasi (GET) - Mengambil Dokumen Terbaru Saja
@app.route("/get_locations", methods=["GET"])
def get_locations():
    try:
        # Pastikan Firestore diinisialisasi
        if db is None:
            return jsonify({"status": "error", "message": "Firestore is not initialized."}), 500

        # Mengambil parameter 'limit' dari query string, default ke 1 karena kita hanya mengambil dokumen terbaru
        limit_param = request.args.get('limit', 1)
        try:
            limit = int(limit_param)
            if limit <= 0 or limit > 100:
                raise ValueError
        except ValueError:
            return jsonify({"status": "error", "message": "Parameter 'limit' harus integer positif antara 1 dan 100."}), 400

        # Query untuk mengambil dokumen terbaru berdasarkan 'created_at' descending
        query = db.collection("locations").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        docs = query.stream()

        locations = []
        for doc in docs:
            doc_data = doc.to_dict()
            created_at = doc_data.get("created_at")
            # Konversi Timestamp Firestore ke ISO format jika ada
            if created_at:
                created_at = created_at.isoformat()
            locations.append({
                "id": doc.id,
                "locations": doc_data.get("locations", []),
                "created_at": created_at
            })

        return jsonify({"status": "success", "data": locations}), 200
    except Exception as e:
        logging.error(f"Error retrieving locations: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Endpoint 4: Optimasi Rute (GET)
@app.route("/solve_vrp", methods=["GET"])
def solve_vrp():
    try:
        # Pastikan Firestore diinisialisasi
        if db is None:
            return jsonify({"status": "error", "message": "Firestore is not initialized."}), 500

        # Ambil dokumen terbaru
        query = db.collection("locations").order_by("created_at", direction=firestore.Query.DESCENDING).limit(1)
        docs = query.stream()

        locations = []
        for doc in docs:
            doc_data = doc.to_dict()
            locations = doc_data.get("locations", [])

        if len(locations) < 2:
            return jsonify({"status": "error", "message": "At least 2 locations are required"}), 400

        # Konversi ke format yang diperlukan OR-Tools
        locations_list = [[loc["latitude"], loc["longitude"]] for loc in locations]

        # VRP Logic
        manager = pywrapcp.RoutingIndexManager(len(locations_list), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Fungsi Jarak (Haversine Distance)
        def distance_callback(from_index, to_index):
            from_node = locations_list[manager.IndexToNode(from_index)]
            to_node = locations_list[manager.IndexToNode(to_index)]
            return int(haversine_distance(from_node[0], from_node[1], to_node[0], to_node[1]))

        # Register callback jarak
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Parameter Pencarian
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.FromSeconds(10)  # Batas waktu pencarian

        # Solve VRP
        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            return jsonify({"status": "error", "message": "No solution found"}), 500

        # Buat rute
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append({
                "latitude": locations_list[node][0],
                "longitude": locations_list[node][1]
            })
            index = solution.Value(routing.NextVar(index))


        return jsonify({"status": "success", "routes": [route]}), 200
    except Exception as e:
        logging.error(f"Error solving VRP: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Endpoint 5: Ambil ID dan Timestamp Terbaru (GET)
@app.route("/get_latest_ids", methods=["GET"])
def get_latest_ids():
    try:
        # Pastikan Firestore diinisialisasi
        if db is None:
            return jsonify({"status": "error", "message": "Firestore is not initialized."}), 500

        # Mengambil parameter 'limit' dari query string, default ke 1 karena kita hanya mengambil dokumen terbaru
        limit_param = request.args.get('limit', 1)
        try:
            limit = int(limit_param)
            if limit <= 0 or limit > 100:
                raise ValueError
        except ValueError:
            return jsonify({"status": "error", "message": "Parameter 'limit' harus integer positif antara 1 dan 100."}), 400

        # Query untuk mengambil dokumen terbaru berdasarkan 'created_at' descending
        query = db.collection("locations").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        docs = query.stream()

        latest_ids = []
        for doc in docs:
            doc_data = doc.to_dict()
            created_at = doc_data.get("created_at")
            # Konversi Timestamp Firestore ke ISO format jika ada
            if created_at:
                created_at = created_at.isoformat()
            latest_ids.append({
                "id": doc.id,
                "created_at": created_at
            })

        return jsonify({"status": "success", "data": latest_ids}), 200
    except Exception as e:
        logging.error(f"Error retrieving latest IDs: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
