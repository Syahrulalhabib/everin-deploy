from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import json
from PIL import Image
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import logging
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('ServiceAccountKey.json')  # Ganti dengan path ke file kunci Anda
firebase_admin.initialize_app(cred)

# Buat instance Firestore client
firestore_client = firestore.client()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize models as None
cnn_model = None
knn_model = None
dataset = None

def create_knn_model(df):
    """Create and train KNN model"""
    try:
        features = df[['Karbohidrat (g)', 'Protein (g)', 'Lemak (g)']].values
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(features)
        logger.info("KNN model created successfully")
        return knn
    except Exception as e:
        logger.error(f"Error creating KNN model: {str(e)}")
        raise

def load_models():
    """Function to load all required models"""
    global cnn_model, knn_model, dataset
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model_path = os.path.join(base_path, 'model', 'v1.h5')
        logger.info(f"Loading CNN model from: {model_path}")
        cnn_model = tf.keras.models.load_model(model_path, compile=False)
        cnn_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("CNN model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CNN model: {str(e)}")
        raise

    try:
        dataset_path = os.path.join(base_path, 'dataset.json')
        logger.info(f"Loading dataset from: {dataset_path}")
        with open(dataset_path) as f:
            dataset = json.load(f)
        logger.info("Dataset loaded successfully")
        
        df = pd.DataFrame(dataset)
        
        knn_model = create_knn_model(df)
        logger.info("KNN model created successfully")
        
    except Exception as e:
        logger.error(f"Error loading dataset or creating KNN model: {str(e)}")
        raise

# Try to load models when app starts
try:
    logger.info("Starting model loading...")
    load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    pass
def predict_food(image_path):
    """Predict food from image and get nutrition info"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = cnn_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        food_name = dataset[predicted_class]["Nama Makanan/Minuman"]
        nutrition = {
            "kalori": dataset[predicted_class]["Kalori (kcal)"],
            "karbohidrat": dataset[predicted_class]["Karbohidrat (g)"],
            "protein": dataset[predicted_class]["Protein (g)"],
            "lemak": dataset[predicted_class]["Lemak (g)"]
        }
        return food_name, nutrition, confidence
    except Exception as e:
        logger.error(f"Error in predict_food: {str(e)}")
        raise

def get_food_recommendations(food_features):
    """Get food recommendations using KNN model"""
    try:
        # Pastikan food_features dikirim dalam format yang benar (list dengan 3 fitur)
        food_features = np.array(food_features).reshape(1, -1)
        
        # Mendapatkan nearest neighbors
        distances, indices = knn_model.kneighbors(food_features)
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            food = dataset[idx]  # Mengakses makanan berdasarkan index KNN
            recommendations.append({
                "nama": food["Nama Makanan/Minuman"],  # Nama makanan/minuman
                "nutrition": {
                    "kalori": food["Kalori (kcal)"],  # Kalori
                    "karbohidrat": food["Karbohidrat (g)"],  # Karbohidrat
                    "protein": food["Protein (g)"],  # Protein
                    "lemak": food["Lemak (g)"]  # Lemak
                },
                "similarity_score": float(1 / (1 + distances[0][i]))  # Similarity score
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error in get_food_recommendations: {str(e)}")
        raise  # Raise the error again to be handled in the calling function
    
def hitung_bmr_tdee(berat_badan, tinggi_badan, umur, jenis_kelamin, tingkat_aktivitas):
    """Calculate BMR and TDEE"""
    if jenis_kelamin.lower() == 'pria':
        bmr = 10 * berat_badan + 6.25 * tinggi_badan - 5 * umur + 5
    else:  # wanita
        bmr = 10 * berat_badan + 6.25 * tinggi_badan - 5 * umur - 161

    activity_multipliers = {
        "ringan": 1.375,
        "sedang": 1.55,
        "berat": 1.725
    }
    
    if tingkat_aktivitas.lower() not in activity_multipliers:
        raise ValueError("Tingkat aktivitas tidak valid. Pilih 'ringan', 'sedang', atau 'berat'.")
        
    tdee = bmr * activity_multipliers[tingkat_aktivitas.lower()]
    return bmr, tdee

def hitung_kebutuhan_makronutrien(tdee):
    """Calculate macronutrient needs"""
    kalori_karbohidrat = tdee * 0.55
    kalori_protein = tdee * 0.20
    kalori_lemak = tdee * 0.25

    gram_karbohidrat = kalori_karbohidrat / 4
    gram_protein = kalori_protein / 4
    gram_lemak = kalori_lemak / 9

    return gram_karbohidrat, gram_protein, gram_lemak

def submit_prediction_to_firestore(data):
    """Submit prediction data to Firestore"""
    try:
        # Cek apakah semua field yang dibutuhkan ada
        required_fields = ['food_name', 'confidence', 'nutrition', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {
                'status': 'error',
                'message': f"Missing required fields: {missing_fields}"
            }, 400  # Mengembalikan status 400 jika ada field yang hilang
        
        # Struktur data untuk Firestore
        submission_data = {
            'food_name': data['food_name'],
            'confidence': data['confidence'],
            'nutrition': data['nutrition'],
            'recommendations': data['recommendations'],
            'timestamp': firestore.SERVER_TIMESTAMP  # Gunakan timestamp dari Firestore
        }
        
        # Menyimpan data ke Firestore
        doc_ref = firestore_client.collection('users').add(submission_data)   

        # Mengembalikan response dengan doc_id yang dihasilkan
        return {
            'status': 'success',
            'doc_id': doc_ref.id,  # ID dokumen Firestore yang dihasilkan
            'submitted_data': submission_data
        }, 200  # Kode status 200 jika berhasil
        
    except Exception as e:
        # Menangani error jika terjadi kegagalan saat submit
        logger.error(f"Error submitting to Firestore: {str(e)}")
        
        return {
            'status': 'error',
            'message': f"Error submitting to Firestore: {str(e)}"
        }, 500  # Kode status 500 jika ada kesalahan server

@app.route('/status/<email>', methods=['GET'])
def get_status_by_email(email):
    """Menampilkan total kalori, karbohidrat, protein, dan lemak berdasarkan email pengguna"""
    try:
        # Referensi ke dokumen total nutrition berdasarkan email
        totals_ref = firestore_client.collection('users').document(email).collection('totals').document('nutrition_totals')
        totals_doc = totals_ref.get()

        # Periksa apakah dokumen ada
        if totals_doc.exists:
            total_nutrition = totals_doc.to_dict()
            return jsonify({
                'status': 'success',
                'email': email,
                'kalori': total_nutrition.get('kalori', 0),
                'karbohidrat': total_nutrition.get('karbohidrat', 0),
                'protein': total_nutrition.get('protein', 0),
                'lemak': total_nutrition.get('lemak', 0)
            }), 200
        else:
            # Jika tidak ada data untuk email tersebut
            return jsonify({
                'status': 'error',
                'message': f'No nutrition data found for email: {email}'
            }), 404

    except Exception as e:
        logger.error(f"Error fetching status for {email}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():

    if cnn_model is None or dataset is None:
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    try:
        file = request.files['file']
        os.makedirs('temp', exist_ok=True)
        
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)
        
        food_name, nutrition, confidence = predict_food(file_path)
        os.remove(file_path)
        

        food_features = [
            nutrition["karbohidrat"],
            nutrition["protein"], 
            nutrition["lemak"]
        ]
        recommendations = get_food_recommendations(food_features)
        
        return jsonify({
            'food_name': food_name,
            'confidence': confidence,
            'nutrition': nutrition,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calculate', methods=['POST'])
def calculate():
    """Endpoint for calculating nutrition needs"""
    try:
        data = request.json
        required_fields = ['weight', 'height', 'age', 'gender', 'activity_level']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        bmr, tdee = hitung_bmr_tdee(
            data['weight'],
            data['height'],
            data['age'],
            data['gender'],
            data['activity_level']
        )
        
        karbohidrat, protein, lemak = hitung_kebutuhan_makronutrien(tdee)
        
        return jsonify({
            'bmr': float(bmr),
            'tdee': float(tdee),
            'kebutuhan_harian': {
                'karbohidrat': float(karbohidrat),
                'protein': float(protein),
                'lemak': float(lemak)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in calculate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for food recommendations based on nutrition values"""
    if knn_model is None or dataset is None:
        logger.error("Models not loaded properly")
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    try:
        # Mengambil data dari request
        data = request.json
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['karbohidrat', 'protein', 'lemak']
        
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            logger.error(f"Missing fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Validasi dan konversi nilai ke tipe data float
        try:
            food_features = [
                float(data['karbohidrat']),
                float(data['protein']),
                float(data['lemak'])
            ]
        except ValueError as e:
            logger.error(f"Invalid numeric values: {str(e)}")
            return jsonify({'error': 'All nutritional values must be numbers'}), 400
            
        # Validasi nilai tidak boleh negatif
        if any(x < 0 for x in food_features):
            logger.error("Negative nutritional values provided")
            return jsonify({'error': 'Nutritional values cannot be negative'}), 400
        
        # Mendapatkan rekomendasi makanan
        recommendations = get_food_recommendations(food_features)
        
        if not recommendations:
            logger.warning("No recommendations found")
            return jsonify({'message': 'No similar foods found', 'recommendations': []}), 200
        
        # Menyusun response dengan nilai input dan rekomendasi
        return jsonify({
            'input_values': {
                'karbohidrat': food_features[0],
                'protein': food_features[1],
                'lemak': food_features[2]
            },
            'recommendations': recommendations
        })
    
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/recommend-by-name', methods=['POST'])
def recommend_by_name():

    if knn_model is None or dataset is None:
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    try:
        data = request.json
        if 'food_name' not in data:
            return jsonify({'error': 'Missing food_name field'}), 400
            
        food_name_input = data['food_name'].lower().strip()
        food_data = None
        
        for food in dataset:
            dataset_food_name = ' '.join(food["Nama Makanan/Minuman"].lower().strip().split())
            if dataset_food_name == food_name_input:
                food_data = food
                break
                
        if food_data is None:
            return jsonify({'error': 'Food not found in database'}), 404
            

        food_features = [
            float(food_data["Karbohidrat (g)"]),
            float(food_data["Protein (g)"]),
            float(food_data["Lemak (g)"])
        ]
        

        recommendations = get_food_recommendations(food_features)
        
        response_data = {
            'input_food': {
                'nama': food_data["Nama Makanan/Minuman"],
                'nutrition': {
                    'kalori': food_data["Kalori (kcal)"],
                    'karbohidrat': food_data["Karbohidrat (g)"],
                    'protein': food_data["Protein (g)"],
                    'lemak': food_data["Lemak (g)"]
                }
            },
            'recommendations': recommendations
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in recommend-by-name endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit_submission():
    """Menambahkan data submission ke Firestore dan memperbarui total nilai berdasarkan email"""
    try:
        # Ambil data dari body request
        data = request.get_json()
        email = data.get('email')  # Gunakan email sebagai user_id
        new_nutrition = data.get('nutrition', {})

        # Validasi input
        if not email or not new_nutrition:
            return jsonify({'error': 'email dan nutrition diperlukan'}), 400

        # Referensi ke dokumen user berdasarkan email
        user_ref = firestore_client.collection('users').document(email)
        submission_ref = user_ref.collection('submissions')

        # Tambahkan data baru ke sub-koleksi 'submissions'
        submission_ref.add({'nutrition': new_nutrition})

        # Dapatkan atau inisialisasi total nutrition
        totals_ref = user_ref.collection('totals').document('nutrition_totals')
        totals_doc = totals_ref.get()

        if totals_doc.exists:
            # Jika total nutrition sudah ada, tambahkan nilai baru
            total_nutrition = totals_doc.to_dict()
            total_nutrition['kalori'] += new_nutrition.get('kalori', 0)
            total_nutrition['karbohidrat'] += new_nutrition.get('karbohidrat', 0)
            total_nutrition['protein'] += new_nutrition.get('protein', 0)
            total_nutrition['lemak'] += new_nutrition.get('lemak', 0)
        else:
            # Jika belum ada, inisialisasi total nutrition
            total_nutrition = {
                'kalori': new_nutrition.get('kalori', 0),
                'karbohidrat': new_nutrition.get('karbohidrat', 0),
                'protein': new_nutrition.get('protein', 0),
                'lemak': new_nutrition.get('lemak', 0)
            }

        # Simpan kembali total nutrition
        totals_ref.set(total_nutrition)

        return jsonify({
            'status': 'success',
            'message': 'Data berhasil ditambahkan dan total diperbarui',
            'totals': total_nutrition
        }), 200

    except Exception as e:
        logger.error(f"Error in /submit: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()  # Load models before starting the app
    app.run(port=8080, debug=True)
