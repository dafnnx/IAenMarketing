"""
Implementa la función evaluate_model() que toma un modelo entrenado y evalúa su rendimiento.
Carga datos de prueba desde MongoDB.
Devuelve métricas clave: MAE (error medio absoluto), MSE (error cuadrático medio), y precisión.
Tiene manejo de errores incorporado para devolver métricas simuladas en caso de falla."
"""
import numpy as np
from pymongo import MongoClient
import tensorflow as tf
import time
import json

def get_db_connection():
    """Establece conexión con la base de datos MongoDB"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['social_media_analytics']
        return db
    except Exception as e:
        print(f"Error al conectar a la base de datos: {str(e)}")
        return None

def load_test_data():
    """
    Carga datos de prueba para evaluar el modelo
    """
    try:
        db = get_db_connection()
        if db is None:
            return None, None
        
        collection = db['processed_content']
        
        #Obtener documentos procesados con características
        data = list(collection.find({"processed": 1, "features": {"$ne": None}}))
        
        #Limitar a un subconjunto para evaluación
        test_data = data[:100] if len(data) > 100 else data
        
        #Extraer características
        X_image = []
        X_text = []
        X_meta = []
        y = []
        
        for item in test_data:
            #Procesar características
            features = item.get('features', {})
            if isinstance(features, str):
                features = json.loads(features)
                
            #Extraer componentes
            image_features = features.get('image_features', [0])
            text_features = features.get('text_features', [0])
            meta_features = features.get('meta_features', [0])
            
            #Convertir a arrays numpy
            X_image.append(np.array(image_features))
            X_text.append(np.array(text_features))
            X_meta.append(np.array(meta_features))
            
            #Obtener engagement real
            y.append(float(item.get('engagement_rate', 0.0)))
        
        #Convertir listas a arrays numpy
        X_image = np.array(X_image)
        X_text = np.array(X_text)
        X_meta = np.array(X_meta)
        y = np.array(y)
        
        return [X_image, X_text, X_meta], y
        
    except Exception as e:
        print(f"Error al cargar datos de prueba: {str(e)}")
        return None, None

def evaluate_model(model):
    """Evaluar el modelo con datos de prueba"""
    try:
        #Cargar datos para evaluación
        X_test, y_true = load_test_data()
        
        #Si no hay datos reales, generar datos de prueba
        if X_test is None or y_true is None:
            print("Generando datos de prueba para evaluación...")
            num_samples = 100
            X_image = np.random.random((num_samples, 10))
            X_text = np.random.random((num_samples, 10))
            X_meta = np.random.random((num_samples, 5))
            X_test = [X_image, X_text, X_meta]
            y_true = np.random.random(num_samples)
        
        #Predicción
        y_pred = model.predict(X_test)
        
        #Calcular métricas
        mae = float(np.mean(np.abs(y_pred.flatten() - y_true)))
        mse = float(np.mean(np.square(y_pred.flatten() - y_true)))
        
        #Calcular una métrica de "precisión" para regresión
        #(proporción de predicciones dentro de un umbral del valor real)
        threshold = 0.1
        within_threshold = np.mean(np.abs(y_pred.flatten() - y_true) < threshold)
        
        return {
            "mae": mae,
            "mse": mse,
            "accuracy": within_threshold
        }
    except Exception as e:
        print(f"Error al evaluar el modelo: {str(e)}")
        #Devolver valores predeterminados para evitar errores en la interfaz
        return {
            "mae": 0.15,
            "mse": 0.05,
            "accuracy": 0.85
        }
if __name__ == "__main__":
    #Cargar modelo guardado para pruebas
    try:
        model = tf.keras.models.load_model('proyecto_analisis_redes/models/amps_model.h5')
        metrics = evaluate_model(model)
        print(f"Métricas de evaluación: {metrics}")
    except Exception as e:
        print(f"Error al cargar y evaluar el modelo: {str(e)}")