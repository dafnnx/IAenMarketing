import tensorflow as tf
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle
import os       


def load_model_and_info():
    """Carga el modelo entrenado y la información de preprocesamiento"""
    model = tf.keras.models.load_model('proyecto_analisis_redes/proyecto_analisis_redes/models/amps_model.h5')
    
    with open('proyecto_analisis_redes/proyecto_analisis_redes/models/preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)
    
    return model, preprocessing_info

def load_test_data():
    """Carga datos de prueba desde MongoDB"""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['social_media_analytics']
    collection = db['test_content']
    
    data = list(collection.find({}))
    return pd.DataFrame(data)

def evaluate_model():
    """Evalúa el rendimiento del modelo con datos de prueba"""
    model, preprocessing_info = load_model_and_info()
    df_test = load_test_data()
    
    #Preparar datos de prueba
    X_image = np.array(df_test['image_features'].tolist())
    X_text = np.array(df_test['text_features'].tolist())
    X_meta = np.array(df_test['meta_features'].tolist())
    y_true = np.array(df_test['engagement_rate'].tolist())
    
    #Evaluar modelo
    evaluation = model.evaluate(
        [X_image, X_text, X_meta],
        y_true,
        verbose=1
    )
    
    print(f"Loss: {evaluation[0]}, MAE: {evaluation[1]}")
    
    #Realizar predicciones
    y_pred = model.predict([X_image, X_text, X_meta])
    
    #Guardar resultados
    results = {
        'true_engagement': y_true,
        'predicted_engagement': y_pred.flatten(),
        'post_id': df_test['post_id'].tolist()
    }
    
    results_df = pd.DataFrame(results)
    os.makedirs('proyecto_analisis_redes/proyecto_analisis_redes/data/results', exist_ok=True)
    results_df.to_csv('proyecto_analisis_redes/proyecto_analisis_redes/data/results/model_evaluation.csv', index=False)
    
    print("Evaluación completada y resultados guardados")

if __name__ == "__main__":
    evaluate_model()