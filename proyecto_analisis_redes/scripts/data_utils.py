"""
Implementa la función get_dataset_statistics() para obtener estadísticas del conjunto de datos.
Proporciona utilidades para conectarse a la base de datos.
Tiene funciones para procesar y normalizar características.
Incluye manejo de errores para manejar diferentes formatos de datos.
"""
import json
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np

def get_db_connection():
    """Establece conexión con la base de datos MongoDB"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['social_media_analytics']
        return db
    except Exception as e:
        print(f"Error al conectar a la base de datos: {str(e)}")
        return None

def process_features(feature_data):
    """
    Procesa y normaliza características para el entrenamiento
    
    Args:
        feature_data: Datos de características que pueden estar en varios formatos
        
    Returns:
        dict: Características procesadas
    """
    if isinstance(feature_data, str):
        try:
            #Intentar parsear JSON si es una string
            return json.loads(feature_data)
        except:
            #Si no es JSON válido, devolver estructura básica
            return {
                "image_features": [0],
                "text_features": [0],
                "meta_features": [0]
            }
    elif isinstance(feature_data, dict):
        return feature_data
    else:
        #Si no es un formato reconocido, devolver estructura básica
        return {
            "image_features": [0],
            "text_features": [0],
            "meta_features": [0]
        }

def count_feature_dimensions(features):
    """
    Cuenta las dimensiones de las características
    Args:   features: Lista o diccionario de características 
    Returns:
        int: Número de dimensiones
    """
    if isinstance(features, list):
        return len(features)
    elif isinstance(features, dict):
        #Contar todas las dimensiones en el diccionario
        total_dims = 0
        for value in features.values():
            if isinstance(value, list):
                total_dims += len(value)
            else:
                total_dims += 1
        return total_dims
    else:
        return 1

def get_dataset_statistics():
    """
    Obtiene estadísticas reales del dataset desde la base de datos MongoDB.
    Returns:        dict: Estadísticas del conjunto de datos
    """
    try:
        #Conectar a la base de datos
        db = get_db_connection()
        if db is None:
            return {
                "error": "Error al conectar a la base de datos",
                "total_examples": 0,
                "training_examples": 0,
                "validation_examples": 0,
                "feature_count": 0,
                "feature_description": "No hay datos disponibles"
            }

        #Obtener los documentos procesados con características
        collection = db['processed_content']
        #Verificar si la colección existe
        if 'processed_content' not in db.list_collection_names():
            collection = db['posts']  #Intentar con colección alternativa
        
        #Buscar documentos con características
        documents = list(collection.find({"processed": 1, "features": {"$ne": None}}))
        
        #Si no hay documentos, buscar alternativas
        if len(documents) == 0:
            documents = list(collection.find({}))
        
        #Calcular estadísticas
        total_examples = len(documents)
        validation_split = 0.2
        training_examples = int(total_examples * (1 - validation_split))
        validation_examples = total_examples - training_examples

        #Contar características
        feature_count = 0
        feature_description = "Características combinadas (imagen, texto, metadatos)"
        
        if total_examples > 0:
            #Intentar obtener información de características
            sample_doc = documents[0]
            
            if 'features' in sample_doc:
                features = sample_doc['features']
                processed_features = process_features(features)
                
                #Contar dimensiones totales sumando todas las categorías
                image_dims = count_feature_dimensions(processed_features.get('image_features', []))
                text_dims = count_feature_dimensions(processed_features.get('text_features', []))
                meta_dims = count_feature_dimensions(processed_features.get('meta_features', []))
                
                feature_count = image_dims + text_dims + meta_dims
            else:
                #Si no hay 'features', intentar contar otros campos
                feature_count = len(sample_doc.keys()) - 2  #Restar _id y processed
        
        return {
            "total_examples": total_examples,
            "training_examples": training_examples,
            "validation_examples": validation_examples,
            "feature_count": feature_count,
            "feature_description": feature_description
        }
        
    except Exception as e:
        print(f"Error al obtener estadísticas del dataset: {str(e)}")
        return {
            "error": f"Error al procesar datos: {str(e)}",
            "total_examples": 0,
            "training_examples": 0,
            "validation_examples": 0,
            "feature_count": 0,
            "feature_description": "Error al procesar"
        }

def load_and_preprocess_data():
    """
    Carga y preprocesa datos para entrenamiento desde MongoDB
    Returns:
        tuple: (X_image, X_text, X_meta, y) arrays de datos para entrenamiento
    """
    db = get_db_connection()
    if db is None:
        return None, None, None, None
        
    try:
        #Obtener colección
        collection = db['processed_content']
        if 'processed_content' not in db.list_collection_names():
            collection = db['posts']  #Intentar con colección alternativa
            
        #Obtener documentos
        documents = list(collection.find({"processed": 1}))
        
        if len(documents) == 0:
            print("No se encontraron documentos procesados")
            return None, None, None, None
            
        #Preparar arrays
        X_image = []
        X_text = []
        X_meta = []
        y = []
        
        for doc in documents:
            features = process_features(doc.get('features', {}))
            
            #Extraer características por tipo
            image_feat = np.array(features.get('image_features', [0]))
            text_feat = np.array(features.get('text_features', [0]))
            meta_feat = np.array(features.get('meta_features', [0]))
            
            #Añadir a los arrays
            X_image.append(image_feat)
            X_text.append(text_feat)
            X_meta.append(meta_feat)
            
            #Obtener valor objetivo (engagement)
            engagement = float(doc.get('engagement_rate', 0.0))
            y.append(engagement)
            
        #Convertir a arrays numpy
        X_image = np.array(X_image)
        X_text = np.array(X_text)
        X_meta = np.array(X_meta)
        y = np.array(y)
        
        return X_image, X_text, X_meta, y
        
    except Exception as e:
        print(f"Error al cargar y preprocesar datos: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    #Prueba de funcionalidad
    stats = get_dataset_statistics()
    print("Estadísticas del dataset:")
    print(json.dumps(stats, indent=2))