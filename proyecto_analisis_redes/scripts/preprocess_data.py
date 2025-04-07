import pandas as pd
import pymongo
import json
import numpy as np

def preprocess_data_for_model():
    #Cargar configuración de la base de datos
    with open("proyecto_analisis_redes/proyecto_analisis_redes/db/config/db_config.json", "r") as f:
        db_config = json.load(f)
    
    #Conectar a MongoDB
    client = pymongo.MongoClient(db_config["connection_string"])
    db = client[db_config["db_name"]]
    collection = db["posts"]
    
    #Obtener datos
    posts = list(collection.find({}))
    
    #Convertir a DataFrame para facilitar el análisis
    df = pd.DataFrame(posts)
    
    #Preprocesamiento básico
    #1. Características numéricas
    df['engagement_scaled'] = df['engagement'] / df['engagement'].max()
    
    #2. Características de texto
    df['hashtag_count'] = df['hashtags'].apply(len)
    df['caption_length'] = df['caption'].apply(len)
    
    #3. Características de imagen (simuladas)
    if 'image_features' in df.columns:
        df['brightness'] = df['image_features'].apply(lambda x: x.get('brightness', 0.5))
        df['contrast'] = df['image_features'].apply(lambda x: x.get('contrast', 0.5))
    
    #Guardar datos preprocesados
    processed_data = df.to_dict(orient='records')
    
    #Guardar en la base de datos en una nueva colección
    processed_collection = db["processed_content"]
    processed_collection.drop()  #Limpiar colección existente
    processed_collection.insert_many(processed_data)
    
    print(f"Se preprocesaron {len(processed_data)} posts y se guardaron en la base de datos")

if __name__ == "__main__":
    preprocess_data_for_model()