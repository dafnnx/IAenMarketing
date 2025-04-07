#Ejemplo para scripts/load_sample_data.py
import json
import pymongo
import os

def load_sample_data():
    #Cargar configuración de la base de datos
    with open("proyecto_analisis_redes/proyecto_analisis_redes/db/config/db_config.json", "r") as f:
        db_config = json.load(f)
    
    #Conectar a MongoDB
    client = pymongo.MongoClient(db_config["connection_string"])
    db = client[db_config["db_name"]]
    collection = db["posts"]
    
    #Leer datos de muestra
    with open("proyecto_analisis_redes/proyecto_analisis_redes/data/sample/sample_posts.json", "r") as f:
        posts = json.load(f)
    
    #Insertar en la base de datos
    result = collection.insert_many(posts)
    print(f"Se insertaron {len(result.inserted_ids)} posts en la base de datos.")

if __name__ == "__main__":
    load_sample_data()