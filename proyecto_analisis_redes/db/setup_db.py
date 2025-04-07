from pymongo import MongoClient
import os
import json

def setup_database():
    """Configura la base de datos MongoDB con las colecciones necesarias sin datos de prueba"""
    #Conectar a MongoDB (crear si no existe)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['social_media_analytics']
    
    #Eliminar la base de datos si ya existe para evitar datos residuales
    client.drop_database('social_media_analytics')
    print("Base de datos eliminada y recreada.")
    
    #Crear colecciones
    #posts: Contiene los posts subidos por los usuarios
    #processed_content: Datos procesados listos para entrenamiento
    #test_content: Datos de prueba para validar el modelo
    #analysis_results: Resultados de los análisis realizados
    collections = ['posts', 'processed_content', 'test_content', 'analysis_results']
    for col in collections:
        db.create_collection(col)
        print(f"Colección '{col}' creada.")
    
    #Crear índices para mejorar el rendimiento
    db.posts.create_index([('post_id', 1)], unique=True)
    db.processed_content.create_index([('post_id', 1)], unique=True)
    db.test_content.create_index([('post_id', 1)], unique=True)
    db.analysis_results.create_index([('timestamp', -1)])
    print("Índices creados correctamente.")
    
    #Crear directorio de configuración si no existe
    os.makedirs('proyecto_analisis_redes/proyecto_analisis_redes/db/config', exist_ok=True)
    
    #Guardar configuración en archivo JSON
    config = {
        'db_name': 'social_media_analytics',
        'collections': collections,
        'connection_string': 'mongodb://localhost:27017/'
    }
    
    with open('proyecto_analisis_redes/proyecto_analisis_redes/db/config/db_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Configuración de base de datos guardada en 'db/config/db_config.json'.")
    print("Base de datos configurada correctamente.")

if __name__ == "__main__":
    print("Ejecutando configuración de base de datos...")
    setup_database()
