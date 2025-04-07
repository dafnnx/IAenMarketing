import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from pymongo import MongoClient
import pickle
import json
from .data_utils import get_db_connection

def load_data_from_mongodb():
    """Carga datos desde MongoDB para entrenamiento"""
    try:
        db = get_db_connection()
        if db is None:
            return pd.DataFrame()
            
        collection = db['processed_content']
        if 'processed_content' not in db.list_collection_names():
            collection = db['posts']  #Intenta con coleccion alternativa
            
        data = list(collection.find({"processed": 1}))
        
        #si no hay datos procesados, sse intenta cargar de la colección original
        if len(data) == 0:
            data = list(collection.find({}))
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()

def process_features(df):
    """Procesa las características de los datos para que sean compatibles con el modelo"""
    #Para características de imagen
    def extract_image_features(item):
        if item is None:
            return np.zeros(10)  #Vector por defecto
        #aqui se puede cambiar el tamaño del vector de imagenes
        if isinstance(item, dict) and 'image_features' in item:
            features = item['image_features']
            if isinstance(features, list):
                return np.array(features)
            return np.zeros(10)
        elif isinstance(item, str):
            try:
                data = json.loads(item)
                if isinstance(data, dict) and 'image_features' in data:
                    return np.array(data['image_features'])
            except:
                pass
        return np.zeros(10)
    
    #Para características de texto
    def extract_text_features(item):
        if item is None:
            return np.zeros(10)  #Vector por defecto
            
        if isinstance(item, dict) and 'text_features' in item:
            features = item['text_features']
            if isinstance(features, list):
                return np.array(features)
            return np.zeros(10)
        elif isinstance(item, str):
            try:
                data = json.loads(item)
                if isinstance(data, dict) and 'text_features' in data:
                    return np.array(data['text_features'])
            except:
                pass
        return np.zeros(10)
    
    #Para metadatos
    def extract_meta_features(item):
        if item is None:
            return np.zeros(5)  #Vector por defecto
            
        if isinstance(item, dict) and 'meta_features' in item:
            features = item['meta_features']
            if isinstance(features, list):
                return np.array(features)
            elif isinstance(features, dict):
                return np.array(list(features.values()))
            return np.zeros(5)
        elif isinstance(item, str):
            try:
                data = json.loads(item)
                if isinstance(data, dict) and 'meta_features' in data:
                    features = data['meta_features']
                    if isinstance(features, list):
                        return np.array(features)
                    elif isinstance(features, dict):
                        return np.array(list(features.values()))
            except:
                pass
        return np.zeros(5)
    
    #Procesar características
    feature_column = 'features' if 'features' in df.columns else None
    
    if feature_column:
        X_image = np.array([extract_image_features(item) for item in df[feature_column]])
        X_text = np.array([extract_text_features(item) for item in df[feature_column]])
        X_meta = np.array([extract_meta_features(item) for item in df[feature_column]])
    else:
        #Si no hay columna de características, usar vectores predeterminados
        X_image = np.array([np.zeros(10) for _ in range(len(df))])
        X_text = np.array([np.zeros(10) for _ in range(len(df))])
        X_meta = np.array([np.zeros(5) for _ in range(len(df))])
    
    #aqui nos debemos asegurar de q  todas las matrices tienen la misma longitud en la primera dimensión
    min_samples = min(len(X_image), len(X_text), len(X_meta), len(df))
    X_image = X_image[:min_samples]
    X_text = X_text[:min_samples]
    X_meta = X_meta[:min_samples]
    
    #aqui nos debemos asegurar de q los tensores tienen las dimensiones correctas
    if len(X_image.shape) == 1:
        X_image = np.expand_dims(X_image, axis=1)
    if len(X_text.shape) == 1:
        X_text = np.expand_dims(X_text, axis=1)
    if len(X_meta.shape) == 1:
        X_meta = np.expand_dims(X_meta, axis=1)
    
    #Obtener engagement rate
    if 'engagement_rate' in df.columns:
        y = np.array([float(rate) if isinstance(rate, (int, float)) else 0.0 
                      for rate in df['engagement_rate'][:min_samples]])
    else:
        #Si no hay engagement_rate, se usan valores aleatorios
        y = np.random.random(min_samples)
    
    return X_image, X_text, X_meta, y

def create_amps_model(input_shape_image, input_shape_text, input_shape_meta):
    """
    Implementación simplificada del modelo AMPS
    (Attention-based Multi-modal Popularity prediction model of Short-form videos)
    """
    #Entrada de imagen
    image_input = tf.keras.layers.Input(shape=input_shape_image, name="image_input")
    image_features = tf.keras.layers.Dense(128, activation='relu')(image_input)
    image_features = tf.keras.layers.Dense(128, activation='relu')(image_features)
    
    #Entrada de texto
    text_input = tf.keras.layers.Input(shape=input_shape_text, name="text_input")
    
    #Si el input es 2D, usar Embedding; si es 1D, usar Dense
    if len(input_shape_text) > 1:
        text_features = tf.keras.layers.Embedding(10000, 128)(text_input)
        text_features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(text_features)
        text_features = tf.keras.layers.GlobalAveragePooling1D()(text_features)
    else:
        text_features = tf.keras.layers.Dense(128, activation='relu')(text_input)
    
    text_features = tf.keras.layers.Dense(128, activation='relu')(text_features)
    
    #Entrada de metadatos
    meta_input = tf.keras.layers.Input(shape=input_shape_meta, name="meta_input")
    meta_features = tf.keras.layers.Dense(64, activation='relu')(meta_input)
    meta_features = tf.keras.layers.Dense(128, activation='relu')(meta_features)
    
    #Fusión de características
    combined = tf.keras.layers.Concatenate()([image_features, text_features, meta_features])
    combined = tf.keras.layers.Dense(256, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    
    #Salida: predicción de engagement
    output = tf.keras.layers.Dense(1, name="engagement_prediction")(combined)
    
    model = tf.keras.Model(
        inputs=[image_input, text_input, meta_input],
        outputs=output
    )
    
    return model

def train_model(epochs=10, batch_size=32, learning_rate=0.001, validation_split=0.2, progress_callback=None):
    """
    Entrena el modelo AMPS con los datos preprocesados
    
    Args:
        epochs: Número de épocas para entrenar
        batch_size: Tamaño del lote para entrenamiento
        learning_rate: Tasa de aprendizaje para el optimizador
        validation_split: Proporción de datos para validación
        progress_callback: Función para reportar progreso
    
    Returns:
        tuple: (modelo entrenado, historial de entrenamiento)
    """
    print(f"Iniciando entrenamiento con {epochs} épocas, batch_size={batch_size}")
    
    #Cargar datos
    df = load_data_from_mongodb()
    
    if df.empty:
        print("No se pudieron cargar datos. Generando datos de prueba...")
        #Crear datos de prueba si no hay datos reales
        num_samples = 500
        X_image = np.random.random((num_samples, 10))
        X_text = np.random.random((num_samples, 10))
        X_meta = np.random.random((num_samples, 5))
        y = np.random.random(num_samples)
    else:
        print(f"Datos cargados: {len(df)} ejemplos")
        #Procesar características
        X_image, X_text, X_meta, y = process_features(df)
    
    #Imprimir dimensiones para depuración
    print(f"X_image shape: {X_image.shape}")
    print(f"X_text shape: {X_text.shape}")
    print(f"X_meta shape: {X_meta.shape}")
    print(f"y shape: {y.shape}")
    
    #Definir dimensiones de entrada
    input_shape_image = X_image.shape[1:]
    input_shape_text = X_text.shape[1:]
    input_shape_meta = X_meta.shape[1:]
    
    print(f"Input shapes - Image: {input_shape_image}, Text: {input_shape_text}, Meta: {input_shape_meta}")
    
    #Crear modelo con tasa de aprendizaje personalizada
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = create_amps_model(input_shape_image, input_shape_text, input_shape_meta)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    #Calcular steps_per_epoch para poder actualizar el progreso
    steps_per_epoch = int(np.ceil(len(X_image) * (1 - validation_split) / batch_size))
    
    #Crear clase de callback personalizada para reportar progreso
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            if progress_callback:
                #inicializar el progreso al inicio del entrenamiento
                progress_callback(0, 0, steps_per_epoch, {'loss': 0, 'mae': 0})
        
        def on_epoch_begin(self, epoch, logs=None):
            if progress_callback:
                #usar el epoca actual para el progreso
                progress_callback(epoch + 1, 0, steps_per_epoch, {'loss': 0, 'mae': 0})
                
        def on_batch_end(self, batch, logs=None):
            if progress_callback:
                #hacer logs de progreso al final de cada batch
                log_data = logs or {'loss': 0, 'mae': 0}
                progress_callback(self.params.get('epoch', 0) + 1, batch + 1, steps_per_epoch, log_data)
                
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                #hacer logs de progreso al final de cada epoca
                log_data = logs or {'loss': 0, 'mae': 0}
                progress_callback(epoch + 1, steps_per_epoch, steps_per_epoch, log_data)
                
    #mostrar resumen del modelo
    model.summary()
    
    #entrenar modelo con callback de progreso
    history = model.fit(
        [X_image, X_text, X_meta],
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[ProgressCallback()],
        verbose=1
    )
    
    #Crear directorio para guardar si no existe
    os.makedirs('proyecto_analisis_redes/models', exist_ok=True)
    
    #Guardar modelo
    model_path = 'proyecto_analisis_redes/models/amps_model.h5'
    model.save(model_path)
    
    #Guardar información del preprocesamiento
    preprocessing_info = {
        'input_shape_image': input_shape_image,
        'input_shape_text': input_shape_text,
        'input_shape_meta': input_shape_meta
    }
    #
    with open('proyecto_analisis_redes/models/preprocessing_info.pkl', 'wb') as f:
        pickle.dump(preprocessing_info, f)
    
    print(f"Modelo entrenado y guardado en {model_path}")
    
    return model, history

if __name__ == "__main__":
    #Prueba de entrenamiento
    def progress_test(epoch, batch, total_batches, logs=None):
        print(f"Época {epoch}/{10}, Batch {batch}/{total_batches}, Loss: {logs.get('loss') if logs else 'N/A'}")
    
    train_model(epochs=2, batch_size=32, progress_callback=progress_test)