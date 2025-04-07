import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import pymongo
import logging
from datetime import datetime
import json
import cv2

#Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("preprocess.log"), logging.StreamHandler()]
)
logger = logging.getLogger("data_preprocessing")

class DataPreprocessor:
    def __init__(self, image_size=(224, 224)):
        """
        Inicializa el preprocesador de datos
        Args:
            image_size (tuple): Tamaño al que se redimensionarán las imágenes (alto, ancho)
        """
        self.image_size = image_size
        
        #Conectar a MongoDB
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["social_media_analytics"]
        self.images_collection = self.db["images"]
        self.videos_collection = self.db["videos"]
        
        #Crear directorios para datos procesados si no existen
        os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/processed/images", exist_ok=True)
        os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/processed/videos", exist_ok=True)
        os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/processed/features", exist_ok=True)
        
        logger.info("Inicializado el preprocesador de datos")
    
    def preprocess_images(self, limit=1000):
    
        logger.info(f"Iniciando preprocesamiento de imágenes (límite: {limit})")
        
        #Obtener datos de imágenes
        cursor = self.images_collection.find().limit(limit)
        images_data = []
        
        processed_count = 0
        for doc in cursor:
            try:
                image_id = doc['id']
                
                #Verificar si la imagen existe
                raw_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/images/{image_id}.jpg"
                if not os.path.exists(raw_path):
                    logger.warning(f"Imagen no encontrada: {raw_path}")
                    continue
                
                #Procesar la imagen
                processed_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/processed/images/{image_id}.jpg"
                success = self._process_image(raw_path, processed_path)
                
                if success:
                    #Extraer características de engagement
                    engagement = self._calculate_engagement(doc)
                    
                    #Crear registro
                    record = {
                        'id': image_id,
                        'processed_path': processed_path,
                        'likes': doc.get('likes', 0),
                        'comments': doc.get('comments', 0),
                        'caption': doc.get('caption', ''),
                        'engagement': engagement,
                        'is_popular': 1 if engagement > 0.05 else 0,  #Umbral arbitrario para popularidad
                        'processed_at': datetime.now()
                    }
                    
                    images_data.append(record)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Procesadas {processed_count} imágenes")
                
            except Exception as e:
                logger.error(f"Error al procesar imagen {doc.get('id', 'desconocido')}: {e}")
        
        #Crear DataFrame
        df = pd.DataFrame(images_data)
        
        #Guardar DataFrame
        csv_path = "proyecto_analisis_redes/proyecto_analisis_redes/data/processed/images_metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Procesadas {processed_count} imágenes. Metadatos guardados en {csv_path}")
        
        return df
    
    def preprocess_videos(self, limit=500):

        logger.info(f"Iniciando preprocesamiento de videos (límite: {limit})")
        
        #Obtener datos de videos
        cursor = self.videos_collection.find().limit(limit)
        videos_data = []
        
        processed_count = 0
        for doc in cursor:
            try:
                video_id = doc['id']
                
                #Verificar si el video existe
                raw_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/videos/{video_id}.mp4"
                if not os.path.exists(raw_path):
                    logger.warning(f"Video no encontrado: {raw_path}")
                    continue
                
                #Procesar el video (extraer frames)
                frames_dir = f"proyecto_analisis_redes/proyecto_analisis_redes/data/processed/videos/{video_id}"
                os.makedirs(frames_dir, exist_ok=True)
                
                success, frame_count = self._extract_video_frames(raw_path, frames_dir)
                
                if success and frame_count > 0:
                    #Extraer características de engagement
                    engagement = self._calculate_engagement(doc)
                    
                    #Crear registro
                    record = {
                        'id': video_id,
                        'frames_dir': frames_dir,
                        'frame_count': frame_count,
                        'likes': doc.get('likes', 0),
                        'comments': doc.get('comments', 0),
                        'views': doc.get('views', 0),
                        'caption': doc.get('caption', ''),
                        'engagement': engagement,
                        'is_popular': 1 if engagement > 0.05 else 0,  #Umbral arbitrario para popularidad
                        'processed_at': datetime.now()
                    }
                    
                    videos_data.append(record)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        logger.info(f"Procesados {processed_count} videos")
                
            except Exception as e:
                logger.error(f"Error al procesar video {doc.get('id', 'desconocido')}: {e}")
        
        #Crear DataFrame
        df = pd.DataFrame(videos_data)
        
        #Guardar DataFrame
        csv_path = "proyecto_analisis_redes/proyecto_analisis_redes/data/processed/videos_metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Procesados {processed_count} videos. Metadatos guardados en {csv_path}")
        
        return df
    
    def _process_image(self, input_path, output_path):
   
        try:
            #Abrir imagen
            img = Image.open(input_path)
            
            #Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            #Redimensionar
            img = img.resize(self.image_size, Image.LANCZOS)
            
            #Guardar imagen procesada
            img.save(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al procesar imagen {input_path}: {e}")
            return False
    
    def _extract_video_frames(self, video_path, output_dir, max_frames=5):
        """
        Extrae frames clave de un video
        Args:
            video_path (str): Ruta al archivo de video
            output_dir (str): Directorio donde guardar los frames
            max_frames (int): Número máximo de frames a extraer
        Returns:
            tuple: (éxito, número de frames extraídos)
        """
        try:
            #Abrir video
            cap = cv2.VideoCapture(video_path)
            
            #Verificar si se abrió correctamente
            if not cap.isOpened():
                logger.error(f"No se pudo abrir el video: {video_path}")
                return False, 0
            
            #Obtener FPS y total de frames
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.error(f"No se detectaron frames en el video: {video_path}")
                return False, 0
            
            #Calcular los índices de los frames a extraer (distribuidos uniformemente)
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
            frame_count = 0
            for i, frame_idx in enumerate(indices):
                #Posicionar en el frame deseado
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                #Leer el frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"No se pudo leer el frame {frame_idx} del video {video_path}")
                    continue
                
                #Guardar el frame como imagen
                frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
                
                #Redimensionar el frame al mismo tamaño que las imágenes
                frame_resized = cv2.resize(frame, self.image_size)
                
                #Guardar el frame
                cv2.imwrite(frame_path, frame_resized)
                
                frame_count += 1
            
            #Liberar los recursos
            cap.release()
            
            logger.info(f"Extraídos {frame_count} frames del video {video_path}")
            return True, frame_count
            
        except Exception as e:
            logger.error(f"Error al procesar el video {video_path}: {e}")
            return False, 0
    
    def _calculate_engagement(self, doc):
     
        try:
            #Obtener métricas relevantes
            likes = doc.get('likes', 0)
            shares = doc.get('shares', 0)
            views = doc.get('views', 0)
            
            #Para calcular el engagement, se usan diferentes fórmulas dependiendo del tipo de publicación
            if views > 0:  #Video
                engagement = ((likes + views + shares)/ followers)*100
            else:
                #Para imágenes, asumimos un alcance basado en el número de seguidores (si está disponible)
                followers = doc.get('followers', 1000)  #Valor por defecto si no está disponible
                engagement = (likes + views*2 + shares*3) / followers
            
            #Normalizar el engagement (limitarlo a un máximo de 1.0)
            engagement = min(engagement, 1.0)
            
            return engagement
            
        except Exception as e:
            logger.error(f"Error al calcular engagement: {e}")
            return 0.0
    
    def extract_features(self, model_path=None):
      
        logger.info("Iniciando extracción de características...")
        
        #Cargar modelo pre-entrenado 
        if model_path:
            model = tf.keras.models.load_model(model_path)
        else:
            logger.info("Cargando modelo ResNet50 pre-entrenado...")
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size[0], self.image_size[1], 3)
            )
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D()
            ])
        
        #Función para preprocesar imágenes
        def preprocess_img(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.image_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            return img_array
        
        #Extraer características de imágenes
        logger.info("Extrayendo características de imágenes...")
        img_df = pd.read_csv('proyecto_analisis_redes/proyecto_analisis_redes/data/processed/images_metadata.csv')
        image_features = {}
        
        for idx, row in img_df.iterrows():
            try:
                img_path = row['processed_path']
                img_id = row['id']
                
                if os.path.exists(img_path):
                    #Preprocesar imagen
                    img_array = preprocess_img(img_path)
                    
                    #Extraer características
                    features = model.predict(img_array)[0]
                    
                    #Guardar características
                    features_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/processed/features/img_{img_id}.npy"
                    np.save(features_path, features)
                    
                    #Guardar referencia
                    image_features[img_id] = features_path
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Procesadas {idx + 1} imágenes para extracción de características")
            
            except Exception as e:
                logger.error(f"Error al extraer características de la imagen {row.get('id', 'desconocida')}: {e}")
        
        #Extraer características de frames de videos
        logger.info("Extrayendo características de frames de videos...")
        vid_df = pd.read_csv('proyecto_analisis_redes/proyecto_analisis_redes/data/processed/videos_metadata.csv')
        video_features = {}
        
        for idx, row in vid_df.iterrows():
            try:
                frames_dir = row['frames_dir']
                video_id = row['id']
                
                if os.path.exists(frames_dir):
                    #Obtener frames
                    frames = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                    
                    if frames:
                        #Extraer características de cada frame
                        all_features = []
                        for frame_path in frames:
                            #Preprocesar frame
                            frame_array = preprocess_img(frame_path)
                            
                            #Extraer características
                            features = model.predict(frame_array)[0]
                            all_features.append(features)
                        
                        #Promediar características de todos los frames
                        avg_features = np.mean(all_features, axis=0)
                        
                        #Guardar características
                        features_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/processed/features/vid_{video_id}.npy"
                        np.save(features_path, avg_features)
                        
                        #Guardar referencia
                        video_features[video_id] = features_path
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Procesados {idx + 1} videos para extracción de características")
            
            except Exception as e:
                logger.error(f"Error al extraer características del video {row.get('id', 'desconocido')}: {e}")
        
        #Guardar referencias a las características
        features_metadata = {
            'images': image_features,
            'videos': video_features
        }
        
        with open('proyecto_analisis_redes/proyecto_analisis_redes/data/processed/features_metadata.json', 'w') as f:
            json.dump(features_metadata, f)
        
        logger.info("Extracción de características completada")
        return features_metadata
    
    def process_all(self, image_limit=1000, video_limit=500, extract_features=True):
        """
        Ejecuta todo el proceso de preprocesamiento
        
        Args:
            image_limit (int): Número máximo de imágenes a procesar
            video_limit (int): Número máximo de videos a procesar
            extract_features (bool): Si se deben extraer características utilizando un modelo pre-entrenado
        
        Returns:
            dict: Resultados del procesamiento
        """
        results = {}
        
        #Procesar imágenes
        logger.info("Iniciando procesamiento de imágenes...")
        img_df = self.preprocess_images(limit=image_limit)
        results['images_processed'] = len(img_df)
        
        #Procesar videos
        logger.info("Iniciando procesamiento de videos...")
        vid_df = self.preprocess_videos(limit=video_limit)
        results['videos_processed'] = len(vid_df)
        
        #Extraer características (opcional)
        if extract_features:
            logger.info("Iniciando extracción de características...")
            features = self.extract_features()
            results['features_extracted'] = {
                'images': len(features['images']),
                'videos': len(features['videos'])
            }
        
        logger.info("Preprocesamiento completo")
        return results

#Si se ejecuta como programa principal
if __name__ == "__main__":
    #Crear el preprocesador
    preprocessor = DataPreprocessor(image_size=(224, 224))
    
    #Ejecutar todo el proceso de preprocesamiento
    results = preprocessor.process_all(image_limit=1000, video_limit=500, extract_features=True)
    
    #Mostrar resultados
    print("Resultados del preprocesamiento:")
    for key, value in results.items():
        print(f"- {key}: {value}")