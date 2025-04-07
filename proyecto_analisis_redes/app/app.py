from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys
import json
import numpy as np
import tensorflow as tf, keras
from pymongo import MongoClient
import pickle
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
import uuid
import threading
import time
import subprocess
import pickle
import cv2
import re 
from sklearn.cluster import KMeans


from bson.objectid import ObjectId
#sys.path.append(r"D:\Usuario\Escritorio\PROYECTO ESTADIAS\PROGRAMA\IA_MK\CODIGO")

#Obtener la ruta del directorio donde se encuentra el script actual para hacer unas pruebitas
script_dir = os.path.dirname(os.path.abspath(__file__))

#Navegar hacia arriba en el arbol de directorios segun sea necesario (como estoy desde app me salgo de esas hasta llegar a CODIGO que es desde donde se ejecuta )
project_root = os.path.dirname(os.path.dirname((os.path.dirname(script_dir))))

#Agregar al path
sys.path.append(project_root)




app = Flask(__name__, static_folder='static')
app.secret_key = 'clave_Secret_aqui'  #Es necesario para usar flash messages
training_thread = None
stop_training = False
#Configuracion
UPLOAD_FOLDER = 'uploads'
basedir = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  #64MB max
#Aqui nos aseguramos que la carpta de subidas existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#Conexión a MongoDB
def get_db_connection():
    try:
        with open("proyecto_analisis_redes/proyecto_analisis_redes/db/config/db_config.json", "r") as f:
            db_config = json.load(f)
        
        client = MongoClient(db_config["connection_string"])
        db = client[db_config["db_name"]]
        return db
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

#Cargar modelo AMPS
def load_model():
    try:
        #Definir ruta del modelo
        model_path = 'proyecto_analisis_redes/proyecto_analisis_redes/models/amps_model.h5'
        preproc_path = 'proyecto_analisis_redes/proyecto_analisis_redes/models/preprocessing_info.pkl'
        
        #Verificar si el modelo existe
        if not os.path.exists(model_path):
            app.logger.warning(f"Modelo no encontrado en {model_path}, usando modelo de respaldo")
            return None, None
        
        model = tf.keras.models.load_model(model_path)
        
        #Cargar informacion de preprocesamiento desde el archivo .pkl
        preprocessing_info = None
        if os.path.exists(preproc_path):
            with open(preproc_path, 'rb') as f:
                preprocessing_info = pickle.load(f)

        return model, preprocessing_info
    
    except Exception as e:
        app.logger.error(f"Error al cargar el modelo: {e}")
        return None, None

#Función para validar extensiones
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Extraer características basicas de imagen
def extract_image_features(file_path):
    """Extrae características de una imagen o video"""
    features = {}
    try:
        #Determinar tipo de archivo
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            #Procesar imagen
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {file_path}")
            
            #Convertir a RGB (OpenCV usa BGR por defecto)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #Calcular brillo promedio (0-1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:,:,2]) / 255.0
            features['brightness'] = float(brightness)
            
            #Calcular contraste
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std() / 255.0
            features['contrast'] = float(contrast)
            
            #Calcular saturacion
            saturation = np.mean(hsv[:,:,1]) / 255.0
            features['saturation'] = float(saturation)
            
            #Extraer paleta de colores dominantes (3 colores)
            resized = cv2.resize(img_rgb, (100, 100))  #Reducir tamaño para procesar mas rapido
            pixels = resized.reshape(-1, 3)
            
            #Usar K-means para identificar los colores dominantes
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(pixels)
            
            #Convertir los colores centroides a hexadecimal
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
                colors.append(hex_color.upper())
            
            features['color_palette'] = colors
            
        elif file_extension in ['mp4', 'mov']:
            #Para videos, extraer un frame y analizar
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {file_path}")
            
            #Leer el primer frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("No se pudo leer el primer frame del video")
            
            #Mismo análisis que para imagenes
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            brightness = np.mean(hsv[:,:,2]) / 255.0
            features['brightness'] = float(brightness)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = gray.std() / 255.0
            features['contrast'] = float(contrast)
            
            saturation = np.mean(hsv[:,:,1]) / 255.0
            features['saturation'] = float(saturation)
            
            #Extraer paleta de colores
            resized = cv2.resize(img_rgb, (100, 100))
            pixels = resized.reshape(-1, 3)
            
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(pixels)
            
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
                colors.append(hex_color.upper())
            
            features['color_palette'] = colors
            
            #Analizar duracion del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            features['duration'] = float(duration)
            
            #Liberar recursos
            cap.release()
            
    except Exception as e:
        app.logger.error(f"Error en extract_image_features: {str(e)}")
        #Devolver valores predeterminados
        features = {
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'color_palette': ['#888888', '#AAAAAA', '#CCCCCC']
        }
    
    return features


#Extraer características de texto
def extract_text_features(caption, hashtags):
    """Extrae características del texto usando NLP"""
    try:
        #Longitud normalizada del caption (0-1)
        caption_length = min(len(caption) / 1000, 1.0)
        
        #Numero de hashtags normalizado (0-1)
        hashtag_count = min(len(hashtags) / 30, 1.0)
        
        #Análisis simple de sentimiento (-1 a 1)
        positive_words = {'feliz', 'increíble', 'genial', 'maravilloso', 'excelente', 'bueno', 'mejor', 'amor', 'éxito'}
        negative_words = {'triste', 'terrible', 'malo', 'peor', 'odio', 'fracaso', 'problema'}
        
        words = caption.lower().split()
        positive_matches = sum(1 for word in words if word in positive_words)
        negative_matches = sum(1 for word in words if word in negative_words)
        
        sentiment = 0.0
        if positive_matches + negative_matches > 0:
            sentiment = (positive_matches - negative_matches) / (positive_matches + negative_matches)
        
        #Diversidad lexica (0-1)
        unique_words = len(set(words))
        lexical_diversity = min(unique_words / max(len(words), 1), 1.0)
        
        #Longitud promedio de palabras
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        avg_word_length_norm = min(avg_word_length / 15, 1.0)
        
        #Presencia de preguntas (0-1)
        questions = sum(1 for char in '?¿' if char in caption) > 0
        question_feature = 1.0 if questions else 0.0
        
        #Presencia de emojis (0-1)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  #emoticons
            u"\U0001F300-\U0001F5FF"  #símbolos & pictogramas
            u"\U0001F680-\U0001F6FF"  #transporte & símbolos
            u"\U0001F700-\U0001F77F"  #alchemical symbols
            u"\U0001F780-\U0001F7FF"  #Geometric Shapes
            u"\U0001F800-\U0001F8FF"  #Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  #Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  #Chess Symbols
            u"\U0001FA70-\U0001FAFF"  #Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  #Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        
        emojis = emoji_pattern.findall(caption)
        emoji_feature = min(len(emojis) / 10, 1.0)
        
        #Presencia de hashtags en el caption
        hashtags_in_caption = sum(1 for word in words if word.startswith('#'))
        hashtags_in_caption_norm = min(hashtags_in_caption / 10, 1.0)
        
        #Características finales
        features = np.array([
            caption_length,
            hashtag_count,
            sentiment,
            lexical_diversity,
            avg_word_length_norm,
            question_feature,
            emoji_feature,
            hashtags_in_caption_norm,
            0.0,  #Espacio para mas características
            0.0   #igualito
        ], dtype=np.float32)
        
        return features
        
    except Exception as e:
        app.logger.error(f"Error en extract_text_features: {str(e)}")
        #Devolver vector predeterminado
        return np.zeros(10, dtype=np.float32)

 #Calcular engagement segun la plataforma
def calculate_engagement(platform, likes, shares, follows, views):
    """Calcula la tasa de engagement según la plataforma"""
    if follows > 0:
        if platform == "Instagram":
            return round(((likes + shares + views) / follows) * 100, 2)
        elif platform == "Twitter":
            return round(((likes + shares + views) / follows) * 100, 2)
        elif platform == "Facebook":
            return round(((likes + shares + views) / follows) * 100, 2)
    
    return 0


#Rutas de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_content():
    if request.method == 'POST':
        #Verificar que se haya enviado un archivo
        if 'contentFile' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['contentFile']
        
        #Verificar que el archivo tenga un nombre
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        #Verificar que el archivo tenga una extensión permitida
        if not allowed_file(file.filename):
            flash('Tipo de archivo no permitido', 'danger')
            return redirect(request.url)
        
        try:
            #Guardar el archivo
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            #Ruta completa para guardar el archivo físicamente
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            image_features = extract_image_features(file_path)

            #Recopilar datos del formulario
            platform = request.form.get('platform')
            content_type = request.form.get('contentType')
            caption = request.form.get('caption')
            hashtags_text = request.form.get('hashtags', '')
            hashtags = [tag.strip() for tag in hashtags_text.split(',') if tag.strip()]
            
            likes = request.form.get('likes', 0)
            shares = request.form.get('shares', 0)
            follows = request.form.get('follows', 0)   
            views = request.form.get('views', 0)      
            
            try:
                likes = int(likes) if likes else 0
                shares = int(shares) if shares else 0
                follows = int(follows) if follows else 0
                views = int(views) if views else 0
            except ValueError:
                likes, shares, follows, views = 0, 0, 0, 0
           
            engagement_rate = calculate_engagement(platform, likes, shares, follows, views)

            #Crear documento para MongoDB
            post_data = {
                "post_id": f"post_{uuid.uuid4().hex[:8]}",
                "platform": platform,
                "content_type": content_type,
                "caption": caption,
                "hashtags": hashtags,
                "engagement": engagement_rate,
                "engagement_rate": engagement_rate,
                "likes": likes,
                "shares": shares,
                "follows": follows,
                "views": views,
                "date_posted": datetime.now().isoformat(),
                "file_path": unique_filename, 
                "image_features": image_features
            }
            
            #Guardar en la base de datos
            db = get_db_connection()
            if db is not None:
                result = db.posts.insert_one(post_data)
                if result.inserted_id:
                    flash(f'Contenido subido correctamente con ID: {post_data["post_id"]}', 'success')
                    
                    #Iniciar el procesamiento de datos en segundo plano
                    try:
                        #Llamada a scripts de procesamiento
                        subprocess.Popen(['python', 'proyecto_analisis_redes/proyecto_analisis_redes/scripts/preprocess_data.py'])
                        flash('Procesamiento de datos iniciado en segundo plano', 'info')
                    except Exception as e:
                        flash(f'Error al iniciar el procesamiento: {str(e)}', 'warning')
                    
                    return redirect(url_for('content_list'))
                else:
                    flash('Error al insertar en la base de datos', 'danger')
                    return redirect(request.url)
            else:
                flash('Error al conectar con la base de datos', 'danger')
                return redirect(request.url)
                
        except Exception as e:
            flash(f'Error al procesar la solicitud: {str(e)}', 'danger')
            return redirect(request.url)
    
    #Si es GET, mostrar el formulario
    return render_template('upload.html')

@app.route('/content')
def content_list():
    """Lista de todo el contenido subido"""
    db = get_db_connection()
    if db is None:
        flash('Error al conectar a la base de datos', 'danger')
        return render_template('content_list.html', posts=[])

    #Obtener todos los posts ordenados por fecha (mas recientes primero)
    posts = list(db.posts.find().sort('date_posted', -1))
    
    #Convertir ObjectId a string para renderizado
    for post in posts:
        if '_id' in post:
            post['_id'] = str(post['_id'])
    
    return render_template('content_list.html', posts=posts)

def generate_suggestions(engagement_score, image_features, text_features, meta_features, platform):
    """
    Genera sugerencias para mejorar el contenido basado en las características
    y la puntuación de engagement predicha.
    """
    suggestions = []
    
    #Sugerencias basadas en el score de engagement
    if engagement_score < 0.4:
        suggestions.append("El engagement predicho es bajo. Considera revisar tu estrategia de contenido.")
    elif engagement_score < 0.7:
        suggestions.append("El engagement predicho es medio. Con algunos ajustes puedes mejorar significativamente.")
    else:
        suggestions.append("¡Enhorabuena! El engagement predicho es alto. Este contenido tiene potencial viral.")
    
    #Sugerencias basadas en características de imagen
    if 'brightness' in image_features:
        brightness = image_features['brightness']
        if brightness < 0.4:
            suggestions.append("La imagen es oscura. Aumentar el brillo podría mejorar la visibilidad y el engagement.")
        elif brightness > 0.8:
            suggestions.append("La imagen es muy brillante. Ajusta el contraste para que los elementos destaquen mejor.")
    
    if 'contrast' in image_features:
        contrast = image_features['contrast']
        if contrast < 0.3:
            suggestions.append("Aumenta el contraste para hacer que los elementos de la imagen destaquen más.")
    
    #Sugerencias basadas en características de texto
    if meta_features and len(meta_features) >= 2:
        caption_length = meta_features[1] 
        
        if caption_length < 50:
            suggestions.append("La descripción es corta. Considera añadir más contexto para generar mayor engagement.")
        elif caption_length > 300:
            suggestions.append("La descripción es muy larga. Considera hacerla más concisa para mayor retención.")
    
    if meta_features and len(meta_features) >= 1:
        hashtag_count = meta_features[0]  #este es len(hashtags)
        
        if platform == 'Instagram':
            if hashtag_count < 5:
                suggestions.append("Añade más hashtags relevantes. En Instagram el ideal es entre 8-15 hashtags.")
            elif hashtag_count > 20:
                suggestions.append("Demasiados hashtags pueden parecer spam. Limita a 15 hashtags más relevantes.")
        elif platform == 'Twitter':
            if hashtag_count > 3:
                suggestions.append("En Twitter, limita tus hashtags a 1-3 de los más relevantes para mejor engagement.")
    
    #Sugerencias basadas en la plataforma
    if platform == 'Instagram':
        suggestions.append("Considera utilizar un carrusel de imágenes para aumentar el tiempo de visualización.")
    elif platform == 'Facebook':
        suggestions.append("Añade una pregunta o llamada a la acción para fomentar comentarios.")
    elif platform == 'Twitter':
        suggestions.append("Manten tu contenido conciso y actual para mayor engagement en Twitter.")
    
    #sw limita a 5 sugerencias para no abrumar al usuario
    return suggestions[:5]

@app.route('/dashboard')
def dashboard():
    """Generar datos para el dashboard"""
    db = get_db_connection()
    if db is None:
        flash('Error al conectar a la base de datos', 'danger')
        return redirect(url_for('index'))

    #Obtener estadisticas generales
    total_contents = db.analysis_results.count_documents({})
    positive_sentiments = db.analysis_results.count_documents({'sentiment': {'$gt': 0}})
    negative_sentiments = db.analysis_results.count_documents({'sentiment': {'$lt': 0}})

    #Hashtags mas utilizados
    hashtags = db.analysis_results.aggregate([
        {'$unwind': '$hashtags'},
        {'$group': {'_id': '$hashtags', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
        {'$limit': 10}
    ])
    top_hashtags = [(h['_id'], h['count']) for h in hashtags]

    #Sentimientos más recurrentes
    sentiment_data = db.analysis_results.aggregate([
        {'$group': {'_id': '$sentiment_label', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}}
    ])
    sentiment_labels = []
    sentiment_counts = []
    for s in sentiment_data:
        sentiment_labels.append(s['_id'])
        sentiment_counts.append(s['count'])

    #Engagement por contenido
    recent_contents = db.analysis_results.find().sort('timestamp', -1).limit(10)
    engagement_labels = [content['filename'] for content in recent_contents]
    engagement_data = [content['engagement_prediction'] * 100 for content in recent_contents]

    #Preparar datos para la plantilla
    stats = {
        'total_contents': total_contents,
        'positive_sentiments': positive_sentiments,
        'negative_sentiments': negative_sentiments,
        'top_hashtags': top_hashtags,
        'sentiment_labels': sentiment_labels,
        'sentiment_data': sentiment_counts,
        'engagement_labels': engagement_labels,
        'engagement_data': engagement_data
    }

    return render_template('dashboard.html', stats=stats)



#Nueva ruta para agregar datos de entrenamiento
@app.route('/training-data', methods=['GET', 'POST'])
def training_data():
    """Agregar datos para entrenar el modelo"""
    if request.method == 'POST':
        #Verificar que se haya enviado un archivo
        if 'contentFile' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['contentFile']
        
        #Verificar que el archivo tenga un nombre
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        #Verificar que el archivo tenga una extensión permitida
        if not allowed_file(file.filename):
            flash('Tipo de archivo no permitido', 'danger')
            return redirect(request.url)
        
        try:
            #Guardar el archivo
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            #Recopilar datos del formulario
            platform = request.form.get('platform')
            content_type = request.form.get('contentType')
            caption = request.form.get('caption')
            hashtags_text = request.form.get('hashtags', '')
            hashtags = [tag.strip() for tag in hashtags_text.split(',') if tag.strip()]
            
            #Valores reales de engagement para entrenamiento
            likes = request.form.get('likes', 0)
            shares = request.form.get('shares', 0)
            follows = request.form.get('follows', 0)  
            views = request.form.get('views', 0)     
            
            #Convertir a numeros
            try:
                likes = int(likes) if likes else 0
                shares = int(shares) if shares else 0
                follows = int(follows) if follows else 0
                views = int(views) if views else 0
            except ValueError:
                likes, shares, follows, views = 0, 0, 0, 0
            
            #Calcular engagement segun la plataforma
            engagement_rate = calculate_engagement(platform, likes, shares, follows, views)
            engagement = likes + shares  #Para mantener compatibilidad
            
            #Crear documento para processed_content (para entrenamiento)
            training_data = {
                "post_id": f"training_{uuid.uuid4().hex[:8]}",
                "platform": platform,
                "content_type": content_type,
                "caption": caption,
                "hashtags": hashtags,
                "engagement": engagement,
                "likes": likes,
                "shares": shares,
                "follows": follows,  
                "views": views,      
                "engagement_rate": engagement_rate,
                "date_added": datetime.now().isoformat(),
                "file_path": file_path,
                "image_features": list(image_features.values())[:3],
                "text_features": text_features.tolist(),
                "meta_features": meta_features
            }
            
            #Guardar en la base de datos
            db = get_db_connection()
            if db is not None:
                #Guardar en processed_content para entrenamiento
                result = db.processed_content.insert_one(training_data)
                
                if result.inserted_id:
                    flash('Datos de entrenamiento agregados correctamente', 'success')
                    
                    #Iniciar el procesamiento de datos en segundo plano
                    try:
                        subprocess.Popen(['python', 'proyecto_analisis_redes/proyecto_analisis_redes/scripts/preprocess_data.py'])
                        flash('Procesamiento de datos iniciado en segundo plano', 'info')
                    except Exception as e:
                        flash(f'Error al iniciar el procesamiento: {str(e)}', 'warning')
                    
                    return redirect(url_for('training_data'))
                else:
                    flash('Error al insertar en la base de datos', 'danger')
            else:
                flash('Error al conectar con la base de datos', 'danger')
                
        except Exception as e:
            flash(f'Error al procesar la solicitud: {str(e)}', 'danger')
            return redirect(request.url)
    
    #Si es GET, mostrar el formulario
    return render_template('training_data.html')

@app.route('/api/recent_analyses')
def recent_analyses():
    """API para obtener analisis recientes en formato JSON"""
    db = get_db_connection()
    if not db:
        return jsonify({'error': 'Error de conexión a la base de datos'}), 500
    
    recent = list(db.analysis_results.find().sort('timestamp', -1).limit(5))
    
    #Convertir ObjectId a string para serializacion JSON
    for item in recent:
        if '_id' in item:
            item['_id'] = str(item['_id'])
    
    return jsonify(recent)

@app.route('/documentation')
def documentation():
    """Pagina de documentación"""
    return render_template('documentation.html')

@app.route('/train', methods=['GET'])
def train_model_page1():
    """Iniciar el entrenamiento del modelo manualmente"""
    return render_template('training_data.html')

@app.route('/train', methods=['POST'])
def train_model_action():
    """Iniciar el entrenamiento del modelo"""
    try:
        #Llamada al script de entrenamiento
        subprocess.Popen(['python', 'proyecto_analisis_redes/proyecto_analisis_redes/scripts/train_model.py'])
        
        flash('Entrenamiento del modelo iniciado en segundo plano. Este proceso puede tardar varios minutos.', 'info')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Error al iniciar el entrenamiento: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))
##############################################33
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/content_detail/<post_id>')
def content_detail(post_id):
    """
    Muestra la pagina de detalle de un contenido especifico
    """
    db = get_db_connection()
    if db is None:
        flash('Error al conectar a la base de datos', 'danger')
        return redirect(url_for('content_list'))

    posts_collection = db['posts']

    #Obtener post de la base de datos
    post = posts_collection.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        flash('Contenido no encontrado', 'danger')
        return redirect(url_for('content_list'))
    
    #Debug: Verificar la existencia y valor de file_path
    print(f"File path: {post.get('file_path', 'No existe')}")
    
    #Convertir ObjectId a string para que sea serializable en el template
    post['_id'] = str(post['_id'])
    
    return render_template('content_detail.html', post=post)

@app.route('/delete_content', methods=['POST'])
def delete_content():
    """
    Elimina un contenido de la base de datos
    """
    post_id = request.form.get('post_id')
    
    if not post_id:
        flash('ID de contenido no proporcionado', 'danger')
        return redirect(url_for('content_list')) 
    
    db = get_db_connection()
    if db is None:
        flash('Error al conectar a la base de datos', 'danger')
        return redirect(url_for('content_list')) 
    posts_collection = db['posts']
    
    post = posts_collection.find_one({"_id": ObjectId(post_id)})
    
    if post:
        if 'file_path' in post:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], post['file_path'])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    flash('Archivo eliminado correctamente', 'success')
                except Exception as e:
                    flash(f'Error al eliminar el archivo: {e}', 'danger')
            else:
                flash('El archivo no existe, solo se eliminará el registro de la base de datos', 'warning')
        
        result = posts_collection.delete_one({"_id": ObjectId(post_id)})
        #aqui eliminamos el registro de la base de datos
        if result.deleted_count > 0:
            flash('Contenido eliminado correctamente', 'success')
        else:
            flash('Error al eliminar el contenido de la base de datos', 'danger')
    else:
        flash('Contenido no encontrado', 'danger')
    
    return redirect(url_for('content_list')) 


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    Endpoint para iniciar el preprocesamiento de datos
    """
    try:
        #Aqui se llama al script que realiza el preprocesamiento :|
        from proyecto_analisis_redes.proyecto_analisis_redes.scripts.preprocess_data import preprocess_data_for_model
        preprocess_data_for_model()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/train_model_page')
def train_model_page():
    """
    Página para mostrar el progreso del entrenamiento del modelo
    """
    return render_template('train_model.html')


@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """API endpoint para iniciar el entrenamiento del modelo"""
    global training_thread, stop_training
    
    #Si ya hay un entrenamiento en curso, devolver error
    if training_thread and training_thread.is_alive():
        return jsonify({"success": False, "error": "Ya hay un entrenamiento en curso"})
    
    #Obtener parametros del request
    data = request.json
    epochs = data.get('epochs', 10)
    batch_size = data.get('batch_size', 32)
    learning_rate = data.get('learning_rate', 0.001)
    validation_split = data.get('validation_split', 0.2)
    
    #Reiniciar la variable de control
    stop_training = False
    
    #Iniciar el entrenamiento en un hilo separado
    training_thread = threading.Thread(
        target=run_training,
        args=(epochs, batch_size, learning_rate, validation_split)
    )
    training_thread.start()
    
    return jsonify({"success": True, "message": "Entrenamiento iniciado"})

@app.route('/api/cancel_training', methods=['POST'])
def api_cancel_training():
    """API endpoint para cancelar el entrenamiento del modelo"""
    global stop_training
    
    if training_thread and training_thread.is_alive():
        stop_training = True
        return jsonify({"success": True, "message": "Solicitud de cancelación enviada"})
    else:
        return jsonify({"success": False, "error": "No hay entrenamiento en curso"})
    
def run_training(epochs, batch_size, learning_rate, validation_split):
    """Función que ejecuta el entrenamiento del modelo en un hilo separado"""
    global training_stats, stop_training
    
    try:
        #Actualizar estadisticas iniciales
        training_stats["total_epochs"] = epochs
        training_stats["current_epoch"] = 0
        training_stats["epoch_progress"] = 0
        training_stats["total_progress"] = 0
        training_stats["training_logs"] = []
        training_stats["metrics"] = {"mae": 0, "mse": 0, "accuracy": 0, "training_time": "00:00"}
        
        #Registrar tiempo de inicio
        start_time = time.time()
        
        #Importar funcion de entrenamiento
        from proyecto_analisis_redes.proyecto_analisis_redes.scripts.train_model import train_model
        
        #funcion callback para actualizar el progreso
        def progress_callback(epoch, current_batch, total_batches, logs=None):
            if stop_training:
                return True  #Detener el entrenamiento
                
            #Calcular progreso
            epoch_progress = int((current_batch / total_batches) * 100)
            total_progress = int(((epoch - 1 + current_batch / total_batches) / epochs) * 100)
            
            #Calcular tiempo restante
            elapsed_time = time.time() - start_time
            progress_fraction = (epoch - 1 + current_batch / total_batches) / epochs
            if progress_fraction > 0:
                estimated_total_time = elapsed_time / progress_fraction
                remaining_seconds = max(0, int(estimated_total_time - elapsed_time))
                minutes = remaining_seconds // 60
                seconds = remaining_seconds % 60
                remaining_time = f"{minutes:02d}:{seconds:02d}"
            else:
                remaining_time = "--:--"
            
            #Actualizar estadisticas globales
            global training_stats
            training_stats["current_epoch"] = epoch
            training_stats["epoch_progress"] = epoch_progress
            training_stats["total_progress"] = total_progress
            training_stats["remaining_time"] = remaining_time
            
            #informacion de logs si esta disponible
            if logs:
                log_message = f"Época {epoch}/{epochs} - Batch {current_batch}/{total_batches} - loss: {logs.get('loss', 'N/A'):.4f}"
                if 'val_loss' in logs:
                    log_message += f", val_loss: {logs['val_loss']:.4f}"
                if 'mae' in logs:
                    log_message += f", mae: {logs['mae']:.4f}"
                training_stats["training_logs"].append({
                    "message": log_message,
                    "type": "info",
                    "timestamp": time.strftime("%H:%M:%S")
                })
            
            #Print para debug
            print(f"Progress update: Epoch {epoch}/{epochs}, Batch {current_batch}/{total_batches}, Progress: {total_progress}%")
            
            return False  #Continuar el entrenamiento
        
        #Llamar a la funcion de entrenamiento real
        model, history = train_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            progress_callback=progress_callback
        )
        
        #Calcular metricas finales
        from proyecto_analisis_redes.proyecto_analisis_redes.scripts.evaluate_model import evaluate_model
        metrics = evaluate_model(model)
        
        #Calcular tiempo total de entrenamiento
        total_time = int(time.time() - start_time)
        minutes = total_time // 60
        seconds = total_time % 60
        formatted_time = f"{minutes:02d}:{seconds:02d}"
        
        #Actualizar metricas finales
        training_stats["metrics"] = {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "accuracy": metrics["accuracy"],
            "training_time": formatted_time
        }
        
        #Guardar el modelo
        model_path = os.path.join("proyecto_analisis_redes/proyecto_analisis_redes", "models", "amps_model.h5")
        model.save(model_path)
        
        #Registrar finalizacion
        training_stats["training_logs"].append({
            "message": "Entrenamiento completado con éxito",
            "type": "success",
            "timestamp": time.strftime("%H:%M:%S")
        })
        training_stats["training_logs"].append({
            "message": f"Modelo guardado en: {model_path}",
            "type": "success",
            "timestamp": time.strftime("%H:%M:%S")
        })
        
    except Exception as e:
        error_message = f"Error en el entrenamiento: {str(e)}"
        print(error_message)
        training_stats["training_logs"].append({
            "message": error_message,
            "type": "error",
            "timestamp": time.strftime("%H:%M:%S")
        })
@app.route('/api/get_training_progress', methods=['GET'])
def api_get_training_progress():
    """API endpoint para obtener el progreso del entrenamiento"""
    global training_stats, training_thread
    
    #Verificar si hay un entrenamiento en curso
    is_training = training_thread and training_thread.is_alive()
    
    #Incluir estado del entrenamiento en la respuesta
    response = {
        "success": True,
        "is_training": is_training,
        "data": training_stats
    }
    
    #Agregar debug info en la respuesta
    response["debug_info"] = {
        "thread_alive": training_thread.is_alive() if training_thread else False,
        "stats_updated": time.strftime("%H:%M:%S")
    }
    
    return jsonify(response)
#Funcion auxiliar para obtener estadisticas del dataset 
def get_dataset_statistics():
    """
    Obtener estadisticas reales del dataset desde la base de datos MongoDB.
    """
    #Conectar a la base de datos
    db = get_db_connection()
    if db is None:
        return {
            "error": "Error al conectar a la base de datos"
        }

    #Obtener los documentos procesados con caracteristicas
    posts_collection = db['posts']
    documents = list(posts_collection.find({"processed": 1, "features": {"$ne": None}}))

    #Calcular estadisticas
    total_examples = len(documents)
    validation_split = 0.2
    training_examples = int(total_examples * (1 - validation_split))
    validation_examples = total_examples - training_examples

    #Contar caracteristicas
    feature_count = 0
    if total_examples > 0 and 'features' in documents[0]:
        sample_features = documents[0]['features']
        if isinstance(sample_features, str): 
            sample_features = json.loads(sample_features)
        feature_count = len(sample_features) if isinstance(sample_features, list) else 0

    return {
        "total_examples": total_examples,
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "feature_count": feature_count,
        "feature_description": "Características combinadas (imagen, texto, metadatos)"
    }


@app.route('/api/get_dataset_info', methods=['GET'])
def api_get_dataset_info():
    """
    API endpoint para obtener información real del dataset desde la base de datos
    """
    try:
        #Importar funciones necesarias para acceder a los datos
        from proyecto_analisis_redes.proyecto_analisis_redes.scripts.data_utils import get_dataset_statistics
    
        #Obtener estadisticas reales del dataset
        stats = get_dataset_statistics()
        
        return jsonify({
            "success": True,
            "data": {
                "status": "Listo para entrenamiento" if stats["total_examples"] > 0 else "No hay suficientes datos",
                "total_examples": stats["total_examples"],
                "training_examples": stats["training_examples"],
                "validation_examples": stats["validation_examples"],
                "feature_count": stats["feature_count"],
                "feature_description": stats["feature_description"]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error al obtener información del dataset: {str(e)}"
        })
    

@app.route('/api/get_training_progressS', methods=['GET'])
def api_get_training_progressS():
    """
    API endpoint para obtener el progreso actual del entrenamiento
    """
    global training_stats
    
    if not training_thread or not training_thread.is_alive():
        return jsonify({
            "success": False, 
            "error": "No hay entrenamiento en curso"
        })
    
    return jsonify({
        "success": True,
        "data": training_stats
    })

#Definir una variable global para almacenar las estadisticas de entrenamiento
training_stats = {
    "current_epoch": 0,
    "total_epochs": 0,
    "epoch_progress": 0,
    "total_progress": 0,
    "remaining_time": "00:00",
    "training_logs": [],
    "metrics": {
        "mae": 0,
        "mse": 0,
        "accuracy": 0,
        "training_time": "00:00"
    }
}

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_content():
    if request.method == 'POST':
        #Verificar si es una solicitud AJAX
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        #Verificar si se envio un archivo
        if 'file' not in request.files:
            if is_ajax:
                return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'})
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        #Obtener el archivo del formulario
        file = request.files['file']
        if file.filename == '':
            if is_ajax:
                return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'})
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        #Verificar si el archivo tiene una extensión permitida
        if not allowed_file(file.filename):
            if is_ajax:
                return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'})
            flash('Tipo de archivo no permitido', 'danger')
            return redirect(request.url)
        
        try:
            #Guardar el archivo
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            #Extraer caracteristicas relevantes
            image_features = extract_image_features(file_path)
            
            #Obtener datos del formulario
            platform = request.form.get('platform', 'Instagram')
            caption = request.form.get('caption', '')
            hashtags_text = request.form.get('hashtags', '')
            hashtags = [tag.strip() for tag in hashtags_text.split(',') if tag.strip()]
            account = request.form.get('account', '')
            
            #Debug: Imprimir valores recibidos
            app.logger.info(f"Platform: {platform}, Caption length: {len(caption)}, Hashtags: {len(hashtags)}")
            app.logger.info(f"Image features: {image_features}")
            
            #Extraer caracteristicas de texto
            text_features = extract_text_features(caption, hashtags)
            
            #Metadatos para el modelo
            meta_features = [min(len(hashtags), 30), min(len(caption), 1000), 0]
            
            #Usar el modelo de ML para la predicción
            engagement_score = 0.0
            try:
                #Cargar modelo
                model, preprocessing_info = load_model()
                
                #Preparar entradas para el modelo
                model_inputs = prepare_model_inputs(image_features, text_features, meta_features)
                
                #Realizar predicción
                prediction = model.predict(model_inputs)
                engagement_score = float(prediction[0][0])
                app.logger.info(f"Predicción exitosa: {engagement_score}")
            except Exception as model_error:
                app.logger.error(f"Error en la predicción del modelo: {str(model_error)}")
                #Implementar un modelo de respaldo basado en reglas
                engagement_score = fallback_prediction(image_features, text_features, hashtags, platform)
                app.logger.info(f"Usando predicción de respaldo: {engagement_score}")
            
            #Asegurar que el score este entre 0 y 1
            engagement_score = max(0.0, min(1.0, engagement_score))
            
            #Generar sugerencias basadas en IA
            suggestions = generate_ai_suggestions(
                engagement_score,
                image_features,
                caption,
                hashtags,
                platform,
                account
            )
            
            #Analisis avanzado para el dashboard
            advanced_analysis = perform_advanced_analysis(
                file_path, image_features, caption, hashtags, platform, engagement_score
            )
            
            #Preparar resultado del análisis
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'file_path': unique_filename,
                'platform': platform,
                'caption': caption,
                'hashtags': hashtags,
                'account': account,
                'engagement_prediction': engagement_score,
                'suggestions': suggestions,
                'image_features': image_features,
                'advanced_analysis': advanced_analysis
            }
            
            #Guardar resultado en la base de datos
            result_id = save_analysis_to_db(analysis_result)
            
            #Responder según el tipo de solicitud
            if is_ajax:
                return jsonify({
                    'success': True,
                    'engagement_prediction': engagement_score,
                    'suggestions': suggestions,
                    'result_id': result_id,
                    'redirect_url': url_for('analysis_results', result_id=result_id)
                })
            
            flash('Análisis completado con éxito', 'success')
            return redirect(url_for('analysis_results', result_id=result_id))
            
        except Exception as e:
            app.logger.error(f"Error general en analyze_content: {str(e)}")
            if is_ajax:
                return jsonify({'success': False, 'error': f'Error durante el análisis: {str(e)}'})
            flash(f'Error durante el análisis: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html')


@app.route('/temp_results/<result_id>')
def temp_results(result_id):
    """Punto de entrada temporal para resultados cuando no hay BD"""
    #Generar resultados de ejemplo
    engagement_score = 0.6  #Valor medio
    suggestions = [
        "Considera publicar en horarios de mayor actividad",
        "Añade una pregunta para fomentar comentarios",
        "Usa hashtags más específicos para tu nicho"
    ]
    
    return render_template(
        'analysis_results.html',
        result={
            '_id': result_id,
            'engagement_prediction': engagement_score,
            'suggestions': suggestions,
            'image_features': {'color_palette': []},
            'advanced_analysis': {'visual_elements': [], 'sentiment_analysis': {}, 'trending_factors': []}
        },
        engagement_level=get_engagement_level(engagement_score),
        color_palette=[]
    )

def inspect_model_input_shapes(model):
    """Imprime la forma esperada de los inputs del modelo"""
    print("Modelo input shapes:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'input_shape'):
            print(f"Layer {i}, name: {layer.name}, input shape: {layer.input_shape}")

#Llamar a esta función despues de cargar el modelo
model, preprocessing_info = load_model()
inspect_model_input_shapes(model)
#2. Verificar los datos antes de convertirlos a arrays
def print_data_info(image_features, text_features, meta_features):
    """Imprime información sobre los datos antes de procesarlos"""
    print(f"Image features: {image_features} (type: {type(image_features)})")
    print(f"Text features: {text_features} (type: {type(text_features)})")
    print(f"Meta features: {meta_features} (type: {type(meta_features)})")
    
    #Verificar si alguno de los valores no es numerico
    for i, val in enumerate(list(image_features.values())[:3]):
        if not isinstance(val, (int, float)) or np.isnan(val):
            print(f"Problema en image_features[{i}]: {val} (tipo: {type(val)})")
    
    for i, val in enumerate(text_features):
        if not isinstance(val, (int, float)) or np.isnan(val):
            print(f"Problema en text_features[{i}]: {val} (tipo: {type(val)})")
    
    for i, val in enumerate(meta_features):
        if not isinstance(val, (int, float)) or np.isnan(val):
            print(f"Problema en meta_features[{i}]: {val} (tipo: {type(val)})")

#3. Alternativa: usar un solo array plano en lugar de múltiples inputs
def alternative_prediction(model, image_features, text_features, meta_features):
    """Intenta una predicción alternativa combinando todos los features en un solo array"""
    #Extraer y convertir image_features a una lista de floats
    img_feats = [float(val) if val is not None else 0.0 for val in list(image_features.values())[:3]]
    
    #Asegurar que text_features sea una lista plana de floats
    txt_feats = [float(val) if val is not None else 0.0 for val in text_features]
    
    #Asegurar que meta_features sea una lista plana de floats
    meta_feats = [float(val) if val is not None else 0.0 for val in meta_features]
    
    #Combinar todos los features en un solo array plano
    combined_features = np.array([img_feats + txt_feats + meta_feats], dtype=float)
    print(f"Combined features shape: {combined_features.shape}")
    
    #Si el modelo espera un solo input, usar esto
    try:
        return model.predict(combined_features)
    except Exception as e:
        print(f"Error con combined_features: {e}")
        return None
 
@app.route('/analysis_results/<result_id>')
def analysis_results(result_id):
    """Mostrar los resultados de un analisis específico"""
    app.logger.info(f"Accediendo a resultados con ID: {result_id}")
    
    db = get_db_connection()
    if db is None:
        flash('Error al conectar a la base de datos', 'danger')
        return redirect(url_for('index'))
    
    #Obtener el resultado del análisis
    try:
        result = db.analysis_results.find_one({"_id": ObjectId(result_id)})
        if not result:
            flash('Resultado de análisis no encontrado', 'danger')
            return redirect(url_for('index'))
        
        #Convertir ObjectId a string para el template si es necesario
        if isinstance(result['_id'], ObjectId):
            result['_id'] = str(result['_id'])
        
        #Preparar datos para visualización
        engagement_level = get_engagement_level(result['engagement_prediction'])
        color_palette = result['image_features'].get('color_palette', [])
        
        return render_template(
            'analysis_results.html',
            result=result,
            engagement_level=engagement_level,
            color_palette=color_palette
        )
    except Exception as e:
        app.logger.error(f"Error al obtener los resultados: {str(e)}")
        flash(f'Error al obtener los resultados: {str(e)}', 'danger')
        return redirect(url_for('index'))

#3. Nuevas funciones para el analisis avanzado
def perform_advanced_analysis(file_path, image_features, caption, hashtags, platform, engagement_score):
    """Realiza un analisis avanzado del contenido"""
    #Determinar el tipo de archivo
    file_extension = file_path.split('.')[-1].lower()
    
    analysis_results = {
        'visual_elements': [],
        'sentiment_analysis': {},
        'color_analysis': {},
        'trending_factors': []
    }
    
    #Analisis visual (simulado)
    if file_extension in ['jpg', 'jpeg', 'png']:
        #Analisis de imagen
        analysis_results['visual_elements'] = [
            'personas' if np.random.random() > 0.5 else None,
            'paisaje' if np.random.random() > 0.5 else None,
            'producto' if np.random.random() > 0.5 else None,
            'texto' if np.random.random() > 0.5 else None
        ]
        analysis_results['visual_elements'] = [e for e in analysis_results['visual_elements'] if e]
        
    elif file_extension in ['mp4', 'mov']:
        #Análisis de video
        analysis_results['visual_elements'] = [
            'personas' if np.random.random() > 0.5 else None,
            'movimiento' if np.random.random() > 0.5 else None,
            'música' if np.random.random() > 0.5 else None,
            'texto' if np.random.random() > 0.5 else None
        ]
        analysis_results['visual_elements'] = [e for e in analysis_results['visual_elements'] if e]
        
        #Analisis de audio (simulado para videos)
        analysis_results['audio_analysis'] = {
            'tempo': f"{np.random.randint(60, 160)} BPM",
            'mood': np.random.choice(['energético', 'relajado', 'emotivo', 'neutral'])
        }
    
    #Analisis de sentimiento del texto
    if caption:
        positive_words = ['increíble', 'fantástico', 'feliz', 'amor', 'genial', 'mejor', 'éxito']
        negative_words = ['mal', 'problema', 'triste', 'difícil', 'odio', 'peor']
        
        #Simulación simple de análisis de sentimiento
        positive_count = sum(1 for word in caption.lower().split() if word in positive_words)
        negative_count = sum(1 for word in caption.lower().split() if word in negative_words)
        
        total = positive_count + negative_count
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
            sentiment_score = max(min(sentiment_score, 1), -1)  #Normalizar entre -1 y 1
        else:
            sentiment_score = 0
        
        #Mapear a categoria
        if sentiment_score > 0.3:
            sentiment = 'positivo'
        elif sentiment_score < -0.3:
            sentiment = 'negativo'
        else:
            sentiment = 'neutral'
        
        analysis_results['sentiment_analysis'] = {
            'score': sentiment_score,
            'category': sentiment,
            'strength': abs(sentiment_score)
        }
    
    #Análisis de colores
    if 'color_palette' in image_features:
        colors = image_features['color_palette']
        bright_colors = any(is_bright_color(color) for color in colors)
        
        analysis_results['color_analysis'] = {
            'palette': colors,
            'bright_colors': bright_colors,
            'dominant_tone': get_dominant_tone(colors)
        }
    
    #Tendencias actuales (simulado)
    platform_trends = {
        'Instagram': ['reels cortos', 'estilo auténtico', 'tonos pastel', 'contenido interactivo'],
        'Facebook': ['videos largos', 'contenido informativo', 'colores vibrantes', 'llamadas a la acción'],
        'Twitter': ['contenido conciso', 'referencias culturales', 'formato simple', 'humor']
    }
    
    #Identificar tendencias que coinciden con el contenido
    if platform in platform_trends:
        trends = platform_trends[platform]
        matching_trends = []
        
        #Simular coincidencias con tendencias
        for trend in trends:
            if np.random.random() > 0.6:  #40% de probabilidad de coincidencia
                matching_trends.append(trend)
        
        analysis_results['trending_factors'] = matching_trends
    
    return analysis_results

def is_bright_color(hex_color):
    """Determina si un color hex es brillante"""
    #Eliminar el #si existe
    hex_color = hex_color.lstrip('#')
    
    #Convertir a RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    #Calcular luminosidad (formula ponderada)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return luminance > 0.65

def get_dominant_tone(colors):
    """Determina el tono dominante de una paleta de colores"""
    #Simplificacion: solo comprueba el primer color
    if not colors:
        return "neutro"
    
    hex_color = colors[0].lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    #Clasificacion simple de tonos
    if r > g and r > b:
        return "cálido"
    elif b > r and b > g:
        return "frío"
    elif g > r and g > b:
        return "natural"
    else:
        return "neutro"

def get_engagement_level(score):
    """Convierte un score numérico a un nivel de engagement"""
    if score < 0.3:
        return {"level": "bajo", "class": "danger"}
    elif score < 0.6:
        return {"level": "medio", "class": "warning"}
    else:
        return {"level": "alto", "class": "success"}


def generate_ai_suggestions(engagement_score, image_features, caption, hashtags, platform, account=''):
    """Genera sugerencias inteligentes basadas en análisis de contenido"""
    suggestions = []
    
    #Análisis basado en plataforma
    platform_suggestions = {
        'Instagram': [
            "Usa entre 5-10 hashtags específicos para maximizar el alcance",
            "Las imágenes con personas suelen tener un 38% más de engagement",
            "Incluye una llamada a la acción clara para aumentar interacciones",
            "Considera usar un carrusel de imágenes para aumentar el tiempo de visualización",
            "Publica en horarios de mayor actividad: entre 11am-1pm o 7pm-9pm"
        ],
        'Facebook': [
            "Los videos cortos (menos de 60 segundos) generan más interacción",
            "Incluye una pregunta al final para aumentar comentarios",
            "Las publicaciones con links suelen tener menos alcance, considera usar imágenes",
            "El contenido informativo o educativo funciona mejor que el promocional",
            "Utiliza historias para promocionar esta publicación"
        ]
    }
    
    #1. Analizar puntos debiles basados en score de engagement
    if engagement_score < 0.4:
        if platform in platform_suggestions:
            #Seleccionar 2 sugerencias aleatorias para la plataforma
            platform_sugs = np.random.choice(
                platform_suggestions.get(platform, []), 
                size=min(2, len(platform_suggestions.get(platform, []))),
                replace=False
            )
            suggestions.extend(platform_sugs)
        
        #Análisis especifico de elementos debiles
        if 'brightness' in image_features and image_features['brightness'] < 0.3:
            suggestions.append("Aumenta el brillo de tu imagen para mejorar la visibilidad en feeds móviles")
        
        if not caption or len(caption) < 50:
            suggestions.append("Añade una descripción más detallada para generar mayor conexión con tu audiencia")
        
        if len(hashtags) == 0:
            suggestions.append(f"Incluye hashtags relevantes para aumentar la visibilidad en {platform}")
    
    #2. Análisis de imagen
    if 'color_palette' in image_features and image_features['color_palette']:
        colors = image_features['color_palette']
        bright_colors = any(is_bright_color(color) for color in colors)
        
        if bright_colors and engagement_score < 0.6:
            suggestions.append("Considera usar una paleta de colores más contrastante para captar mayor atención")
    
    #3. Análisis de texto
    if caption:
        if "?" not in caption and engagement_score < 0.7:
            suggestions.append("Añade una pregunta relevante al final para fomentar comentarios")
        
        #Detectar emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  #emoticons
            u"\U0001F300-\U0001F5FF"  #simbolos & pictogramas
            u"\U0001F680-\U0001F6FF"  #transporte & simbolos
            u"\U0001F700-\U0001F77F"  #alchemical symbols
            u"\U0001F780-\U0001F7FF"  #Geometric Shapes
            u"\U0001F800-\U0001F8FF"  #Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  #Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  #Chess Symbols
            u"\U0001FA70-\U0001FAFF"  #Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  #Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        
        has_emojis = bool(emoji_pattern.search(caption))
        if not has_emojis:
            suggestions.append("Considera incluir emojis relevantes para hacer tu texto más atractivo")
    
    #4. Sugerencias especificas para cuenta/marca
    if account:
        #Simulación de análisis de cuenta (en producción esto vendria de datos históricos)
        if platform == 'Instagram':
            suggestions.append(f"Basado en tu historial, el contenido de tipo {get_best_content_type(account)} genera más engagement para tu cuenta")
    
    #5. Análisis de tendencias (simulado - en produccion vendria de datos reales)
    trending_suggestion = get_trending_suggestion(platform)
    if trending_suggestion:
        suggestions.append(trending_suggestion)
    
    #Asegurar que no haya duplicados y limitar a máximo 4 sugerencias
    unique_suggestions = list(set(suggestions))
    return unique_suggestions[:4]

def get_best_content_type(account):
    """Simula análisis de mejor tipo de contenido para una cuenta"""
    #En producción, esto vendria de análisis históricos
    content_types = ["carousel", "video corto", "imagen con texto", "imagen con personas"]
    return np.random.choice(content_types)

def get_trending_suggestion(platform):
    """Simula sugerencias basadas en tendencias actuales"""
    trending_content = {
        'Instagram': [
            "Las publicaciones con tonos pastel están teniendo un 22% más de engagement este mes",
            "Los Reels cortos (menos de 15 segundos) están generando más seguidores nuevos",
            "Contenido que muestra procesos behind-the-scenes está teniendo mayor retención"
        ],
        'Facebook': [
            "Las transmisiones en vivo están generando 3x más engagement que el contenido grabado",
            "Contenido con formato de storytelling está incrementando el tiempo de visualización",
            "Las imágenes que generan nostalgia están teniendo más shares este mes"
        ]
    }
    
    if platform in trending_content:
        return np.random.choice(trending_content[platform])
    return None



def prepare_model_inputs(image_features, text_features, meta_features):
    """Prepara correctamente los inputs para el modelo"""
    #Extraer caracteristicas numericas de la imagen
    img_features = []
    for key in ['brightness', 'contrast', 'saturation']:
        if key in image_features and isinstance(image_features[key], (int, float)):
            img_features.append(float(image_features[key]))
        else:
            img_features.append(0.0)  #Valor predeterminado
    
    #Extraer caracteristicas adicionales de color si es necesario
    if 'color_palette' in image_features and isinstance(image_features['color_palette'], list):
        #Convertir colores hexadecimales a valores RGB normalizados
        for color in image_features['color_palette'][:2]:  #Tomar solo los 2 primeros colores
            #Convertir hex a RGB y normalizar
            try:
                hex_color = color.lstrip('#')
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                img_features.extend([r, g, b])
            except:
                #Si hay error, añadir ceros
                img_features.extend([0.0, 0.0, 0.0])
    
    #Asegurar que tenemos exactamente 10 características de imagen 
    while len(img_features) < 10:
        img_features.append(0.0)
    img_features = img_features[:10]  #Truncar si hay demasiadas
    
    #Asegurar que text_features tenga el tamaño correcto
    text_feats = list(text_features)
    while len(text_feats) < 10:  
        text_feats.append(0.0)
    text_feats = text_feats[:10]  #Truncar si hay demasiadas
    
   #Asegurar que meta_features tenga exactamente 5 valores
    while len(meta_features) < 5:
        meta_features.append(0.0)  #Rellenar con ceros
    meta_features = meta_features[:5]  #Truncar si hay más de 5 valores

    #Convertir a arrays numpy con las formas correctas
    image_input = np.array([list(image_features.values())[:3]], dtype=np.float32)
    text_input = np.array([text_features], dtype=np.float32)
    meta_input = np.array([meta_features], dtype=np.float32)

    return [image_input, text_input, meta_input]


def fallback_prediction(image_features, text_features, hashtags, platform):
    """Modelo de respaldo basado en reglas cuando falla el modelo de ML"""
    score = 0.5  #Punto medio
    
    #Ajustes basados en características de imagen
    if 'brightness' in image_features:
        #Las imágenes con brillo medio-alto suelen tener mejor engagement
        brightness = image_features['brightness']
        if 0.4 <= brightness <= 0.7:
            score += 0.05
        elif brightness > 0.9:
            score -= 0.03  #Demasiado brillante
    
    if 'contrast' in image_features:
        #Mayor contraste suele ser mejor
        contrast = image_features['contrast']
        if contrast > 0.5:
            score += 0.04
    
    #Ajustes basados en hashtags
    if platform == 'Instagram':
        #En Instagram, más hashtags (hasta un punto) mejoran alcance
        if 5 <= len(hashtags) <= 15:
            score += 0.07
        elif len(hashtags) > 15:
            score -= 0.02  #Demasiados hashtags
    elif platform == 'Twitter':
        #En Twitter, menos hashtags funcionan mejor
        if 1 <= len(hashtags) <= 2:
            score += 0.05
        elif len(hashtags) > 3:
            score -= 0.03
    
    #Ajustes basados en texto
    caption_length = len(text_features) > 0 and text_features[0] or 0
    sentiment = len(text_features) > 2 and text_features[2] or 0
    
    #Longitud de caption moderada suele funcionar mejor
    if 0.2 <= caption_length <= 0.6:
        score += 0.04
    
    #Sentimiento positivo suele tener mejor engagement
    if sentiment > 0.3:
        score += 0.06
    elif sentiment < -0.3:
        score -= 0.03
    
    #Limitar el score final entre 0.1 y 0.9
    return max(0.1, min(0.9, score))

def save_analysis_to_db(analysis_result):
    """Guarda el resultado del análisis en la base de datos"""
    try:
        db = get_db_connection()
        if db is not None:
            #Convertir numpy arrays a listas para MongoDB
            if 'text_features' in analysis_result and isinstance(analysis_result['text_features'], np.ndarray):
                analysis_result['text_features'] = analysis_result['text_features'].tolist()
            
            #Insertar en la base de datos
            result_id = db.analysis_results.insert_one(analysis_result).inserted_id
            return str(result_id)
        else:
            #Si no hay conexion a DB, generar un ID temporal
            return str(uuid.uuid4())
    except Exception as e:
        app.logger.error(f"Error al guardar en base de datos: {str(e)}")
        return str(uuid.uuid4())
if __name__ == '__main__':
    app.run(debug=True)