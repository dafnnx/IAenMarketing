#idea pero no esta implementada 
import os
import time
import random
import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import pymongo
from urllib.parse import quote

#Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger("social_scraper")

#Conexión a MongoDB
def connect_to_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["social_media_analytics"]
    return db

class SocialMediaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        })
        self.db = connect_to_mongodb()
        self.image_collection = self.db["images"]
        self.video_collection = self.db["videos"]
        
        #Crear directorio para guardar medios si no existe
        os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/images", exist_ok=True)
        os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/videos", exist_ok=True)
    
    def _random_delay(self):
        #aqui se introduce un retraso aleatorio entre 3 y 7 segundos para evitar bloqueos
        time.sleep(random.uniform(3, 7))
    
    def scrape_instagram_hashtag(self, hashtag, max_posts=100):
        """
        Obtiene publicaciones públicas de Instagram basadas en un hashtag
       Esta es una aproximación simplificada, Instagram limita el acceso no autorizado lol
        """
        logger.info(f"Iniciando scraping de Instagram con hashtag #{hashtag}")
        
        url = f"https://www.instagram.com/explore/tags/{quote(hashtag)}/"
        
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            #Buscar el contenido JSON en la página
            scripts = soup.find_all('script')
            data_script = None
            
            for script in scripts:
                if script.text and "window._sharedData" in script.text:
                    data_script = script.text
                    break
            
            if not data_script:
                logger.error("No se pudo encontrar datos en la página de Instagram")
                return []
            
            #Extraer el JSON
            json_text = data_script.split('window._sharedData = ')[1].split(';</script>')[0]
            data = json.loads(json_text)
            
            #Extraer posts
            posts = []
            try:
                hashtag_data = data['entry_data']['TagPage'][0]['graphql']['hashtag']
                edges = hashtag_data['edge_hashtag_to_media']['edges']
                
                count = 0
                for edge in edges:
                    if count >= max_posts:
                        break
                        
                    node = edge['node']
                    post = {
                        'id': node['id'],
                        'shortcode': node['shortcode'],
                        'timestamp': node['taken_at_timestamp'],
                        'likes': node['edge_liked_by']['count'],
                        'comments': node['edge_media_to_comment']['count'],
                        'caption': node.get('edge_media_to_caption', {}).get('edges', [{}])[0].get('node', {}).get('text', ''),
                        'is_video': node['is_video'],
                        'url': f"https://www.instagram.com/p/{node['shortcode']}/",
                        'thumbnail': node['thumbnail_src'],
                        'scraped_at': datetime.now(),
                    }
                    
                    #Guardar en la colección correspondiente
                    if post['is_video']:
                        self.video_collection.insert_one(post)
                    else:
                        self.image_collection.insert_one(post)
                        
                    #Descargar la miniatura
                    self._download_media(post['thumbnail'], post['id'], post['is_video'])
                    
                    posts.append(post)
                    count += 1
                    self._random_delay()
                
            except KeyError as e:
                logger.error(f"Error al extraer datos de Instagram: {e}")
            
            logger.info(f"Scraping de Instagram completado. Obtenidos {len(posts)} posts con hashtag #{hashtag}")
            return posts
            
        except Exception as e:
            logger.error(f"Error al acceder a Instagram: {e}")
            return []
    
    def scrape_facebook_page(self, page_name, max_posts=50):
        """
        Obtiene publicaciones públicas de una página de Facebook
        Nota: Esta es una aproximación simplificada, Facebook limita el acceso no autorizado
        """
        logger.info(f"Iniciando scraping de Facebook para la página: {page_name}")
        
        url = f"https://www.facebook.com/{page_name}/"
        
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            posts = []
            post_elements = soup.find_all('div', {'class': 'userContentWrapper'})
            
            count = 0
            for element in post_elements:
                if count >= max_posts:
                    break
                
                try:
                    #Intentar extraer texto
                    text_element = element.find('div', {'data-testid': 'post_message'})
                    text = text_element.text if text_element else ""
                    
                    #Intentar encontrar imágenes
                    images = element.find_all('img', {'class': 'scaledImageFitWidth'})
                    image_urls = [img.get('src') for img in images if img.get('src')]
                    
                    #Intentar encontrar videos
                    videos = element.find_all('video')
                    video_urls = [video.get('src') for video in videos if video.get('src')]
                    
                    #Extraer likes, comments, shares
                    reactions = element.find('span', {'class': '_3dlh'})
                    reaction_count = int(reactions.text.replace(',', '')) if reactions else 0
                    
                    #Crear objeto de post
                    post = {
                        'id': f"fb_{int(time.time())}_{count}",
                        'page': page_name,
                        'text': text,
                        'image_urls': image_urls,
                        'video_urls': video_urls,
                        'reaction_count': reaction_count,
                        'scraped_at': datetime.now(),
                    }
                    
                    #Guardar en MongoDB
                    if video_urls:
                        self.video_collection.insert_one(post)
                        #Descargar el primer video si existe
                        if video_urls[0]:
                            self._download_media(video_urls[0], post['id'], True)
                    elif image_urls:
                        self.image_collection.insert_one(post)
                        #Descargar la primera imagen si existe
                        if image_urls[0]:
                            self._download_media(image_urls[0], post['id'], False)
                    
                    posts.append(post)
                    count += 1
                
                except Exception as e:
                    logger.error(f"Error al procesar post de Facebook: {e}")
                
                self._random_delay()
            
            logger.info(f"Scraping de Facebook completado. Obtenidos {len(posts)} posts de la página {page_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error al acceder a Facebook: {e}")
            return []
    
    def _download_media(self, url, file_id, is_video):
        """Descarga imágenes o videos y los guarda localmente"""
        try:
            response = self.session.get(url, stream=True)
            if response.status_code == 200:
                if is_video:
                    file_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/videos/{file_id}.mp4"
                else:
                    file_path = f"proyecto_analisis_redes/proyecto_analisis_redes/data/images/{file_id}.jpg"
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                logger.info(f"Medio descargado: {file_path}")
            else:
                logger.error(f"Error al descargar medio. Código: {response.status_code}")
        except Exception as e:
            logger.error(f"Error al descargar medio: {e}")

if __name__ == "__main__":
    scraper = SocialMediaScraper()
    
    hashtags = ["marketing", "socialmedia", "digitalmarketing"]
    for hashtag in hashtags:
        scraper.scrape_instagram_hashtag(hashtag, max_posts=20)
        
    pages = ["facebook", "cocacola", "nike"]
    for page in pages:
        scraper.scrape_facebook_page(page, max_posts=10)