
import json
import pymongo
import random
import os
from datetime import datetime, timedelta

#Conectar a MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["social_media_analytics"]
collection = db["posts"]

#Obtener el último post insertado
last_post = collection.find_one({}, sort=[("post_id", pymongo.DESCENDING)])

#Determinar el índice inicial
if last_post and "post_id" in last_post:
    last_index = int(last_post["post_id"].split("_")[1])  #Extraer número del post_id
else:
    last_index = -1  #Si no hay posts en la base, empieza desde 0


def generate_sample_posts(count=20):
    platforms = ["Facebook", "Instagram", "Twitter"]
    hashtags = ["#trending", "#viral", "#funny", "#meme", "#photooftheday", "#marketing", "#socialmedia", "#digitalmarketing", "#branding", "#content"]
    
    posts = []  #Lista para almacenar los nuevos posts

    for i in range(last_index + 1, last_index + 1 + count):
        engagement = random.randint(10, 10000)
        likes = random.randint(5, engagement)
        shares = engagement - likes

        post = {
            "post_id": f"post_{i}",
            "platform": random.choice(platforms),
            "content_type": random.choice(["image", "video"]),
            "caption": f"Sample post {i} with {random.choice(hashtags)}",
            "hashtags": [random.choice(hashtags) for _ in range(random.randint(1, 5))],
            "engagement": engagement,
            "likes": likes,
            "shares": shares,
            "date_posted": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "image_features": {
                "brightness": random.uniform(0.2, 0.9),
                "contrast": random.uniform(0.3, 0.8),
                "color_palette": ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(3)]
            }
        }
        posts.append(post)

    #Guardar en archivo JSON
    os.makedirs("proyecto_analisis_redes/proyecto_analisis_redes/data/sample", exist_ok=True)
    with open("proyecto_analisis_redes/proyecto_analisis_redes/data/sample/sample_posts.json", "w") as f:
        json.dump(posts, f, indent=4)

    #Insertar en la base de datos
    collection.insert_many(posts)

    print(f"Se generaron {count} nuevos posts desde post_{last_index + 1} hasta post_{last_index + count}.")

if __name__ == "__main__":
    generate_sample_posts()