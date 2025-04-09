🧠 Análisis de Imágenes y Videos para Campañas en Redes Sociales

Este proyecto permite procesar imágenes y videos, entrenar un modelo de IA y analizar contenido para redes sociales. A continuación se describen los pasos para instalar y ejecutar el sistema.

📁 Estructura del Proyecto
1. Crear una carpeta raíz con el nombre que desees.
2. Dentro de esa carpeta, crea una subcarpeta llamada proyecto_analisis_redes.
3. Dentro de proyecto_analisis_redes, clona el repositorio de GitHub o copia todo el contenido del proyecto.
4. Mueve los siguientes archivos a la carpeta raíz (la principal):
   
  Archivo app
  Archivo estructura
  Archivo requirements.txt
  
Tu estructura debe quedar así:
/MiCarpetaPrincipal
│
├── app.txt
├── estructura.txt
├── proyecto_analisis_redes/
│   └── (código clonado del repositorio)
├── requirements.txt

🐍 Crear entorno virtual (Windows)
1. Abre CMD o PowerShell en la carpeta principal.
2. Ejecuta:  python -m venv venv

🟢 Activar el entorno virtual
1. Desde PowerShell:                 .\venv\Scripts\Activate
2. Desde CMD:                         venv\Scripts\activate

📦 Instalar dependencias

Una vez activado el entorno virtual:  
                                    pip install -r requirements.txt

🍃 Instalar MongoDB (Windows)
1. Descarga el instalador desde [ https://www.mongodb.com/try/download/community](https://www.mongodb.com/docs/manual/installation/)
2. Instálalo con los valores por defecto.
3. Asegúrate de que esté seleccionado "Install MongoDB as a Service".
3. Verifica que MongoDB esté corriendo:
4. Abre CMD y escribe: mongod
  O busca "MongoDB Server" en servicios de Windows y asegúrate de que esté en ejecución.

🗂️ Inicializar la base de datos

Con el entorno virtual activo, y ubicado en la carpeta principal, ejecuta:

                                  py .\proyecto_analisis_redes\proyecto_analisis_redes\db\setup_db.py

🚀 Ejecutar la aplicación

Una vez creada la base de datos, ejecuta:
                                  py .\proyecto_analisis_redes\proyecto_analisis_redes\app\app.py
                                  
Esto iniciará la aplicación Flask. Deberás ver un mensaje como: Running on http://127.0.0.1:5000/



