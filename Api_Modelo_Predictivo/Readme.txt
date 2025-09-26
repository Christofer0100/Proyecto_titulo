0. Crear entorno:
   python -m venv .venv 


1. Activar el entorno virtual:
   .\.venv\Scripts\Activate
   .\.venv\Scripts\activate


2. Instalar dependencias (solo si cambió requirements.txt):
   pip install -r requirements.txt


3. Levantar el servidor FastAPI:
   uvicorn main:app --reload


4. Abrir en el navegador:
   - Swagger UI: http://127.0.0.1:8000/docs
   - Redoc:      http://127.0.0.1:8000/redoc





