Borrar entorno:
   rmdir .venv -Recurse -Force

Crear entorno:
   py -3.13 -m venv .venv

   .\.venv\Scripts\activate

Instaladores: 
   pip install -r requirements.txt


Entrenar:
   cd backend
   python .\train_and_export.py


Levantar la API (FastAPI + CORS):   
   python -m uvicorn main:app --reload --port 8000


Servir el frontend en otra terminal:
   cd frontend
   py -m http.server 5500

consultar en verify_probabilities:
   python verify_probabilities.py --kind origen --day 2 --hour 10

Navegador:
   http://127.0.0.1:8000/docs
   http://127.0.0.1:5500/solicitudes.html
