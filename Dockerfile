# Image officielle Python
FROM python:3.11-slim

# Définir le dossier de travail dans le container
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer dépendances
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port (si FastAPI)
EXPOSE 8000

# Commande de lancement API
CMD ["uvicorn", "src.mlops_tp.api:app", "--host", "0.0.0.0", "--port", "8000"]