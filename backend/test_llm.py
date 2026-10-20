import os
from dotenv import load_dotenv
from google import genai

# Charger la clé API depuis .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY introuvable. Vérifie le fichier .env."
    )

# Connexion à Gemini
client = genai.Client(api_key=api_key)

# Envoyer une question
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explique le RAG en français simple en trois phrases."
)

print("\nRéponse de Gemini :")
print(response.text)
