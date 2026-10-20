
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Charger le même modèle utilisé pour l'indexation
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Se connecter à la base Qdrant locale
client = QdrantClient(path="./qdrant_data")

# 3. Saisir une question
question = input("Pose ta question sur le PDF : ")

# 4. Transformer la question en embedding
question_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# 5. Rechercher les 3 passages les plus pertinents
results = client.query_points(
    collection_name="documents",
    query=question_embedding,
    limit=3,
    with_payload=True
).points

# 6. Afficher les résultats
if not results:
    print("Aucun passage trouvé.")

for i, result in enumerate(results, start=1):
    print(f"\n--- Résultat {i} ---")
    print(f"Page : {result.payload['page']}")
    print(f"Score : {result.score:.4f}")
    print(f"Texte : {result.payload['text']}")

client.close()
