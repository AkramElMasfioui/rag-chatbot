
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from pdf_processor import process_pdf
from uuid import uuid4

# 1. Lire et découper le PDF
chunks = process_pdf("documents/Demande-de-candidature-AkramElMasfioui.pdf")

if not chunks:
    raise ValueError("Aucun texte extrait du PDF.")

print(f"Nombre de chunks : {len(chunks)}")

# 2. Charger le modèle d'embeddings local
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"Dimension d'un embedding : {len(embeddings[0])}")

# 3. Créer une base Qdrant locale
client = QdrantClient(path="./qdrant_data")
collection = "documents"

# Recréer la collection pour ce premier test
if client.collection_exists(collection):
    client.delete_collection(collection)

client.create_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

# 4. Enregistrer les vecteurs et les textes
points = []

for chunk, embedding in zip(chunks, embeddings):
    points.append(
        models.PointStruct(
            id=str(uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "page": chunk["page"]
            }
        )
    )

client.upsert(
    collection_name=collection,
    points=points
)

print(f"{len(points)} chunks enregistrés dans Qdrant !")

client.close()
