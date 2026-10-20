from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_complete_rag_flow():
    # 1. Importer le PDF de test
    with open("documents/Demande-de-candidature-AkramElMasfioui.pdf", "rb") as pdf:
        upload_response = client.post(
            "/upload",
            files={
                "file": (
                    "Demande-de-candidature-AkramElMasfioui.pdf",
                    pdf,
                    "application/pdf"
                )
            }
        )

    assert upload_response.status_code == 200

    upload_data = upload_response.json()
    document_id = upload_data["document_id"]

    assert upload_data["chunks_indexed"] > 0

    # 2. Poser une question sur le document
    chat_response = client.post(
        "/chat",
        json={
            "document_id": document_id,
            "question": "Quel est le sujet principal du document ?"
        }
    )

    assert chat_response.status_code == 200

    chat_data = chat_response.json()

    assert chat_data["answer"]
    assert len(chat_data["sources"]) > 0

    print("Réponse :", chat_data["answer"])
    print("Sources :", chat_data["sources"])