
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "RAG Chatbot API opérationnelle"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "document.txt",
                b"ceci n'est pas un PDF",
                "text/plain"
            )
        }
    )
    assert response.status_code == 400


def test_chat_rejects_empty_question():
    response = client.post(
        "/chat",
        json={
            "document_id": "document-inexistant",
            "question": "   "
        }
    )
    assert response.status_code == 400
