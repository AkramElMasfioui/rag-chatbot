
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/upload", formData);
  return response.data;
}

export async function sendQuestion(documentId, question) {
  const response = await api.post("/chat", {
    document_id: documentId,
    question,
  });

  return response.data;
}
