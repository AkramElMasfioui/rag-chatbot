import { useState, useRef, useEffect } from "react";
import { uploadPDF, sendQuestion } from "./services/api";
import "./App.css";

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function App() {
  const [file, setFile] = useState(null);
  const [documentId, setDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  function pickFile(selected) {
    if (!selected) return;
    if (selected.type !== "application/pdf" && !selected.name.toLowerCase().endsWith(".pdf")) {
      setError("Le fichier doit être un PDF.");
      return;
    }
    setFile(selected);
    setDocumentId("");
    setMessages([]);
    setError("");
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    pickFile(event.dataTransfer.files?.[0]);
  }

  async function handleUpload() {
    if (!file) {
      setError("Sélectionnez d'abord un fichier PDF.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await uploadPDF(file);
      setDocumentId(data.document_id);
      setMessages([
        {
          role: "assistant",
          text: `Le document « ${data.filename} » est prêt. Posez votre première question.`,
        },
      ]);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Impossible d'importer le PDF. Vérifiez que le backend fonctionne."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleSend(event) {
    event.preventDefault();

    const currentQuestion = question.trim();

    if (!documentId) {
      setError("Importez un PDF avant de poser une question.");
      return;
    }

    if (!currentQuestion || loading) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", text: currentQuestion },
    ]);
    setQuestion("");
    setLoading(true);
    setError("");

    try {
      const data = await sendQuestion(documentId, currentQuestion);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Une erreur est survenue pendant la réponse."
      );
    } finally {
      setLoading(false);
    }
  }

  const ready = Boolean(documentId);

  return (
    <div className="shell">
      <aside className="sidebar">
        <header className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 3h7l5 5v13H7z" />
              <path d="M14 3v5h5" />
              <path d="M10 13h6M10 17h4" />
            </svg>
          </div>
          <div>
            <h1>RAG Chatbot</h1>
            <p>Discutez avec vos documents</p>
          </div>
        </header>

        <section className="doc" aria-label="Document">
          <h2>Document</h2>

          {!file ? (
            <div
              className={`dropzone ${dragging ? "is-over" : ""}`}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
            >
              <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 16V4M7 9l5-5 5 5" />
                <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
              </svg>
              <strong>Déposez un PDF ici</strong>
              <span>ou cliquez pour parcourir</span>
            </div>
          ) : (
            <div className={`file-card ${ready ? "is-ready" : ""}`}>
              <div className="file-icon" aria-hidden="true">PDF</div>
              <div className="file-meta">
                <strong title={file.name}>{file.name}</strong>
                <span>
                  {formatSize(file.size)} ·{" "}
                  {ready ? "Prêt pour vos questions" : "Pas encore importé"}
                </span>
              </div>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            hidden
            onChange={(event) => {
              pickFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />

          {file && (
            <div className="doc-actions">
              {!ready && (
                <button className="btn primary" onClick={handleUpload} disabled={loading}>
                  {loading ? "Analyse en cours…" : "Importer le PDF"}
                </button>
              )}
              <button
                className="btn ghost"
                onClick={() => inputRef.current?.click()}
                disabled={loading}
              >
                Changer de fichier
              </button>
            </div>
          )}
        </section>
      </aside>

      <main className="chat">
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              <h2>Posez vos questions à votre PDF</h2>
              <p>
                Importez un document dans le panneau de gauche. Les réponses
                indiquent les pages d'où elles viennent.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${index}-${message.role}`}>
              <div className="bubble">
                <p>{message.text}</p>

                {message.sources?.length > 0 && (
                  <div className="sources">
                    <span className="sources-title">Sources dans le document</span>
                    {message.sources.map((source, sourceIndex) => (
                      <details key={sourceIndex}>
                        <summary>
                          <span className="page-chip">p. {source.page}</span>
                          <span className="source-preview">{source.text}</span>
                        </summary>
                        <p>{source.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}

          {loading && ready && (
            <div className="message assistant">
              <div className="bubble typing" aria-label="Réponse en cours">
                <i></i><i></i><i></i>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <footer className="composer">
          {error && <p className="error" role="alert">{error}</p>}
          <form className="question-form" onSubmit={handleSend}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={ready ? "Posez une question sur votre PDF" : "Importez un PDF pour commencer"}
              disabled={!ready || loading}
            />
            <button
              className="btn primary send"
              type="submit"
              disabled={!ready || loading || !question.trim()}
            >
              Envoyer
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

export default App;
