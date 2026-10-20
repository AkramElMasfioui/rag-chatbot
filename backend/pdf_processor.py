
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys


def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            chunks.append({
                "page": page_number,
                "text": chunk
            })

    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilisation : python pdf_processor.py chemin_du_pdf.pdf")
        sys.exit(1)

    chunks = process_pdf(sys.argv[1])

    print(f"Nombre de chunks : {len(chunks)}")

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} | Page {chunk['page']} ---")
        print(chunk["text"][:300])
