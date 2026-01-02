import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

# Google Document AI
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# Embedding + Chroma
from sentence_transformers import SentenceTransformer
import chromadb


# ----------------------------
# Text extraction helpers
# ----------------------------
def _get_text_from_anchor(doc: documentai.Document, text_anchor: documentai.Document.TextAnchor) -> str:
    """Extract text from Document.text using TextAnchor segments."""
    if not text_anchor.text_segments:
        return ""
    pieces = []
    full = doc.text or ""
    for seg in text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index is not None else 0
        end = int(seg.end_index) if seg.end_index is not None else 0
        if 0 <= start < end <= len(full):
            pieces.append(full[start:end])
    return "".join(pieces)


def doc_to_paragraph_text(doc: documentai.Document) -> str:
    """
    Prefer paragraphs (keeps line/paragraph structure better).
    Fallback to doc.text.
    """
    out_lines = []
    if doc.pages:
        for page in doc.pages:
            # paragraphs are best for "down dòng/đoạn"
            if page.paragraphs:
                for p in page.paragraphs:
                    t = _get_text_from_anchor(doc, p.layout.text_anchor).strip()
                    if t:
                        out_lines.append(t)
                out_lines.append("")  # blank line between pages
            else:
                # fallback: blocks -> lines
                if page.lines:
                    for ln in page.lines:
                        t = _get_text_from_anchor(doc, ln.layout.text_anchor).strip()
                        if t:
                            out_lines.append(t)
                    out_lines.append("")
    text = "\n".join(out_lines).strip()
    return text if text else (doc.text or "").strip()


# ----------------------------
# Chunking: 300 chars + overlap
# ----------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])  # keep line breaks
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ----------------------------
# Document AI OCR
# ----------------------------
def make_docai_client(location: str) -> documentai.DocumentProcessorServiceClient:
    # regional endpoint
    endpoint = f"{location}-documentai.googleapis.com"
    return documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )


def process_pdf_with_docai(
    client: documentai.DocumentProcessorServiceClient,
    project_id: str,
    location: str,
    processor_id: str,
    pdf_path: Path,
) -> documentai.Document:
    name = client.processor_path(project_id, location, processor_id)

    raw = pdf_path.read_bytes()
    req = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(
            content=raw,
            mime_type="application/pdf",
        ),
    )
    result = client.process_document(request=req)
    return result.document


# ----------------------------
# Main pipeline: OCR -> chunk -> embed -> chroma
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--location", required=True, help="e.g. us, eu, asia-southeast1 (if supported)")
    ap.add_argument("--processor_id", required=True)

    ap.add_argument("--input_dir", default="raw", help="Folder chứa PDF scan")
    ap.add_argument("--out_text_dir", default="output/docai_text", help="Lưu text OCR (mỗi PDF 1 file .txt)")
    ap.add_argument("--chunk_size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=60)

    ap.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--chroma_dir", default="chroma_db")
    ap.add_argument("--collection", default="usth_docs")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_text_dir = Path(args.out_text_dir)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Không thấy PDF trong: {input_dir.resolve()}")
        return

    # 1) OCR client
    client = make_docai_client(args.location)

    # 2) Embedder
    embedder = SentenceTransformer(args.embed_model)

    # 3) Chroma
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)
    col = chroma_client.get_or_create_collection(name=args.collection)

    # Process all PDFs
    all_ids = []
    all_docs = []
    all_metas = []
    all_embeds = []

    for pdf in tqdm(pdfs, desc="Document AI OCR"):
        try:
            doc = process_pdf_with_docai(client, args.project_id, args.location, args.processor_id, pdf)
            text = doc_to_paragraph_text(doc)

            # Save OCR text for inspection
            txt_path = out_text_dir / f"{pdf.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")

            # Chunk
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            if not chunks:
                continue

            # Embed
            embeds = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

            # Prepare for Chroma
            for idx, (ch, emb) in enumerate(zip(chunks, embeds)):
                cid = f"{pdf.stem}::chunk{idx}"
                all_ids.append(cid)
                all_docs.append(ch)
                all_metas.append({
                    "source_file": pdf.name,
                    "chunk_index": idx,
                    "chunk_size": args.chunk_size,
                    "overlap": args.overlap,
                })
                all_embeds.append(emb.tolist())

        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    if all_ids:
        # Upsert to Chroma
        col.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=all_embeds,
        )
        print(f"\n done. Upserted {len(all_ids)} chunks into Chroma collection '{args.collection}'.")
        print(f"OCR text saved to: {out_text_dir.resolve()}")
        print(f"Chroma persisted at: {Path(args.chroma_dir).resolve()}")
    else:
        print("Không có chunk nào để lưu (text rỗng hoặc OCR lỗi).")


if __name__ == "__main__":
    # Google auth must be set:
    # export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS chưa được set.")
        print("Ví dụ: export GOOGLE_APPLICATION_CREDENTIALS=$HOME/keys/docai-sa.json")
    else:
        main()


# OCR text saved to: /home/ngochieu/USTH_chatbot_Rag/data/output/docai_text
# Chroma persisted at: /home/ngochieu/USTH_chatbot_Rag/data/chroma_db