# 📄 DocSense AI

**DocSense AI** is a fast document analyzer built with Streamlit that accepts PDFs (including scanned PDFs), images, and returns: extracted text, image descriptions, concise summaries, and answers to user questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Key Features

- Accepts input: PDF, scanned PDF, and common image formats (PNG, JPG, JPEG, TIFF, BMP, GIF).
- Fast hybrid extraction: native PDF text extraction + Tesseract OCR for scanned pages and images.
- Parallel page processing for speed (ThreadPoolExecutor + batched processing).
- Image analysis (fast OCR-based descriptions and optional deeper Llama analysis).
- Summarization using an LLM (configured here to call Llama 3.2 via an Ollama-style local API).
- Semantic search over document chunks using SentenceTransformers + FAISS for RAG-style QA.
- Optional web augmentation using DuckDuckGo when document context is insufficient.
- Streamlit UI for uploading documents, asking questions, and inspecting results.

---

## 🧩 Architecture Overview

1. **Upload**: User uploads a PDF or image via Streamlit sidebar.
2. **Extraction**: PDF pages are checked for native text; scanned pages are rasterized and passed to Tesseract OCR. Images are OCRed directly.
3. **Image processing**: Extracted images on pages receive quick OCR descriptions. (Full LLM-based image analysis is available but disabled by default for speed.)
4. **Chunking & Embeddings**: Extracted text is chunked into overlapping pieces; SentenceTransformer (`all-MiniLM-L6-v2`) creates embeddings.
5. **Indexing**: Chunks are stored in a session FAISS index for fast similarity search scoped to the current document.
6. **RAG QA**: User question → embed query → search FAISS → build prompt (document excerpts + optional web context) → LLM answer via local Llama 3.2 API.

---

## ⚙️ Requirements

> Recommended to run inside a Python 3.8+ virtual environment. For large documents and production settings, a machine with more RAM and optional GPU will help.

Install required packages (example):

```bash
pip install -r requirements.txt
```

Example `requirements.txt` (based on the provided script):

```
streamlit
PyMuPDF
Pillow
pytesseract
sentence-transformers
faiss-cpu
numpy
requests
duckduckgo-search
```

**System dependencies**
- Tesseract-OCR binary installed and available in PATH. On Ubuntu:
  ```bash
  sudo apt update && sudo apt install tesseract-ocr
  ```
- (Optional) If using GPU for models, install appropriate FAISS / PyTorch packages for GPU.

---

## 🔧 Configuration

- **LLM Endpoint**: The app calls a local Ollama-style endpoint at `http://localhost:11434/api/generate` configured for `llama3.2`. Modify `query_llama()` in the code to point to your LLM endpoint or to use another API.

- **Tesseract Path**: If `pytesseract` cannot find the Tesseract binary, set it manually in the environment or in code:

```python
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # or C:\Program Files\Tesseract-OCR\tesseract.exe
```

- **Embedding model**: The script uses `sentence-transformers/all-MiniLM-L6-v2` by default. You can change this in `load_embedding_model()`.

---

## 🧭 How to run

1. Create and activate a virtualenv (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Tesseract is installed and reachable.

4. (Optional) Run your LLM server (e.g., Ollama) locally if you want Llama 3.2 responses.

5. Start the Streamlit app:

```bash
streamlit run final_pdf_ocr.py
```

Open the UI in your browser (Streamlit will print the local URL, usually `http://localhost:8501`).

---

## 🧪 Usage Examples

- Upload a scanned PDF of a contract — the app will OCR pages and provide a summary and searchable chunks.
- Upload a multi-page report PDF with images — extract text, quick image descriptions, then ask domain questions (e.g., "What are the main findings?").
- Upload a photo of a receipt — extract the text and ask for specific values like totals or dates.

---

## ✨ Tips & Performance

- For **very large PDFs**, increase `batch_size` and tune `ThreadPoolExecutor` workers carefully to balance CPU and memory.
- To speed up OCR, consider pre-processing images (grayscale, thresholding) before `pytesseract`.
- Keep the LLM analysis in quick mode for images to improve responsiveness. Enable full LLM image analysis only when necessary.

---

## ⚠️ Limitations & Privacy

- The app sends prompts to a local LLM endpoint by default. If you change it to a remote/cloud LLM, sensitive document contents will be transmitted externally — review privacy requirements before doing so.
- OCR accuracy depends on image quality and language support. Tesseract works best on clearly-scanned, high-contrast documents.
- FAISS index and embeddings are stored in-memory for the session and cleared when the session ends.

---

## 🛠️ Extending & Customization

- Swap the embedding model for a larger/higher-quality SentenceTransformer for better semantic search.
- Replace the local Llama call with an API client for OpenAI, Anthropic, or another provider (ensure you follow their API terms).
- Persist indexes and metadata to disk or a database for multi-session search.
- Add language detection and multi-language OCR support.

---


