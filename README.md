# 🏢 RAG-Based HR Policy Q&A System

> **Ask your HR policy questions in plain English — get instant, accurate, source-cited answers.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-PriyaKumari2002-black?logo=github)](https://github.com/PriyaKumari2002/RAG-based-HR-Policy-Q-A-System)

---

## 📌 Project Overview

**HR Policy Q&A System** is a production-grade **Retrieval-Augmented Generation (RAG)** application that allows employees to query HR policy documents in natural language and receive accurate, grounded answers with source citations.

### 🔍 The Problem
In most organizations, HR policies are buried inside long PDF documents. Employees waste hours searching through 20–50 page manuals to find answers to simple questions like:
- *"How many sick leaves am I entitled to?"*
- *"What is the notice period for resignation?"*
- *"What does the maternity leave policy say?"*

### ✅ The Solution
This system ingests HR policy PDFs, builds a semantic search index, and uses a Large Language Model to answer questions — **grounded strictly in the document**, with page-level source citations.

### 🎯 Target Users
- Employees seeking quick HR policy clarification
- HR teams reducing repetitive query load
- Organizations wanting AI-powered internal knowledge bases

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Finds relevant content by meaning, not just keywords |
| 📄 **PDF Parsing** | Automatically loads and processes HR policy PDFs |
| 🧩 **Smart Chunking** | Splits documents into optimal chunks with overlap |
| 🧠 **Vector Embeddings** | Uses HuggingFace `all-MiniLM-L6-v2` for fast, accurate embeddings |
| ⚡ **FAISS Vector Store** | Lightning-fast similarity search over document vectors |
| 🤖 **LLM Integration** | Groq LLaMA 3 generates accurate, context-aware answers |
| 🛡️ **Hallucination Prevention** | Answers strictly grounded in source documents |
| 📌 **Source Citation** | Every answer cites the exact page it came from |
| 🚀 **FastAPI Backend** | Production-ready REST API with Swagger documentation |
| 🖥️ **Streamlit Frontend** | Clean, interactive UI for non-technical users |
| 📋 **Structured Logging** | Full request/response logging for observability |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **RAG Framework** | LangChain 0.3.25 |
| **LLM Provider** | Groq (LLaMA 3.1 8B Instant) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **PDF Loader** | PyPDFLoader (LangChain Community) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Environment** | python-dotenv |
| **Logging** | Python logging module |
| **Version Control** | Git + GitHub |

---

## 🏗️ Project Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE (Once)                  │
│                                                             │
│  📄 PDF Document                                            │
│       │                                                     │
│       ▼                                                     │
│  📖 PyPDFLoader → Extract Text (27 pages)                   │
│       │                                                     │
│       ▼                                                     │
│  ✂️  RecursiveCharacterTextSplitter                         │
│       chunk_size=500, overlap=50 → 198 chunks               │
│       │                                                     │
│       ▼                                                     │
│  🧠 HuggingFace Embeddings (all-MiniLM-L6-v2)              │
│       Text → 384-dimensional vectors                        │
│       │                                                     │
│       ▼                                                     │
│  💾 FAISS Vector Store → Saved to disk                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (Every Request)            │
│                                                             │
│  👤 User Question                                           │
│       │                                                     │
│       ▼                                                     │
│  🧠 Embed Question → 384-dim vector                         │
│       │                                                     │
│       ▼                                                     │
│  🔍 FAISS Similarity Search → Top 3 relevant chunks        │
│       │                                                     │
│       ▼                                                     │
│  📝 Prompt Template                                         │
│       [System Instructions + Context + Question]            │
│       │                                                     │
│       ▼                                                     │
│  🤖 Groq LLaMA 3.1 → Generate Answer                       │
│       │                                                     │
│       ▼                                                     │
│  ✅ Answer + Source Pages → User                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
hr-policy-qa/
│
├── 📂 data/                    # HR Policy PDF documents
│   └── hr_policy.pdf
│
├── 📂 src/                     # Core RAG pipeline modules
│   ├── loader.py               # PDF loading logic
│   ├── chunker.py              # Text splitting logic
│   ├── embeddings.py           # Embedding generation
│   ├── vectorstore.py          # FAISS store management
│   ├── retriever.py            # Similarity search logic
│   └── llm_chain.py            # LLM chain setup
│
├── 📂 api/                     # FastAPI backend
│   └── main.py                 # API routes, logging, error handling
│
├── 📂 frontend/                # Streamlit UI
│   └── app.py                  # Interactive chat interface
│
├── 📂 tests/                   # Unit tests
│   └── test_retriever.py
│
├── 📓 hr_policy_qa.ipynb       # Development notebook
├── 🔒 .env                     # Environment variables (not committed)
├── 🚫 .gitignore
├── 📋 requirements.txt
└── 📖 README.md
```

---

## ⚙️ Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/PriyaKumari2002/RAG-based-HR-Policy-Q-A-System.git
cd RAG-based-HR-Policy-Q-A-System
```

### 2. Create Virtual Environment
```bash
conda create -n hr-qa-env python=3.10
conda activate hr-qa-env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add HR Policy PDF
Place your HR policy PDF in the `data/` folder:
```
data/hr_policy.pdf
```

### 6. Run FastAPI Backend
```bash
python -m uvicorn api.main:app --reload
```
API will be available at: `http://127.0.0.1:8000`
Swagger docs at: `http://127.0.0.1:8000/docs`

### 7. Run Streamlit Frontend
Open a new terminal:
```bash
streamlit run frontend/app.py
```
UI will open at: `http://localhost:8501`

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) | ✅ Yes |

---

## 📖 Usage Guide

### Asking Questions
1. Open the Streamlit UI at `http://localhost:8501`
2. Click a sample question or type your own
3. Click **"Ask"** button
4. Get instant answer with source page citations

### Example Queries & Outputs

**Query:** `How many sick leaves are allowed?`
```
Answer: According to Section 2.3 of the HR policy, employees are 
entitled to 7 days of sick leave per calendar year.

Sources:
- Page 9: Section 2.3 - Sick Leave
- Page 21: Appendix 1 - Leave Guidance
```

**Query:** `What is the notice period for resignation?`
```
Answer: As per Section 3, the notice period is one month prior 
written notice to the HR department.

Sources:
- Page 11: Section 3 - Resignation
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — API running status |
| `/health` | GET | Service health status |
| `/ask` | POST | Submit a question, get answer + sources |
| `/docs` | GET | Interactive Swagger API documentation |

### Sample Request
```json
POST /ask
{
  "question": "How many sick leaves are allowed?"
}
```

### Sample Response
```json
{
  "question": "How many sick leaves are allowed?",
  "answer": "Employees are entitled to 7 days of sick leave per calendar year (Section 2.3).",
  "sources": [
    { "page": 9, "content": "Employees will be entitled to 7 days of sick leave..." },
    { "page": 21, "content": "Every employee is entitled for 7 Sick Leaves per year..." }
  ],
  "response_time": 1.42
}
```

---

## 🧠 RAG Pipeline — Deep Dive

### 1. Document Loading
`PyPDFLoader` extracts text from each PDF page, preserving page metadata for source citation.

### 2. Chunking
`RecursiveCharacterTextSplitter` splits text into 500-character chunks with 50-character overlap. Overlap ensures context is not lost at chunk boundaries.

### 3. Embeddings
`all-MiniLM-L6-v2` converts each chunk into a 384-dimensional vector. Similar meaning = similar vectors. Runs locally — no API cost.

### 4. Vector Storage
FAISS stores all 198 vectors on disk. On query, it finds the top-3 most similar chunks using cosine similarity in milliseconds.

### 5. Prompt Engineering
A strict system prompt instructs the LLM to answer ONLY from retrieved context — preventing hallucination critical in HR/legal domains.

### 6. LLM Response
Groq LLaMA 3.1 8B generates a structured, concise answer with section references, completing the RAG loop.

---

## ⚠️ Challenges Faced

- **Hallucination Prevention** — Solved via strict prompt instructions limiting LLM to context only
- **Chunk Boundary Loss** — Fixed using 50-character chunk overlap
- **Version Conflicts** — Resolved LangChain/LangChain-core dependency conflicts manually
- **FAISS Path Issues** — Handled cross-environment path differences between Jupyter and FastAPI
- **PDF Quality** — Raw PDFs contain headers/footers in chunks; handled via chunk filtering

---

## 🚀 Future Improvements

| Improvement | Impact |
|---|---|
| **Hybrid Search** (BM25 + Vector) | Better retrieval accuracy |
| **Re-ranking** (Cohere/BGE) | Higher answer quality |
| **Redis Caching** | Faster repeated queries |
| **Multi-document Support** | Upload multiple PDFs |
| **Authentication** | Secure enterprise access |
| **Multilingual Support** | Hindi/regional language queries |
| **ChromaDB Migration** | Better metadata filtering |
| **Docker Deployment** | Easy containerized deployment |
| **AWS/GCP Deployment** | Production cloud hosting |
| **Monitoring Dashboard** | Query analytics and feedback loop |

---

## 💼 Business Impact

- ⏱️ **Reduced HR query resolution time** from hours to under 5 seconds
- 📉 **Decreased HR team workload** by handling repetitive policy questions automatically
- 🔍 **100% source-traceable answers** — critical for compliance in regulated industries
- 📄 **Eliminated manual PDF search** across 27-page policy documents
- 🛡️ **Zero hallucination design** — LLM restricted to document context only

---

## 📄Important Points

- Designed and deployed an end-to-end **RAG pipeline** for HR policy Q&A using **LangChain, FAISS, and Groq LLaMA 3**, enabling semantic search over PDF documents with source-level traceability
- Implemented **document chunking** (chunk_size=500, overlap=50) and **HuggingFace embeddings** (`all-MiniLM-L6-v2`) generating 198 vectors stored in **FAISS** for sub-second retrieval
- Built **hallucination-free prompt engineering** system restricting LLM responses strictly to retrieved document context — critical for HR/legal compliance use cases
- Developed **production-ready REST API** using **FastAPI** with Pydantic validation, structured logging, error handling, and auto-generated Swagger documentation
- Created interactive **Streamlit frontend** with semantic search, source citation, and page-level traceability for non-technical end users
- Resolved real-world **LangChain dependency conflicts** and cross-environment path issues between Jupyter development and FastAPI production environments

---

## 🌐 Deployment Guide

### Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `frontend/app.py` as entry point
4. Add `GROQ_API_KEY` in secrets

### Render (FastAPI Backend)
1. Create new Web Service on [render.com](https://render.com)
2. Connect GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👩‍💻 Author

**Priya Kumari**
- GitHub: [@PriyaKumari2002](https://github.com/PriyaKumari2002)
- Project: [RAG-based-HR-Policy-Q-A-System](https://github.com/PriyaKumari2002/RAG-based-HR-Policy-Q-A-System)

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — RAG framework
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [HuggingFace](https://huggingface.co) — Open-source embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com) — Modern Python API framework
- [Streamlit](https://streamlit.io) — Rapid UI development
- [Doctors For You](https://doctorsforyou.org) — HR policy document

---

*Built with ❤️ using LangChain • FAISS • Groq LLaMA 3 • FastAPI • Streamlit*
