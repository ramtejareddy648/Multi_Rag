# 🤖 Multimodal RAG Chatbot with LangGraph

A production-ready Retrieval-Augmented Generation (RAG) chatbot that supports text, PDF, and image documents. Built with LangGraph for agentic workflows, AstraDB for vector storage, Neon PostgreSQL for conversation persistence, and Streamlit for the frontend UI.

---

## ✨ Features

- **Multimodal Ingestion** — supports PDF, TXT, and image files (PNG, JPG, JPEG)
- **Multi-Vector Retrieval** — stores parent documents and child chunk summaries separately for better retrieval accuracy
- **Agentic Workflow** — LangGraph pipeline with document evaluation, web search fallback, and context refinement
- **Persistent Conversations** — full chat history saved to Neon (PostgreSQL) via LangGraph checkpointing
- **Voice Input** — speech-to-text via Sarvam AI API
- **Web Search Fallback** — DuckDuckGo search when retrieved documents are not relevant enough
- **Image Understanding** — GPT-4o Vision for image description and text extraction from PDFs

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Retriever Node (AstraDB Multi-Vector)
    │
    ▼
Document Evaluator (Relevance Scoring)
    │
    ├── CORRECT ──► Refine Node ──► Generate Node ──► Answer
    │
    └── INCORRECT / AMBIGUOUS ──► Rewrite Query ──► Web Search ──► Refine ──► Generate ──► Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b) + OpenAI (GPT-4o-mini Vision) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | AstraDB |
| Parent Store | AstraDB ByteStore |
| Graph Orchestration | LangGraph |
| Checkpointing | LangGraph + Neon PostgreSQL |
| Web Search | DuckDuckGo |
| Voice | Sarvam AI |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
cagproject/
├── backend1.py          # RAG pipeline, LangGraph workflow, DB connections
├── frontend.py          # Streamlit UI, voice input, chat interface
├── .env                 # Environment variables (not committed)
├── .env.example         # Example env file
├── .gitignore
├── requirements.txt
└── uploads_doc/         # Temp folder for uploaded files (not committed)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate virtual environment

```bash
python -m venv myvenv

# Windows
myvenv\Scripts\activate

# Mac/Linux
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### 5. Run the app

```bash
streamlit run frontend.py
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Groq
GROQ_API_KEY=your_groq_api_key

# AstraDB
ASTRA_DB_API_ENDPOINT=your_astradb_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astradb_token
VECTOR_COLLECTION_NAME=rag_vector
PARENT_STORE_COLLECTION_NAME=rag_parents

# Neon PostgreSQL (use direct connection URL, not pooler)
POSTGRES_URL=postgresql://user:password@your-neon-host.neon.tech/dbname?sslmode=require

# Tavily (optional)
TAVILY_API_KEY=your_tavily_key
```

> ⚠️ **Important:** Use the **direct** Neon connection URL (without `-pooler` in the hostname), not the pooled URL, since `psycopg_pool` manages its own connection pooling.

---

## 📦 Requirements

Key packages (see `requirements.txt` for full list):

```
streamlit
langchain
langchain-openai
langchain-groq
langchain-community
langchain-astradb
langgraph
langgraph-checkpoint-postgres
psycopg[binary]
psycopg-pool
python-dotenv
pypdf
pymupdf
streamlit-mic-recorder
```

---

## 🚀 Usage

### Upload & Index Documents

1. Use the sidebar file uploader to upload PDFs, TXTs, or images
2. Click **"Process your files"** to index them into AstraDB
3. Wait for the success message

### Chat

- Type a question in the chat input or use the **voice recorder** in the sidebar
- The app will retrieve relevant documents, evaluate them, and generate an answer
- If documents are not relevant, it automatically falls back to web search

### Manage Conversations

- Click **"New Chat"** to start a fresh conversation
- Previous conversations appear in the sidebar — click to reload
- Click 🗑️ to delete a conversation

---

## 🗄️ Database Setup

The Neon PostgreSQL tables are created automatically on first run via `check_point.setup()`. No manual migration needed.

Tables created:
- `checkpoints`
- `checkpoint_blobs`
- `checkpoint_writes`

---

## 🔒 .gitignore

Make sure your `.gitignore` includes:

```
.env
uploads_doc/
myvenv/
__pycache__/
*.pyc
chatbot.db
```

---

## 📝 License

MIT License — feel free to use and modify.

---

## 🙋 Contributing

Pull requests are welcome. For major changes, please open an issue first.
