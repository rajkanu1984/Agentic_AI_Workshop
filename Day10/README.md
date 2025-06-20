# 🤖 Agentic AI: Product Lifecycle Management (PLM)

A fully automated, end-to-end agentic AI system for **Product Lifecycle Management** using:
- **Streamlit** for UI
- **LangGraph** for multi-agent workflow orchestration
- **Gemini 1.5 Flash / Pro** (Google Generative AI) for content generation
- **LangChain Agents** for zero-shot task execution
- **FAISS**, **TextBlob**, **SentenceTransformers**, and more for semantic search, sentiment analysis, and embeddings.

---

## 🚀 Features

### 🧠 Multi-Agent Workflow
- Talent acquisition evaluation
- Feature prioritization using RICE
- Sprint progress monitoring
- GTM (Go-to-Market) strategy generation
- Feedback analysis with roadmap updates

### 📄 Resume Matching
- Upload resumes (PDF) and compare against job description using semantic similarity.

### 🧮 RICE Roadmap Planner
- Input features and calculate RICE scores
- Visualize and rank roadmap items

### 📈 Sprint Progress Analyzer
- Summarize blockers and velocity visually

### 🚀 GTM Generator
- Generate GTM content: email, social media copy, PR headlines, etc.

### 💬 Feedback Processor
- Analyze customer sentiment and auto-prioritize product features

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-ai-plm.git
cd agentic-ai-plm
```

### 2. Create and Activate Environment (optional)

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

> Create `requirements.txt` using:
```bash
pip freeze > requirements.txt
```

### 4. Configure Gemini API Key

#### Option A: Using `.env` file

Create a file named `.env` in the project root:

```dotenv
GEMINI_API_KEY=your_google_generative_ai_key
```

#### Option B: Using Streamlit Secrets

Create a file at `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_google_generative_ai_key"
```

---

## ▶️ Run the Application

```bash
streamlit run agentic_ai_plm.py
```

---

## 🧩 Key Libraries Used

- `streamlit`
- `google.generativeai`
- `langchain`, `langchain_google_genai`
- `langgraph`
- `sentence-transformers`
- `textblob`
- `faiss`
- `plotly`
- `PyMuPDF` (fitz)

---

## 📌 Notes

- You may encounter a **429 Resource Exhausted** error from Gemini if you exceed the quota.
- Always keep your API key secure and do **not** commit `.env` or `secrets.toml` files to public repositories.
- You can customize the multi-agent flow using LangGraph nodes.

---

## 🙏 Acknowledgements

Built with ❤️ by combining cutting-edge agentic AI tools and frameworks.