
# 💼 Career Preparation Assistant using Agentic AI (Gemini + Streamlit)

This project is a smart **Career Preparation Assistant** that uses **Agentic AI** (powered by Google Gemini) to help students prepare for company placements. It analyzes **company documents** and **student resumes**, and provides **readiness evaluation**, **FAQ responses**, and **personalized guidance**.

---

## 🚀 Features

This app includes **4 intelligent agents**:

1. **Drive Parser Agent**  
   Upload a company PDF/JSON to extract:
   - Placement timelines  
   - Eligibility criteria  
   - Test formats  

2. **Readiness Evaluator Agent**  
   Upload your resume to get:
   - Skill gaps  
   - Academic comparison  
   - Project relevance  
   - Readiness % score  

3. **FAQ Retrieval Agent**  
   Ask questions like:
   - "Am I eligible for this drive?"
   - "How should I prepare for the test format?"
   - "What are my weak areas?"

4. **Reminder & Guidance Agent**  
   Get:
   - Countdown to deadlines  
   - Weekly preparation plans  
   - Resources to close skill gaps  
   - Personalized improvement tips  

---

## 🧠 Built With

- Python 🐍  
- Streamlit (UI)  
- Google Gemini API (via `google.generativeai`)  
- PyPDF2  
- JSON & PDF file processing  

---

## 📂 Folder Structure

```
career-prep-ai/
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── README.md             # Project guide
```

---

## ✅ Prerequisites

Before you begin, ensure you have:

1. **Python 3.8+** installed  
2. A **Google Gemini API Key** – [Get one here](https://makersuite.google.com/app)

---

## 🔧 Installation & Setup

Follow these steps to run the app on your local system:

### 1. 🔽 Clone this Repository

```bash
git clone https://github.com/yourusername/career-prep-ai.git
cd career-prep-ai
```

### 2. 📦 Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # for macOS/Linux
venv\Scripts\activate     # for Windows
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. 🔐 Add Gemini API Key

Set the Gemini API key as an environment variable:

#### Option 1: Add to `.streamlit/secrets.toml`

Create a file named `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

#### Option 2: Set Environment Variable

```bash
export GEMINI_API_KEY=your_actual_api_key_here   # macOS/Linux
set GEMINI_API_KEY=your_actual_api_key_here      # Windows
```

---

## ▶️ Run the App

Once everything is set up, launch the app:

```bash
streamlit run app.py
```

You’ll see the interface in your browser at `http://localhost:8501`

---

## 📥 How to Use the App

1. **Upload a company document** (PDF/JSON) → Get parsed timelines, eligibility, and test formats.  
2. **Upload your resume** → Get AI evaluation of your readiness.  
3. **Ask career-related questions** → Get personalized AI answers.  
4. **Generate reminders & tips** → Get guidance tailored to your gaps and deadlines.

---

## 📌 Notes

- PDF extraction works best with text-based (not scanned) resumes.  
- Only the **first 10,000 characters** of documents are processed to stay within Gemini limits.  
- Make sure you upload a company document **before** uploading the resume.

---

## 🧪 Requirements File (`requirements.txt`)

You can create this file with the following:

```txt
streamlit
PyPDF2
pandas
google-generativeai
```

---

## 🧑‍💻 Contributing

PRs are welcome! For major changes, open an issue first.

---

## 📄 License

MIT License (or your preferred license)
