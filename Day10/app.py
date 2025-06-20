# agentic_ai_plm.py
# Fully Automated End-to-End Agentic AI for Product Lifecycle Management
# Using Streamlit + LangGraph + Gemini (Gemini 2.0 Flash) + AgentExecutor hybrid

# agentic_ai_plm.py
# Fully Automated End-to-End Agentic AI for Product Lifecycle Management
# Using Streamlit + LangGraph + Gemini (Gemini 2.0 Flash) + AgentExecutor hybrid

import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import faiss
import requests
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from textblob import TextBlob
from apscheduler.schedulers.background import BackgroundScheduler
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
import plotly.express as px

# =============================
# Load .env if exists
# =============================
from dotenv import load_dotenv
load_dotenv()

# =============================
# Gemini API Configuration (Safe and Consistent)
# =============================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("\u274c Gemini API key not found in env or secrets. Please set it in your .env or .streamlit/secrets.toml.")
        st.stop()

try:
    configure(api_key=api_key)
except Exception as e:
    st.error(f"\u274c Failed to configure Gemini API: {str(e)}")
    st.stop()

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
model = GenerativeModel("models/gemini-1.5-flash")

st.set_page_config(page_title="Agentic AI - PLM", layout="wide")

# Inject custom CSS for better visuals
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9fbfd;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2rem;
        }
        .stTextArea textarea, .stTextInput input {
            background-color: #f0f2f6;
            color: #333;
            font-size: 15px;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-size: 15px;
        }
        .stButton>button:hover {
            background-color: #004d99;
        }
        .stExpanderHeader {
            font-size: 18px;
            font-weight: bold;
        }
        .stMarkdown, .stText, .stCaption {
            font-size: 15px;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🤖 Agentic AI: Product Lifecycle Management")

# ... the rest of your code remains unchanged ...

# =============================
# AgentExecutor Setup (Reusable Agent for Advanced Tasks)
# =============================
def initialize_feedback_agent():
    def summarize_feedback_tool(input_text: str) -> str:
        blob = TextBlob(input_text)
        sentiment = blob.sentiment
        return f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}"

    def generate_roadmap_tool(input_text: str) -> str:
        return model.generate_content(f"Generate a product roadmap based on the following insight: {input_text}").text

    tools = [
        Tool(
            name="Feedback Summary Tool",
            func=summarize_feedback_tool,
            description="Summarizes the sentiment in feedback using polarity and subjectivity."
        ),
        Tool(
            name="Roadmap Generation Tool",
            func=generate_roadmap_tool,
            description="Generates roadmap items based on insight text."
        )
    ]

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent_executor

# =============================
# LangGraph Setup for Multi-Agent Flow
# =============================
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentState:
    step: str
    memory: Dict[str, Any]

def talent_agent_node(state: AgentState) -> AgentState:
    jd = state.memory.get("jd", "")
    agent = initialize_feedback_agent()
    result = agent.run(f"Summarize this job description: {jd}")
    state.memory["talent_summary"] = result
    state.step = "roadmap"
    return state

def roadmap_agent_node(state: AgentState) -> AgentState:
    features = state.memory.get("features", [])
    agent = initialize_feedback_agent()
    result = agent.run(f"Validate and prioritize the following features: {features}")
    state.memory["roadmap_summary"] = result
    state.step = "progress"
    return state

def progress_agent_node(state: AgentState) -> AgentState:
    updates = state.memory.get("sprint_updates", "")
    agent = initialize_feedback_agent()
    result = agent.run(f"Analyze this sprint update and identify blockers: {updates}")
    state.memory["progress_summary"] = result
    state.step = "gtm"
    return state

def gtm_agent_node(state: AgentState) -> AgentState:
    features = state.memory.get("features", [])
    agent = initialize_feedback_agent()
    result = agent.run(f"Generate GTM plan for features: {features}")
    state.memory["gtm_summary"] = result
    state.step = "sales"
    return state

def sales_agent_node(state: AgentState) -> AgentState:
    feedback = state.memory.get("feedback", "")
    agent = initialize_feedback_agent()
    result = agent.run(f"Analyze feedback and suggest roadmap updates: {feedback}")
    state.memory["sales_insight"] = result
    state.step = "end"
    return state

@st.cache_resource
def compile_graph():
    graph = StateGraph(AgentState)
    graph.add_node("talent", talent_agent_node)
    graph.add_node("roadmap", roadmap_agent_node)
    graph.add_node("progress", progress_agent_node)
    graph.add_node("gtm", gtm_agent_node)
    graph.add_node("sales", sales_agent_node)

    graph.set_entry_point("talent")
    graph.add_edge("talent", "roadmap")
    graph.add_edge("roadmap", "progress")
    graph.add_edge("progress", "gtm")
    graph.add_edge("gtm", "sales")
    graph.add_edge("sales", END)
    return graph.compile()

if st.button("🔁 Run LangGraph End-to-End"):
    memory = {
        "jd": st.session_state.get("jd", "Sample JD"),
        "features": [f["Feature"] for f in st.session_state.get("features", [])],
        "sprint_updates": st.session_state.get("last_summary", "Sprint update example"),
        "feedback": st.session_state.get("feedback_text", "User feedback example")
    }
    workflow = compile_graph()
    result_state = workflow.invoke(AgentState(step="talent", memory=memory))
    st.success("✅ LangGraph Execution Complete")
    st.json(result_state.memory)

# =============================
# Shared Utilities
# =============================
@st.cache_data
def embed_texts(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)

def parse_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def calculate_rice_score(reach, impact, confidence, effort):
    try:
        score = (reach * impact * confidence) / effort
        return round(score, 2)
    except:
        return 0

# =============================
# Agent Interfaces
# =============================
# 1. Talent Acquisition Agent
with st.expander("👥 1. Talent Acquisition Agent"):
    jd = st.text_area("📄 Job Description")
    resumes = st.file_uploader("📂 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    if st.button("🔍 Match Candidates") and jd and resumes:
        jd_emb = embed_texts([jd])[0]
        results = []
        for resume in resumes:
            text = parse_pdf(resume)
            res_emb = embed_texts([text])[0]
            score = float(cosine_similarity([jd_emb], [res_emb])[0][0])
            results.append((resume.name, round(score * 100, 2)))
        results.sort(key=lambda x: x[1], reverse=True)
        st.success("🎯 Candidates Ranked:")
        for name, score in results:
            st.write(f"{name}: {score}% match")

    with st.expander("📅 Auto-Schedule Interview (Simulated)"):
        selected_candidate = st.text_input("Select Candidate to Schedule Interview")
        date = st.date_input("Select Date")
        time_input = st.time_input("Select Time")
        if st.button("📨 Schedule Interview"):
            st.success(f"✅ Interview scheduled for {selected_candidate} on {date} at {time_input} (simulated)")

    if st.button("🧠 Run AgentExecutor for Candidate Insight") and jd:
        agent = initialize_feedback_agent()
        result = agent.run(f"Summarize the key hiring traits from this JD and how we might improve: {jd}")
        st.success("✅ AgentExecutor Insight:")
        st.write(result)

# 2. Roadmap Planning Agent
with st.expander("🗺️ 2. Roadmap Planning Agent"):
    features_df = pd.DataFrame(columns=["Feature", "Reach", "Impact", "Confidence", "Effort", "RICE Score"])
    with st.form("rice_form"):
        st.write("🧮 Enter feature and RICE attributes")
        feature = st.text_input("Feature Name")
        reach = st.number_input("Reach", min_value=1)
        impact = st.number_input("Impact", min_value=1)
        confidence = st.number_input("Confidence", min_value=1)
        effort = st.number_input("Effort", min_value=1)
        submitted = st.form_submit_button("➕ Add to Roadmap")
        if submitted:
            rice = calculate_rice_score(reach, impact, confidence, effort)
            st.session_state.setdefault("features", []).append({
                "Feature": feature,
                "Reach": reach,
                "Impact": impact,
                "Confidence": confidence,
                "Effort": effort,
                "RICE Score": rice
            })

    if "features" in st.session_state:
        df = pd.DataFrame(st.session_state["features"])
        st.dataframe(df.sort_values(by="RICE Score", ascending=False))
        fig = px.bar(df.sort_values(by="RICE Score", ascending=False), x="Feature", y="RICE Score", title="Prioritized Roadmap by RICE Score")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Run AgentExecutor to Validate Roadmap"):
        agent = initialize_feedback_agent()
        roadmap_summary = agent.run("""Given the following features, suggest refinement:
{}""".format([f["Feature"] for f in st.session_state.get("features", [])]))
        st.success("✅ AgentExecutor Suggestion:")
        st.write(roadmap_summary)

# 3. Progress Monitoring Agent
with st.expander("📈 3. Progress Monitoring Agent"):
    sprint_updates = st.text_area("📋 Paste Sprint Progress Updates")
    velocity = st.slider("📊 Sprint Velocity", 0, 100, 50)
    blockers = st.text_area("🚧 Known Blockers")
    if st.button("📤 Monitor Progress"):
        summary = model.generate_content(f"""Summarize progress:
{sprint_updates}
Blockers: {blockers}""")
        st.success("✅ Sprint Summary:")
        st.write(summary.text)
        st.progress(velocity)
        st.session_state["last_summary"] = summary.text

    if st.button("🤖 Run AgentExecutor on Sprint Report") and sprint_updates:
        agent = initialize_feedback_agent()
        progress_insight = agent.run(f"Analyze this sprint update and generate a roadmap or feedback summary: {sprint_updates}")
        st.success("✅ AgentExecutor Insight:")
        st.write(progress_insight)

# 4. GTM Strategy Agent
with st.expander("🚀 4. GTM Strategy Agent"):
    product_features = st.text_area("🧩 Product Features")
    persona = st.text_input("🎯 Target Persona")
    if st.button("📣 Create GTM Plan"):
        gtm_prompt = f"""
        Features: {product_features}
        Persona: {persona}
        Generate:
        1. Email marketing copy
        2. Social media post
        3. Press release title
        4. Launch date suggestion
        """
        gtm = model.generate_content(gtm_prompt)
        st.write(gtm.text)
        st.download_button("📥 Download Campaign Kit", gtm.text, file_name="gtm_campaign.txt")

    if st.button("🤖 Run AgentExecutor for GTM Strategy") and product_features:
        agent = initialize_feedback_agent()
        gtm_output = agent.run(f"Generate GTM content for the following features: {product_features} targeting {persona}")
        st.success("✅ AgentExecutor GTM Output:")
        st.write(gtm_output)

# 5. Sales & Feedback Agent
with st.expander("💬 5. Sales & Feedback Agent"):
    feedback_text = st.text_area("🗣️ Customer Reviews/Feedback")
    if st.button("🧠 Analyze Feedback"):
        blob = TextBlob(feedback_text)
        sent = blob.sentiment
        st.write(f"Polarity: {sent.polarity}, Subjectivity: {sent.subjectivity}")
        if sent.polarity < 0:
            insight = "Users are unhappy. Improve onboarding, UI, and performance."
        elif sent.polarity > 0.5:
            insight = "Users are very happy. Consider requesting testimonials and expanding reach."
        else:
            insight = "Feedback is neutral. Follow up with surveys."
        st.code(insight)
        st.session_state["roadmap_insight"] = insight

    if st.button("🔁 Run AgentExecutor on Feedback") and feedback_text:
        agent = initialize_feedback_agent()
        result = agent.run(f"Analyze this customer feedback and generate roadmap items: {feedback_text}")
        st.success("✅ AgentExecutor Result:")
        st.write(result)

    if st.session_state.get("roadmap_insight"):
        st.markdown("**🔄 Send Feedback to Roadmap Agent**")
        if st.button("📨 Forward Insight"):
            st.session_state.setdefault("features", []).append({
                "Feature": st.session_state["roadmap_insight"],
                "Reach": 50,
                "Impact": 50,
                "Confidence": 80,
                "Effort": 20,
                "RICE Score": calculate_rice_score(50, 50, 80, 20)
            })
            st.success("✅ Insight integrated into Roadmap Agent")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangGraph, AgentExecutor, and Gemini")
