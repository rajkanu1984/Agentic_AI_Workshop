import os
import streamlit as st
from datetime import datetime, timedelta
import json
from PyPDF2 import PdfReader
from google.generativeai import GenerativeModel, configure
import pandas as pd
import time

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY") # Use either secrets or env var

# Initialize Gemini model
model = GenerativeModel('gemini-2.0-flash-exp')

# Initialize session state variables
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'student_data' not in st.session_state:
    st.session_state.student_data = None
if 'faq_data' not in st.session_state:
    st.session_state.faq_data = None

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF files"""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_json(uploaded_file):
    """Extract text from JSON files"""
    data = json.load(uploaded_file)
    return json.dumps(data)

def parse_company_document(uploaded_file):
    """Drive Parser Agent: Extract company information from documents"""
    try:
        if uploaded_file.name.endswith('.pdf'):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            text = extract_text_from_json(uploaded_file)
        else:
            text = str(uploaded_file.read())
        
        prompt = f"""Extract the following information from the provided company document:
        1. Placement timelines (important dates and deadlines)
        2. Eligibility criteria (CGPA requirements, backlogs allowed, etc.)
        3. Test formats (coding rounds, aptitude tests, interview rounds)
        
        Return the information in JSON format with keys: timelines, eligibility, test_formats.
        
        Document content: {text[:10000]}"""  # Limiting to first 10k chars for Gemini
        
        response = model.generate_content(prompt)
        
        try:
            # Try to parse the JSON from the response
            start_idx = response.text.find('{')
            end_idx = response.text.rfind('}') + 1
            json_str = response.text[start_idx:end_idx]
            company_data = json.loads(json_str)
            return company_data
        except json.JSONDecodeError:
            # If automatic parsing fails, ask Gemini to fix it
            fix_prompt = f"""The following text should be valid JSON but it's not. 
            Please convert it to proper JSON format with keys: timelines, eligibility, test_formats.
            
            Text: {response.text}"""
            
            fixed_response = model.generate_content(fix_prompt)
            start_idx = fixed_response.text.find('{')
            end_idx = fixed_response.text.rfind('}') + 1
            json_str = fixed_response.text[start_idx:end_idx]
            company_data = json.loads(json_str)
            return company_data
            
    except Exception as e:
        st.error(f"Error processing company document: {str(e)}")
        return None

def evaluate_readiness(resume_file, company_data):
    """Readiness Evaluator Agent: Compare student resume against company requirements"""
    try:
        resume_text = extract_text_from_pdf(resume_file)
        
        prompt = f"""Analyze this student resume against the company requirements and provide:
        1. Skill gaps (missing skills mentioned in company requirements)
        2. Academic comparison (how GPA compares to requirements)
        3. Project relevance (how projects align with company's domain)
        4. Overall readiness percentage (0-100%)
        
        Return in JSON format with keys: skill_gaps, academic_comparison, project_relevance, readiness_percentage.
        
        Company Requirements: {json.dumps(company_data)}
        Resume Content: {resume_text[:8000]}"""  # Limiting resume content
        
        response = model.generate_content(prompt)
        
        try:
            # Try to parse the JSON from the response
            start_idx = response.text.find('{')
            end_idx = response.text.rfind('}') + 1
            json_str = response.text[start_idx:end_idx]
            evaluation = json.loads(json_str)
            return evaluation
        except json.JSONDecodeError:
            # If automatic parsing fails, ask Gemini to fix it
            fix_prompt = f"""The following text should be valid JSON but it's not. 
            Please convert it to proper JSON format with keys: skill_gaps, academic_comparison, project_relevance, readiness_percentage.
            
            Text: {response.text}"""
            
            fixed_response = model.generate_content(fix_prompt)
            start_idx = fixed_response.text.find('{')
            end_idx = fixed_response.text.rfind('}') + 1
            json_str = fixed_response.text[start_idx:end_idx]
            evaluation = json.loads(json_str)
            return evaluation
            
    except Exception as e:
        st.error(f"Error evaluating readiness: {str(e)}")
        return None

def answer_faqs(question, company_data, evaluation_data):
    """FAQ Retrieval Agent: Answer common questions using RAG approach"""
    try:
        prompt = f"""You're a career assistant helping students prepare for company placements.
        Answer this question: {question}
        
        Context:
        - Company requirements: {json.dumps(company_data)[:3000]}
        - Student evaluation: {json.dumps(evaluation_data)[:3000] if evaluation_data else "No evaluation data"}
        
        Provide a concise, helpful answer based on the available information."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating FAQ answer: {str(e)}")
        return "Sorry, I couldn't process that question right now."

def generate_reminders(company_data, evaluation_data):
    """Reminder & Guidance Agent: Generate personalized reminders and tips"""
    try:
        if not company_data or not evaluation_data:
            return "Please complete company and resume analysis first."
            
        prompt = f"""Generate personalized preparation reminders and guidance for a student based on:
        1. Upcoming company drive timelines: {company_data.get('timelines', 'No timeline data')}
        2. Identified skill gaps: {evaluation_data.get('skill_gaps', 'No skill gap data')}
        3. Academic comparison: {evaluation_data.get('academic_comparison', 'No academic data')}
        4. Project relevance: {evaluation_data.get('project_relevance', 'No project data')}
        
        Provide:
        - A countdown to the nearest important date
        - Weekly preparation plan until the drive
        - Specific resources to address skill gaps
        - Tips to improve in weak areas
        
        Format the output with clear headings and bullet points."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating reminders: {str(e)}")
        return "Sorry, I couldn't generate reminders right now."

# Streamlit UI
st.title("Career Preparation Assistant")

# Agent 1: Drive Parser Agent
st.header("1. Drive Parser Agent")
company_file = st.file_uploader("Upload Company Document (PDF/JSON/TXT)", type=['pdf', 'json', 'txt'], key="company_upload")

if company_file:
    with st.spinner("Extracting company information..."):
        company_data = parse_company_document(company_file)
        if company_data:
            st.session_state.company_data = company_data
            st.success("Company data extracted successfully!")
            st.json(company_data)

# Agent 2: Readiness Evaluator Agent
st.header("2. Readiness Evaluator Agent")
if st.session_state.company_data:
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=['pdf'], key="resume_upload")
    
    if resume_file:
        with st.spinner("Evaluating your readiness..."):
            evaluation_data = evaluate_readiness(resume_file, st.session_state.company_data)
            if evaluation_data:
                st.session_state.student_data = evaluation_data
                st.success("Readiness evaluation complete!")
                st.json(evaluation_data)
else:
    st.warning("Please upload and process company document first.")

# Agent 3: FAQ Retrieval Agent
st.header("3. FAQ Retrieval Agent")
if st.session_state.company_data and st.session_state.student_data:
    faq_question = st.text_input("Ask a question about the company or preparation:")
    
    if faq_question:
        with st.spinner("Finding the best answer..."):
            answer = answer_faqs(faq_question, st.session_state.company_data, st.session_state.student_data)
            st.markdown(f"**Answer:** {answer}")
else:
    st.warning("Please complete both company document and resume analysis first.")

# Agent 4: Reminder & Guidance Agent
st.header("4. Reminder & Guidance Agent")
if st.session_state.company_data and st.session_state.student_data:
    if st.button("Generate Personalized Reminders"):
        with st.spinner("Creating your personalized plan..."):
            reminders = generate_reminders(st.session_state.company_data, st.session_state.student_data)
            st.markdown(reminders)
else:
    st.warning("Please complete both company document and resume analysis first.")

# Add some styling
st.markdown("""
<style>
    .stHeader {
        padding-top: 0.5rem;
    }
    .stSpinner {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)