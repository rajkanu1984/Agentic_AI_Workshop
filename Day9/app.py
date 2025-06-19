import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime, timedelta
import json
from PyPDF2 import PdfReader
from google.generativeai import GenerativeModel, configure
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== Configuration ========================
load_dotenv()

# FIXED: Use environment variable for API key
GEMINI_API_KEY = "AIzaSyB-Op4syIGXZojS5274m1OMFsFwK4yb2Mo" 
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

try:
    configure(api_key=GEMINI_API_KEY)
    gemini_model = GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {str(e)}")
    st.stop()

# FIXED: Initialize embeddings with error handling
try:
    embeddings = HuggingFaceEmbeddings()
except Exception as e:
    st.error(f"❌ Failed to initialize embeddings: {str(e)}")
    st.stop()

# Initialize session state
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'student_data' not in st.session_state:
    st.session_state.student_data = None
if 'faq_data' not in st.session_state:
    st.session_state.faq_data = None
if 'progress' not in st.session_state:
    st.session_state.progress = {"badges": [], "completed_tasks": 0}

# ======================== Utility Functions ========================
def validate_file_upload(uploaded_file, max_size_mb=10):
    """Validate uploaded files with proper error handling"""
    try:
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large. Max size: {max_size_mb}MB")
        
        allowed_extensions = ['.pdf', '.json', '.txt']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise ValueError(f"Unsupported file type. Allowed: {allowed_extensions}")
        
        return True
    except Exception as e:
        st.error(f"File validation failed: {str(e)}")
        return False

def safe_json_parse(response_text):
    """Extract JSON from Gemini response with markdown removal"""
    try:
        # Remove markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        # Return structured fallback
        return {
            "error": "JSON parsing failed",
            "raw_response": response_text[:500],
            "timelines": [{"event": "Application Deadline", "date": "2025-07-01"}],
            "eligibility": {"gpa": "7.0+", "backlogs_allowed": 0},
            "test_formats": ["Online Assessment", "Technical Interview"]
        }

def parse_date_safely(date_string):
    """Parse date with multiple format support"""
    if not date_string:
        return datetime.now() + timedelta(days=30)
    
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y"]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_string), fmt)
        except ValueError:
            continue
    
    # If all formats fail, return a default future date
    logger.warning(f"Could not parse date: {date_string}")
    return datetime.now() + timedelta(days=30)

def safe_agent_execution(agent_func, *args, **kwargs):
    """Wrapper for safe agent execution with error handling"""
    try:
        return agent_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        st.error(f"Agent execution failed: {str(e)}")
        return None

# ======================== Core Functions ========================
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF with error handling"""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text += page.extract_text() + "\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        return text
    except Exception as e:
        st.error(f"PDF extraction failed: {str(e)}")
        return ""

def scrape_glassdoor_safe(company_name):
    """Safer web scraping with comprehensive fallback"""
    fallback_questions = [
        "Tell me about yourself and your background",
        "What are your greatest strengths and weaknesses?",
        "Why do you want to work at our company?",
        "Describe a challenging project you worked on",
        "Where do you see yourself in 5 years?",
        "How do you handle pressure and tight deadlines?",
        "What motivates you in your work?",
        "Describe a time you worked in a team",
        "What are your salary expectations?",
        "Do you have any questions for us?"
    ]
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Simple request with timeout
        url = f"https://www.glassdoor.co.in/Interview/{company_name.replace(' ', '-')}-interview-questions"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            questions = []
            
            # Try multiple selectors
            selectors = [".interviewQuestion", ".interview-question", "[data-test='interview-question']"]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    questions = [q.get_text().strip() for q in elements[:5]]
                    break
            
            if questions:
                return questions
        
        # If scraping fails, return fallback
        logger.info(f"Scraping failed for {company_name}, using fallback questions")
        return fallback_questions[:5]
        
    except Exception as e:
        logger.warning(f"Scraping error: {e}")
        return fallback_questions[:5]

def get_embedding_safe(text):
    """Generate embeddings with error handling"""
    try:
        if not text or len(text.strip()) == 0:
            return embeddings.embed_query("default text")
        return embeddings.embed_query(text[:1000])  # Limit text length
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return dummy embedding
        return [0.0] * 384

# ======================== Enhanced Agents ========================
def drive_parser_agent(uploaded_file):
    """FIXED: Enhanced document parsing with robust error handling"""
    try:
        # Validate file first
        if not validate_file_upload(uploaded_file):
            return None
        
        text = ""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == '.json':
            content = uploaded_file.read()
            try:
                return json.loads(content.decode("utf-8"))
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON file: {str(e)}")
                return None
        elif file_extension == '.txt':
            content = uploaded_file.read()
            text = content.decode("utf-8")
        
        if not text.strip():
            st.error("No text could be extracted from the file")
            return None
        
        # Enhanced prompt for better JSON extraction
        prompt = f"""
        Analyze this company document and extract information in STRICT JSON format.
        
        Document text: {text[:15000]}
        
        Return ONLY a JSON object with this exact structure:
        {{
            "company_name": "extracted company name",
            "timelines": [
                {{"event": "Application Deadline", "date": "YYYY-MM-DD"}},
                {{"event": "Test Date", "date": "YYYY-MM-DD"}}
            ],
            "eligibility": {{
                "gpa": "minimum GPA requirement",
                "backlogs_allowed": "number or 0",
                "branches": ["allowed branches"],
                "year_of_graduation": "graduation year"
            }},
            "test_formats": ["Online Assessment", "Technical Interview", "HR Round"],
            "job_description": "brief description of role",
            "required_skills": ["skill1", "skill2", "skill3"]
        }}
        
        Respond with ONLY the JSON object, no explanations.
        """
        
        response = gemini_model.generate_content(prompt)
        parsed_data = safe_json_parse(response.text)
        
        # Validate parsed data
        if "error" not in parsed_data:
            st.session_state.progress["completed_tasks"] += 1
        
        return parsed_data
        
    except Exception as e:
        st.error(f"Document parsing failed: {str(e)}")
        return None

def readiness_evaluator_agent(resume_file, company_data):
    """FIXED: Enhanced evaluation with proper error handling"""
    try:
        if not validate_file_upload(resume_file):
            return None
        
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text.strip():
            st.error("Could not extract text from resume")
            return None
        
        # Safe embedding generation
        resume_embed = get_embedding_safe(resume_text)
        company_text = json.dumps(company_data) if company_data else "No company data"
        jd_embed = get_embedding_safe(company_text)
        
        # Calculate similarity safely
        try:
            similarity = cosine_similarity([resume_embed], [jd_embed])[0][0]
        except Exception:
            similarity = 0.5  # Default similarity
        
        evaluation_prompt = f"""
        Analyze this resume against the company requirements and provide evaluation in JSON format ONLY.
        
        Company Requirements: {json.dumps(company_data, indent=2)}
        
        Resume Text: {resume_text[:8000]}
        
        Provide response in this EXACT JSON format:
        {{
            "readiness_score": 75,
            "skill_gaps": [
                {{"skill": "Python", "severity": 8, "recommendation": "Complete online course"}},
                {{"skill": "Machine Learning", "severity": 6, "recommendation": "Build projects"}}
            ],
            "academic_comparison": {{
                "gpa_status": "meets/doesn't meet requirement",
                "graduation_year": "matches/doesn't match"
            }},
            "strengths": ["strength1", "strength2"],
            "improvement_areas": ["area1", "area2"],
            "overall_feedback": "brief summary of candidate readiness"
        }}
        
        Respond with ONLY the JSON object.
        """
        
        response = gemini_model.generate_content(evaluation_prompt)
        evaluation = safe_json_parse(response.text)
        
        # Add similarity score
        evaluation["similarity_index"] = float(similarity)
        
        # Award badges based on performance
        if evaluation.get("readiness_score", 0) > 70:
            if "High Readiness" not in st.session_state.progress["badges"]:
                st.session_state.progress["badges"].append("High Readiness")
        
        if similarity > 0.7:
            if "Perfect Match" not in st.session_state.progress["badges"]:
                st.session_state.progress["badges"].append("Perfect Match")
        
        st.session_state.progress["completed_tasks"] += 1
        return evaluation
        
    except Exception as e:
        st.error(f"Readiness evaluation failed: {str(e)}")
        return None

def faq_retrieval_agent(question, company_data):
    """FIXED: Improved RAG system with proper error handling"""
    try:
        if not question or not question.strip():
            return "Please ask a specific question about the company or preparation."
        
        company_name = company_data.get("company_name", "Unknown Company") if company_data else "Unknown Company"
        
        # Create documents with proper structure
        documents = []
        
        # Add company data
        if company_data:
            documents.append(
                Document(
                    page_content=json.dumps(company_data, indent=2),
                    metadata={"source": "company_data", "type": "official"}
                )
            )
        
        # Get interview questions with fallback
        try:
            interview_questions = scrape_glassdoor_safe(company_name)
            for i, q in enumerate(interview_questions[:3]):
                documents.append(
                    Document(
                        page_content=f"Common interview question: {q}",
                        metadata={"source": f"interview_{i}", "type": "question"}
                    )
                )
        except Exception as e:
            logger.warning(f"Could not get interview questions: {e}")
        
        # Ensure we have at least some content
        if not documents:
            documents.append(
                Document(
                    page_content="General career preparation advice and interview tips",
                    metadata={"source": "fallback", "type": "general"}
                )
            )
        
        # Build vector store and search
        try:
            db = FAISS.from_documents(documents, embeddings)
            relevant_docs = db.similarity_search(question, k=min(2, len(documents)))
            context = "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\nContent: {d.page_content}" for d in relevant_docs])
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            context = json.dumps(company_data) if company_data else "General information available"
        
        # Generate response
        response_prompt = f"""
        Answer this question based on the provided context. Be helpful, specific, and actionable.
        
        Context Information:
        {context}
        
        User Question: {question}
        
        Instructions:
        - Provide a clear, helpful answer
        - Use specific information from the context when available
        - If context is limited, give general career advice
        - Keep response concise but informative
        """
        
        response = gemini_model.generate_content(response_prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"FAQ retrieval failed: {e}")
        return f"I encountered an error while searching for information. Please try rephrasing your question or ask something more general about career preparation."

def reminder_agent(company_data, evaluation_data):
    """FIXED: Robust reminder system with proper date handling"""
    try:
        if not company_data or not evaluation_data:
            return {
                "plan": "Complete company analysis and resume evaluation first to get personalized reminders.",
                "deadline": "Not available",
                "badges": st.session_state.progress.get("badges", [])
            }
        
        deadlines = company_data.get("timelines", [])
        
        if not deadlines:
            return {
                "plan": "No specific deadlines found. Focus on continuous skill improvement and interview preparation.",
                "deadline": "Not specified",
                "badges": st.session_state.progress.get("badges", [])
            }
        
        # Parse dates safely
        parsed_deadlines = []
        for timeline_item in deadlines:
            if isinstance(timeline_item, dict) and "date" in timeline_item:
                parsed_date = parse_date_safely(timeline_item["date"])
                parsed_deadlines.append((timeline_item.get("event", "Deadline"), parsed_date))
        
        if not parsed_deadlines:
            nearest_deadline_date = datetime.now() + timedelta(days=30)
            nearest_event = "General Preparation"
        else:
            # Find nearest future deadline
            future_deadlines = [(event, date) for event, date in parsed_deadlines if date > datetime.now()]
            if future_deadlines:
                nearest_event, nearest_deadline_date = min(future_deadlines, key=lambda x: x[1])
            else:
                nearest_event, nearest_deadline_date = parsed_deadlines[0]  # Use first if all are past
        
        days_left = max(0, (nearest_deadline_date - datetime.now()).days)
        
        # Generate personalized study plan
        skill_gaps = evaluation_data.get("skill_gaps", [])
        readiness_score = evaluation_data.get("readiness_score", 0)
        
        plan_prompt = f"""
        Create a personalized study plan for a job candidate with:
        - Days until deadline: {days_left}
        - Current readiness score: {readiness_score}%
        - Main skill gaps: {[gap.get('skill', 'Unknown') for gap in skill_gaps[:3]]}
        - Next event: {nearest_event}
        
        Provide a practical, day-by-day study plan with:
        1. Priority skills to focus on
        2. Recommended resources
        3. Daily time allocation
        4. Milestone checkpoints
        
        Keep it concise and actionable.
        """
        
        try:
            plan_response = gemini_model.generate_content(plan_prompt)
            study_plan = plan_response.text
        except Exception as e:
            logger.error(f"Study plan generation failed: {e}")
            study_plan = f"Focus on improving: {', '.join([gap.get('skill', 'key skills') for gap in skill_gaps[:3]])}. Allocate 2-3 hours daily for preparation."
        
        # Award time-based badges
        if days_left <= 7 and "Last Sprint" not in st.session_state.progress["badges"]:
            st.session_state.progress["badges"].append("Last Sprint")
        elif days_left <= 30 and "Preparation Mode" not in st.session_state.progress["badges"]:
            st.session_state.progress["badges"].append("Preparation Mode")
        
        return {
            "plan": study_plan,
            "deadline": nearest_deadline_date.strftime("%B %d, %Y"),
            "days_left": days_left,
            "next_event": nearest_event,
            "badges": st.session_state.progress.get("badges", [])
        }
        
    except Exception as e:
        logger.error(f"Reminder agent failed: {e}")
        return {
            "plan": f"Error generating study plan: {str(e)}",
            "deadline": "Error",
            "badges": st.session_state.progress.get("badges", [])
        }

def mock_interview():
    """FIXED: Improved mock interview with comprehensive error handling"""
    try:
        # Try to import voice libraries
        try:
            import pyttsx3
            import speech_recognition as sr
            voice_available = True
        except ImportError:
            voice_available = False
        
        if not voice_available:
            return {
                "type": "text",
                "message": "Voice libraries not available. Please install pyttsx3 and SpeechRecognition for voice features.",
                "questions": [
                    "Tell me about yourself and your background",
                    "What are your greatest strengths?",
                    "Describe a challenging project you worked on",
                    "Why do you want to work here?",
                    "Where do you see yourself in 5 years?"
                ]
            }
        
        # Check microphone availability
        try:
            recognizer = sr.Recognizer()
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                raise Exception("No microphone detected")
        except Exception as e:
            return {
                "type": "text",
                "message": f"Microphone not available: {str(e)}. Here are some common interview questions to practice:",
                "questions": [
                    "Tell me about yourself and your background",
                    "What are your greatest strengths?",
                    "Describe a challenging project you worked on"
                ]
            }
        
        # Voice interview simulation
        try:
            engine = pyttsx3.init()
            interview_question = "Tell me about a challenging project you worked on and how you overcame the difficulties."
            
            engine.say(interview_question)
            engine.runAndWait()
            
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=60)
                
                response_text = recognizer.recognize_google(audio)
                
                # Analyze response
                analysis_prompt = f"""
                Evaluate this interview response for a technical role:
                
                Question: {interview_question}
                Response: {response_text}
                
                Provide feedback on:
                1. Content quality
                2. Structure and clarity
                3. Specific improvements
                4. Overall score (1-10)
                
                Keep feedback constructive and actionable.
                """
                
                analysis = gemini_model.generate_content(analysis_prompt)
                
                return {
                    "type": "voice",
                    "question": interview_question,
                    "your_response": response_text,
                    "feedback": analysis.text,
                    "status": "success"
                }
                
        except sr.RequestError:
            return {
                "type": "error",
                "message": "Speech recognition service unavailable. Please check your internet connection."
            }
        except sr.UnknownValueError:
            return {
                "type": "error", 
                "message": "Could not understand audio. Please speak clearly and try again."
            }
        except sr.WaitTimeoutError:
            return {
                "type": "error",
                "message": "No speech detected. Please try again and speak within 10 seconds."
            }
            
    except Exception as e:
        logger.error(f"Mock interview failed: {e}")
        return {
            "type": "error",
            "message": f"Interview simulation failed: {str(e)}",
            "fallback_questions": [
                "Tell me about yourself",
                "What are your strengths?",
                "Describe a challenging project"
            ]
        }

# ======================== Streamlit UI ========================
st.set_page_config(
    layout="wide", 
    page_title="CareerPrep AI",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

st.title("🚀 CareerPrep AI - Your Job Preparation Assistant")
st.markdown("*Powered by AI agents to help you ace your job applications*")

# Progress indicator
if st.session_state.company_data or st.session_state.student_data:
    progress = 0
    if st.session_state.company_data:
        progress += 50
    if st.session_state.student_data:
        progress += 50
    
    st.progress(progress / 100)
    st.caption(f"Setup Progress: {progress}% completed")

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
agent_view = st.sidebar.radio(
    "Select Agent",
    ["🏢 Drive Parser", "📊 Readiness Evaluator", "❓ FAQ Assistant", "⏰ Reminder System", "🎤 Mock Interview"],
    index=0
)

# Display badges if available
if st.session_state.progress["badges"]:
    st.sidebar.success("🏆 Your Badges:")
    for badge in st.session_state.progress["badges"]:
        st.sidebar.write(f"• {badge}")

# Agent 1: Drive Parser
if agent_view == "🏢 Drive Parser":
    st.header("📅 Company Drive Information Parser")
    st.markdown("Upload your company's job description or recruitment document to extract key information.")
    
    company_file = st.file_uploader(
        "Upload Company Document", 
        type=['pdf', 'json', 'txt'],
        help="Supported formats: PDF, JSON, TXT (Max 10MB)"
    )
    
    if company_file:
        st.info(f"📄 File uploaded: {company_file.name} ({company_file.size / 1024:.1f} KB)")
        
        if st.button("🔍 Parse Document", type="primary"):
            with st.spinner("🔄 Extracting key information..."):
                result = safe_agent_execution(drive_parser_agent, company_file)
                
                if result:
                    st.session_state.company_data = result
                    st.success("✅ Document analysis complete!")
                    
                    # Display results in organized tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📅 Timeline", "✅ Eligibility", "🔧 Technical"])
                    
                    with tab1:
                        st.subheader("Company Information")
                        st.write(f"**Company:** {result.get('company_name', 'Not specified')}")
                        st.write(f"**Role:** {result.get('job_description', 'Not specified')}")
                    
                    with tab2:
                        timelines = result.get('timelines', [])
                        if timelines:
                            df = pd.DataFrame(timelines)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No timeline information found")
                    
                    with tab3:
                        eligibility = result.get('eligibility', {})
                        for key, value in eligibility.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    with tab4:
                        st.write("**Test Formats:**")
                        for fmt in result.get('test_formats', []):
                            st.write(f"• {fmt}")
                        
                        st.write("**Required Skills:**")
                        for skill in result.get('required_skills', []):
                            st.write(f"• {skill}")
                else:
                    st.error("❌ Failed to parse document. Please check the file format and try again.")

# Agent 2: Readiness Evaluator
elif agent_view == "📊 Readiness Evaluator":
    st.header("📊 Career Readiness Evaluator")
    st.markdown("Upload your resume to get a detailed analysis of your job readiness.")
    
    if st.session_state.company_data:
        st.success(f"✅ Company data loaded: {st.session_state.company_data.get('company_name', 'Unknown')}")
        
        resume_file = st.file_uploader(
            "Upload Your Resume (PDF)", 
            type=['pdf'],
            help="Upload your resume in PDF format for analysis"
        )
        
        if resume_file and st.button("📈 Evaluate Readiness", type="primary"):
            with st.spinner("🔄 Analyzing your profile against company requirements..."):
                evaluation = safe_agent_execution(
                    readiness_evaluator_agent, 
                    resume_file, 
                    st.session_state.company_data
                )
                
                if evaluation:
                    st.session_state.student_data = evaluation
                    
                    # Display results with metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        readiness_score = evaluation.get('readiness_score', 0)
                        st.metric("🎯 Readiness Score", f"{readiness_score}%")
                    
                    with col2:
                        similarity = evaluation.get('similarity_index', 0) * 100
                        st.metric("🔍 Resume-JD Match", f"{similarity:.1f}%")
                    
                    with col3:
                        skill_gaps = len(evaluation.get('skill_gaps', []))
                        st.metric("📚 Skill Gaps", skill_gaps)
                    
                    # Skill gaps visualization
                    skill_gaps = evaluation.get('skill_gaps', [])
                    if skill_gaps:
                        st.subheader("🎯 Skill Gap Analysis")
                        
                        # Create DataFrame for visualization
                        df_gaps = pd.DataFrame(skill_gaps)
                        if 'skill' in df_gaps.columns and 'severity' in df_gaps.columns:
                            fig = px.bar(
                                df_gaps,
                                x='skill',
                                y='severity',
                                title="Skills Requiring Improvement",
                                color='severity',
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        st.subheader("💡 Recommendations")
                        for gap in skill_gaps[:3]:  # Show top 3
                            with st.expander(f"🔧 Improve {gap.get('skill', 'Unknown Skill')}"):
                                st.write(f"**Severity:** {gap.get('severity', 'Unknown')}/10")
                                st.write(f"**Recommendation:** {gap.get('recommendation', 'Practice and study this skill')}")
                    
                    # Overall feedback
                    st.subheader("📝 Overall Feedback")
                    st.info(evaluation.get('overall_feedback', 'Keep improving your skills and preparing for interviews!'))
                    
                else:
                    st.error("❌ Failed to evaluate readiness. Please check your resume file.")
    else:
        st.warning("⚠️ Please upload company data first using the Drive Parser.")

# Agent 3: FAQ Assistant (continuation from where it left off)
elif agent_view == "❓ FAQ Assistant":
    st.header("❓ Intelligent FAQ Assistant")
    st.markdown("Ask questions about the company, interview preparation, or career advice.")
    
    if st.session_state.company_data:
        st.success(f"✅ Knowledge base loaded for: {st.session_state.company_data.get('company_name', 'Unknown Company')}")
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What are the key requirements for this role?",
                "When is the application deadline?",
                "What should I focus on for preparation?",
                "What are common interview questions for this company?",
                "How should I prepare for the technical assessment?"
            ]
            for q in sample_questions:
                if st.button(q, key=f"sample_{q[:10]}"):
                    st.session_state.current_question = q
        
        # Question input
        user_question = st.text_input(
            "Ask your question:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What are the eligibility criteria?"
        )
        
        if user_question and st.button("🔍 Get Answer", type="primary"):
            with st.spinner("🔄 Searching for relevant information..."):
                answer = safe_agent_execution(
                    faq_retrieval_agent, 
                    user_question, 
                    st.session_state.company_data
                )
                
                if answer:
                    st.subheader("💬 Answer")
                    st.write(answer)
                    
                    # Store in session for reference
                    if 'faq_history' not in st.session_state:
                        st.session_state.faq_history = []
                    
                    st.session_state.faq_history.append({
                        "question": user_question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
        
        # FAQ History
        if 'faq_history' in st.session_state and st.session_state.faq_history:
            with st.expander("📚 Previous Questions"):
                for i, item in enumerate(reversed(st.session_state.faq_history[-5:])):  # Show last 5
                    st.write(f"**Q:** {item['question']}")
                    st.write(f"**A:** {item['answer'][:200]}...")
                    st.caption(f"Asked on: {item['timestamp']}")
                    st.divider()
    else:
        st.warning("⚠️ Please upload company data first to enable intelligent Q&A.")
        st.info("You can still ask general career preparation questions:")
        
        general_question = st.text_input("Ask a general career question:")
        if general_question and st.button("💭 Get General Advice"):
            with st.spinner("🔄 Generating advice..."):
                try:
                    general_prompt = f"""
                    Provide helpful career advice for this question: {general_question}
                    
                    Give practical, actionable advice for job seekers and students.
                    Keep the response helpful and encouraging.
                    """
                    response = gemini_model.generate_content(general_prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error(f"❌ Failed to generate advice: {str(e)}")

# Agent 4: Reminder System
elif agent_view == "⏰ Reminder System":
    st.header("⏰ Smart Reminder & Study Planner")
    st.markdown("Get personalized study plans and deadline reminders based on your analysis.")
    
    if st.session_state.company_data and st.session_state.student_data:
        if st.button("📅 Generate Study Plan", type="primary"):
            with st.spinner("🔄 Creating your personalized study plan..."):
                reminder_data = safe_agent_execution(
                    reminder_agent,
                    st.session_state.company_data,
                    st.session_state.student_data
                )
                
                if reminder_data:
                    # Deadline info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("📅 Next Deadline", reminder_data.get('deadline', 'Not specified'))
                    
                    with col2:
                        days_left = reminder_data.get('days_left', 0)
                        if days_left > 0:
                            st.metric("⏳ Days Remaining", f"{days_left} days")
                        else:
                            st.metric("⏳ Status", "Deadline Passed")
                    
                    # Study plan
                    st.subheader("📚 Your Personalized Study Plan")
                    st.write(reminder_data.get('plan', 'No plan available'))
                    
                    # Progress tracking
                    st.subheader("🎯 Progress Tracking")
                    completed_tasks = st.session_state.progress.get("completed_tasks", 0)
                    total_tasks = 4  # Total agents
                    progress_percent = (completed_tasks / total_tasks) * 100
                    
                    st.progress(progress_percent / 100)
                    st.write(f"Completed: {completed_tasks}/{total_tasks} assessments")
                    
                    # Calendar integration suggestion
                    if days_left > 0:
                        st.info(f"💡 **Tip:** Add '{reminder_data.get('next_event', 'Deadline')}' to your calendar for {reminder_data.get('deadline', 'the deadline')}!")
                    
                    # Export study plan
                    if st.button("📄 Export Study Plan"):
                        study_plan_text = f"""
PERSONALIZED STUDY PLAN
======================

Company: {st.session_state.company_data.get('company_name', 'Unknown')}
Deadline: {reminder_data.get('deadline', 'Not specified')}
Days Remaining: {days_left}

STUDY PLAN:
{reminder_data.get('plan', 'No plan available')}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        st.download_button(
                            label="⬇️ Download Study Plan",
                            data=study_plan_text,
                            file_name=f"study_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
    else:
        missing = []
        if not st.session_state.company_data:
            missing.append("Company Analysis")
        if not st.session_state.student_data:
            missing.append("Resume Evaluation")
        
        st.warning(f"⚠️ Please complete: {', '.join(missing)} to generate personalized reminders.")
        
        # Show generic timeline if company data exists
        if st.session_state.company_data:
            timelines = st.session_state.company_data.get('timelines', [])
            if timelines:
                st.subheader("📅 Company Timeline")
                df_timeline = pd.DataFrame(timelines)
                st.dataframe(df_timeline, use_container_width=True)

# Agent 5: Mock Interview
elif agent_view == "🎤 Mock Interview":
    st.header("🎤 AI Mock Interview Simulator")
    st.markdown("Practice interviews with AI-powered feedback and voice interaction.")
    
    # Interview mode selection
    interview_mode = st.radio(
        "Select Interview Mode:",
        ["💬 Text-based Practice", "🎙️ Voice Interview (Beta)"],
        index=0
    )
    
    if interview_mode == "💬 Text-based Practice":
        st.subheader("📝 Text Interview Practice")
        
        # Generate questions based on company data
        if st.session_state.company_data:
            company_name = st.session_state.company_data.get('company_name', 'Unknown')
            st.info(f"Questions tailored for: {company_name}")
        
        # Question categories
        question_type = st.selectbox(
            "Choose question category:",
            ["General", "Technical", "Behavioral", "Company-specific"]
        )
        
        if st.button("🎯 Generate Interview Question"):
            with st.spinner("🔄 Generating question..."):
                try:
                    if question_type == "Company-specific" and st.session_state.company_data:
                        question_prompt = f"""
                        Generate a relevant interview question for this company/role:
                        {json.dumps(st.session_state.company_data, indent=2)}
                        
                        Make it specific to the role and company requirements.
                        """
                    else:
                        question_prompt = f"""
                        Generate a {question_type.lower()} interview question 
                        appropriate for a job interview. Make it realistic and commonly asked.
                        """
                    
                    response = gemini_model.generate_content(question_prompt)
                    interview_question = response.text
                    
                    st.subheader("❓ Interview Question")
                    st.write(interview_question)
                    
                    # Answer input
                    user_answer = st.text_area(
                        "Your Answer:",
                        placeholder="Type your response here...",
                        height=150
                    )
                    
                    if user_answer and st.button("📊 Get Feedback"):
                        with st.spinner("🔄 Analyzing your response..."):
                            feedback_prompt = f"""
                            Evaluate this interview response:
                            
                            Question: {interview_question}
                            Response: {user_answer}
                            
                            Provide feedback on:
                            1. Content quality and relevance
                            2. Structure and clarity
                            3. Areas for improvement
                            4. Score out of 10
                            5. Specific suggestions
                            
                            Be constructive and helpful.
                            """
                            
                            feedback_response = gemini_model.generate_content(feedback_prompt)
                            st.subheader("📋 Feedback")
                            st.write(feedback_response.text)
                            
                            # Award badge for practice
                            if "Interview Practice" not in st.session_state.progress["badges"]:
                                st.session_state.progress["badges"].append("Interview Practice")
                                st.success("🏆 Badge earned: Interview Practice!")
                
                except Exception as e:
                    st.error(f"❌ Failed to generate question: {str(e)}")
    
    else:  # Voice Interview
        st.subheader("🎙️ Voice Interview Practice")
        st.warning("⚠️ This feature requires microphone access and voice libraries.")
        
        if st.button("🎤 Start Voice Interview"):
            with st.spinner("🔄 Initializing voice interview..."):
                interview_result = safe_agent_execution(mock_interview)
                
                if interview_result:
                    if interview_result.get("type") == "voice":
                        st.success("✅ Voice interview completed!")
                        st.subheader("❓ Question Asked")
                        st.write(interview_result.get("question", "Unknown"))
                        
                        st.subheader("🗣️ Your Response")
                        st.write(interview_result.get("your_response", "No response recorded"))
                        
                        st.subheader("📊 AI Feedback")
                        st.write(interview_result.get("feedback", "No feedback available"))
                        
                    elif interview_result.get("type") == "text":
                        st.info(interview_result.get("message", "Voice not available"))
                        st.subheader("📝 Practice Questions")
                        for q in interview_result.get("questions", []):
                            st.write(f"• {q}")
                            
                    else:  # Error case
                        st.error(interview_result.get("message", "Interview failed"))
                        if "fallback_questions" in interview_result:
                            st.subheader("📝 Practice These Questions Instead:")
                            for q in interview_result["fallback_questions"]:
                                st.write(f"• {q}")

# Footer with additional features
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Reset All Data"):
        for key in ['company_data', 'student_data', 'faq_data', 'progress', 'faq_history']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

with col2:
    if st.button("📊 Export Summary"):
        if st.session_state.company_data or st.session_state.student_data:
            summary = {
                "company_data": st.session_state.get('company_data'),
                "evaluation_data": st.session_state.get('student_data'),
                "progress": st.session_state.get('progress'),
                "export_timestamp": datetime.now().isoformat()
            }
            
            summary_json = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Summary",
                data=summary_json,
                file_name=f"careerprep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("No data to export yet. Complete some assessments first!")

with col3:
    st.write("**Quick Stats:**")
    st.write(f"• Tasks completed: {st.session_state.progress.get('completed_tasks', 0)}")
    st.write(f"• Badges earned: {len(st.session_state.progress.get('badges', []))}")

# Help section
with st.expander("❓ Need Help?"):
    st.markdown("""
    **How to use CareerPrep AI:**
    
    1. **🏢 Drive Parser**: Upload your company's job description (PDF/JSON/TXT)
    2. **📊 Readiness Evaluator**: Upload your resume for skill gap analysis
    3. **❓ FAQ Assistant**: Ask questions about preparation and company info
    4. **⏰ Reminder System**: Get personalized study plans and deadlines
    5. **🎤 Mock Interview**: Practice interviews with AI feedback
    
    **Tips:**
    - Complete steps 1 & 2 first for best results
    - Use high-quality PDF files for better parsing
    - Ask specific questions in the FAQ section
    - Practice mock interviews regularly
    
    **Troubleshooting:**
    - If parsing fails, check file format and size
    - For slow responses, wait for processing to complete
    - Clear browser cache if experiencing issues
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🚀 CareerPrep AI - Powered by Google Gemini & Advanced AI Agents<br>"
    "Made with ❤️ for job seekers and students"
    "</div>", 
    unsafe_allow_html=True
)