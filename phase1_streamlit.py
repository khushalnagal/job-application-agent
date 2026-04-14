"""
Phase 1 Frontend — Job Description Analyzer
Streamlit UI for the LangChain extraction pipeline
"""

import os
import streamlit as st
from phase1_backend import analyze_job_description

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="JD Analyzer — Phase 1",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Root theme */
    :root {
        --bg: #0d0d0d;
        --surface: #161616;
        --surface2: #1e1e1e;
        --border: #2a2a2a;
        --accent: #00ff88;
        --accent2: #00c4ff;
        --text: #e8e8e8;
        --muted: #666;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: var(--text) !important;
    }

    .stTextArea textarea {
        background-color: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #000 !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3) !important;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .card-accent {
        border-left: 3px solid var(--accent);
    }

    .card-blue {
        border-left: 3px solid var(--accent2);
    }

    .skill-tag {
        display: inline-block;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        color: var(--accent);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-family: 'Space Mono', monospace;
        margin: 3px;
    }

    .skill-tag-preferred {
        background: rgba(0, 196, 255, 0.1);
        border-color: rgba(0, 196, 255, 0.3);
        color: var(--accent2);
    }

    .metric-box {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: var(--accent);
    }

    .metric-label {
        font-size: 12px;
        color: var(--muted);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--muted);
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .stExpander {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .stTextInput input {
        background-color: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: var(--text) !important;
        line-height: 1.7 !important;
    }

    .gap-section {
        background: var(--surface2);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid var(--border);
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.8;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — API Key + Instructions
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title">⚡ JD<br>Analyzer</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-title">🔑 Configuration</p>', unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✅ Key set!")

    st.markdown("---")
    st.markdown('<p class="section-title">📚 Phase 1 Concepts</p>', unsafe_allow_html=True)
    concepts = {
        "🤖 LLM": "ChatGroq with Llama 3.3-70b",
        "📝 Prompt Template": "ChatPromptTemplate with system + human messages",
        "🔗 LCEL Chain": "prompt | llm | parser",
        "📦 StrOutputParser": "Returns plain string",
        "🏗️ PydanticOutputParser": "Returns typed Python object",
    }
    for concept, desc in concepts.items():
        st.markdown(f"**{concept}**")
        st.caption(desc)
        st.markdown("")

    st.markdown("---")
    st.caption("Phase 1 of AI Job Application Agent")

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown('<h1 style="font-family: Space Mono, monospace; margin-bottom: 4px;">Job Description Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #666; margin-bottom: 32px;">Paste any JD → Get structured extraction + skills gap analysis using LangChain</p>', unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<p class="section-title">📋 Job Description</p>', unsafe_allow_html=True)
    jd_input = st.text_area(
        label="jd",
        label_visibility="collapsed",
        placeholder="""Paste the full job description here...

Example:
We are hiring a Machine Learning Engineer to build AI agents 
using LangChain and LangGraph. You will design RAG pipelines,
deploy models with FastAPI, and collaborate with our AI team...

Requirements:
- 2+ years Python experience
- LangChain / LLM experience
- etc.""",
        height=350
    )

with col2:
    st.markdown('<p class="section-title">🧑‍💻 Your Current Skills</p>', unsafe_allow_html=True)
    skills_input = st.text_area(
        label="skills",
        label_visibility="collapsed",
        placeholder="""List your current skills (comma separated or one per line):

Python, Machine Learning, LangChain,
LangGraph, RAG, FAISS, Streamlit,
SQL, Git, Docker basics...""",
        height=200
    )

    st.markdown("")
    run_gap = st.checkbox("Include Skills Gap Analysis", value=True)

    st.markdown("")
    analyze_btn = st.button("🚀 Analyze JD", use_container_width=True)

# ─────────────────────────────────────────────
# Output section
# ─────────────────────────────────────────────
if analyze_btn:
    if not jd_input.strip():
        st.error("⚠️ Please paste a job description first.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        candidate_skills = skills_input if (run_gap and skills_input.strip()) else None

        with st.spinner("🤖 Analyzing job description..."):
            try:
                results = analyze_job_description(jd_input, candidate_skills)

                st.markdown("---")
                st.markdown("## 📊 Results")

                # ── Tab layout for results ──
                tab1, tab2, tab3 = st.tabs(["📝 Summary", "🏗️ Structured Data", "📈 Skills Gap"])

                # Tab 1: Summary
                with tab1:
                    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
                    st.markdown("#### Quick Summary")
                    st.write(results["summary"])
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown('<p class="section-title">💡 What happened under the hood</p>', unsafe_allow_html=True)
                    with st.expander("See the chain that ran"):
                        st.code("""
# Chain 1: Summary Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR analyst..."),
    ("human", "Summarize this JD: {job_description}")
])

chain = prompt | ChatGroq() | StrOutputParser()
result = chain.invoke({"job_description": jd_text})
# result is a plain string ✅
                        """, language="python")

                # Tab 2: Structured Data
                with tab2:
                    jd_data = results["structured_data"]

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{len(jd_data.required_skills)}</div>
                            <div class="metric-label">Required Skills</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{len(jd_data.preferred_skills)}</div>
                            <div class="metric-label">Preferred Skills</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{len(jd_data.responsibilities)}</div>
                            <div class="metric-label">Responsibilities</div>
                        </div>""", unsafe_allow_html=True)
                    with m4:
                        loc = jd_data.location or "N/A"
                        loc_short = loc[:8] + "..." if len(loc) > 8 else loc
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="font-size:18px">{loc_short}</div>
                            <div class="metric-label">Location</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")

                    # Details
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"**🎯 Role:** {jd_data.role}")
                        if jd_data.company:
                            st.markdown(f"**🏢 Company:** {jd_data.company}")
                        st.markdown(f"**📅 Experience:** {jd_data.experience_required}")
                        if jd_data.education:
                            st.markdown(f"**🎓 Education:** {jd_data.education}")
                        if jd_data.job_type:
                            st.markdown(f"**💼 Type:** {jd_data.job_type}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**📌 Responsibilities**")
                        for r in jd_data.responsibilities:
                            st.markdown(f"• {r}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with d2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**✅ Required Skills**")
                        skills_html = " ".join([f'<span class="skill-tag">{s}</span>' for s in jd_data.required_skills])
                        st.markdown(skills_html, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**⭐ Preferred Skills**")
                        pref_html = " ".join([f'<span class="skill-tag skill-tag-preferred">{s}</span>' for s in jd_data.preferred_skills])
                        st.markdown(pref_html, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
                        st.markdown("**🔍 Role Summary**")
                        st.write(jd_data.summary)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("See the chain that ran"):
                        st.code("""
# Chain 2: Structured Extraction Chain
class JobDescription(BaseModel):
    role: str
    required_skills: List[str]
    # ... more fields

parser = PydanticOutputParser(pydantic_object=JobDescription)

prompt = ChatPromptTemplate.from_messages([
    ("system", "...{format_instructions}"),
    ("human", "Extract from this JD: {job_description}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | ChatGroq(temperature=0) | parser
result = chain.invoke({"job_description": jd_text})
# result.role, result.required_skills etc ✅
                        """, language="python")

                # Tab 3: Skills Gap
                with tab3:
                    if "gap_analysis" in results:
                        st.markdown('<div class="gap-section">', unsafe_allow_html=True)
                        st.write(results["gap_analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)

                        with st.expander("See the chain that ran"):
                            st.code("""
# Chain 3: Skills Gap Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a career coach..."),
    ("human", '''
        JD: {job_description}
        My Skills: {candidate_skills}
        Analyze my fit...
    ''')
])

chain = prompt | ChatGroq(temperature=0.2) | StrOutputParser()
result = chain.invoke({
    "job_description": jd_text,
    "candidate_skills": skills
})
                            """, language="python")
                    else:
                        st.info("💡 Enable 'Skills Gap Analysis' and add your skills to see this.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.markdown("**Common fixes:**")
                st.markdown("- Check your Groq API key is correct")
                st.markdown("- Make sure `pip install langchain-groq langchain-core pydantic` is done")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color: #444; font-size: 12px; font-family: Space Mono, monospace;">'
    'AI Job Application Agent · Phase 1 · LangChain Foundations'
    '</p>',
    unsafe_allow_html=True
)
