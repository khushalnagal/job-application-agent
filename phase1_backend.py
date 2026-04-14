"""
Phase 1 Backend — Job Description Extractor
Concepts: LLMs, ChatPromptTemplate, LCEL chains, PydanticOutputParser, StrOutputParser
"""

import os
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# STEP 1: Define the Pydantic Model (Data Schema)
# This tells the LLM exactly what structure to return
# ─────────────────────────────────────────────
class JobDescription(BaseModel):
    """Structured representation of a Job Description"""

    role: str = Field(description="The job title or role name")
    company: Optional[str] = Field(description="Company name if mentioned")
    required_skills: List[str] = Field(description="List of required technical and soft skills")
    preferred_skills: List[str] = Field(description="List of preferred or nice-to-have skills")
    experience_required: str = Field(description="Years or level of experience required")
    education: Optional[str] = Field(description="Education requirement if mentioned")
    responsibilities: List[str] = Field(description="Key job responsibilities")
    location: Optional[str] = Field(description="Job location or remote status")
    job_type: Optional[str] = Field(description="Full-time, part-time, contract, etc.")
    summary: str = Field(description="2-3 sentence summary of what this role is about")


# ─────────────────────────────────────────────
# STEP 2: Initialize the LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.0):
    """
    Returns a ChatGroq LLM instance.
    Temperature 0.0 = deterministic (good for extraction tasks).
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # free on Groq
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# STEP 3: Build Chain 1 — Simple Summary Chain
# Uses: ChatPromptTemplate + LLM + StrOutputParser
# ─────────────────────────────────────────────
def build_summary_chain():
    """
    Simple chain that returns a plain text summary of the JD.
    Good for understanding StrOutputParser.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert HR analyst with 10+ years of experience. 
            Your job is to read job descriptions and give a clear, concise summary 
            that helps a job seeker quickly understand if they are a good fit."""
        ),
        (
            "human",
            """Please read this job description and give me a 3-4 line summary 
            covering: what the role is, who they are looking for, and what makes 
            this role interesting.

            Job Description:
            {job_description}
            """
        )
    ])

    # LCEL chain: prompt → llm → string parser
    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# STEP 4: Build Chain 2 — Structured Extraction Chain
# Uses: PydanticOutputParser — returns a JobDescription object
# ─────────────────────────────────────────────
def build_extraction_chain():
    """
    Structured chain that extracts all key details from JD
    into a typed Python object using PydanticOutputParser.
    """
    llm = get_llm(temperature=0.0)

    # Parser knows the schema and injects format instructions into prompt
    parser = PydanticOutputParser(pydantic_object=JobDescription)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert HR analyst. Extract structured information 
            from job descriptions accurately and completely.
            
            {format_instructions}"""
        ),
        (
            "human",
            """Extract all relevant information from this job description:

            {job_description}
            
            Return ONLY the JSON object, nothing else."""
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    # LCEL chain: prompt → llm → pydantic parser
    chain = prompt | llm | parser
    return chain


# ─────────────────────────────────────────────
# STEP 5: Build Chain 3 — Skills Gap Analyzer
# Combines structured extraction + gap analysis in sequence
# ─────────────────────────────────────────────
def build_skills_gap_chain():
    """
    Takes JD + user's skills → returns a gap analysis.
    Shows how to pass multiple inputs to a chain.
    """
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a career coach who specializes in helping people 
            identify skill gaps and plan their learning path. 
            Be honest but encouraging."""
        ),
        (
            "human",
            """Analyze the gap between this candidate's skills and the job requirements.

            Job Description:
            {job_description}

            Candidate's Current Skills:
            {candidate_skills}

            Please provide:
            1. ✅ Skills they already have (matching skills)
            2. ❌ Skills they are missing (must-learn)
            3. 📈 Skills they should improve
            4. 🎯 Overall fit score (out of 10)
            5. 💡 Top 3 recommendations to become a stronger candidate
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# STEP 6: Main runner function
# ─────────────────────────────────────────────
def analyze_job_description(jd_text: str, candidate_skills: str = None):
    """
    Main function that runs all 3 chains on a given JD.
    Returns dict with summary, structured_data, and gap_analysis.
    """
    results = {}

    # Run summary chain
    print("🔍 Generating summary...")
    summary_chain = build_summary_chain()
    results["summary"] = summary_chain.invoke({"job_description": jd_text})

    # Run extraction chain
    print("📊 Extracting structured data...")
    extraction_chain = build_extraction_chain()
    results["structured_data"] = extraction_chain.invoke({"job_description": jd_text})

    # Run gap analysis if skills provided
    if candidate_skills:
        print("📈 Analyzing skills gap...")
        gap_chain = build_skills_gap_chain()
        results["gap_analysis"] = gap_chain.invoke({
            "job_description": jd_text,
            "candidate_skills": candidate_skills
        })

    return results


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_jd = """
    We are looking for a Machine Learning Engineer to join our AI team at TechCorp.
    
    Responsibilities:
    - Design and implement ML pipelines using Python
    - Work with LangChain and LangGraph to build AI agents
    - Deploy models using FastAPI and Docker
    - Collaborate with data scientists and product teams
    - Optimize model performance and monitor in production
    
    Requirements:
    - 2+ years of experience in ML or AI development
    - Strong Python skills
    - Experience with LangChain, LLMs, RAG systems
    - Knowledge of vector databases (FAISS, Pinecone)
    - Familiarity with cloud platforms (AWS/GCP)
    - Bachelor's degree in CS or related field
    
    Nice to have:
    - Experience with LangGraph or multi-agent systems
    - Knowledge of MLOps practices
    - Contributions to open source
    
    Location: Remote (India preferred)
    Type: Full-time
    """

    sample_skills = """
    Python, LangChain, LangGraph, RAG, FAISS, Streamlit, 
    Machine Learning basics, SQL, Git
    """

    results = analyze_job_description(sample_jd, sample_skills)

    print("\n" + "="*60)
    print("📝 SUMMARY")
    print("="*60)
    print(results["summary"])

    print("\n" + "="*60)
    print("📊 STRUCTURED DATA")
    print("="*60)
    jd = results["structured_data"]
    print(f"Role: {jd.role}")
    print(f"Company: {jd.company}")
    print(f"Required Skills: {', '.join(jd.required_skills)}")
    print(f"Experience: {jd.experience_required}")
    print(f"Summary: {jd.summary}")

    print("\n" + "="*60)
    print("📈 SKILLS GAP ANALYSIS")
    print("="*60)
    print(results["gap_analysis"])
