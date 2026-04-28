"""
JD Analyzer — Backend
Handles JD parsing, structured extraction, and skills gap analysis.
"""

import os
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────
class JobDescription(BaseModel):
    """Structured representation of a job description."""

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
# LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.0):
    """
    Returns a ChatGroq instance.
    Temperature 0.0 for extraction tasks, slightly higher for generative ones.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# Summary Chain
# ─────────────────────────────────────────────
def build_summary_chain():
    """Returns a plain text summary of the job description."""

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

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# Structured Extraction Chain
# ─────────────────────────────────────────────
def build_extraction_chain():
    """Extracts structured fields from the JD and returns a JobDescription object."""

    llm = get_llm(temperature=0.0)

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

    chain = prompt | llm | parser
    return chain


# ─────────────────────────────────────────────
# Skills Gap Chain
# ─────────────────────────────────────────────
def build_skills_gap_chain():
    """Compares the candidate's skills against the JD and returns a gap analysis."""

    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a career coach. Identify skill gaps clearly and suggest a practical path forward."""
        ),
        (
            "human",
            """Analyze the gap between this candidate's skills and the job requirements.

            Job Description:
            {job_description}

            Candidate's Current Skills:
            {candidate_skills}

            Please provide:
            1. Skills they already have (matching skills)
            2. Skills they are missing (must-learn)
            3. Skills they should improve
            4. Overall fit score (out of 10)
            5. Top 3 recommendations to become a stronger candidate
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def analyze_job_description(jd_text: str, candidate_skills: str = None):
    """Runs all three chains against the provided JD. Returns summary, structured data, and optional gap analysis."""

    results = {}

    summary_chain = build_summary_chain()
    results["summary"] = summary_chain.invoke({"job_description": jd_text})

    extraction_chain = build_extraction_chain()
    results["structured_data"] = extraction_chain.invoke({"job_description": jd_text})

    if candidate_skills:
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

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results["summary"])

    print("\n" + "=" * 60)
    print("STRUCTURED DATA")
    print("=" * 60)
    jd = results["structured_data"]
    print(f"Role: {jd.role}")
    print(f"Company: {jd.company}")
    print(f"Required Skills: {', '.join(jd.required_skills)}")
    print(f"Experience: {jd.experience_required}")
    print(f"Summary: {jd.summary}")

    print("\n" + "=" * 60)
    print("SKILLS GAP ANALYSIS")
    print("=" * 60)
    print(results["gap_analysis"])