"""
Resume Engine — Parser
Extracts and structures resume content from an uploaded PDF.
"""

import os
import re
import pdfplumber
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# Data Schemas
# ─────────────────────────────────────────────
class ResumeData(BaseModel):
    name: Optional[str] = Field(description="Full name of the candidate")
    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    location: Optional[str] = Field(description="City or region")
    summary: Optional[str] = Field(description="Professional summary or objective if present")
    skills: List[str] = Field(description="All technical and soft skills mentioned")
    experience: List[str] = Field(description="Work experience entries as plain text")
    education: List[str] = Field(description="Education entries as plain text")
    certifications: Optional[List[str]] = Field(default=None, description="Certifications or courses if mentioned")
    projects: Optional[List[str]] = Field(default=None, description="Projects if mentioned")


class ResumeScore(BaseModel):
    ats_score: int = Field(description="ATS compatibility score out of 100")
    ats_reasons: List[str] = Field(description="Reasons for ATS score — what helps and what hurts")
    market_score: int = Field(description="Market competitiveness score out of 100 based on current industry demand")
    market_reasons: List[str] = Field(description="Reasons for market score — how strong this profile is vs real competition")
    missing_keywords: List[str] = Field(description="Important keywords missing from the resume that ATS systems look for")
    weak_sections: List[str] = Field(description="Sections that are weak or missing entirely")
    top_suggestions: List[str] = Field(description="Top 5 actionable suggestions to improve this resume")
    overall_verdict: str = Field(description="2-3 sentence honest verdict on this resume's current state")


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.0):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# PDF Text Extraction + Spacing Fix
# ─────────────────────────────────────────────
def fix_spacing(text: str) -> str:
    """
    Fixes common PDF text extraction artifacts — missing spaces between words,
    stuck digits and letters, and missing spaces after punctuation.
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'([.,;:])([a-zA-Z])', r'\1 \2', text)
    return text.strip()

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return fix_spacing(text.strip())


# ─────────────────────────────────────────────
# Resume Structuring Chain
# ─────────────────────────────────────────────
def structure_resume(raw_text: str, llm) -> ResumeData:
    parser = PydanticOutputParser(pydantic_object=ResumeData)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional resume parser. Extract structured information
            from the resume text accurately and completely.
            {format_instructions}"""
        ),
        (
            "human",
            """Parse this resume and extract all relevant information:

            {resume_text}

            Return ONLY the JSON object, nothing else."""
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke({"resume_text": raw_text})


# ─────────────────────────────────────────────
# Resume Scoring Chain
# ─────────────────────────────────────────────
def score_resume(raw_text: str, structured: ResumeData, llm) -> ResumeScore:
    parser = PydanticOutputParser(pydantic_object=ResumeScore)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a senior technical recruiter and career coach with 15+ years of experience.
            You know exactly how ATS systems work and what the current job market demands.
            Be honest and strict with scoring — most resumes score between 40-75.
            Only truly exceptional resumes score above 85.

            {format_instructions}"""
        ),
        (
            "human",
            """Analyze this resume thoroughly and provide scores and feedback.

            RESUME TEXT:
            {resume_text}

            EXTRACTED SKILLS: {skills}
            EXPERIENCE ENTRIES: {experience_count}
            HAS PROJECTS: {has_projects}
            HAS CERTIFICATIONS: {has_certifications}

            Evaluate:
            1. ATS Score — formatting, keywords, standard sections, quantified achievements
            2. Market Score — how competitive this profile is vs real candidates in today's market
            3. Missing keywords that ATS systems commonly scan for
            4. Weak or missing sections
            5. Top 5 specific actionable suggestions
            6. Honest overall verdict

            Return ONLY the JSON object, nothing else."""
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke({
        "resume_text": raw_text,
        "skills": ", ".join(structured.skills),
        "experience_count": len(structured.experience),
        "has_projects": "Yes" if structured.projects else "No",
        "has_certifications": "Yes" if structured.certifications else "No"
    })


# ─────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────
def wrap_text(text: str, width: int = 57, indent: str = "  ") -> None:
    words = text.split()
    line = indent
    for word in words:
        if len(line) + len(word) > width:
            print(line)
            line = indent + word + " "
        else:
            line += word + " "
    print(line)


def print_results(raw_text: str, structured: ResumeData, score: ResumeScore):
    div = "=" * 60
    thin = "-" * 60

    print(f"\n{div}")
    print("RESUME PARSER — FULL REPORT")
    print(f"{div}\n")

    print("BASIC INFORMATION")
    print(thin)
    print(f"Name        : {structured.name or 'Not found'}")
    print(f"Email       : {structured.email or 'Not found'}")
    print(f"Phone       : {structured.phone or 'Not found'}")
    print(f"Location    : {structured.location or 'Not found'}")
    print()

    if structured.summary:
        print("SUMMARY")
        print(thin)
        wrap_text(structured.summary)
        print()

    print("SKILLS")
    print(thin)
    skills_line = ""
    for i, skill in enumerate(structured.skills):
        skills_line += skill
        if i < len(structured.skills) - 1:
            skills_line += " | "
        if len(skills_line) > 55:
            print(f"{skills_line}")
            skills_line = ""
    if skills_line:
        print(f"{skills_line}")
    print()

    print("EXPERIENCE")
    print(thin)
    for i, exp in enumerate(structured.experience, 1):
        print(f"{i}.{exp}")
    print()

    print("EDUCATION")
    print(thin)
    for i, edu in enumerate(structured.education, 1):
        print(f"{i}.{edu}")
    print()

    if structured.projects:
        print("PROJECTS")
        print(thin)
        for i, proj in enumerate(structured.projects, 1):
            print(f"{i}.{proj}")
        print()

    if structured.certifications:
        print("CERTIFICATIONS")
        print(thin)
        for i, cert in enumerate(structured.certifications, 1):
            print(f"{i}. {cert}")
        print()

    print(f"\n{div}")
    print("  SCORES AND ANALYSIS")
    print(f"{div}\n")

    ats_bar = "#" * (score.ats_score // 5) + "-" * (20 - score.ats_score // 5)
    print(f"ATS SCORE : {score.ats_score}/100")
    print(f"[{ats_bar}]")
    print()
    for reason in score.ats_reasons:
        print(f"-{reason}")
    print()

    mkt_bar = "#" * (score.market_score // 5) + "-" * (20 - score.market_score // 5)
    print(f"MARKET SCORE : {score.market_score}/100")
    print(f"[{mkt_bar}]")
    print()
    for reason in score.market_reasons:
        print(f"  - {reason}")
    print()

    print("MISSING KEYWORDS")
    print(thin)
    print(f"  {',  '.join(score.missing_keywords)}")
    print()

    print("WEAK SECTIONS")
    print(thin)
    for section in score.weak_sections:
        print(f"  - {section}")
    print()

    print("TOP SUGGESTIONS")
    print(thin)
    for i, suggestion in enumerate(score.top_suggestions, 1):
        print(f"  {i}. {suggestion}")
    print()

    print("OVERALL VERDICT")
    print(thin)
    wrap_text(score.overall_verdict)
    print(f"\n{div}\n")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def parse_resume(pdf_file, llm) -> dict:
    raw_text = extract_text_from_pdf(pdf_file)
    structured = structure_resume(raw_text, llm)
    score = score_resume(raw_text, structured, llm)

    return {
        "raw_text": raw_text,
        "structured": structured,
        "score": score
    }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    llm = get_llm()

    result = parse_resume(
        r"C:\Users\khush\OneDrive\Desktop\Docs\Khushal_s_Resume.pdf",
        llm
    )

    print_results(result["raw_text"], result["structured"], result["score"])