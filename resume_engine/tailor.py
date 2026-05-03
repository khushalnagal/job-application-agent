"""
Resume Engine — Tailor
Rewrites resume content to match a specific job description.
"""

import os
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────
class TailoredResume(BaseModel):
    summary: str = Field(description="Rewritten professional summary tailored to the JD")
    skills: List[str] = Field(description="Reordered and updated skills list — most relevant to JD first, added missing keywords naturally")
    experience: List[str] = Field(description="Rewritten experience entries with JD keywords added and achievements quantified where possible")
    projects: Optional[List[str]] = Field(default=None, description="Rewritten project entries highlighting aspects relevant to the JD")
    keywords_added: List[str] = Field(description="List of keywords that were added from the JD into the resume")
    keywords_missing: List[str] = Field(description="Important JD keywords that couldn't be added because candidate has no evidence of them")
    match_score: int = Field(description="How well the tailored resume matches the JD, out of 100")
    tailoring_notes: List[str] = Field(description="3-5 notes explaining what was changed and why")


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.3):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# Tailor Chain
# ─────────────────────────────────────────────
def build_tailor_chain(llm):
    parser = PydanticOutputParser(pydantic_object=TailoredResume)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert resume writer and career coach with 15+ years of experience.
            Your job is to tailor resumes to specific job descriptions without lying or fabricating experience.

            Rules:
            - Never invent experience or skills the candidate does not have
            - Reorder and rephrase existing content to highlight relevant parts
            - Add JD keywords naturally where the candidate genuinely has that experience
            - Quantify achievements where possible based on context clues
            - Make the summary laser focused on this specific role
            - Put the most JD-relevant skills first

            {format_instructions}"""
        ),
        (
            "human",
            """Tailor this resume to match the job description below.

            JOB DESCRIPTION
            ---------------
            {job_description}

            CANDIDATE RESUME
            ----------------
            Name: {name}

            Current Summary:
            {summary}

            Current Skills:
            {skills}

            Experience:
            {experience}

            Projects:
            {projects}

            Education:
            {education}

            Rewrite the resume content to best match this JD.
            Return ONLY the JSON object, nothing else."""
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


# ─────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────
def print_tailored(tailored: TailoredResume, original_name: str):
    div = "=" * 60
    thin = "-" * 60

    print(f"\n{div}")
    print("  TAILORED RESUME OUTPUT")
    print(f"{div}\n")

    print(f"  Candidate : {original_name}")
    print()

    print("  SUMMARY")
    print(thin)
    words = tailored.summary.split()
    line = "  "
    for word in words:
        if len(line) + len(word) > 57:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    print(line)
    print()

    print("  SKILLS  (ranked by JD relevance)")
    print(thin)
    skills_line = ""
    for i, skill in enumerate(tailored.skills):
        skills_line += skill
        if i < len(tailored.skills) - 1:
            skills_line += "  |  "
        if len(skills_line) > 55:
            print(f"  {skills_line}")
            skills_line = ""
    if skills_line:
        print(f"  {skills_line}")
    print()

    print("  EXPERIENCE")
    print(thin)
    for i, exp in enumerate(tailored.experience, 1):
        print(f"  {i}. {exp}")
    print()

    if tailored.projects:
        print("  PROJECTS")
        print(thin)
        for i, proj in enumerate(tailored.projects, 1):
            print(f"  {i}. {proj}")
        print()

    bar_filled = tailored.match_score // 5
    bar = "#" * bar_filled + "-" * (20 - bar_filled)
    print(f"  JD MATCH SCORE  : {tailored.match_score}/100")
    print(f"  [{bar}]")
    print()

    print("  KEYWORDS ADDED")
    print(thin)
    print(f"  {',  '.join(tailored.keywords_added)}")
    print()

    print("  KEYWORDS MISSING  (no supporting evidence in resume)")
    print(thin)
    print(f"  {',  '.join(tailored.keywords_missing)}")
    print()

    print("  WHAT WAS CHANGED AND WHY")
    print(thin)
    for i, note in enumerate(tailored.tailoring_notes, 1):
        print(f"  {i}. {note}")

    print(f"\n{div}\n")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def tailor_resume(structured_resume, job_description: str, llm=None) -> TailoredResume:
    if llm is None:
        llm = get_llm()

    chain = build_tailor_chain(llm)

    return chain.invoke({
        "job_description": job_description,
        "name": structured_resume.name or "Candidate",
        "summary": structured_resume.summary or "Not provided",
        "skills": ", ".join(structured_resume.skills),
        "experience": "\n".join(structured_resume.experience),
        "projects": "\n".join(structured_resume.projects) if structured_resume.projects else "None",
        "education": "\n".join(structured_resume.education)
    })


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_resume

    llm = get_llm()

    result = parse_resume(
        r"C:\Users\khush\OneDrive\Desktop\Docs\Khushal_s_Resume.pdf",
        llm
    )
    structured = result["structured"]

    sample_jd = """
    We are hiring a Machine Learning Engineer to build AI-powered tools.

    Requirements:
    - 2+ years Python experience
    - Experience with LangChain, LLMs, RAG systems
    - Knowledge of vector databases (FAISS, Pinecone)
    - Familiarity with FastAPI and Docker
    - Strong data analysis and problem solving skills
    - Experience with cloud platforms (AWS/GCP)

    Nice to have:
    - LangGraph or multi-agent systems
    - MLOps experience
    - Open source contributions

    Location: Remote
    Type: Full-time
    """

    tailored = tailor_resume(structured, sample_jd, llm)
    print_tailored(tailored, structured.name or "Candidate")