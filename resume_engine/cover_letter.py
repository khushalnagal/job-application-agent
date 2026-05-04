"""
Resume Engine — Cover Letter Generator
Generates a tailored cover letter based on resume and job description.
"""

import os
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────
class CoverLetter(BaseModel):
    subject_line: str = Field(description="Email subject line for the application")
    opening_paragraph: str = Field(description="Opening paragraph — who you are and what role you are applying for")
    body_paragraph_1: str = Field(description="First body paragraph — most relevant skills and experience matching the JD")
    body_paragraph_2: str = Field(description="Second body paragraph — specific projects or achievements that demonstrate fit")
    closing_paragraph: str = Field(description="Closing paragraph — call to action and professional sign off")
    tone_used: str = Field(description="The tone used — formal, confident, or concise")
    key_points_highlighted: List[str] = Field(description="Key points from the resume that were emphasized in the letter")


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.4):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# Cover Letter Chain
# ─────────────────────────────────────────────
def build_cover_letter_chain(llm, tone: str = "confident"):
    parser = PydanticOutputParser(pydantic_object=CoverLetter)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert cover letter writer with 15+ years of experience placing candidates 
            at top companies. You write cover letters that are human, specific, and compelling — 
            never generic or templated sounding.

            Tone guidelines:
            - formal    : professional, structured, traditional
            - confident : direct, assertive, shows personality without being casual
            - concise   : short and to the point, no filler sentences

            Current tone requested: {tone}

            Rules:
            - Never fabricate experience or skills
            - If the candidate does not have experience with a tool or technology mentioned in the JD, do not claim they do — instead frame it as eagerness to learn that specific tool
            - Only reference tools and technologies that appear in the candidate's actual resume
            - Reference specific details from both the resume and JD
            - Do not use cliche phrases like 'I am writing to express my interest'
            - Keep each paragraph focused — no rambling
            - The letter should feel written by the candidate, not by AI
            {format_instructions}"""
        ),
        (
            "human",
            """Write a cover letter for this candidate applying to this job.

            JOB DESCRIPTION
            ---------------
            {job_description}

            CANDIDATE DETAILS
            -----------------
            Name         : {name}
            Email        : {email}
            Summary      : {summary}
            Top Skills   : {skills}
            Experience   : {experience}
            Projects     : {projects}
            Education    : {education}

            Return ONLY the JSON object, nothing else."""
        )
    ]).partial(
        format_instructions=parser.get_format_instructions(),
        tone=tone
    )

    return prompt | llm | parser


# ─────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────
def wrap_text(text: str, width: int = 65, indent: str = "  ") -> None:
    words = text.split()
    line = indent
    for word in words:
        if len(line) + len(word) > width:
            print(line)
            line = indent + word + " "
        else:
            line += word + " "
    print(line)


def print_cover_letter(letter: CoverLetter, name: str):
    div = "=" * 65
    thin = "-" * 65

    print(f"\n{div}")
    print("  COVER LETTER OUTPUT")
    print(f"{div}\n")

    print(f"  Candidate  : {name}")
    print(f"  Tone       : {letter.tone_used}")
    print()

    print("  SUBJECT LINE")
    print(thin)
    print(f"  {letter.subject_line}")
    print()

    print("  FULL LETTER")
    print(thin)
    print()
    wrap_text(letter.opening_paragraph, width=65)
    print()
    wrap_text(letter.body_paragraph_1, width=65)
    print()
    wrap_text(letter.body_paragraph_2, width=65)
    print()
    wrap_text(letter.closing_paragraph, width=65)
    print()

    print(thin)
    print("  KEY POINTS HIGHLIGHTED")
    print(thin)
    for i, point in enumerate(letter.key_points_highlighted, 1):
        print(f"  {i}. {point}")

    print(f"\n{div}\n")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def generate_cover_letter(
    structured_resume,
    job_description: str,
    tone: str = "confident",
    llm=None
) -> CoverLetter:
    if llm is None:
        llm = get_llm()

    chain = build_cover_letter_chain(llm, tone)

    return chain.invoke({
        "job_description": job_description,
        "name": structured_resume.name or "Candidate",
        "email": structured_resume.email or "Not provided",
        "summary": structured_resume.summary or "Not provided",
        "skills": ", ".join(structured_resume.skills[:10]),
        "experience": "\n".join(structured_resume.experience),
        "projects": "\n".join(structured_resume.projects) if structured_resume.projects else "None",
        "education": structured_resume.education[0] if structured_resume.education else "Not provided"
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

    Nice to have:
    - LangGraph or multi-agent systems
    - MLOps experience

    Location: Remote
    Type: Full-time
    """

    # Test all three tones
    for tone in ["formal", "confident", "concise"]:
        print(f"\n  [ TONE: {tone.upper()} ]")
        letter = generate_cover_letter(structured, sample_jd, tone=tone, llm=llm)
        print_cover_letter(letter, structured.name or "Candidate")