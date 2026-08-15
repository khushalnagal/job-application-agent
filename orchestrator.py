"""
Orchestrator — LangGraph Agent
Connects JD Intelligence and Resume Engine into a stateful pipeline.

Graph Flow:
    START → analyze_jd → parse_resume → tailor_resume → generate_cover_letter → END
"""

import os
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from jd_analyzer import analyze_job_description
from resume_engine.parser import parse_resume, get_llm
from resume_engine.tailor import tailor_resume
from resume_engine.cover_letter import generate_cover_letter

load_dotenv()


# ─────────────────────────────────────────────
# State Schema
# Shared state that passes between every node
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    # Inputs
    jd_text: str
    resume_path: str
    candidate_skills: Optional[str]
    cover_letter_tone: Optional[str]

    # JD Intelligence outputs
    jd_summary: Optional[str]
    jd_structured: Optional[object]
    jd_gap_analysis: Optional[str]

    # Resume Engine outputs
    resume_raw: Optional[str]
    resume_structured: Optional[object]
    resume_score: Optional[object]
    tailored_resume: Optional[object]
    cover_letter: Optional[object]


# ─────────────────────────────────────────────
# Node 1 — JD Intelligence
# ─────────────────────────────────────────────
def analyze_jd_node(state: AgentState) -> AgentState:
    print("  [1/4] Analyzing job description...")
    results = analyze_job_description(
        state["jd_text"],
        state.get("candidate_skills")
    )
    return {
        **state,
        "jd_summary": results["summary"],
        "jd_structured": results["structured_data"],
        "jd_gap_analysis": results.get("gap_analysis")
    }


# ─────────────────────────────────────────────
# Node 2 — Resume Parsing + Scoring
# ─────────────────────────────────────────────
def parse_resume_node(state: AgentState) -> AgentState:
    print("  [2/4] Parsing and scoring resume...")
    llm = get_llm()
    result = parse_resume(state["resume_path"], llm)
    return {
        **state,
        "resume_raw": result["raw_text"],
        "resume_structured": result["structured"],
        "resume_score": result["score"]
    }


# ─────────────────────────────────────────────
# Node 3 — Resume Tailoring
# ─────────────────────────────────────────────
def tailor_resume_node(state: AgentState) -> AgentState:
    print("  [3/4] Tailoring resume to JD...")
    tailored = tailor_resume(
        structured_resume=state["resume_structured"],
        job_description=state["jd_text"]
    )
    return {
        **state,
        "tailored_resume": tailored
    }


# ─────────────────────────────────────────────
# Node 4 — Cover Letter Generation
# ─────────────────────────────────────────────
def cover_letter_node(state: AgentState) -> AgentState:
    print("  [4/4] Generating cover letter...")
    tone = state.get("cover_letter_tone") or "confident"
    letter = generate_cover_letter(
        structured_resume=state["resume_structured"],
        job_description=state["jd_text"],
        tone=tone
    )
    return {
        **state,
        "cover_letter": letter
    }


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_jd", analyze_jd_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("tailor_resume", tailor_resume_node)
    graph.add_node("generate_cover_letter", cover_letter_node)

    graph.add_edge(START, "analyze_jd")
    graph.add_edge("analyze_jd", "parse_resume")
    graph.add_edge("parse_resume", "tailor_resume")
    graph.add_edge("tailor_resume", "generate_cover_letter")
    graph.add_edge("generate_cover_letter", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def run_agent(
    jd_text: str,
    resume_path: str,
    candidate_skills: str = None,
    cover_letter_tone: str = "confident"
) -> AgentState:
    agent = build_agent()

    initial_state: AgentState = {
        "jd_text": jd_text,
        "resume_path": resume_path,
        "candidate_skills": candidate_skills,
        "cover_letter_tone": cover_letter_tone,
        "jd_summary": None,
        "jd_structured": None,
        "jd_gap_analysis": None,
        "resume_raw": None,
        "resume_structured": None,
        "resume_score": None,
        "tailored_resume": None,
        "cover_letter": None
    }

    print("\n  Agent starting...\n")
    final_state = agent.invoke(initial_state)
    print("\n  Agent complete.\n")

    return final_state

if __name__ == "__main__":
    sample_jd = """
    We are looking for a Machine Learning Engineer.
    Requirements: Python, LangChain, LangGraph, RAG, FastAPI.
    Experience: 2+ years. Location: Remote.
    """

    result = run_agent(
        jd_text=sample_jd,
        resume_path="resume_engine/sample_resume.pdf",
        candidate_skills="Python, LangChain, RAG, Streamlit, SQL",
        cover_letter_tone="confident"
    )

    print("\nJD Summary:", result["jd_summary"])
    print("\nCover Letter:", result["cover_letter"])