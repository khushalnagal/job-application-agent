# AI Job Application Agent

An AI-powered assistant that helps you analyze job descriptions, tailor your resume, and generate cover letters - built with LangChain, Groq, and Streamlit.

---

## What It Does

### JD Intelligence
Paste any job description and get:
- A plain English summary of the role
- Structured extraction of skills, responsibilities, experience, and location
- A skills gap analysis comparing your profile against the JD

### Resume Engine
Upload your resume PDF and get:
- Full structured extraction of your resume content
- ATS compatibility score and market competitiveness score
- Missing keywords and weak section identification
- A rewritten resume tailored to a specific JD
- A cover letter in your chosen tone - formal, confident, or concise

---

## Concepts Covered

- LLMs with ChatGroq (Llama 3.3-70b, free)
- ChatPromptTemplate (system + human messages)
- LCEL Chains using the `|` pipe operator
- StrOutputParser  plain string output
- PydanticOutputParser - structured Python object output
- pdfplumber - PDF text extraction
- Multiple chains for different tasks

---

## Project Structure

```
job-application-agent/
├── app.py                  # Streamlit UI for JD Intelligence
├── jd_analyzer.py          # JD extraction and skills gap chains
├── resume_engine/
│   ├── __init__.py
│   ├── parser.py           # PDF extraction, structuring, ATS and market scoring
│   ├── tailor.py           # Resume rewriting to match a target JD
│   └── cover_letter.py     # Cover letter generation with tone selection
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/khushalnagal/job-application-agent
cd job-application-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Visit: https://console.groq.com
- Sign up (free)
- Create an API key
- Copy it

### 4. Create a `.env` file in the project root
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the JD analyzer
```bash
streamlit run app.py
```

---

## Chains Built

### Chain 1 - Summary Chain
```python
chain = prompt | ChatGroq() | StrOutputParser()
```
Returns a plain string summary of the job description.

### Chain 2 - Extraction Chain
```python
chain = prompt | ChatGroq(temperature=0) | PydanticOutputParser(...)
```
Returns a typed `JobDescription` object with fields:
- `role`, `company`, `required_skills`, `preferred_skills`
- `experience_required`, `education`, `responsibilities`
- `location`, `job_type`, `summary`

### Chain 3 - Skills Gap Chain
```python
chain = prompt | ChatGroq(temperature=0.2) | StrOutputParser()
```
Takes JD + your skills and returns a gap analysis with fit score.

### Chain 4 - Resume Scoring Chain
```python
chain = prompt | ChatGroq(temperature=0) | PydanticOutputParser(...)
```
Returns ATS score, market score, missing keywords, and improvement suggestions.

### Chain 5 - Resume Tailor Chain
```python
chain = prompt | ChatGroq(temperature=0.3) | PydanticOutputParser(...)
```
Rewrites resume content to match a target JD without fabricating experience.

### Chain 6 - Cover Letter Chain
```python
chain = prompt | ChatGroq(temperature=0.4) | PydanticOutputParser(...)
```
Generates a tailored cover letter in formal, confident, or concise tone.

---

## Free Resources Used

| Tool | Cost | Link |
|------|------|------|
| Groq API | Free tier | console.groq.com |
| LangChain | Open source | python.langchain.com |
| Streamlit | Open source | streamlit.io |
| pdfplumber | Open source | github.com/jsvine/pdfplumber |

---

## Roadmap

- JD Intelligence - extract, structure, and analyze job descriptions
- Resume Engine - parse, score, tailor, and generate cover letters
- Agent Orchestrator - LangGraph agent that runs the full pipeline autonomously
- Job Discovery - find matching job openings automatically
- Interview Coach - generate practice questions and score responses

---

## Author

Khushal Nagal
khushalnagal@gmail.com
github.com/khushalnagal
