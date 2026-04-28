#  AI Job Application Agent 
## Job Description Analyzer using LangChain

---

## Concepts Covered
- LLMs with ChatGroq (Llama 3.3-70b, free)
- ChatPromptTemplate (system + human messages)
- LCEL Chains using the `|` pipe operator
- StrOutputParser → plain string output
- PydanticOutputParser → structured Python object output
- Multiple chains for different tasks

---

## Files
```
phase1_backend.py     → All LangChain logic (3 chains)
phase1_streamlit.py   → Streamlit UI
requirements.txt      → Dependencies
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free Groq API key
Visit: https://console.groq.com
- Sign up (free)
- Create API key
- Copy it

### 3. Run the app
```bash
streamlit run phase1_streamlit.py
```

### 4. Enter your Groq key in the sidebar → Paste a JD → Analyze!

---

## 3 Chains Built

### Chain 1 — Summary Chain
```python
chain = prompt | ChatGroq() | StrOutputParser()
```
Returns a plain string summary of the JD.

### Chain 2 — Extraction Chain
```python
chain = prompt | ChatGroq(temperature=0) | PydanticOutputParser(...)
```
Returns a typed `JobDescription` object with fields:
- `role`, `company`, `required_skills`, `preferred_skills`
- `experience_required`, `education`, `responsibilities`
- `location`, `job_type`, `summary`

### Chain 3 — Skills Gap Chain
```python
chain = prompt | ChatGroq(temperature=0.2) | StrOutputParser()
```
Takes JD + your skills → returns gap analysis with fit score.

---

##  Free Resources Used
| Tool | Cost | Link |
|------|------|------|
| Groq API | Free tier | console.groq.com |
| LangChain | Open source | python.langchain.com |
| Streamlit | Open source | streamlit.io |

---

## ➡️ Next: Phase 2
Add RAG — upload your resume PDF and let the agent compare it with the JD.
