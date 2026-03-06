import os
import re
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants & Sample Data
# ---------------------------------------------------------------------------

BIAS_KEYWORDS = [
    "young", "old", "age", "gender", "male", "female", "he", "she", "his",
    "her", "married", "single", "race", "ethnicity", "nationality", "religion",
    "church", "mosque", "temple", "disability", "blind", "deaf", "wheelchair",
]

SKILL_WEIGHTS = {
    "python": 1.5, "machine learning": 2.0, "deep learning": 2.0,
    "data science": 1.8, "sql": 1.4, "java": 1.3, "javascript": 1.3,
    "react": 1.2, "node": 1.2, "aws": 1.5, "docker": 1.4, "kubernetes": 1.4,
    "tensorflow": 1.8, "pytorch": 1.8, "nlp": 1.9, "computer vision": 1.9,
    "flask": 1.2, "django": 1.2, "fastapi": 1.3, "pandas": 1.3,
    "leadership": 1.1, "communication": 1.0, "teamwork": 1.0,
}

SAMPLE_JOB_ROLES = [
    {
        "id": 1,
        "title": "Senior ML Engineer",
        "description": "Develop and deploy machine learning models using Python, TensorFlow, PyTorch. "
                       "Experience with NLP, computer vision, and MLOps pipelines. AWS or GCP cloud experience.",
        "required_skills": ["python", "machine learning", "tensorflow", "pytorch", "aws", "docker"],
        "min_experience_years": 3,
    },
    {
        "id": 2,
        "title": "Full Stack Developer",
        "description": "Build scalable web applications using React, Node.js, and Python backends. "
                       "Experience with REST APIs, SQL databases, Docker, and CI/CD pipelines.",
        "required_skills": ["javascript", "react", "node", "python", "sql", "docker"],
        "min_experience_years": 2,
    },
    {
        "id": 3,
        "title": "Data Scientist",
        "description": "Analyze large datasets and build predictive models. "
                       "Proficiency in Python, pandas, scikit-learn, SQL, and data visualization.",
        "required_skills": ["python", "data science", "pandas", "sql", "machine learning"],
        "min_experience_years": 2,
    },
    {
        "id": 4,
        "title": "DevOps Engineer",
        "description": "Manage cloud infrastructure on AWS using Kubernetes, Docker, Terraform. "
                       "CI/CD automation with Jenkins or GitHub Actions.",
        "required_skills": ["aws", "kubernetes", "docker", "python"],
        "min_experience_years": 2,
    },
]

# ---------------------------------------------------------------------------
# Core ML Functions
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove special chars, lowercase, strip extra whitespace."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def detect_bias(resume_text: str) -> dict:
    """Scan resume text for potentially biased personal identifiers."""
    text_lower = resume_text.lower()
    found = [kw for kw in BIAS_KEYWORDS if re.search(r"\b" + kw + r"\b", text_lower)]
    score = max(0.0, 1.0 - len(found) * 0.1)
    return {
        "bias_free_score": round(score, 2),
        "flagged_terms": found,
        "recommendation": (
            "Resume appears bias-free." if not found
            else f"Consider removing or anonymising: {', '.join(found)}"
        ),
    }


def extract_years_experience(text: str) -> int:
    """Heuristic: find the largest year-span number mentioned."""
    matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text.lower())
    return max((int(m) for m in matches), default=0)


def score_skills(resume_text: str) -> dict:
    """Weighted skill match score (0-100)."""
    text_lower = resume_text.lower()
    matched, total_weight = {}, 0.0
    for skill, weight in SKILL_WEIGHTS.items():
        if skill in text_lower:
            matched[skill] = weight
            total_weight += weight
    max_possible = sum(SKILL_WEIGHTS.values())
    score = min(100.0, round((total_weight / max_possible) * 100 * 2.5, 1))
    return {"skill_score": score, "matched_skills": list(matched.keys()), "skill_count": len(matched)}


def tfidf_match(resume_text: str, job_description: str) -> float:
    """Cosine similarity between resume and job description via TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_description)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100, 1)
    except Exception:
        return 0.0


def rank_candidates(candidates: list, job: dict) -> list:
    """Score and rank multiple candidates against a job role."""
    ranked = []
    for cand in candidates:
        text = cand.get("resume_text", "")
        skill_data = score_skills(text)
        tfidf_score = tfidf_match(text, job["description"])
        exp_years = extract_years_experience(text)
        exp_score = min(100.0, (exp_years / max(job["min_experience_years"], 1)) * 100)

        req_skills = job.get("required_skills", [])
        text_lower = text.lower()
        req_hit = sum(1 for s in req_skills if s in text_lower)
        req_score = (req_hit / len(req_skills) * 100) if req_skills else 50.0

        composite = round(
            0.30 * tfidf_score +
            0.25 * skill_data["skill_score"] +
            0.25 * req_score +
            0.20 * exp_score,
            1,
        )
        ranked.append({
            "candidate_id": cand.get("id", "unknown"),
            "name": cand.get("name", "Anonymous"),
            "composite_score": composite,
            "tfidf_match": tfidf_score,
            "skill_score": skill_data["skill_score"],
            "matched_skills": skill_data["matched_skills"],
            "required_skills_hit": req_hit,
            "experience_years": exp_years,
            "bias_analysis": detect_bias(text),
        })
    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SmartHire AI - Enterprise Hiring Assistant</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; }
    header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 24px 40px; }
    header h1 { font-size: 2rem; } header p { opacity: .85; margin-top: 4px; }
    .container { max-width: 960px; margin: 32px auto; padding: 0 20px; }
    .card { background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
            padding: 28px; margin-bottom: 24px; }
    .card h2 { font-size: 1.2rem; color: #1a73e8; margin-bottom: 16px; }
    textarea { width: 100%; border: 1px solid #ddd; border-radius: 8px; padding: 12px;
               font-size: .9rem; resize: vertical; min-height: 120px; }
    select { width: 100%; border: 1px solid #ddd; border-radius: 8px; padding: 10px;
             font-size: .9rem; margin-top: 8px; }
    button { background: #1a73e8; color: white; border: none; border-radius: 8px;
             padding: 12px 28px; font-size: 1rem; cursor: pointer; margin-top: 12px; }
    button:hover { background: #1558b0; }
    #result { display: none; }
    .score-badge { display: inline-block; padding: 6px 16px; border-radius: 20px;
                   font-weight: bold; font-size: 1.1rem; }
    .high { background: #e8f5e9; color: #2e7d32; }
    .mid  { background: #fff8e1; color: #f57f17; }
    .low  { background: #ffebee; color: #c62828; }
    .tag  { display: inline-block; background: #e3f2fd; color: #1565c0;
            border-radius: 4px; padding: 3px 10px; margin: 3px; font-size: .82rem; }
    .flag { display: inline-block; background: #fff3e0; color: #e65100;
            border-radius: 4px; padding: 3px 10px; margin: 3px; font-size: .82rem; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th { background: #1a73e8; color: white; padding: 10px; text-align: left; }
    td { padding: 10px; border-bottom: 1px solid #eee; }
    tr:hover td { background: #f5f9ff; }
  </style>
</head>
<body>
  <header>
    <h1>SmartHire AI</h1>
    <p>AI-powered enterprise hiring assistant - bias-free, data-driven candidate matching</p>
  </header>
  <div class="container">
    <div class="card">
      <h2>Screen a Resume</h2>
      <label>Paste Resume Text</label>
      <textarea id="resumeText" placeholder="Paste resume content here..."></textarea>
      <label style="margin-top:12px;display:block">Select Job Role</label>
      <select id="jobSelect">
        <option value="1">Senior ML Engineer</option>
        <option value="2">Full Stack Developer</option>
        <option value="3">Data Scientist</option>
        <option value="4">DevOps Engineer</option>
      </select>
      <button onclick="screenResume()">Analyse Resume</button>
    </div>
    <div class="card" id="result">
      <h2>Analysis Results</h2>
      <p><strong>Composite Match Score:</strong>
        <span id="scoreValue" class="score-badge"></span></p>
      <p style="margin-top:12px"><strong>TF-IDF Semantic Match:</strong> <span id="tfidf"></span>%</p>
      <p style="margin-top:4px"><strong>Skill Score:</strong> <span id="skillScore"></span>%</p>
      <p style="margin-top:4px"><strong>Experience Detected:</strong> <span id="expYears"></span> years</p>
      <p style="margin-top:16px"><strong>Matched Skills:</strong></p>
      <div id="matchedSkills" style="margin-top:8px"></div>
      <p style="margin-top:16px"><strong>Bias Analysis:</strong>
        <span id="biasScore"></span> bias-free score</p>
      <div id="biasFlags" style="margin-top:8px"></div>
      <p id="biasRec" style="margin-top:8px;font-style:italic;color:#555"></p>
    </div>
  </div>
  <script>
    async function screenResume() {
      const resumeText = document.getElementById("resumeText").value.trim();
      const jobId = parseInt(document.getElementById("jobSelect").value);
      if (!resumeText) { alert("Please paste a resume."); return; }
      const res = await fetch("/api/screen", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({resume_text: resumeText, job_id: jobId})
      });
      const data = await res.json();
      if (!data.success) { alert(data.error); return; }
      const r = data.result;
      const scoreEl = document.getElementById("scoreValue");
      scoreEl.textContent = r.composite_score + "%";
      scoreEl.className = "score-badge " + (r.composite_score >= 65 ? "high" : r.composite_score >= 40 ? "mid" : "low");
      document.getElementById("tfidf").textContent = r.tfidf_match;
      document.getElementById("skillScore").textContent = r.skill_score;
      document.getElementById("expYears").textContent = r.experience_years;
      document.getElementById("matchedSkills").innerHTML = r.matched_skills.map(s => `<span class="tag">${s}</span>`).join("");
      document.getElementById("biasScore").textContent = (r.bias_analysis.bias_free_score * 100).toFixed(0) + "%";
      document.getElementById("biasFlags").innerHTML = r.bias_analysis.flagged_terms.map(t => `<span class="flag">${t}</span>`).join("") || "<span class='tag'>None found</span>";
      document.getElementById("biasRec").textContent = r.bias_analysis.recommendation;
      document.getElementById("result").style.display = "block";
    }
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/screen", methods=["POST"])
def screen_resume():
    """Single resume screening endpoint."""
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "").strip()
    job_id = int(data.get("job_id", 1))
    if not resume_text:
        return jsonify({"success": False, "error": "resume_text is required"}), 400
    job = next((j for j in SAMPLE_JOB_ROLES if j["id"] == job_id), SAMPLE_JOB_ROLES[0])
    skill_data = score_skills(resume_text)
    tfidf_score = tfidf_match(resume_text, job["description"])
    exp_years = extract_years_experience(resume_text)
    exp_score = min(100.0, (exp_years / max(job["min_experience_years"], 1)) * 100)
    req_skills = job.get("required_skills", [])
    text_lower = resume_text.lower()
    req_hit = sum(1 for s in req_skills if s in text_lower)
    req_score = (req_hit / len(req_skills) * 100) if req_skills else 50.0
    composite = round(0.30 * tfidf_score + 0.25 * skill_data["skill_score"] + 0.25 * req_score + 0.20 * exp_score, 1)
    result = {
        "composite_score": composite,
        "tfidf_match": tfidf_score,
        "skill_score": skill_data["skill_score"],
        "matched_skills": skill_data["matched_skills"],
        "experience_years": exp_years,
        "required_skills_hit": req_hit,
        "bias_analysis": detect_bias(resume_text),
        "job_title": job["title"],
    }
    return jsonify({"success": True, "result": result})


@app.route("/api/rank", methods=["POST"])
def rank_candidates_endpoint():
    """Batch ranking: score multiple candidates against a job."""
    data = request.get_json(force=True)
    candidates = data.get("candidates", [])
    job_id = int(data.get("job_id", 1))
    if not candidates:
        return jsonify({"success": False, "error": "candidates list is required"}), 400
    job = next((j for j in SAMPLE_JOB_ROLES if j["id"] == job_id), SAMPLE_JOB_ROLES[0])
    ranked = rank_candidates(candidates, job)
    return jsonify({"success": True, "job": job["title"], "ranked_candidates": ranked, "total": len(ranked)})


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify({"success": True, "jobs": SAMPLE_JOB_ROLES})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SmartHire AI", "version": "1.0.0"})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"SmartHire AI starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
