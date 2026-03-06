# SmartHire AI

### Eliminate hiring bias with AI-powered resume screening and intelligent candidate matching

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## About

**SmartHire AI** is an enterprise-grade, AI-powered hiring assistant designed to transform the way organizations screen and select candidates. By combining TF-IDF semantic analysis, weighted skill matching, and automated bias detection, SmartHire AI ensures every candidate is evaluated fairly and objectively — removing unconscious bias from the hiring pipeline and surfacing the best talent efficiently.

Built for the **TECH HACKS 2.0** hackathon (SRM Institute of Science and Technology), this project sits at the intersection of **Enterprise technology**, **Machine Learning/AI**, and **Social Good**.

---

## Features

- **Resume Screening** — Paste any resume and get an instant composite match score against a target job role
- **Bias Detection** — Flags potentially biased language (age, gender, religion, disability) and provides an actionable bias-free score
- **TF-IDF Semantic Matching** — Measures contextual similarity between resume and job description using NLP
- **Weighted Skill Scoring** — Domain-aware skill detection with role-specific importance weights
- **Batch Candidate Ranking** — Rank multiple candidates against a job role in a single API call
- **REST API** — Clean JSON endpoints for integration into existing HR systems
- **Responsive Web UI** — Clean, mobile-friendly dashboard for HR teams

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, Flask 2.x |
| ML/NLP | scikit-learn, NLTK, TF-IDF, Cosine Similarity |
| Data Processing | pandas, numpy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Docker-ready, runs on any WSGI server |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/muhibwqr/smarthire-ai.git
cd smarthire-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

The app will be live at `http://localhost:5000`.

---

## Usage

### Web Interface
Navigate to `http://localhost:5000`, paste a resume into the text area, select a job role, and click **Analyse Resume**. Results appear instantly showing:
- Composite match score with colour-coded grade
- Semantic TF-IDF match percentage
- Detected skills with tags
- Bias analysis report

### API Reference

**Screen a single resume**
```http
POST /api/screen
Content-Type: application/json

{
  "resume_text": "Experienced Python developer with 5 years in machine learning...",
  "job_id": 1
}
```

**Rank multiple candidates**
```http
POST /api/rank
Content-Type: application/json

{
  "job_id": 3,
  "candidates": [
    {"id": "c1", "name": "Alice", "resume_text": "..."},
    {"id": "c2", "name": "Bob",   "resume_text": "..."}
  ]
}
```

**List available job roles**
```http
GET /api/jobs
```

---

## Scoring Algorithm

The composite score is a weighted blend of four signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| TF-IDF Semantic Match | 30% | Contextual similarity to job description |
| Weighted Skill Score | 25% | Domain-weighted skill detection |
| Required Skills Hit | 25% | Fraction of must-have skills present |
| Experience Score | 20% | Years of experience vs. job requirement |

---

## Team

Built with passion at TECH HACKS 2.0 by a team dedicated to making hiring fairer and smarter.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
