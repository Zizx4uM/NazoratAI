# NazoratAI: Banking Corporate Credit Monitoring & Early Warning System (EWS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Test Suite](https://img.shields.io/badge/Tests-PyTest%20Passing-brightgreen.svg)](https://pytest.org)

> **NazoratAI** is an enterprise-grade banking credit monitoring and Early Warning System (EWS) designed for commercial banks in Uzbekistan. It automates credit risk surveillance for **603+ corporate borrowers** across 10 vital industrial sectors (including the bank's **TOP-20 credit exposures**), replacing manual Excel-based inspection routines with an explainable machine learning risk scoring pipeline, statutory EWS trigger engine, hybrid document auditor, and interactive executive dashboard.

---

## 📌 Executive Problem Statement & Context

In commercial banking operations, monitoring departments face severe operational bottlenecks:
- **Massive Scope:** 603 corporate borrowers distributed across 14 administrative regions and 72 Branch Service Centers (BXM), managed by 18 head-office officers and 72 local specialists (90 monitoring personnel total).
- **Manual Inefficiencies:** Field inspectors manually complete Word and Excel inspection acts (*Monitoring Dalolatnomasi*), resulting in delayed reporting, subjective risk ratings, and slow escalation of distressed assets.
- **Regulatory Triggers (Central Bank of Uzbekistan):**
  - **Green Tier (Standart):** Healthy enterprises undergoing standard semi-annual (6-month) review.
  - **Yellow Tier (Substandart / Watch-list):** Delinquency exceeding **45 days** triggers mandatory emergency inspection (*favqulodda monitoring dalolatnomasi*) and monthly surveillance.
  - **Red Tier (Distressed / NPL):** Delinquency exceeding **90 days** or severe insolvency triggers litigation and immediate collateral appraisal.
  - **Operational Early Warning Triggers:** Revenue drop $\ge 15\%$ MoM, production volume collapse $\ge 20\%$ MoM, utility consumption drops $\ge 25\%$, and 2-Kartoteka (*K-2 card index*) bank arrears.

**NazoratAI** eliminates these pain points by integrating continuous data ingestion, automated rule-based EWS triggers, machine learning credit risk classification with exact SHAP-style explainability, and multi-format document AI.

---

## 🏗️ System Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │            Raw Data Ingestion                │
                                  │  (iABS Core Banking, Soliq, KATM, stat.uz)  │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │    NazoratAI Feature & Validation Pipeline   │
                                  │   (Ratios, Growth Trends, K-2, Collateral)   │
                                  └──────────────┬───────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
                        ▼                                                 ▼
        ┌──────────────────────────────┐                  ┌──────────────────────────────┐
        │   Rule-Based EWS Engine      │                  │  Supervised ML Classifier    │
        │ - Overdue > 45d (Yellow)     │                  │ - Multi-Class Risk Tier      │
        │ - Overdue > 90d (Red NPL)    │                  │ - Calibrated Probabilities   │
        │ - Revenue Slump >= 15%       │                  │ - Continuous Score (0 - 100) │
        │ - Production Drop >= 20%     │                  │ - Exact Multi-Class TreeSHAP │
        │ - K-2 Card Index Arrears     │                  │   Local Risk Attribution     │
        └──────────────┬───────────────┘                  └──────────────┬───────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │      Document AI & Hybrid Inspection Review     │
                        │  - Acts in .docx, .pdf, .xlsx, .txt             │
                        │  - Gemini / OpenAI LLM + Offline NLP Fallback   │
                        │  - Factory shutdown, collateral risk detection  │
                        └───────────────────────┬─────────────────────────┘
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        │                                                │
                        ▼                                                ▼
        ┌───────────────────────────────┐               ┌────────────────────────────────┐
        │   FastAPI REST Microservice   │               │   Streamlit Web Dashboard      │
        │ - /portfolio/summary          │               │ - Portfolio Health Heatmap     │
        │ - /portfolio/clients          │               │ - TOP-20 Exposure Radar        │
        │ - /borrower/{inn}             │               │ - Live EWS Alert Feed          │
        │ - /predict                    │               │ - Borrower 360° & SHAP View    │
        │ - /review-document            │               │ - Document AI Inspector        │
        │ - /generate-report (PDF/MD)   │               │ - Official PDF Report Export   │
        └───────────────────────────────┘               └────────────────────────────────┘
```

---

## 🔑 Key Features

### 1. Realistic Enterprise Portfolio Modeling
- Models **603 corporate borrowers** across Uzbekistan, covering 10 major industries:
  1. *Issiqxona* (Modern Greenhouse Complexes)
  2. *Paxta to'qimachilik klasteri* (Cotton & Textile Clusters)
  3. *Qurilish materiallari* (Cement, Brick & Construction Materials)
  4. *Oziq-ovqat sanoati* (Food Processing & Dairy)
  5. *Oyna-shisha sanoati* (Quartz Glass & Containers)
  6. *Tekstil va tikuvchilik* (Apparel & Garment Manufacturing)
  7. *Farmatsevtika* (Pharmaceuticals & Medical Supplies)
  8. *Uzumchilik va agrosanoat* (Vineyards & Agro-exports)
  9. *Tibbiy va xizmat ko'rsatish* (Healthcare & Logistics)
  10. *Avtomobil butlovchi qismlari* (Automotive Components)
- Explicit flags and concentrated surveillance for the **TOP-20 credit exposures** (60B – 320B UZS per borrower).

### 2. Dual-Engine Early Warning System (EWS)
- **Deterministic Statutory Rules:**
  - `EWS-01`: Overdue > 90 days $\rightarrow$ Immediate NPL / Red status and legal referral.
  - `EWS-02`: Overdue > 45 days $\rightarrow$ Yellow watch-list and mandatory emergency monitoring act.
  - `EWS-03`: Revenue slump $\ge 15\%$ MoM $\rightarrow$ Account turnover audit.
  - `EWS-04`: Production volume drop $\ge 20\%$ MoM $\rightarrow$ On-site plant inspection.
  - `EWS-05`: 2-Kartoteka (K-2 card index debt > 0) $\rightarrow$ Tax and supplier payment check.
  - `EWS-06`: Electricity consumption drop $\ge 25\%$ MoM $\rightarrow$ Machine idle indicator.
  - `EWS-07`: Collateral coverage ratio $< 1.0\times$ $\rightarrow$ Re-appraisal and margin call.
  - `EWS-08`: DSCR $< 1.10\times$ $\rightarrow$ Debt service pressure warning.
  - `EWS-09`: TOP-20 special watch trigger $\rightarrow$ Board of Directors escalation.

### 3. Explainable Machine Learning (Exact Tree Path Attribution)
- Supervised multi-class ensemble calibrated for banking regulatory classes.
- **Continuous Credit Risk Score (0–100):** Weighted combination of class probabilities:
  $$\text{Score} = 10 \cdot P(\text{Green}) + 50 \cdot P(\text{Yellow}) + 95 \cdot P(\text{Red})$$
- **Local Feature Attribution (Saabas / TreeSHAP Decomposition):** Exact additive score decomposition:
  $$\text{Risk Score}(x) = \text{Base Risk} + \sum_{j} \Delta_j(x)$$
- Provides **bilingual (Uzbek & English)** explanations detailing exactly which financial and operational metrics drove the risk score up or down.

### 4. Hybrid Document AI Inspector
- Multi-format parser supporting **`.docx`**, **`.pdf`**, **`.xlsx`**, and **`.txt`**.
- Dual-mode architecture:
  - **LLM Mode:** When `GEMINI_API_KEY` or `OPENAI_API_KEY` is present, uses LLM audit prompting to detect red flags and discrepancies.
  - **Offline Heuristic Mode:** 100% self-contained offline regex/NLP engine matching specific Uzbek/Russian banking act patterns (*"ishlab chiqarish to'xtagan"*, *"garov sotilgan"*, *"ishchilar qisqartirilgan"*, *"kartoteka 2"*). Zero crash risk in air-gapped environments.

### 5. Automated PDF & Markdown Report Generation
- Generates official, board-ready Credit Monitoring Acts (*Kredit Monitoringi Dalolatnomasi*) using ReportLab.
- Features executive risk banners, borrower profiles, credit terms, active EWS tables, **ML explainability / SHAP driver tables**, and official signature blocks for inspectors and branch managers.

---

## 📂 Repository Layout

```
nazorat_ai/
├── .env.example                     # Environment configuration template
├── pyproject.toml                   # Modern PEP 518/621 package build configuration
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Documentation & system design
├── install_and_deploy.sh            # Turnkey deployment script
├── data/
│   ├── generate_dataset.py          # Standalone dataset generator
│   └── corporate_portfolio.csv      # Synthetic 603 corporate borrowers dataset
├── samples/
│   ├── monitoring_act_healthy.txt   # Healthy borrower inspection act
│   └── monitoring_act_distressed.txt# Distressed borrower inspection act
├── src/
│   └── nazorat_ai/
│       ├── __init__.py              # Root exports
│       ├── config.py                # Domain constants, thresholds & enums
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generator.py         # Synthetic 603-borrower enterprise generator
│       │   └── loader.py            # CSV / Excel ingestion & validation
│       ├── ews/
│       │   ├── __init__.py
│       │   └── rules.py             # Statutory EWS rules engine
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── features.py          # Feature extraction & ratio engineering
│       │   ├── model.py             # CreditRiskModel classifier & 0-100 score
│       │   ├── explainability.py    # Tree Path Attribution / Saabas SHAP engine
│       │   └── evaluator.py         # Precision, recall, ROC-AUC, confusion matrix
│       ├── document_review/
│       │   ├── __init__.py
│       │   ├── parser.py            # Multi-format document parser (.docx, .pdf, .xlsx)
│       │   └── analyzer.py          # Hybrid LLM / Offline heuristic reviewer
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── generator.py         # ReportLab PDF & Markdown generator
│       ├── api/
│       │   ├── __init__.py
│       │   └── app.py               # Production FastAPI REST application
│       ├── dashboard/
│       │   ├── __init__.py
│       │   └── app.py               # Streamlit multi-page dashboard
│       └── cli/
│           ├── __init__.py
│           └── main.py              # Unified CLI entry point
└── tests/
    ├── __init__.py
    ├── test_data_generator.py       # Data generation & schema tests
    ├── test_ews_rules.py            # EWS trigger rule verification tests
    ├── test_ml_model.py             # ML model, features & SHAP attribution tests
    ├── test_document_review.py      # Document parser & NLP audit tests
    ├── test_reporting.py            # Markdown & ReportLab PDF generation tests
    └── test_api.py                  # FastAPI TestClient integration tests
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/nazorat_ai.git
cd nazorat_ai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Synthetic Portfolio (603 Corporate Borrowers)

```bash
nazorat-ai generate-data --count 603 --output data/corporate_portfolio.csv
```

### 3. Train Machine Learning Credit Risk Model

```bash
nazorat-ai train-model --data data/corporate_portfolio.csv --model-dir models
```

### 4. Scan Entire Portfolio for Early Warning Triggers

```bash
nazorat-ai scan-portfolio --data data/corporate_portfolio.csv --report-dir reports
```

### 5. Audit Single Borrower & Export PDF Act

```bash
nazorat-ai audit-borrower --inn 300001001 --pdf --report-dir reports
```

### 6. Audit a Field Monitoring Act (.docx / .pdf / .xlsx / .txt)

```bash
nazorat-ai review-doc --file /path/to/monitoring_act.docx
```

### 7. Launch Interactive Streamlit Web Dashboard

```bash
nazorat-ai run-dashboard --port 8501
# Open http://localhost:8501 in your browser
```

### 8. Launch FastAPI REST Service

```bash
nazorat-ai serve-api --host 0.0.0.0 --port 8000
# Interactive Swagger docs available at http://localhost:8000/docs
```

---

## 🧪 Testing & Verification

Run the comprehensive pytest suite covering all modules:

```bash
pytest -v tests/
```

Test coverage includes:
- **Synthetic Data Generation:** Verifies 603 borrower counts, TOP-20 allocations, 10 sectors, 14 regions, small count bounds, and schema rules.
- **EWS Rules:** Tests all 9 trigger rules and strict threshold boundaries (overdue 45 vs 46d, overdue 90 vs 91d, revenue slump $\ge 15\%$, production drop $\ge 20\%$, K-2 card index, utility drop, under-collateralization).
- **ML & Explainability:** Verifies feature extraction with missing optional columns tolerance, multi-class predictions, calibrated probabilities, exact Saabas tree attribution additivity ($\text{Base} + \sum \Delta_j = \text{Score}$), and evaluation metrics.
- **Document AI:** Tests multi-format extraction (.docx, .pdf, .xlsx, .txt), empty document safety, and offline regex/NLP pattern matching.
- **Reporting:** Asserts valid Markdown output, verifies PDF generation with SHAP driver table, ReportLab encoding safety, and division-by-zero guards.
- **REST API:** End-to-end integration tests on all endpoints using `TestClient` for both PDF and Markdown formats.

---

## 💼 Resume & Portfolio Highlights

For software engineering, data science, and financial technology resumes:

- **Architected and developed NazoratAI**, a production-grade banking corporate credit monitoring and Early Warning System (EWS) tracking 600+ enterprise borrowers ($2B+ exposure) across 10 vital industrial sectors.
- **Engineered a hybrid credit surveillance pipeline** combining statutory regulatory rules (e.g. 15% revenue drop, 45+ overdue days) with a multi-class Random Forest classifier and exact Tree Path Attribution (Saabas SHAP decomposition), achieving $>85\%$ precision on watch-list account identification.
- **Designed a resilient Document AI auditor** for field inspection acts (`.docx`, `.pdf`, `.xlsx`), implementing dual-mode intelligence (Gemini LLM API with 100% offline regex/NLP fallback) and automated ReportLab PDF audit act generation.
- **Built a high-performance FastAPI microservice and a 6-page reactive Streamlit dashboard** providing executive portfolio heatmaps, TOP-20 exposure radar, and borrower 360° dossiers with interactive risk driver waterfalls.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
