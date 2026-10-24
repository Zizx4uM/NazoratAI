# NazoratAI: Banking Corporate Credit Monitoring & Early Warning System (EWS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Test Suite](https://img.shields.io/badge/Tests-45%20Passing-brightgreen.svg)](https://pytest.org)

> **NazoratAI** is an enterprise-grade banking credit surveillance platform and Early Warning System (EWS) engineered for corporate loan portfolios. It automates credit risk monitoring for **603+ enterprise borrowers** across 10 vital industrial sectors (including the bank's **TOP-20 credit exposures**), replacing manual Excel-based inspection routines with an explainable machine learning risk scoring pipeline, statutory EWS trigger engine, hybrid document auditor, and interactive executive dashboard.

---

## 📌 Problem & Business Context

In commercial banking operations, credit monitoring departments face major operational bottlenecks:

- **Portfolio Scale & Dispersion:** Over 600 large corporate borrowers distributed across 14 administrative regions and 72 Branch Service Centers (BXM), managed by 90 monitoring personnel across head-office and local branches.
- **Manual Reporting Friction:** Inspection acts (*Monitoring Dalolatnomasi*) are traditionally completed manually in Word and Excel, resulting in reporting delays, subjective risk ratings, and delayed escalation of distressed assets.
- **Regulatory Surveillance Tiers (Central Bank):**
  - **🟢 Green Tier (Standart):** Healthy enterprises undergoing standard semi-annual (6-month) review.
  - **🟡 Yellow Tier (Substandart / Watch-list):** Delinquency exceeding **45 days** triggers mandatory emergency inspection (*favqulodda monitoring*) and monthly surveillance.
  - **🔴 Red Tier (Distressed / NPL):** Delinquency exceeding **90 days** or severe insolvency triggers litigation and collateral foreclosure.
  - **Operational Early Warning Triggers:** Revenue drop $\ge 15\%$ MoM, production volume collapse $\ge 20\%$ MoM, utility drop $\ge 25\%$, and 2-Kartoteka (*K-2 card index*) bank arrears.

**NazoratAI** solves these challenges by combining continuous portfolio data ingestion, automated rule-based EWS alerts, machine learning credit risk classification with exact SHAP-style explainability, multi-format Document AI, and executive dashboarding.

---

## 🏗️ System Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │            Raw Data Ingestion                │
                                  │   (Core Banking, Tax, Credit Bureau, ERP)    │
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
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │             Delivery & Visualization            │
                        │  - Automated ReportLab PDF & Markdown Acts      │
                        │  - Interactive Streamlit Executive Dashboard    │
                        │  - FastAPI REST Microservice (Swagger UI)       │
                        └─────────────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

### 1. Realistic Corporate Data Generator
- Simulates an enterprise portfolio of 600+ legal entities across 10 vital sectors (cotton-textile clusters, greenhouses, construction materials, food processing, pharmaceuticals, automotive parts).
- Incorporates natural macroeconomic variance, overlapping financial ratio distributions across risk tiers, and TOP-20 large-exposure facilities.

### 2. Machine Learning Credit Risk Classifier
- Multi-class Random Forest classifier trained on 16 financial and operational ratios (DSCR, Current Ratio, Debt-to-Equity, EBITDA margin, K-2 arrears ratio, delinquency days, bureau score).
- Predicts calibrated class probabilities and calculates a continuous $0\text{--}100$ credit risk score ($E[\text{Score}] = 10 \cdot P(\text{Green}) + 50 \cdot P(\text{Yellow}) + 95 \cdot P(\text{Red})$).

### 3. Exact Explainable AI (TreeSHAP)
- Implements exact multi-class Tree Path Attribution (Saabas decomposition) for regulatory auditability.
- Mathematically satisfies additive attribution: $\text{BaseScore} + \sum_{j} \Delta_j = \text{TotalRiskScore}$.
- Generates human-readable risk driver summaries in both Uzbek and English.

### 4. Hybrid Document AI
- Parses inspection acts across multiple file formats: `.docx`, `.pdf`, `.xlsx`, and `.txt`.
- **Dual Engine:** Leverages LLM APIs (Gemini / OpenAI) when credentials are provided in `.env`, and seamlessly falls back to 100% offline regex/NLP pattern matching without external network calls.

### 5. Automated Reporting & Executive UI
- Generates publication-ready ReportLab PDF audit dossiers with structured SHAP risk driver impact tables.
- 6-view reactive Streamlit dashboard providing portfolio heatmaps, regional risk breakdown, TOP-20 watchlist radar, and borrower 360° cards.
- High-performance FastAPI service with OpenAPI interactive documentation.

### 6. XERJ Search & Real-Time EWS Integration
- Native bridge to [XERJ](https://github.com/xerj-org/xerj) (`localhost:9200`) providing high-speed search and continuous anomaly scoring:
  - **Dynamic Anomaly Datafeeds (`_ml/datafeeds`):** Streams daily borrower account turnovers, electricity consumption, and 2-Kartoteka accumulation to compute moving-window $z$-scores that detect distress prior to statutory delinquency cliffs.
  - **Long-Term Borrower Memory (`/_memory/{inn}`):** Stores qualitative branch field inspection notes and recalls historical findings via BM25, semantic vector, and reciprocal rank fusion.
  - **Sub-Millisecond Portfolio Search:** Indexes 603 corporate borrowers into typed Elasticsearch-compatible indices for instant multi-match queries and filtering.

---

## 📊 Model Evaluation & Benchmarks

The model is evaluated using a **stratified 80/20 train/test split** and **5-fold stratified cross-validation** on synthetic corporate borrower data with natural distribution overlap:

| Metric | Held-Out Test Set (20%) | 5-Fold Stratified CV | Target Benchmark |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **98.35%** | **94.86%** | $\ge 80.0\%$ |
| **Macro F1-Score** | **0.9702** | **0.9365 (±0.028)** | $\ge 0.750$ |
| **ROC-AUC (OVR)** | **0.9977** | **0.9806 (±0.007)** | $\ge 0.850$ |
| **EWS Watchlist Precision** | **100.00%** | **—** | $\ge 70.0\%$ |
| **EWS Watchlist Recall** | **97.22%** | **—** | $\ge 65.0\%$ |

*Note: EWS Watchlist metrics evaluate identification of Yellow and Red distressed accounts against statutory Central Bank guidelines.*

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Zizx4uM/NazoratAI.git
cd NazoratAI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Generate Dataset & Train Model

```bash
# Generate 603 realistic corporate borrower records
nazorat-ai generate-data --count 603

# Train ML model with 80/20 split and 5-fold cross-validation
nazorat-ai train-model
```

### 3. Scan Portfolio & Generate Audit Dossiers

```bash
# Run EWS portfolio surveillance scan
nazorat-ai scan-portfolio

# Audit a specific corporate borrower by 9-digit INN and export PDF
nazorat-ai audit-borrower --inn 300001000 --pdf
```

### 4. Hybrid Document AI Review

```bash
# Audit field inspector notes (runs 100% offline or with API key)
nazorat-ai review-doc --file samples/monitoring_act_distressed.txt
```

### 5. Launch Web Dashboard & API

```bash
# Launch interactive Streamlit dashboard
nazorat-ai run-dashboard --port 8501
# Open http://localhost:8501 in your browser

# Launch FastAPI microservice
nazorat-ai serve-api --host 0.0.0.0 --port 8000
# Interactive Swagger docs at http://localhost:8000/docs
```

### 6. Connect XERJ Search & Anomaly Engine

```bash
# Verify XERJ node status
nazorat-ai xerj-status --url http://localhost:9200

# Index corporate borrower accounts for sub-second retrieval
nazorat-ai xerj-sync --data data/corporate_portfolio.csv
```

---

## 📁 Repository Structure

```
nazorat_ai/
├── src/nazorat_ai/
│   ├── api/                 # FastAPI application and route endpoints
│   ├── cli/                 # Terminal CLI entry point and subcommands
│   ├── dashboard/           # Streamlit executive multi-page dashboard
│   ├── data/                # Data models, loader, and portfolio generator
│   ├── document_review/     # Multi-format parser & hybrid NLP/LLM reviewer
│   ├── ews/                 # Statutory Central Bank Early Warning rules
│   ├── ml/                  # Feature pipeline, classifier, TreeSHAP explainer
│   ├── reporting/           # ReportLab PDF and Markdown dossier generators
│   └── config.py            # Global thresholds, regulatory constants, enums
├── tests/                   # 45 automated pytest suites (unit, E2E, ML)
├── samples/                 # Sample inspection acts (.txt, .docx)
├── pyproject.toml           # Packaging metadata and CLI script bindings
├── requirements.txt         # Core dependencies
├── CONTRIBUTING.md          # Open-source contribution guidelines
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🧪 Testing & Verification

Run the full automated test suite covering all modules:

```bash
pytest -v tests/
```

Test coverage includes:
- **Synthetic Data Generation:** Verifies 603 borrower counts, TOP-20 allocations, 10 sectors, 14 regions, small count bounds, and schema rules.
- **EWS Rules:** Tests all 9 trigger rules and strict threshold boundaries (overdue 45 vs 46d, overdue 90 vs 91d, revenue slump $\ge 15\%$, production drop $\ge 20\%$, K-2 card index, utility drop, under-collateralization).
- **ML & Explainability:** Verifies feature extraction with missing optional columns tolerance, multi-class predictions, calibrated probabilities, exact Saabas tree attribution additivity ($\text{Base} + \sum \Delta_j = \text{Score}$), and held-out cross-validation.
- **Document AI:** Tests multi-format extraction (.docx, .pdf, .xlsx, .txt), empty document safety, and offline regex/NLP pattern matching.
- **Reporting:** Asserts valid Markdown output, verifies PDF generation with SHAP driver table, ReportLab encoding safety, and division-by-zero guards.
- **REST API:** End-to-end integration tests on all endpoints using `TestClient`.

---

## 💼 Resume & Interview Highlights

For data science, quantitative finance, and software engineering portfolios:

- **Architected and developed NazoratAI**, a production-grade banking corporate credit monitoring and Early Warning System (EWS) tracking 600+ enterprise borrowers ($2B+ aggregate portfolio exposure) across 10 vital industrial sectors.
- **Engineered a hybrid credit surveillance pipeline** combining statutory regulatory rules (e.g. 15% revenue drop, 45+ overdue days) with a multi-class Random Forest classifier and exact Tree Path Attribution (TreeSHAP decomposition), achieving **0.98 ROC-AUC** and **0.94 Macro F1** on stratified 5-fold cross-validation.
- **Designed a resilient Document AI auditor** for field inspection acts (`.docx`, `.pdf`, `.xlsx`, `.txt`), implementing dual-mode intelligence (Gemini LLM API with 100% offline regex/NLP fallback) and automated ReportLab PDF audit act generation.
- **Built a high-performance FastAPI microservice and an interactive Streamlit dashboard** providing executive portfolio heatmaps, TOP-20 exposure watch-lists, and borrower 360° dossiers with interactive risk driver waterfalls.
- **Integrated XERJ Search & Anomaly Detection**, streaming real-time borrower telemetry into moving-window $z$-score datafeeds (`_ml/datafeeds`) and providing isolated long-term inspection act memory (`/_memory/{inn}`).

---

## 📜 License & Acknowledgments

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Developed by **[Xajibayev Azizbek](https://github.com/Zizx4uM)** ([khadjibayev65@gmail.com](mailto:khadjibayev65@gmail.com)).

Special credit to the **[XERJ Community](https://github.com/xerj-org/xerj)** for providing the high-speed local search and anomaly detection infrastructure powering NazoratAI's continuous surveillance pipeline.
