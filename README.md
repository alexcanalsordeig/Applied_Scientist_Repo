# ML & Data Engineering Portfolio

Alexandre Canals Ordeig — selected projects across the full analytics value chain:
data engineering and deployment, machine learning, and analytics.

---

## Projects

### [`mdm-golden-record-pipeline/`](./mdm-golden-record-pipeline)
**Master Data Management — Golden Record Pipeline**
End-to-end MDM pipeline consolidating 28 conflicting records from 3 source systems into 12
audited golden records. Fully event-driven **AWS** architecture (Glue, Step Functions, Lambda,
EventBridge, SNS/SQS, Athena) provisioned entirely with **Terraform**; identical **PySpark**
logic runs locally or on Glue. Entity matching with a corroborating-signal rule and field-level
survivorship with full source provenance, over a two-layer PostgreSQL design.
`Python · PySpark · AWS · Terraform · PostgreSQL`

### [`recommendation-system/`](./recommendation-system)
**Recommendation System (ALS · Collaborative Filtering)**
Matrix-factorization recommender on MovieLens 100K with implicit feedback, rigorously evaluated
on NDCG@K, MAP, Precision@K and Recall@K. Modular, reproducible codebase with optional
GPU-accelerated ALS and MLflow experiment tracking.
`Python · NumPy · SciPy · MLflow`

### [`stock-screener/`](./stock-screener)
**Signal — Systematic Stock Screener**
Streamlit app screening the full S&P 500 on a momentum + quality factor model, building a
weighted portfolio and backtesting it against SPY on Sharpe, CAGR and max drawdown — a
reproducible research framework, not a price predictor.
`Python · Streamlit · Pandas · Plotly`

### [`ecommerce-sql-analytics/`](./ecommerce-sql-analytics)
**E-commerce SQL Analytics Platform**
Analytics on a synthetic PostgreSQL dataset (500 users, 8k orders, 40k clickstream events):
cohort retention, conversion funnels, RFM segmentation, rolling revenue with window functions,
and market-basket analysis.
`Python · PostgreSQL · Pandas`

---

## About

Data & ML engineer with a background spanning naval, maritime and industrial domains.
Each project here is a self-contained, reproducible codebase — see its own folder for
setup and run instructions.
