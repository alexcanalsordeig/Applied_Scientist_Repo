# Collaborative Filtering Recommendation System

Matrix factorization-based recommender system built on the MovieLens 100K dataset.
Implements ALS (Alternating Least Squares) with implicit feedback, evaluated with
standard ranking metrics (NDCG, MAP, Precision@K).

---

## Project Structure

```
recommendation-system/
├── src/
│   ├── data_loader.py      # Dataset download, preprocessing, train/test split
│   ├── evaluation.py       # Ranking metrics: NDCG@K, MAP, Precision@K, Recall@K
│   └── model.py            # ALS matrix factorization (implicit feedback)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset stats, sparsity, rating distributions
│   ├── 02_matrix_factorization.ipynb # Model training and hyperparameter search
│   └── 03_model_evaluation.ipynb     # Ranking evaluation and results analysis
└── README.md
```

---

## Problem

Standard collaborative filtering treats ratings as explicit feedback (1–5 stars).
This project treats them as **implicit feedback** — a user rating a film signals
interest, regardless of the score. This better reflects real-world recommendation
scenarios where purchase history or click data is available but explicit ratings are not.

---

## Approach

**Algorithm:** ALS (Alternating Least Squares) matrix factorization via the `implicit` library.

ALS decomposes the user-item interaction matrix R ≈ U × Vᵀ into latent factor matrices,
alternating between fixing U and solving for V, and vice versa. Each step has a
closed-form solution, making ALS efficient and scalable.

**Key design choices:**
- Confidence weighting: `c_ui = 1 + α × r_ui` (higher interaction → higher confidence)
- L2 regularisation to prevent overfitting on popular items
- Evaluation on held-out users (leave-last-out split)

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings
from 943 users across 1,682 movies.

| Stat              | Value     |
|-------------------|-----------|
| Users             | 943       |
| Items (movies)    | 1,682     |
| Interactions      | 100,000   |
| Matrix sparsity   | ~93.7%    |
| Rating scale      | 1–5 stars |

---

## Evaluation Metrics

Ranking metrics computed on held-out test interactions (top-K recommendations):

| Metric         | Description                                              |
|----------------|----------------------------------------------------------|
| **NDCG@K**     | Normalised Discounted Cumulative Gain — rewards rank position |
| **MAP**        | Mean Average Precision across all users                  |
| **Precision@K** | Fraction of top-K recommendations that are relevant     |
| **Recall@K**   | Fraction of relevant items found in top-K               |

---

## How to Run

**Requirements:** Python 3.9+

```bash
# 1. Install dependencies
pip install numpy scipy pandas matplotlib seaborn tqdm jupyter implicit

# 2. Launch notebooks (recommended — step-by-step walkthrough)
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Or use the src modules directly
python -c "
from src.data_loader import load_movielens
from src.model import ALSModel
from src.evaluation import evaluate_ranking

train, test = load_movielens()
model = ALSModel(factors=64, regularization=0.1, alpha=40)
model.fit(train)
metrics = evaluate_ranking(model, train, test, K=10)
print(metrics)
"
```

**Optional — GPU-accelerated ALS** (for large datasets):
```bash
pip install implicit>=0.7.0   # ships with CUDA support
```

**Optional — experiment tracking:**
```bash
pip install mlflow>=2.5.0
mlflow ui   # view runs at http://localhost:5000
```

---

## Stack

- **Matrix factorization**: `implicit` (ALS with confidence weighting)
- **Numerical computing**: numpy · scipy (sparse matrices)
- **Data processing**: pandas
- **Visualisation**: matplotlib · seaborn
- **Experiment tracking**: mlflow (optional)
- **Testing**: pytest · pytest-cov

---

## Skills Demonstrated

- Implicit feedback modelling (vs. explicit rating prediction)
- Matrix factorization with ALS — understanding of the closed-form update step
- Information retrieval evaluation: NDCG, MAP, Precision/Recall @K
- Modular code structure separating data, model, and evaluation concerns
- Sparse matrix handling with scipy for memory-efficient computation
