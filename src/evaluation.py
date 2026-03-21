"""
evaluation.py
-------------
Ranking metrics for implicit feedback recommender systems.
 
In collaborative filtering, we care about the *rank* of relevant items in
the recommendation list, not just whether they appear. Standard metrics:
 
    - Precision@K:   fraction of top-K recs that are relevant
    - Recall@K:      fraction of relevant items that appear in top-K
    - NDCG@K:        normalized discounted cumulative gain — rewards
                     putting relevant items higher in the list
    - MAP@K:         mean average precision
    - MRR:           mean reciprocal rank — how high is the FIRST hit?
    - Hit Rate@K:    did at least one relevant item appear in top-K?
 
Why these and not RMSE? Because we're predicting *rankings*, not ratings.
A user doesn't care if the score for an item is 4.2 vs 4.5 — they care
whether it shows up in their top-10 feed.
"""
 
import logging
from typing import Dict, List, Optional
 
import numpy as np
from scipy.sparse import csr_matrix
 
logger = logging.getLogger(__name__)
 
 
# ------------------------------------------------------------------
# Core metric functions (operate on arrays, no model dependency)
# ------------------------------------------------------------------
 
def precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Precision@K = |recommended[:k] ∩ relevant| / k
 
    Measures exactness: of the K items we showed, how many were relevant?
    """
    if k == 0:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & set(relevant)) / k
 
 
def recall_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Recall@K = |recommended[:k] ∩ relevant| / |relevant|
 
    Measures coverage: of all relevant items, how many did we surface?
    """
    if len(relevant) == 0:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & set(relevant)) / len(relevant)
 
 
def ndcg_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
 
    DCG@K = Σ_{i=1}^{K} rel_i / log2(i+1)
    where rel_i = 1 if recommended[i] is relevant, else 0.
 
    IDCG@K = DCG of the ideal ranking (all relevant items first).
 
    NDCG penalizes relevant items appearing lower in the list.
    This is the primary metric at Amazon/Netflix for offline evaluation.
    """
    relevant_set = set(relevant)
    gains = []
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            gains.append(1.0 / np.log2(i + 2))  # i+2 because log2(1+1)=1
 
    dcg = sum(gains)
 
    # Ideal DCG: best possible ranking
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))
 
    return dcg / idcg if idcg > 0 else 0.0
 
 
def average_precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    AP@K = (1 / |relevant|) * Σ_{i=1}^{K} P@i * rel_i
 
    Average precision rewards systems that rank all relevant items high,
    not just the first one. MAP = mean of AP across users.
    """
    if len(relevant) == 0:
        return 0.0
    relevant_set = set(relevant)
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)
 
 
def reciprocal_rank(recommended: np.ndarray, relevant: np.ndarray) -> float:
    """
    RR = 1 / rank_of_first_hit
 
    MRR = mean reciprocal rank across users. Useful when only the
    top result matters (e.g., autocomplete, voice assistants).
    """
    relevant_set = set(relevant)
    for i, item in enumerate(recommended):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0
 
 
def hit_rate_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    HR@K = 1 if any relevant item is in top-K, else 0.
 
    Binary metric. Measures whether the system is "useful" for the user.
    """
    return float(len(set(recommended[:k]) & set(relevant)) > 0)
 
 
# ------------------------------------------------------------------
# Evaluator class
# ------------------------------------------------------------------
 
class RecommenderEvaluator:
    """
    Evaluates a fitted recommendation model against held-out interactions.
 
    Protocol:
    1. For each user in the test set, generate top-K recommendations
       (excluding training items).
    2. Compare recommendations against held-out test items.
    3. Average metrics across all users.
 
    Args:
        k_values: List of cutoffs to evaluate (e.g., [5, 10, 20]).
        n_test_users: If set, sample this many users for faster evaluation.
    """
 
    def __init__(self, k_values: List[int] = None, n_test_users: Optional[int] = None):
        self.k_values = k_values or [5, 10, 20]
        self.n_test_users = n_test_users
 
    def evaluate(
        self,
        model,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run full evaluation and return metric dictionary.
 
        Args:
            model: Fitted ALSModel (or any model with .recommend()).
            train_matrix: Matrix used for training (to filter seen items).
            test_matrix: Held-out interactions to evaluate against.
 
        Returns:
            dict like {"NDCG@10": 0.123, "Recall@10": 0.234, ...}
        """
        max_k = max(self.k_values)
        n_users = test_matrix.shape[0]
 
        # Select users who have at least one test interaction
        test_users = np.where(np.diff(test_matrix.indptr) > 0)[0]
        if self.n_test_users and len(test_users) > self.n_test_users:
            rng = np.random.RandomState(42)
            test_users = rng.choice(test_users, self.n_test_users, replace=False)
 
        logger.info(f"Evaluating on {len(test_users)} users with k={self.k_values}")
 
        # Accumulators
        metrics_sum: Dict[str, float] = {}
        for k in self.k_values:
            for name in ["Precision", "Recall", "NDCG", "MAP", "HR"]:
                metrics_sum[f"{name}@{k}"] = 0.0
        metrics_sum["MRR"] = 0.0
 
        n_evaluated = 0
        for u in test_users:
            relevant = test_matrix[u].indices
            if len(relevant) == 0:
                continue
 
            recs = model.recommend(u, train_matrix, n=max_k,
                                   filter_already_seen=True)
            recommended = np.array([item for item, _ in recs])
 
            for k in self.k_values:
                metrics_sum[f"Precision@{k}"] += precision_at_k(recommended, relevant, k)
                metrics_sum[f"Recall@{k}"] += recall_at_k(recommended, relevant, k)
                metrics_sum[f"NDCG@{k}"] += ndcg_at_k(recommended, relevant, k)
                metrics_sum[f"MAP@{k}"] += average_precision_at_k(recommended, relevant, k)
                metrics_sum[f"HR@{k}"] += hit_rate_at_k(recommended, relevant, k)
 
            metrics_sum["MRR"] += reciprocal_rank(recommended, relevant)
            n_evaluated += 1
 
        # Average over users
        results = {k: v / n_evaluated for k, v in metrics_sum.items()}
 
        if verbose:
            self._print_results(results)
 
        return results
 
    def compare_models(
        self,
        models: Dict[str, object],
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models and return comparison dict.
        Useful for hyperparameter search or ablation studies.
        """
        all_results = {}
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            all_results[name] = self.evaluate(model, train_matrix, test_matrix,
                                              verbose=False)
 
        # Print comparison table
        print("\n" + "=" * 72)
        print(f"{'Model':<20}", end="")
        k = self.k_values[0]
        cols = [f"NDCG@{k}", f"Recall@{k}", f"MAP@{k}", f"HR@{k}", "MRR"]
        for col in cols:
            print(f"{col:>10}", end="")
        print()
        print("-" * 72)
        for name, res in all_results.items():
            print(f"{name:<20}", end="")
            for col in cols:
                print(f"{res.get(col, 0.0):>10.4f}", end="")
            print()
        print("=" * 72)
 
        return all_results
 
    def _print_results(self, results: Dict[str, float]):
        print("\n" + "=" * 52)
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 52)
        for key in sorted(results):
            print(f"{key:<20} {results[key]:>10.4f}")
        print("=" * 52)
 
 
# ------------------------------------------------------------------
# Baseline models (for comparison)
# ------------------------------------------------------------------
 
class PopularityBaseline:
    """
    Recommends the globally most popular items.
 
    Why include this? A good ML model must beat simple popularity.
    If it doesn't, the ML complexity is not justified.
    At Amazon, this is the standard sanity-check baseline.
    """
 
    def __init__(self):
        self.item_popularity: Optional[np.ndarray] = None
        self._is_fitted = False
 
    def fit(self, train_matrix: csr_matrix) -> "PopularityBaseline":
        # Popularity = number of users who interacted with each item
        self.item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
        self._is_fitted = True
        return self
 
    def recommend(self, user_idx: int, train_matrix: csr_matrix,
                  n: int = 10, filter_already_seen: bool = True):
        scores = self.item_popularity.copy()
        if filter_already_seen:
            seen = train_matrix[user_idx].indices
            scores[seen] = -np.inf
        top_n = np.argpartition(scores, -n)[-n:]
        top_n = top_n[np.argsort(-scores[top_n])]
        return [(int(i), float(scores[i])) for i in top_n]
 
 
class RandomBaseline:
    """Recommends random unseen items. Lower bound for any useful model."""
 
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        self.n_items: Optional[int] = None
        self._is_fitted = False
 
    def fit(self, train_matrix: csr_matrix) -> "RandomBaseline":
        self.n_items = train_matrix.shape[1]
        self._is_fitted = True
        return self
 
    def recommend(self, user_idx: int, train_matrix: csr_matrix,
                  n: int = 10, filter_already_seen: bool = True):
        seen = set(train_matrix[user_idx].indices)
        candidates = [i for i in range(self.n_items) if i not in seen]
        chosen = self.rng.choice(candidates, min(n, len(candidates)), replace=False)
        return [(int(i), 0.0) for i in chosen]