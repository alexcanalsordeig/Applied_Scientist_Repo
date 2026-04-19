"""
model.py
--------
Alternating Least Squares (ALS) for implicit feedback collaborative filtering.
 
Based on: Hu, Koren & Volinsky (2008) — "Collaborative Filtering for
Implicit Feedback Datasets". This is the canonical algorithm used at
Amazon, Netflix, Spotify for large-scale recommendations.
 
Key insight: Instead of treating missing data as "disliked", we treat it as
"unknown" and assign lower confidence to those entries. We then optimize:
 
    min_{U, V} Σ c_ui (p_ui - u_i^T v_i)^2 + λ(||U||² + ||V||²)
 
where:
    p_ui = 1 if user u interacted with item i, else 0  (preference)
    c_ui = 1 + α * r_ui  (confidence; r_ui = raw interaction count)
"""
 
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple
 
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
 
logger = logging.getLogger(__name__)
 
 
class ALSModel:
    """
    Implicit ALS with closed-form user/item factor updates.
 
    Why ALS over SGD for implicit feedback?
    - ALS has an exact closed-form solution per factor update (no LR tuning).
    - Each user/item update is independent → embarrassingly parallelizable.
    - Handles confidence-weighted interactions naturally.
    - Convergent behavior, easier to reason about in production.
 
    Args:
        n_factors: Dimensionality of the latent space.
        regularization: L2 penalty to prevent overfitting.
        alpha: Confidence scaling for implicit feedback (Hu et al.).
        n_iterations: Number of ALS sweeps (alternate U then V).
        random_state: For reproducibility.
    """
 
    def __init__(
        self,
        n_factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        n_iterations: int = 20,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.random_state = random_state
 
        self.user_factors: Optional[np.ndarray] = None  # (n_users, n_factors)
        self.item_factors: Optional[np.ndarray] = None  # (n_items, n_factors)
        self._is_fitted = False
 
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
 
    def fit(self, train_matrix: csr_matrix,
            val_matrix: Optional[csr_matrix] = None,
            eval_fn=None) -> "ALSModel":
        """
        Fit ALS factors to the training matrix.
 
        Args:
            train_matrix: (n_users, n_items) CSR matrix of implicit feedback.
            val_matrix: Optional validation matrix for loss tracking.
            eval_fn: Callable(model, val_matrix) -> dict of metrics.
        """
        n_users, n_items = train_matrix.shape
        rng = np.random.RandomState(self.random_state)
 
        # Initialize factors with small random values
        # Xavier init: scale by 1/sqrt(n_factors) for stable gradient flow
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)
 
        # Confidence matrix: C = 1 + alpha * R
        # We work with the sparse R and add 1 implicitly to avoid n_users*n_items memory
        confidence = train_matrix.copy().astype(np.float32)
        confidence.data = 1.0 + self.alpha * confidence.data
 
        # Precompute item-side for efficiency: R^T for item updates
        confidence_T = confidence.T.tocsr()
 
        logger.info(
            f"Starting ALS: {n_users} users, {n_items} items, "
            f"factors={self.n_factors}, α={self.alpha}, λ={self.regularization}"
        )
 
        reg_eye = self.regularization * np.eye(self.n_factors, dtype=np.float32)
        history = []
 
        for iteration in range(self.n_iterations):
            t0 = time.time()
 
            # --- Update user factors ---
            # For each user u: solve (V^T C_u V + λI) u_u = V^T C_u p_u
            item_sq = self.item_factors.T @ self.item_factors  # (f, f)
            self._update_factors(
                self.user_factors, self.item_factors,
                confidence, item_sq, reg_eye
            )
 
            # --- Update item factors ---
            user_sq = self.user_factors.T @ self.user_factors
            self._update_factors(
                self.item_factors, self.user_factors,
                confidence_T, user_sq, reg_eye
            )
 
            elapsed = time.time() - t0
            train_loss = self._weighted_mse(confidence, self.user_factors,
                                            self.item_factors)
 
            log_entry = {"iteration": iteration + 1, "train_loss": train_loss,
                         "time_s": elapsed}
 
            if val_matrix is not None and eval_fn is not None:
                val_metrics = eval_fn(self, val_matrix)
                log_entry.update(val_metrics)
 
            history.append(log_entry)
            logger.info(
                f"Iter {iteration+1:02d}/{self.n_iterations} | "
                f"loss={train_loss:.4f} | {elapsed:.1f}s"
            )
 
        self._is_fitted = True
        self.history_ = history
        return self
 
    def _update_factors(
        self,
        target_factors: np.ndarray,
        other_factors: np.ndarray,
        confidence: csr_matrix,
        other_sq: np.ndarray,
        reg_eye: np.ndarray,
    ):
        """
        Closed-form ALS update for one side (users or items).
 
        Derivation: differentiate the objective w.r.t. u_u, set to zero:
            A_u = V^T V + V^T (C_u - I) V + λI
            b_u = V^T C_u p_u
            u_u = A_u^{-1} b_u
        """
        n = target_factors.shape[0]
 
        for idx in range(n):
            # Sparse row: indices and confidence values for this user/item
            start, end = confidence.indptr[idx], confidence.indptr[idx + 1]
            item_indices = confidence.indices[start:end]
            conf_values = confidence.data[start:end]  # c_ui values
 
            if len(item_indices) == 0:
                # Cold-start: no interactions — keep random init
                continue
 
            # Factors of interacted items: (k, n_factors)
            V_u = other_factors[item_indices]
 
            # A = V^T V + V^T (C_u - I) V + λI
            # Since C_u - I is nonzero only at interacted items:
            # A = other_sq + V_u^T diag(c_ui - 1) V_u + λI
            conf_minus_one = conf_values - 1.0  # (k,)
            A = other_sq + (V_u * conf_minus_one[:, None]).T @ V_u + reg_eye
 
            # b = V^T C_u p_u  (p_u=1 for interacted items)
            b = (V_u * conf_values[:, None]).sum(axis=0)  # (n_factors,)
 
            # Solve linear system (more numerically stable than explicit inverse)
            target_factors[idx] = np.linalg.solve(A, b)
 
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
 
    def recommend(
        self,
        user_idx: int,
        train_matrix: csr_matrix,
        n: int = 10,
        filter_already_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N item recommendations for a single user.
 
        Returns list of (item_idx, score) tuples, sorted descending.
        """
        self._check_fitted()
        scores = self.user_factors[user_idx] @ self.item_factors.T  # (n_items,)
 
        if filter_already_seen:
            seen = train_matrix[user_idx].indices
            scores[seen] = -np.inf
 
        top_n = np.argpartition(scores, -n)[-n:]
        top_n = top_n[np.argsort(-scores[top_n])]
        return [(int(i), float(scores[i])) for i in top_n]
 
    def recommend_batch(
        self,
        user_indices: np.ndarray,
        train_matrix: csr_matrix,
        n: int = 10,
    ) -> np.ndarray:
        """
        Vectorized batch recommendations.
        Returns (len(user_indices), n) array of item indices.
        More efficient than calling recommend() in a loop.
        """
        self._check_fitted()
        U = self.user_factors[user_indices]        # (batch, f)
        scores = U @ self.item_factors.T           # (batch, n_items)
 
        # Mask seen items (vectorized)
        for i, u in enumerate(user_indices):
            seen = train_matrix[u].indices
            scores[i, seen] = -np.inf
 
        # Top-N per user
        top_n_indices = np.argpartition(scores, -n, axis=1)[:, -n:]
        # Sort each user's top-N
        sorted_top = np.array([
            top_n_indices[i][np.argsort(-scores[i, top_n_indices[i]])]
            for i in range(len(user_indices))
        ])
        return sorted_top
 
    def get_similar_items(self, item_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items via cosine similarity in the latent space.
        Useful for item-to-item recommendations (e.g., "customers also bought").
        """
        self._check_fitted()
        v = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        sims = (self.item_factors @ v) / (norms * np.linalg.norm(v) + 1e-9)
        sims[item_idx] = -np.inf  # exclude self
        top_n = np.argpartition(sims, -n)[-n:]
        top_n = top_n[np.argsort(-sims[top_n])]
        return [(int(i), float(sims[i])) for i in top_n]
 
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
 
    def save(self, path: str):
        """Save model factors to disk (NumPy compressed format)."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            n_factors=self.n_factors,
            regularization=self.regularization,
            alpha=self.alpha,
        )
        logger.info(f"Model saved to {path}")
 
    @classmethod
    def load(cls, path: str) -> "ALSModel":
        """Load saved model factors."""
        data = np.load(path)
        model = cls(
            n_factors=int(data["n_factors"]),
            regularization=float(data["regularization"]),
            alpha=float(data["alpha"]),
        )
        model.user_factors = data["user_factors"]
        model.item_factors = data["item_factors"]
        model._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return model
 
    # ------------------------------------------------------------------
    # Internal utils
    # ------------------------------------------------------------------
 
    def _weighted_mse(self, confidence: csr_matrix,
                      U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute weighted MSE loss on observed interactions only.
        Full-matrix computation is O(n_users * n_items) — too slow.
        We approximate using only the nonzero entries.
        """
        loss = 0.0
        for u in range(U.shape[0]):
            start, end = confidence.indptr[u], confidence.indptr[u + 1]
            if start == end:
                continue
            items = confidence.indices[start:end]
            c = confidence.data[start:end]
            pred = U[u] @ V[items].T
            loss += np.sum(c * (1.0 - pred) ** 2)
        # Regularization term
        loss += self.regularization * (np.sum(U ** 2) + np.sum(V ** 2))
        return float(loss)
 
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")