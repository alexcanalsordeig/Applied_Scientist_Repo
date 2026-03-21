"""
data_loader.py
--------------
Loads and preprocesses implicit feedback data for collaborative filtering.
 
Implicit feedback: user interactions (clicks, views, purchases) that do NOT
explicitly say "I liked this" — the absence of interaction is ambiguous, not
negative. This is the standard setting at Amazon-scale recommender systems.
"""
 
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
 
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class InteractionDataset:
    """Container for a processed user-item interaction dataset."""
    train_matrix: csr_matrix       # (n_users, n_items) implicit feedback
    val_matrix: csr_matrix         # held-out interactions for evaluation
    test_matrix: csr_matrix
    user_id_map: dict              # original_id -> internal index
    item_id_map: dict
    reverse_user_map: dict         # internal index -> original_id
    reverse_item_map: dict
    n_users: int
    n_items: int
 
    @property
    def density(self) -> float:
        nnz = self.train_matrix.nnz
        return nnz / (self.n_users * self.n_items)
 
 
class ImplicitDataLoader:
    """
    Loads interaction logs and builds sparse user-item matrices.
 
    Design decisions (relevant for interviews):
    - Uses CSR (Compressed Sparse Row) format: O(1) row slicing, ideal for
      iterating over a user's history during training.
    - Binarizes feedback: confidence weighting (Hu et al., 2008) is applied
      in the model, not here — keeps preprocessing concerns separate.
    - Temporal split instead of random split: avoids data leakage. We train
      on past interactions and evaluate on future ones, mimicking production.
    """
 
    def __init__(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def load_from_csv(self, filepath: str, user_col: str = "user_id",
                      item_col: str = "item_id", rating_col: Optional[str] = None,
                      timestamp_col: Optional[str] = None) -> InteractionDataset:
        """
        Load interaction data from a CSV file.
 
        Args:
            filepath: Path to CSV with at least user and item columns.
            user_col: Column name for user identifiers.
            item_col: Column name for item identifiers.
            rating_col: Optional explicit ratings (will be binarized).
            timestamp_col: If provided, uses temporal train/val/test split.
        """
        logger.info(f"Loading interactions from {filepath}")
        df = pd.read_csv(filepath)
        return self._process_dataframe(df, user_col, item_col, rating_col, timestamp_col)
 
    def load_movielens_100k(self, data_dir: str = "data/") -> InteractionDataset:
        """
        Convenience loader for the MovieLens 100K benchmark dataset.
        Downloads if not present. Good for reproducible experiments.
        """
        filepath = Path(data_dir) / "ml-100k" / "u.data"
        if not filepath.exists():
            self._download_movielens(data_dir)
 
        df = pd.read_csv(
            filepath, sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
        # Treat any rating as implicit positive (user engaged with item)
        logger.info(f"Loaded MovieLens 100K: {len(df):,} interactions")
        return self._process_dataframe(
            df, "user_id", "item_id", "rating", "timestamp"
        )
 
    def generate_synthetic(self, n_users: int = 1000, n_items: int = 500,
                           density: float = 0.02,
                           random_state: int = 42) -> InteractionDataset:
        """
        Generates synthetic implicit feedback for unit tests and demos.
        Simulates power-law distributed interactions (realistic for e-commerce).
        """
        rng = np.random.RandomState(random_state)
 
        # Power-law user activity: most users are occasional, few are heavy
        user_activity = rng.zipf(1.5, n_users).clip(1, n_items // 2)
        # Power-law item popularity: blockbuster vs. long-tail
        item_pop = rng.zipf(1.8, n_items)
        item_probs = item_pop / item_pop.sum()
 
        rows, cols, data = [], [], []
        for u in range(n_users):
            n_interactions = min(user_activity[u], n_items)
            items = rng.choice(n_items, size=n_interactions,
                               replace=False, p=item_probs)
            for i in items:
                rows.append(u)
                cols.append(i)
                data.append(1.0)
 
        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        train, val, test = self._temporal_split(matrix)
 
        id_map = {i: i for i in range(n_users)}
        item_map = {i: i for i in range(n_items)}
 
        logger.info(
            f"Generated synthetic dataset: {n_users} users, {n_items} items, "
            f"density={matrix.nnz/(n_users*n_items):.4f}"
        )
        return InteractionDataset(
            train_matrix=train, val_matrix=val, test_matrix=test,
            user_id_map=id_map, item_id_map=item_map,
            reverse_user_map={v: k for k, v in id_map.items()},
            reverse_item_map={v: k for k, v in item_map.items()},
            n_users=n_users, n_items=n_items,
        )
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _process_dataframe(self, df: pd.DataFrame, user_col: str,
                           item_col: str, rating_col: Optional[str],
                           timestamp_col: Optional[str]) -> InteractionDataset:
        df = df[[c for c in [user_col, item_col, rating_col, timestamp_col]
                 if c is not None]].copy()
        df = df.dropna(subset=[user_col, item_col])
 
        # k-core filtering: remove sparse users and items iteratively
        df = self._kcore_filter(df, user_col, item_col)
 
        # Encode string IDs to contiguous integers
        user_ids = sorted(df[user_col].unique())
        item_ids = sorted(df[item_col].unique())
        user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_map = {iid: idx for idx, iid in enumerate(item_ids)}
 
        df["user_idx"] = df[user_col].map(user_map)
        df["item_idx"] = df[item_col].map(item_map)
 
        n_users, n_items = len(user_ids), len(item_ids)
 
        # Binarize: any interaction = 1 (implicit positive)
        values = np.ones(len(df), dtype=np.float32)
 
        if timestamp_col and timestamp_col in df.columns:
            matrix = self._build_temporal_matrix(
                df, n_users, n_items, values, timestamp_col
            )
        else:
            matrix = csr_matrix(
                (values, (df["user_idx"].values, df["item_idx"].values)),
                shape=(n_users, n_items)
            )
 
        train, val, test = self._temporal_split(matrix)
 
        logger.info(
            f"Dataset: {n_users} users, {n_items} items | "
            f"train nnz={train.nnz:,} | density={train.nnz/(n_users*n_items):.5f}"
        )
 
        return InteractionDataset(
            train_matrix=train, val_matrix=val, test_matrix=test,
            user_id_map=user_map, item_id_map=item_map,
            reverse_user_map={v: k for k, v in user_map.items()},
            reverse_item_map={v: k for k, v in item_map.items()},
            n_users=n_users, n_items=n_items,
        )
 
    def _kcore_filter(self, df: pd.DataFrame, user_col: str,
                      item_col: str) -> pd.DataFrame:
        """Iteratively remove users/items below minimum interaction threshold."""
        prev_len = -1
        while prev_len != len(df):
            prev_len = len(df)
            user_counts = df[user_col].value_counts()
            df = df[df[user_col].isin(
                user_counts[user_counts >= self.min_user_interactions].index
            )]
            item_counts = df[item_col].value_counts()
            df = df[df[item_col].isin(
                item_counts[item_counts >= self.min_item_interactions].index
            )]
        logger.info(f"After {self.min_user_interactions}-core filter: {len(df):,} interactions remain")
        return df
 
    def _build_temporal_matrix(self, df, n_users, n_items, values, timestamp_col):
        """Sort by time and deduplicate (keep first interaction)."""
        df = df.sort_values(timestamp_col)
        df = df.drop_duplicates(subset=["user_idx", "item_idx"], keep="first")
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32),
             (df["user_idx"].values, df["item_idx"].values)),
            shape=(n_users, n_items)
        )
 
    def _temporal_split(self, matrix: csr_matrix
                        ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        """
        Per-user leave-last-k-out split.
        Keeps chronological order within each user's history.
        This is the standard evaluation protocol for sequential recommendations.
        """
        train_rows, train_cols = [], []
        val_rows, val_cols = [], []
        test_rows, test_cols = [], []
 
        for u in range(matrix.shape[0]):
            items = matrix[u].indices
            n = len(items)
            if n < 3:
                # Not enough interactions to split — put all in train
                train_rows.extend([u] * n)
                train_cols.extend(items.tolist())
                continue
 
            n_val = max(1, int(n * self.val_ratio))
            n_test = max(1, int(n * self.test_ratio))
            n_train = n - n_val - n_test
 
            train_rows.extend([u] * n_train)
            train_cols.extend(items[:n_train].tolist())
            val_rows.extend([u] * n_val)
            val_cols.extend(items[n_train:n_train + n_val].tolist())
            test_rows.extend([u] * n_test)
            test_cols.extend(items[n_train + n_val:].tolist())
 
        shape = matrix.shape
        ones = lambda r: np.ones(len(r), dtype=np.float32)
 
        return (
            csr_matrix((ones(train_rows), (train_rows, train_cols)), shape=shape),
            csr_matrix((ones(val_rows), (val_rows, val_cols)), shape=shape),
            csr_matrix((ones(test_rows), (test_rows, test_cols)), shape=shape),
        )
 
    def _download_movielens(self, data_dir: str):
        import urllib.request, zipfile
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        os.makedirs(data_dir, exist_ok=True)
        zip_path = Path(data_dir) / "ml-100k.zip"
        logger.info(f"Downloading MovieLens 100K from {url}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        logger.info("Download complete.")