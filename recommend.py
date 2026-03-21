"""
recommend.py
------------
Loads the trained ALS model and prints actual film recommendations by name.
Run from the project root:
    python recommend.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.data_loader import ImplicitDataLoader
from src.model import ALSModel

# ── 1. Load MovieLens data ──────────────────────────────────────────
print("Loading MovieLens 100K...")
loader = ImplicitDataLoader(min_user_interactions=5, min_item_interactions=5)
dataset = loader.load_movielens_100k(data_dir="data/")

# ── 2. Load film titles ─────────────────────────────────────────────
titles_path = os.path.join("data", "ml-100k", "u.item")
item_titles = {}
with open(titles_path, encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        item_id = int(parts[0])
        title   = parts[1]
        item_titles[item_id] = title

# Map internal index → film title using the dataset's item map
def get_title(internal_idx):
    original_id = dataset.reverse_item_map[internal_idx]
    return item_titles.get(original_id, f"Item {original_id}")

# ── 3. Train model ──────────────────────────────────────────────────
print("Training ALS model...")
model = ALSModel(n_factors=64, regularization=0.01, alpha=40.0,
                 n_iterations=20, random_state=42)
model.fit(dataset.train_matrix)
print("Done.\n")

# ── 4. Show recommendations for a few users ─────────────────────────
def show_recommendations(user_internal_idx, n=10):
    original_user_id = dataset.reverse_user_map[user_internal_idx]

    # Films this user has already seen (training set)
    seen_indices = dataset.train_matrix[user_internal_idx].indices
    seen_titles  = [get_title(i) for i in seen_indices[:8]]

    # Top-N recommendations (unseen films only)
    recs = model.recommend(user_internal_idx, dataset.train_matrix, n=n,
                           filter_already_seen=True)

    print("=" * 60)
    print(f"User {original_user_id}")
    print("-" * 60)
    print(f"Films they've watched (sample of {len(seen_indices)} total):")
    for title in seen_titles:
        print(f"   • {title}")
    print(f"\nTop {n} recommendations:")
    for rank, (item_idx, score) in enumerate(recs, 1):
        title = get_title(item_idx)
        print(f"   {rank:2d}. {title}  (score: {score:.3f})")
    print()

# Show recommendations for users 0, 1, and 2
for user_idx in [17]:
    show_recommendations(user_idx, n=10)
