"""
generate_data.py
----------------
Fills the ecommerce_analytics database with realistic fake data.

What gets generated:
  - 500   users       (signup dates spread over 2 years)
  - 8     categories  (Amazon-style hierarchy)
  - 200   products    (prices follow realistic distributions)
  - 8,000 orders      (power-law: few users buy a lot)
  - ~24,000 order_items
  - ~40,000 events    (view → add_to_cart → purchase funnel)
  - ~3,000 reviews    (only from users who actually bought)

HOW TO RUN:
  python generate_data.py

BEFORE RUNNING — update the DB_CONFIG below with your credentials.
"""

import random
import psycopg2
from psycopg2.extras import execute_values
from faker import Faker
from datetime import datetime, timedelta, date
import math

# ──────────────────────────────────────────────
# 1. CONFIG — update these to match your setup
# ──────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "ecommerce_analytics",   # <-- change this
    "user":     "postgres",
    "password": "sql_project",        # <-- change this
}

# ──────────────────────────────────────────────
# 2. CONSTANTS
# ──────────────────────────────────────────────
NUM_USERS      = 500
NUM_PRODUCTS   = 200
NUM_ORDERS     = 8_000
START_DATE     = date(2023, 1, 1)
END_DATE       = date(2024, 12, 31)

fake = Faker()
random.seed(42)
Faker.seed(42)

# ──────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────

def random_date(start: date, end: date) -> date:
    """Pick a random date between start and end."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def random_datetime(start: date, end: date) -> datetime:
    """Pick a random datetime between start and end."""
    base = random_date(start, end)
    hour   = random.randint(6, 23)   # people shop between 6am and midnight
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(base.year, base.month, base.day, hour, minute, second)

def seasonal_weight(d: date) -> float:
    """
    More orders in Nov-Dec (holiday season) and Jul (Prime Day).
    Returns a weight between 1.0 and 3.0.
    """
    if d.month in (11, 12):
        return 3.0   # holiday spike
    if d.month == 7:
        return 2.0   # Prime Day
    if d.month in (1, 2):
        return 0.7   # post-holiday slump
    return 1.0

def power_law_user_weights(n: int) -> list:
    """
    Power-law distribution: user 1 buys ~100x more than user 500.
    This mirrors real e-commerce (a small % of users drive most revenue).
    """
    return [1.0 / math.log(i + 2) for i in range(n)]

# ──────────────────────────────────────────────
# 4. GENERATORS
# ──────────────────────────────────────────────

def generate_categories():
    """8 top-level Amazon-style categories."""
    return [
        ("Electronics",       None),
        ("Books",             None),
        ("Clothing",          None),
        ("Home & Kitchen",    None),
        ("Sports & Outdoors", None),
        ("Toys & Games",      None),
        ("Beauty",            None),
        ("Grocery",           None),
    ]

def generate_users(n: int) -> list:
    """
    Users with signup dates spread over 2 years.
    Segments assigned based on how long ago they signed up.
    """
    countries = ["US","GB","DE","FR","CA","AU","ES","IT","NL","BR"]
    country_weights = [40,15,10,8,7,5,4,3,3,5]  # US-heavy, realistic
    rows = []
    for _ in range(n):
        signup = random_date(START_DATE, END_DATE)
        days_since = (END_DATE - signup).days
        # Segment based on account age
        if days_since < 30:
            segment = "new"
        elif days_since < 180:
            segment = "active"
        elif days_since < 365:
            segment = "vip"
        else:
            segment = "churned"
        rows.append((
            fake.unique.email(),
            random.choices(countries, weights=country_weights)[0],
            signup,
            segment,
            datetime.combine(signup, datetime.min.time()),
        ))
    return rows

def generate_products(n: int, category_ids: list) -> list:
    """
    Products with realistic price distributions per category.
    Electronics are expensive, books are cheap, etc.
    """
    price_ranges = [
        (50,  2000),   # Electronics
        (5,   50),     # Books
        (10,  200),    # Clothing
        (15,  300),    # Home & Kitchen
        (20,  500),    # Sports
        (5,   150),    # Toys
        (8,   100),    # Beauty
        (2,   50),     # Grocery
    ]
    rows = []
    for _ in range(n):
        cat_id = random.choice(category_ids)
        lo, hi = random.choice(price_ranges)
        price = round(random.uniform(lo, hi), 2)
        avg_rating = round(random.gauss(4.1, 0.6), 2)  # skewed high like Amazon
        avg_rating = max(1.0, min(5.0, avg_rating))
        review_count = int(random.expovariate(1/50))    # most have few, some have thousands
        review_count = max(0, min(review_count, 5000))
        rows.append((
            cat_id,
            fake.catch_phrase(),
            price,
            round(avg_rating, 2),
            review_count,
            True,
            fake.date_time_between(start_date="-2y", end_date="-1m"),
        ))
    return rows

def generate_orders_and_items(
    num_orders: int,
    user_ids: list,
    product_ids: list,
    product_prices: dict,
) -> tuple:
    """
    Orders follow power-law user distribution + seasonal weighting.
    Each order has 1-6 items.
    Returns (orders_rows, items_rows).
    """
    user_weights = power_law_user_weights(len(user_ids))

    # Pre-build a pool of dates weighted by season
    all_dates = [START_DATE + timedelta(days=i)
                 for i in range((END_DATE - START_DATE).days)]
    date_weights = [seasonal_weight(d) for d in all_dates]

    orders_rows = []
    items_rows  = []
    statuses = ["delivered","delivered","delivered","shipped","confirmed","cancelled","refunded"]
    # delivered is most common — weighted list

    for _ in range(num_orders):
        user_id    = random.choices(user_ids, weights=user_weights)[0]
        order_date = random.choices(all_dates, weights=date_weights)[0]
        ordered_at = datetime(
            order_date.year, order_date.month, order_date.day,
            random.randint(6, 23), random.randint(0, 59), random.randint(0, 59)
        )
        status     = random.choice(statuses)
        n_items    = random.choices([1,2,3,4,5,6], weights=[40,25,15,10,6,4])[0]

        chosen_products = random.sample(product_ids, min(n_items, len(product_ids)))
        total = 0.0
        order_item_rows = []
        for pid in chosen_products:
            qty        = random.choices([1,2,3,4], weights=[70,20,7,3])[0]
            unit_price = product_prices[pid]
            total     += unit_price * qty
            order_item_rows.append((pid, qty, unit_price))

        orders_rows.append((user_id, ordered_at, status, round(total, 2), order_date))
        items_rows.append(order_item_rows)

    return orders_rows, items_rows

def generate_events(
    user_ids: list,
    product_ids: list,
    orders_data: list,   # list of (order_id, user_id, ordered_at, product_ids)
) -> list:
    """
    Realistic clickstream: users view products, some add to cart, some purchase.
    Also generates organic views (no purchase) for users who browsed but didn't buy.
    """
    event_types = ["view","add_to_cart","remove_from_cart","checkout_start","purchase","search"]
    rows = []

    # 1. Purchase funnel events — tied to real orders
    for order_id, user_id, ordered_at, item_product_ids in orders_data:
        session_id = fake.uuid4()
        # View event always happens before purchase
        for pid in item_product_ids:
            view_time = ordered_at - timedelta(minutes=random.randint(5, 120))
            rows.append((user_id, pid, session_id, "view", view_time, None))
            # ~60% add to cart
            if random.random() < 0.6:
                cart_time = view_time + timedelta(minutes=random.randint(1, 10))
                rows.append((user_id, pid, session_id, "add_to_cart", cart_time, None))
            # Purchase event
            rows.append((user_id, pid, session_id, "purchase", ordered_at, None))

    # 2. Organic browse events (users who viewed but didn't buy)
    for _ in range(15_000):
        user_id    = random.choice(user_ids)
        pid        = random.choice(product_ids)
        session_id = fake.uuid4()
        event_time = random_datetime(START_DATE, END_DATE)
        rows.append((user_id, pid, session_id, "view", event_time, None))
        if random.random() < 0.3:
            rows.append((user_id, pid, session_id, "add_to_cart",
                         event_time + timedelta(minutes=random.randint(1,5)), None))
        if random.random() < 0.1:
            rows.append((user_id, pid, session_id, "remove_from_cart",
                         event_time + timedelta(minutes=random.randint(5,15)), None))

    return rows

def generate_reviews(
    purchases: list,   # list of (user_id, product_id)
    pct: float = 0.35, # ~35% of purchases get a review
) -> list:
    """
    Only users who actually bought a product can review it.
    Ratings are slightly skewed high (like real Amazon).
    """
    rows = []
    seen = set()
    for user_id, product_id in purchases:
        if (user_id, product_id) in seen:
            continue
        seen.add((user_id, product_id))
        if random.random() < pct:
            rating = random.choices([1,2,3,4,5], weights=[3,4,10,25,58])[0]
            reviewed_at = random_datetime(START_DATE, END_DATE)
            rows.append((user_id, product_id, rating, reviewed_at, True))
    return rows

# ──────────────────────────────────────────────
# 5. MAIN — connect and insert everything
# ──────────────────────────────────────────────

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # Clean slate before every run
        conn.autocommit = True
        cur.execute("""
            TRUNCATE TABLE reviews, events, order_items, orders, 
            products, categories, users RESTART IDENTITY CASCADE
        """)
        conn.autocommit = False

        # ── Categories ──────────────────────────────
        print("Inserting categories...")
        cats = generate_categories()
        execute_values(cur,
            "INSERT INTO categories (name, parent_id) VALUES %s",
            cats
        )
        cur.execute("SELECT category_id FROM categories")
        category_ids = [r[0] for r in cur.fetchall()]

        # ── Users ───────────────────────────────────
        print("Inserting users...")
        users = generate_users(NUM_USERS)
        execute_values(cur,
            "INSERT INTO users (email, country, signup_date, segment, created_at) VALUES %s ON CONFLICT (email) DO NOTHING",
            users
        )
        cur.execute("SELECT user_id FROM users")
        user_ids = [r[0] for r in cur.fetchall()]

        # ── Products ────────────────────────────────
        print("Inserting products...")
        products = generate_products(NUM_PRODUCTS, category_ids)
        execute_values(cur,
            """INSERT INTO products
               (category_id, title, price, avg_rating, review_count, is_active, created_at)
               VALUES %s""",
            products
        )
        cur.execute("SELECT product_id, price FROM products")
        rows = cur.fetchall()
        product_ids    = [r[0] for r in rows]
        product_prices = {r[0]: float(r[1]) for r in rows}

        # ── Orders ──────────────────────────────────
        print("Inserting orders and order_items (this may take a moment)...")
        orders_rows, items_rows = generate_orders_and_items(
            NUM_ORDERS, user_ids, product_ids, product_prices
        )
        execute_values(cur,
            """INSERT INTO orders (user_id, ordered_at, status, total_amount, order_date)
               VALUES %s""",
            orders_rows
        )

        # Fetch the order_ids that were just created (in insertion order)
        cur.execute("SELECT order_id FROM orders ORDER BY order_id")
        order_ids = [r[0] for r in cur.fetchall()]

        # ── Order items ─────────────────────────────
        all_items = []
        purchase_pairs = []   # (user_id, product_id) — used for reviews + events
        for i, (order_id, order_row) in enumerate(zip(order_ids, orders_rows)):
            user_id = order_row[0]
            for pid, qty, unit_price in items_rows[i]:
                all_items.append((order_id, pid, qty, unit_price))
                purchase_pairs.append((user_id, pid))

        execute_values(cur,
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES %s",
            all_items
        )

        # ── Events ──────────────────────────────────
        print("Inserting events...")
        # Build lightweight orders_data for event generation
        orders_data = []
        for i, (order_id, order_row) in enumerate(zip(order_ids, orders_rows)):
            user_id    = order_row[0]
            ordered_at = order_row[1]
            item_pids  = [t[1] for t in items_rows[i]]
            orders_data.append((order_id, user_id, ordered_at, item_pids))

        events = generate_events(user_ids, product_ids, orders_data)
        execute_values(cur,
            """INSERT INTO events (user_id, product_id, session_id, event_type, event_time, metadata)
               VALUES %s""",
            events,
            page_size=500
        )

        # ── Reviews ─────────────────────────────────
        print("Inserting reviews...")
        reviews = generate_reviews(purchase_pairs)
        execute_values(cur,
            """INSERT INTO reviews (user_id, product_id, rating, reviewed_at, verified_purchase)
               VALUES %s ON CONFLICT (user_id, product_id) DO NOTHING""",
            reviews
        )

        conn.commit()
        print("\nDone! Row counts:")

        for table in ["users","categories","products","orders","order_items","events","reviews"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table:<15} {count:>7,} rows")

    except Exception as e:
        conn.rollback()
        print(f"\nError: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()