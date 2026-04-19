-- ============================================================
-- schema.sql  —  E-commerce Analytics Platform
-- PostgreSQL 15+
-- ============================================================

-- ── Dimension: users ──────────────────────────────────────────
CREATE TABLE users (
    user_id      BIGSERIAL PRIMARY KEY,
    email        VARCHAR(255) NOT NULL UNIQUE,
    country      CHAR(2)      NOT NULL,                 -- ISO 3166-1 alpha-2
    signup_date  DATE         NOT NULL DEFAULT CURRENT_DATE,
    segment      VARCHAR(50)  NOT NULL DEFAULT 'new'    -- new | active | vip | churned
                 CHECK (segment IN ('new','active','vip','churned')),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ── Dimension: categories (self-referencing hierarchy) ────────
CREATE TABLE categories (
    category_id BIGSERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    parent_id   BIGINT REFERENCES categories(category_id)  -- NULL = top-level
);

-- ── Dimension: products ───────────────────────────────────────
CREATE TABLE products (
    product_id   BIGSERIAL PRIMARY KEY,
    category_id  BIGINT       NOT NULL REFERENCES categories(category_id),
    title        VARCHAR(500) NOT NULL,
    price        NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    avg_rating   NUMERIC(3,2)  CHECK (avg_rating BETWEEN 1 AND 5),
    review_count INT           NOT NULL DEFAULT 0 CHECK (review_count >= 0),
    is_active    BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE orders (
    order_id      BIGSERIAL PRIMARY KEY,
    user_id       BIGINT        NOT NULL REFERENCES users(user_id),
    ordered_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    status        VARCHAR(20)   NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled','refunded')),
    total_amount  NUMERIC(12,2) NOT NULL CHECK (total_amount >= 0),
    order_date    DATE          NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE order_items (
    item_id      BIGSERIAL PRIMARY KEY,
    order_id     BIGINT        NOT NULL REFERENCES orders(order_id),
    product_id   BIGINT        NOT NULL REFERENCES products(product_id),
    quantity     INT           NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit_price   NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0)
);

-- ── Fact: events (clickstream — feeds funnel + sessionisation) ─
CREATE TABLE events (
    event_id    BIGSERIAL    PRIMARY KEY,
    user_id     BIGINT       REFERENCES users(user_id),  -- NULL = anonymous
    product_id  BIGINT       REFERENCES products(product_id),
    session_id  VARCHAR(64)  NOT NULL,
    event_type  VARCHAR(30)  NOT NULL
                CHECK (event_type IN ('view','add_to_cart','remove_from_cart',
                                      'checkout_start','purchase','search')),
    event_time  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata    JSONB         -- flexible: search_query, referrer, device_type, etc.
);

-- ── Fact: reviews ─────────────────────────────────────────────
CREATE TABLE reviews (
    review_id          BIGSERIAL   PRIMARY KEY,
    user_id            BIGINT      NOT NULL REFERENCES users(user_id),
    product_id         BIGINT      NOT NULL REFERENCES products(product_id),
    rating             SMALLINT    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    reviewed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_purchase  BOOLEAN     NOT NULL DEFAULT FALSE,
    UNIQUE (user_id, product_id)   -- one review per user per product
);

-- ============================================================
-- INDEXES  —  tuned for the analytics queries in this project
-- ============================================================

-- Cohort retention: scan by signup month, then join to orders by user
CREATE INDEX idx_users_signup_date     ON users(signup_date);
CREATE INDEX idx_users_country_segment ON users(country, segment);

-- Time-series order scans (revenue dashboards, rolling windows)
CREATE INDEX idx_orders_ordered_at     ON orders(ordered_at DESC);
CREATE INDEX idx_orders_user_ordered   ON orders(user_id, ordered_at DESC);
CREATE INDEX idx_orders_date_status    ON orders(order_date, status);

-- Market basket: find all orders containing a given product
CREATE INDEX idx_items_product_id      ON order_items(product_id);
CREATE INDEX idx_items_order_id        ON order_items(order_id);

-- Funnel + sessionisation: event-stream scans by time and session
CREATE INDEX idx_events_user_time      ON events(user_id, event_time DESC);
CREATE INDEX idx_events_session        ON events(session_id, event_time);
CREATE INDEX idx_events_type_time      ON events(event_type, event_time DESC);

-- Product catalogue lookups
CREATE INDEX idx_products_category     ON products(category_id) WHERE is_active;

-- Reviews: fast per-product aggregation
CREATE INDEX idx_reviews_product       ON reviews(product_id, rating);
CREATE INDEX idx_reviews_verified      ON reviews(product_id) WHERE verified_purchase;

-- ============================================================
-- VIEWS  —  pre-baked for the analytics layer
-- ============================================================

-- Revenue by day — feeds rolling-window and MoM queries
CREATE VIEW daily_revenue AS
SELECT
    order_date,
    COUNT(DISTINCT order_id)  AS orders,
    COUNT(DISTINCT user_id)   AS unique_buyers,
    SUM(total_amount)         AS gmv,
    AVG(total_amount)         AS aov
FROM orders
WHERE status NOT IN ('cancelled','refunded')
GROUP BY order_date;

-- User activity spine — one row per (user, month), feeds cohort analysis
CREATE VIEW user_monthly_activity AS
SELECT
    user_id,
    DATE_TRUNC('month', ordered_at)::date AS activity_month,
    COUNT(DISTINCT order_id)              AS orders,
    SUM(total_amount)                     AS spend
FROM orders
WHERE status NOT IN ('cancelled','refunded')
GROUP BY user_id, DATE_TRUNC('month', ordered_at);

