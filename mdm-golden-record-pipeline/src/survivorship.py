"""
Phase 4 — SURVIVORSHIP
======================
Phase 3 told me WHICH records belong to the same company. It didn't resolve the
fact that those records still DISAGREE with each other: three systems, three
different employee counts, two different revenue figures, three spellings of the
name. Survivorship is where I decide, field by field, which value wins — and
record why.

The output is one clean row per company, plus an audit trail saying which source
system each surviving value actually came from.

WHY THESE RULES
---------------
The naive approach is "most recent record wins" or "the CRM is our system of
record, use that". Both are wrong, because trustworthiness isn't a property of a
SYSTEM — it's a property of a system AND a field. Billing knows what a company
pays; it has no opinion worth hearing about their industry. The CRM knows the
sales relationship; its revenue figure is a salesperson's estimate.

So authority is assigned PER FIELD:

  * BILLING wins legal and financial fields (legal name, revenue, VAT id,
    billing address). This data is contractual and audited — someone gets sued
    if it's wrong — which makes it the most reliable source for money and identity.

  * CRM wins sales-owned fields (industry classification, contact domain). It's
    the system where humans actively maintain the relationship.

  * MARKETING is self-entered by leads at event signup, so it's the least
    reliable. It never wins a contested field. It's used ONLY to fill gaps where
    the better sources have nothing — which is still valuable, because a slightly
    uncertain value beats a null.

  * RECENCY breaks ties within an equally-trusted source, so a stale record never
    beats a fresh one from the same system.

  * NULLS are skipped rather than winning, so the golden record ends up as
    complete as the underlying data allows.

The result maximises both trust (each system is used for what it's actually good
at) and completeness — which is what any firm needs from one reliable view of a
client company.

HOW IT'S IMPLEMENTED: window functions
--------------------------------------
A window lets me rank and compare rows WITHIN each match group without collapsing
the group. I partition by match_group_id, order the rows by (source priority,
then recency), and take the first non-null value — that IS the surviving value.
The same ordering simultaneously tells me which source it came from, which is
what makes the audit trail essentially free.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# 0. WHERE TO READ AND WRITE — the one seam between local and cloud
# ---------------------------------------------------------------------------
# awsglue only exists in the Glue runtime, so this import is guarded. Locally it
# throws, MDM_BASE stays ".", and the same relative paths resolve to my disk.
import os, sys
try:
    from awsglue.utils import getResolvedOptions
    os.environ["MDM_BASE"] = getResolvedOptions(sys.argv, ["MDM_BASE"])["MDM_BASE"]
except Exception:
    pass  # not on Glue — running locally
MDM_BASE = os.environ.get("MDM_BASE", ".").rstrip("/")
import mdm_io


# Lower rank = more trusted. Two different orderings because trust is per-field:
# the system that knows the money is not the system that knows the industry.
FINANCE_PRIORITY = {"billing": 1, "crm": 2, "marketing": 3}   # legal / financial
SALES_PRIORITY   = {"crm": 1, "marketing": 2, "billing": 3}   # sales-owned


def priority_col(mapping):
    """Turn a {source: rank} dict into a Spark column of ranks.

    This becomes the primary sort key of the window: rank 1 sorts first, so the
    most trusted source's value is the first one the window sees.
    """
    col = F.when(F.col("source_system") == "billing", F.lit(mapping["billing"]))
    col = col.when(F.col("source_system") == "crm", F.lit(mapping["crm"]))
    col = col.when(F.col("source_system") == "marketing", F.lit(mapping["marketing"]))
    # An unknown source sorts last rather than crashing — a new system appearing
    # upstream shouldn't take down the pipeline, it should just be distrusted.
    return col.otherwise(F.lit(99))


def _window(priority):
    """The ordering that encodes the entire survivorship policy.

    Within one company: most-trusted source first, then most-recent first. Every
    survive() and winning_source() call below reads from this same ordering, which
    is why the surviving value and its provenance can never disagree.
    """
    return (Window.partitionBy("match_group_id")
                  .orderBy(priority_col(priority).asc(),
                           F.col("last_activity_date").desc()))


def survive(field, priority):
    """The winning value for `field` within each match group.

    first(ignorenulls=True) walks the ordered rows and takes the first value that
    actually exists. That single call does all three jobs at once: source
    authority (from the ordering), recency (from the ordering), and null-skipping
    (from ignorenulls) — which is why marketing can fill a gap without ever being
    able to overrule billing.
    """
    return F.first(F.col(field), ignorenulls=True).over(_window(priority))


def winning_source(field, priority):
    """Which source_system supplied the winning value for `field`.

    The trick: I null out source_system on any row where the field itself is null,
    then take the first surviving one. So this reports the source of the value that
    ACTUALLY won — not merely the highest-priority source present in the group.
    Without that, a group where billing had a null revenue would be credited to
    billing anyway, and the audit trail would be a lie.
    """
    src = F.when(F.col(field).isNotNull(), F.col("source_system"))
    return F.first(src, ignorenulls=True).over(_window(priority))


def main():
    _builder = SparkSession.builder.appName("mdm-phase4-survivorship")
    if not MDM_BASE.startswith("s3://"):
        _builder = _builder.master("local[*]")
    spark = _builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(f"{MDM_BASE}/data/processed/matched.csv",
                        header=True, inferSchema=True)

    # ------------------------------------------------------------------------
    # Money and its unit must survive TOGETHER.
    #
    # Surviving revenue_amount and revenue_currency independently is a subtle trap:
    # if one record had an amount with no currency, and another had a currency with
    # no amount, the two first-non-null calls could pair billing's EUR *figure*
    # with a USD *label* from somewhere else. The number would look perfectly
    # plausible and be completely wrong.
    #
    # So I pack them into one struct, survive that as a single unit, and unpack it
    # afterwards. The amount and its currency now come from the same row, always.
    # ------------------------------------------------------------------------
    df = df.withColumn(
        "revenue_pair",
        F.when(F.col("revenue_amount").isNotNull(),
               F.struct(F.col("revenue_amount").alias("amt"),
                        F.col("revenue_currency").alias("cur")))
    )

    enriched = (df
        # --- legal / financial fields: billing has the authority ---
        .withColumn("g_name",      survive("raw_name",    FINANCE_PRIORITY))
        .withColumn("g_name_src",  winning_source("raw_name", FINANCE_PRIORITY))
        .withColumn("g_country",   survive("country_std", FINANCE_PRIORITY))
        .withColumn("g_city",      survive("city",        FINANCE_PRIORITY))
        .withColumn("g_employees", survive("employees",   FINANCE_PRIORITY))
        .withColumn("g_emp_src",   winning_source("employees", FINANCE_PRIORITY))

        # revenue survives as an inseparable (amount, currency) pair
        .withColumn("g_revenue_pair", survive("revenue_pair", FINANCE_PRIORITY))
        .withColumn("g_rev_src",      winning_source("revenue_pair", FINANCE_PRIORITY))

        .withColumn("g_vat",       survive("vat_id",      FINANCE_PRIORITY))

        # --- sales-owned fields: the CRM has the authority ---
        .withColumn("g_industry",  survive("industry",     SALES_PRIORITY))
        .withColumn("g_domain",    survive("email_domain", SALES_PRIORITY))
    )

    # Unpack the surviving struct back into two flat columns for the output.
    enriched = (enriched
        .withColumn("g_revenue", F.col("g_revenue_pair.amt"))
        .withColumn("g_rev_cur", F.col("g_revenue_pair.cur")))

    # ------------------------------------------------------------------------
    # Collapse to ONE row per company.
    #
    # The window already wrote the identical winning value onto every row in the
    # group, so F.first() here is just picking one of N identical values — it isn't
    # making any decision. All the deciding happened in the window. This step is
    # purely "go from N rows to 1".
    #
    # sourced_from and source_records are the exceptions: they're group-level facts
    # (which systems contributed, how many records merged) rather than survived
    # values, so they're genuinely aggregated here.
    # ------------------------------------------------------------------------
    golden = (enriched.groupBy("match_group_id")
        .agg(
            F.first("g_name").alias("company_name"),
            F.first("g_name_src").alias("name_from"),
            F.first("g_industry").alias("industry"),
            F.first("g_country").alias("country"),
            F.first("g_city").alias("city"),
            F.first("g_employees").alias("employees"),
            F.first("g_emp_src").alias("employees_from"),
            F.first("g_revenue").alias("revenue"),
            F.first("g_rev_cur").alias("revenue_currency"),
            F.first("g_rev_src").alias("revenue_from"),
            F.first("g_vat").alias("vat_id"),
            F.first("g_domain").alias("email_domain"),
            F.max("last_activity_date").alias("last_activity"),
            F.collect_set("source_system").alias("sourced_from"),
            F.count("*").alias("source_records"),
        )
        .orderBy("match_group_id"))

    print("\n=== GOLDEN RECORDS (one row per company) ===")
    golden.select("match_group_id", "company_name", "country",
                  "employees", "revenue", "revenue_currency",
                  "vat_id", "source_records").show(20, truncate=False)

    # The audit trail is the point of the whole exercise. A golden record nobody
    # can interrogate is just another opinion — this is what lets a data steward
    # ask "where did this number come from?" and get a real answer.
    print("\n=== AUDIT TRAIL (which source won the key fields) ===")
    golden.select("match_group_id", "company_name",
                  "name_from", "employees", "employees_from",
                  "revenue", "revenue_currency", "revenue_from").show(20, truncate=False)

    # Flatten the array to a string so it survives the trip through CSV.
    out = golden.withColumn("sourced_from", F.concat_ws(";", F.col("sourced_from")))
    mdm_io.write_csv(out.toPandas(), "data/processed/golden.csv")
    print("\nWritten: data/processed/golden.csv")

    spark.stop()


if __name__ == "__main__":
    main()