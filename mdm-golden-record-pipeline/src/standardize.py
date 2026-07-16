"""
Phase 2 — STANDARDISATION
=========================
The three source files all describe the same companies, but none of them agree
on how to say it: different column names, different date formats, different
country spellings, different currencies. Before I can decide whether two records
are the same company (Phase 3), I have to make them comparable.

So this stage deliberately does NOT merge anything. It only levels the playing
field:
  - map each source's columns onto one common schema
  - reduce company names to a clean match-token (no punctuation, no legal suffix)
  - canonicalise countries (USA / US / United States -> United States)
  - parse three different date formats into real dates
  - tag revenue with its currency rather than pretending USD and EUR are the same

The last one matters: billing reports EUR and CRM reports USD. Averaging them
would produce a confident, wrong number. I keep the currency alongside the amount
so the conflict stays visible instead of being silently averaged away.

------------------------------------------------------------------------------
PySpark notes to myself:
  * SparkSession -> starts the engine.
  * DataFrame    -> a distributed table (rows + named columns).
  * withColumn() -> add or replace a column.
  * F.<fn>       -> column functions (lower, trim, regexp_replace, when, to_date).
                    They act on a WHOLE column at once — I never loop over rows.
  * lazy eval    -> nothing actually runs until an ACTION (.show(), .count()).
------------------------------------------------------------------------------
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F   # convention: F.<function_name>

# ---------------------------------------------------------------------------
# 0. WHERE TO READ AND WRITE — the one seam between local and cloud
# ---------------------------------------------------------------------------
# This is the only part of the file that knows whether it's running on my laptop
# or on AWS Glue. Locally MDM_BASE is "." so paths resolve to ./data/... exactly
# as they always did. On Glue, Terraform passes --MDM_BASE s3://<bucket> and the
# SAME relative paths below resolve to S3 instead.
#
# awsglue only exists inside the Glue runtime, so the import is guarded: on my
# machine it throws, I swallow it, and the default base path takes over. One
# codebase, two environments, no second "cloud version" to drift out of sync.
import os, sys
try:
    from awsglue.utils import getResolvedOptions
    os.environ["MDM_BASE"] = getResolvedOptions(sys.argv, ["MDM_BASE"])["MDM_BASE"]
except Exception:
    pass  # not on Glue — running locally
MDM_BASE = os.environ.get("MDM_BASE", ".").rstrip("/")
import mdm_io  # handles the local-vs-S3 read/write


# ---------------------------------------------------------------------------
# 1. REFERENCE DATA — the small lookup tables I clean against
# ---------------------------------------------------------------------------

# Legal suffixes I strip so "Acme Corp." and "Acme Corporation" both reduce to
# the token "acme". Each is removed as a WHOLE WORD (\b boundaries below), which
# is what stops "corp" from chewing the middle out of "corporation".
#
# Caveat I'm aware of: "co" is in here, so a company legitimately named with a
# standalone "Co" token loses it. For this dataset that's the right trade — the
# suffix noise is far more common than the false positive — but on real data I'd
# want a curated suffix list per country rather than one global list.
LEGAL_SUFFIXES = [
    "corporation", "corp", "incorporated", "inc", "llc", "ltd", "limited",
    "gmbh", "bv", "b v", "plc", "co", "company", "group", "partners",
]

# Every spelling I've seen in the sources, mapped to one canonical value.
# Without this, "USA" and "United States" would look like different countries and
# the same company would never match across systems.
COUNTRY_CANON = {
    "usa": "United States", "us": "United States",
    "united states": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "united kingdom": "United Kingdom",
    "great britain": "United Kingdom", "gb": "United Kingdom",
    "de": "Germany", "germany": "Germany", "deutschland": "Germany",
    "nl": "Netherlands", "netherlands": "Netherlands", "holland": "Netherlands",
}

# The CRM calls it "industry", marketing calls it "sector", and they use different
# vocabularies for the same thing. If I left them alone, "Tech" and "Technology"
# would split one company across two labels and every downstream count would be wrong.
INDUSTRY_CANON = {
    "tech": "Technology", "technology": "Technology",
    "finance": "Financial Services", "financial services": "Financial Services",
    "industrial": "Manufacturing", "manufacturing": "Manufacturing",
    "healthcare": "Healthcare",
    "food & beverage": "Food & Beverage",
    "aerospace & defense": "Aerospace & Defense",
    "conglomerate": "Conglomerate",
}


# ---------------------------------------------------------------------------
# 2. CLEANING HELPERS — each takes a column, returns a transformed column
# ---------------------------------------------------------------------------

def clean_name(col):
    """Reduce a company name to a clean match-token.

    lowercase -> strip punctuation -> remove legal suffixes -> collapse spaces.
    Built as a chain of column operations, so Spark applies each step to the
    whole column at once. No row loops anywhere.
    """
    c = F.lower(col)

    # Anything that isn't a letter, number or space becomes a space. This kills
    # the punctuation differences ("Acme, Inc." vs "Acme Inc") that would
    # otherwise make two identical companies look different.
    c = F.regexp_replace(c, r"[^a-z0-9 ]", " ")

    # \b = word boundary, so "inc" only matches the standalone word, not the
    # "inc" inside "Incorporated" or a name that happens to contain those letters.
    for suffix in LEGAL_SUFFIXES:
        c = F.regexp_replace(c, r"\b" + suffix + r"\b", " ")

    # Stripping suffixes leaves double spaces behind, so tidy up at the end.
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def canon_country(col):
    """Map a country column onto its canonical name.

    The when/otherwise chain is Spark's if/elif/else, applied across the whole
    column. Anything I haven't seen falls through to .otherwise() and keeps its
    original value (title-cased) rather than being dropped — I'd rather carry an
    unrecognised country forward than silently lose the record.
    """
    norm = F.trim(F.lower(col))
    result = F.when(norm.isNull(), None)
    for raw_value, canonical in COUNTRY_CANON.items():
        result = result.when(norm == raw_value, F.lit(canonical))
    return result.otherwise(F.initcap(F.trim(col)))


def canon_industry(col):
    """Same when/otherwise pattern as canon_country, over the industry taxonomy."""
    norm = F.trim(F.lower(col))
    result = F.when(norm.isNull(), None)
    for raw_value, canonical in INDUSTRY_CANON.items():
        result = result.when(norm == raw_value, F.lit(canonical))
    return result.otherwise(F.initcap(F.trim(col)))


def domain_from_email(col):
    """'alex@acme.com' -> 'acme.com'.

    The domain turns out to be one of my strongest matching signals in Phase 3 —
    two records sharing a domain are almost certainly the same company, which is
    exactly the corroborating evidence fuzzy name matching can't provide on its own.
    """
    return F.when(col.isNull(), None).otherwise(F.element_at(F.split(col, "@"), 2))


# The one common schema all three sources are mapped onto. Defining it once here
# means every source lines up column-for-column, so unionByName below is safe.
COMMON_COLUMNS = [
    "source_system", "source_id", "raw_name", "std_name",
    "industry", "country_std", "city",
    "employees", "revenue_amount", "revenue_currency",
    "email_domain", "vat_id", "last_activity_date",
]


# ---------------------------------------------------------------------------
# 3. PER-SOURCE TRANSFORMS — one function per source, all returning COMMON_COLUMNS
# ---------------------------------------------------------------------------
# One function per source rather than a generic mapper, because real source
# systems don't just differ in column names — they differ in quirks. Billing has
# a VAT id nobody else has; marketing has no city at all; every system uses a
# different date format. Handling each explicitly is honest about that, and it's
# where I'd add source-specific validation in a production version.

def transform_crm(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return (
        df
        .withColumn("source_system", F.lit("crm"))            # F.lit = a constant column
        .withColumn("source_id", F.col("crm_id"))
        .withColumn("raw_name", F.trim(F.col("account_name")))
        .withColumn("std_name", clean_name(F.col("account_name")))
        .withColumn("industry", canon_industry(F.col("industry")))
        .withColumn("country_std", canon_country(F.col("country")))
        .withColumn("city", F.trim(F.col("city")))
        .withColumn("employees", F.col("employees"))
        .withColumn("revenue_amount", F.col("annual_revenue_usd"))
        .withColumn("revenue_currency", F.lit("USD"))         # CRM books revenue in USD
        .withColumn("email_domain", domain_from_email(F.col("primary_contact_email")))
        .withColumn("vat_id", F.lit(None).cast("string"))     # CRM doesn't hold a VAT id
        .withColumn("last_activity_date", F.to_date(F.col("last_modified"), "yyyy-MM-dd"))
        .select(COMMON_COLUMNS)
    )


def transform_marketing(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return (
        df
        .withColumn("source_system", F.lit("marketing"))
        .withColumn("source_id", F.col("lead_id"))
        .withColumn("raw_name", F.trim(F.col("company")))
        .withColumn("std_name", clean_name(F.col("company")))
        .withColumn("industry", canon_industry(F.col("sector")))   # marketing says "sector"
        .withColumn("country_std", canon_country(F.col("country")))
        .withColumn("city", F.lit(None).cast("string"))            # marketing has no city

        # Deliberate NON-mapping: marketing has a num_attendees column, and it is
        # tempting to treat it as employees because both are "a headcount-ish
        # number". It isn't — it's how many people showed up to an event. Mapping
        # it would inject plausible-looking garbage into the golden record, and
        # because it looks reasonable nobody would ever catch it. I leave it null.
        .withColumn("employees", F.lit(None).cast("int"))

        .withColumn("revenue_amount", F.lit(None).cast("double"))
        .withColumn("revenue_currency", F.lit(None).cast("string"))
        .withColumn("email_domain", F.col("email_domain"))
        .withColumn("vat_id", F.lit(None).cast("string"))

        # Marketing dates are European: 02/11/2025 is 2 November, not 11 February.
        # Parsing this with the wrong format silently produces valid-but-wrong
        # dates, which would then corrupt my recency tiebreak in survivorship.
        .withColumn("last_activity_date", F.to_date(F.col("signup_date"), "dd/MM/yyyy"))
        .select(COMMON_COLUMNS)
    )


def transform_billing(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return (
        df
        .withColumn("source_system", F.lit("billing"))
        .withColumn("source_id", F.col("customer_no"))
        .withColumn("raw_name", F.trim(F.col("legal_name")))
        .withColumn("std_name", clean_name(F.col("legal_name")))
        .withColumn("industry", F.lit(None).cast("string"))   # billing doesn't classify industry
        .withColumn("country_std", canon_country(F.col("billing_country")))
        .withColumn("city", F.trim(F.col("billing_city")))
        .withColumn("employees", F.col("headcount"))          # billing says "headcount"
        .withColumn("revenue_amount", F.col("arr_eur"))
        .withColumn("revenue_currency", F.lit("EUR"))         # billing books revenue in EUR
        .withColumn("email_domain", F.lit(None).cast("string"))
        .withColumn("vat_id", F.col("vat_id"))                # only billing has this
        .withColumn("last_activity_date", F.to_date(F.col("invoice_date"), "yyyy-MM-dd"))
        .select(COMMON_COLUMNS)
    )


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main():
    # Locally I need to tell Spark to run on this machine ("local[*]" = all cores).
    # On Glue I must NOT set a master — Glue provides its own managed Spark cluster
    # and setting one would fight it. So the master is only added when I'm local.
    _builder = SparkSession.builder.appName("mdm-phase2-standardise")
    if not MDM_BASE.startswith("s3://"):
        _builder = _builder.master("local[*]")
    spark = _builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # Spark's INFO logs drown out my own output

    raw = f"{MDM_BASE}/data/raw"

    crm = transform_crm(spark, f"{raw}/source_crm.csv")
    mkt = transform_marketing(spark, f"{raw}/source_marketing.csv")
    bil = transform_billing(spark, f"{raw}/source_billing.csv")

    # unionByName lines the columns up by NAME, not by position. union() would
    # stack them positionally and happily put a city into the country column if
    # I ever reordered COMMON_COLUMNS. unionByName fails loudly instead.
    unified = crm.unionByName(mkt).unionByName(bil)

    # --- Actions: this is where Spark actually computes anything --------------
    print("\n=== STANDARDISED & UNIFIED (raw_name -> std_name) ===")
    unified.select(
        "source_system", "source_id", "raw_name", "std_name",
        "country_std", "revenue_currency", "last_activity_date",
    ).show(40, truncate=False)

    print(f"Total records after standardisation: {unified.count()}")

    # toPandas() collects the whole DataFrame onto the driver. That's fine here —
    # 28 rows — and it gives me a single clean CSV instead of the folder of
    # part-files Spark would otherwise write. At real scale this is exactly the
    # wrong move (it would OOM the driver); I'd write partitioned Parquet directly
    # to S3 and skip pandas entirely.
    mdm_io.write_csv(unified.toPandas(), "data/processed/standardized.csv")
    print("Written: data/processed/standardized.csv")

    spark.stop()


if __name__ == "__main__":
    main()