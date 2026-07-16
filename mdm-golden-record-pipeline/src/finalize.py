"""
Phase 5 — FINALISE THE GOLDEN RECORDS
=====================================
Phase 4 gave me 12 correct golden records. This stage turns them into a finished
master that a downstream system could actually depend on:

  1. STABLE golden_id — a permanent, deterministic id per company.
  2. HIERARCHY — link subsidiaries to their parents WITHOUT merging them.
  3. DATA-QUALITY REPORT — metrics about the run, so a steward can decide whether
     to trust this output before it flows anywhere.

WHY THE ID HAS TO BE DETERMINISTIC
----------------------------------
This is the requirement people miss. It would be easy to hand out sequential ids
(GLD-001, GLD-002...) as records come out of the pipeline. That works exactly once.
Re-run the pipeline with the rows in a different order — or add one new company —
and every id shifts. Every downstream system holding a reference to GLD-004 is now
pointing at a different company, and nothing errors. It just quietly becomes wrong.

So the id is derived from the CONTENT of the record (normalised name + country),
not from its position in a list. The same company always hashes to the same id, on
every run, forever. That's what makes the master id safe to reference from outside.

WHY HIERARCHY IS *LINKED*, NOT MERGED
-------------------------------------
Acme Cloud Services is a real subsidiary of Acme Corporation. They're related but
they are NOT the same company — they have separate revenue, separate headcount,
separate contracts. Merging them would double-count Acme's revenue. Ignoring the
relationship would lose the fact that they're connected.

The signal I use is the same one from Phase 3, read the other way round: a name-
subset relationship WITHOUT a shared domain or city meant "don't merge". Here that
same absence becomes positive evidence of a parent/child structure. The thing that
prevented the merge is the thing that reveals the hierarchy.
"""

import os
import sys
import json
import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# 0. WHERE TO READ AND WRITE — the one seam between local and cloud
# ---------------------------------------------------------------------------
# awsglue only exists in the Glue runtime, so the import is guarded. Locally it
# throws, MDM_BASE stays ".", and the same relative paths point at my disk.
try:
    from awsglue.utils import getResolvedOptions
    os.environ["MDM_BASE"] = getResolvedOptions(sys.argv, ["MDM_BASE"])["MDM_BASE"]
except Exception:
    pass  # not on Glue — running locally
MDM_BASE = os.environ.get("MDM_BASE", ".").rstrip("/")
import mdm_io


# Known duplication: this list also lives in standardize.py. It has to produce the
# SAME normalisation, because the golden_id is built from these tokens and Phase 3
# matched on them — if the two lists ever drifted apart, ids would silently stop
# corresponding to the records they came from, and a golden id that changes is worse
# than useless.

# I hit exactly this failure mode earlier in this project: I kept deployment copies
# of the scripts in a separate folder, edited the originals, and for a while the
# deployed pipeline was running stale code with nothing complaining. Any time a human
# is responsible for keeping two copies in sync, they eventually drift.

# The right fix is one shared normalisation module imported by both. I've left it
# duplicated and flagged rather than pretend it isn't a smell.

LEGAL_SUFFIXES = ["corporation", "corp", "incorporated", "inc", "llc", "ltd",
                  "limited", "gmbh", "bv", "b v", "plc", "co", "company",
                  "group", "partners"]


def norm_name_col(col):
    """Same normalisation as Phase 2's clean_name: lowercase, strip punctuation,
    drop legal suffixes. Used for both the id and the hierarchy token comparison."""
    c = F.lower(col)
    c = F.regexp_replace(c, r"[^a-z0-9 ]", " ")
    for s in LEGAL_SUFFIXES:
        c = F.regexp_replace(c, r"\b" + s + r"\b", " ")
    return F.trim(F.regexp_replace(c, r"\s+", " "))


def main():
    _builder = SparkSession.builder.appName("mdm-phase5-finalise")
    if not MDM_BASE.startswith("s3://"):
        _builder = _builder.master("local[*]")
    spark = _builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    g = spark.read.csv(f"{MDM_BASE}/data/processed/golden.csv",
                       header=True, inferSchema=True)

    # ==========================================================================
    # STEP 1 — THE STABLE golden_id
    #
    # Shape: GLD-<FIRST WORD>-<4 hex chars>, e.g. GLD-ACME-3f2a
    #
    # The readable prefix is for humans — an id you can eyeball and recognise is
    # worth a lot when you're debugging a data issue at 2am. The hash is what makes
    # it unique and, crucially, DETERMINISTIC: md5 of (normalised name | country)
    # always produces the same digits for the same company.
    #
    # Known limitation: 4 hex characters is only 65,536 possible values, so by the
    # birthday paradox collisions become likely in the low thousands of companies.
    # It's fine for 12. For a real master I'd take more of the hash (or use a UUID
    # and keep the readable part purely cosmetic) — the readability is a nice-to-
    # have, the uniqueness isn't.
    # Second known limitation: the readable prefix is only the FIRST token, so
    # Acme Corp. and Acme Cloud Services both render as GLD-ACME-xxxx, and the two
    # unrelated Stark companies both render as GLD-STARK-xxxx. The ids are still
    # unique and stable — the hash does the real work — but the prefix fails at the
    # one job it was added for, which was being eyeball-recognisable. If I wanted it
    # to actually distinguish, I'd use the first two or three tokens.
    # ==========================================================================
    g = g.withColumn("norm_name", norm_name_col(F.col("company_name")))
    g = g.withColumn("name_prefix",
                     F.upper(F.element_at(F.split(F.col("norm_name"), " "), 1)))
    g = g.withColumn("hash4", F.substring(
            F.md5(F.concat_ws("|", F.col("norm_name"), F.col("country"))), 1, 4))
    g = g.withColumn("golden_id", F.concat_ws("-", F.lit("GLD"),
                                              F.col("name_prefix"), F.col("hash4")))

    # ==========================================================================
    # STEP 2 — HIERARCHY (parent / subsidiary)
    #
    # B is the PARENT of A when all four hold:
    #   - same country
    #   - B's name tokens are contained in A's   ({acme} within {acme, cloud, services})
    #   - A's name is strictly LONGER            (the subsidiary carries the extra words)
    #   - they DON'T share a city or domain      (the subsidiary is operationally distinct)
    #
    # That last condition is the interesting one. In Phase 3 a missing shared signal
    # was the reason NOT to merge. Here the very same absence is the evidence that
    # these are two related-but-separate legal entities. One signal, used twice, in
    # opposite directions.
    #
    # A self-join is how I compare every golden record against every other. That's
    # O(n^2), which is completely fine at 12 records and would need blocking at scale.

    # WHAT THIS GETS WRONG (and why I'm comfortable with it anyway)
    #
    # This is an INFERENCE, not ground truth. It will be wrong in two ways:
    #   - two unrelated companies that happen to share a name lineage could get
    #     wrongly linked as parent/child
    #   - a real subsidiary that DOES share its parent's domain would be wrongly
    #     merged back in Phase 3 and never reach this step at all
    #
    # The design errs deliberately toward NOT merging. A wrong hierarchy link is
    # cheap: it's a link, you unlink it, and both records still exist. A wrong
    # merge is expensive: you've destroyed two separate records and you're now
    # double-counting revenue. So when the evidence is thin, this fails safe
    # rather than failing clever.
    #
    # In production, corporate hierarchy is a fact you LOOK UP, not one you infer
    # from strings — a company registry, D&B, or legal entity identifiers. The
    # value of this heuristic isn't that it's right; it's that it FLAGS candidates
    # for a data steward to confirm, instead of silently deciding.
    # ==========================================================================
    g = g.withColumn("tokens", F.split(F.col("norm_name"), " "))
    a = g.alias("a")   # candidate child  (longer name)
    b = g.alias("b")   # candidate parent (shorter name)

    # Every one of b's tokens also appears in a.
    subset = (F.size(F.array_intersect(F.col("b.tokens"), F.col("a.tokens")))
              == F.size(F.col("b.tokens")))

    # Direction matters: the LONGER name is the child. Without this the join would
    # produce the relationship in both directions and I couldn't tell which is which.
    strictly_longer = F.size(F.col("a.tokens")) > F.size(F.col("b.tokens"))

    # The same corroborating signals from Phase 3 — shared city or shared domain.
    shares_signal = (
        (F.col("a.city").isNotNull() & (F.col("a.city") == F.col("b.city"))) |
        (F.col("a.email_domain").isNotNull() &
         (F.col("a.email_domain") == F.col("b.email_domain")))
    )

    hierarchy = (a.join(b, (F.col("a.country") == F.col("b.country")) &
                           (F.col("a.golden_id") != F.col("b.golden_id")) &
                           subset & strictly_longer & (~shares_signal))
                  .select(F.col("a.golden_id").alias("child_id"),
                          F.col("b.golden_id").alias("parent_golden_id"),
                          F.col("a.company_name").alias("child_name"),
                          F.col("b.company_name").alias("parent_name")))

    print("\n=== HIERARCHY LINKS (subsidiary -> parent) ===")
    hierarchy.select("child_name", "parent_name",
                     "parent_golden_id").show(20, truncate=False)

    # LEFT join: a company with no parent keeps a null parent_golden_id and stays in
    # the master. An inner join here would silently delete every top-level company —
    # the kind of bug that produces a beautifully clean, completely wrong output.
    final = (g.join(hierarchy.select("child_id", "parent_golden_id"),
                    g.golden_id == F.col("child_id"), "left")
              .drop("child_id", "tokens", "norm_name", "name_prefix", "hash4"))

    print("\n=== FINAL GOLDEN MASTER ===")
    final.select("golden_id", "company_name", "country",
                 "employees", "revenue", "revenue_currency",
                 "parent_golden_id").orderBy("golden_id").show(20, truncate=False)

    # ==========================================================================
    # STEP 3 — DATA-QUALITY REPORT
    #
    # A pipeline that only emits data is half a pipeline. This one also reports on
    # itself: how many records collapsed, how many hierarchy links it found, and how
    # complete each field ended up. That's what lets someone decide whether to TRUST
    # the output, rather than just consume it.
    #
    # It's also an operational tripwire. If completeness on revenue suddenly drops
    # from 92% to 40%, something upstream broke — and this is the number that says so.
    # In production these metrics would go to CloudWatch and alert; here I emit JSON
    # so the pipeline returns structured results rather than console prints.
    # ==========================================================================
    pdf = final.toPandas()
    total_raw = int(pdf["source_records"].sum())
    n_golden = len(pdf)
    collapsed = total_raw - n_golden
    n_hier = int(pdf["parent_golden_id"].notna().sum())

    fields = ["company_name", "industry", "country", "city",
              "employees", "revenue", "vat_id", "email_domain"]
    completeness = {f: round(pdf[f].notna().mean() * 100) for f in fields}

    lines = [
        "# Data-Quality Summary",
        "",
        f"- **Raw records ingested:** {total_raw} (across 3 source systems)",
        f"- **Golden records produced:** {n_golden} distinct companies",
        f"- **Duplicate/overlapping records collapsed:** {collapsed} "
        f"({total_raw} → {n_golden})",
        f"- **Hierarchy links (subsidiary → parent):** {n_hier}",
        "",
        "## Field completeness (share of golden records with a value)",
    ]
    for f, pct in completeness.items():
        lines.append(f"- {f}: {pct}%")
    summary = "\n".join(lines)

    print("\n=== DATA-QUALITY SUMMARY ===")
    print(summary)

    # The golden master goes into its own golden/ subfolder. In the cloud the Glue
    # crawler points at exactly this prefix, so it catalogues ONE clean table for
    # Athena instead of hoovering up every intermediate file in data/processed/.
    mdm_io.write_csv(pdf, "data/processed/golden/golden_master.csv")
    mdm_io.write_text(summary + "\n", "docs/data_quality_summary.md")

    report = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw_records_ingested": total_raw,
        "golden_records": n_golden,
        "records_collapsed": collapsed,
        "hierarchy_links": n_hier,
        "field_completeness_pct": completeness,
        "outputs": ["data/processed/golden/golden_master.csv",
                    "docs/data_quality_summary.md"],
    }
    mdm_io.write_text(json.dumps(report, indent=2), "run_report.json")

    print("\nWritten: data/processed/golden/golden_master.csv")
    print("Written: docs/data_quality_summary.md")
    print("Written: run_report.json")

    spark.stop()


if __name__ == "__main__":
    main()