"""
Phase 3 — MATCHING
==================
This is where the actual MDM problem lives. I have 28 standardised records and I
need to decide which of them describe the same real company, tagging each with a
shared `match_group_id` so Phase 4 can collapse each group into one golden record.

Two goals pull against each other, and the whole design is about not sacrificing
one for the other:

  * RECALL    -> catch records that ARE the same company.
                 'globex' and 'globex industries' are one company, abbreviated
                 in one system. These must merge.

  * PRECISION -> never merge records that are NOT the same company.
                 'acme corp' and 'acme cloud services' are parent and subsidiary.
                 'stark industries' and 'stark manufacturing' are unrelated
                 companies that happen to share a founder's surname.
                 These must NOT merge.

WHY FUZZY NAME MATCHING ALONE DOESN'T WORK
------------------------------------------
I measured this rather than assuming it. Run a similarity score over the pairs and
you find that 'globex' / 'globex industries' (must merge) and 'acme' / 'acme cloud
services' (must NOT merge) score *identically*. They're structurally the same
shape: one name is a subset of the other.

No threshold can separate them, because the difference isn't in the names at all.
Turn the threshold up and I lose Globex; turn it down and I wrongly swallow Acme's
subsidiary into its parent. The information needed to tell them apart simply isn't
present in the name.

So I don't merge on name similarity. Name similarity only makes a pair a
CANDIDATE. To actually merge, I require a SECOND, corroborating signal — a shared
email domain or a shared city. Globex's records share a domain, so they merge.
Acme and Acme Cloud Services don't, so they stay separate and get flagged as a
hierarchy candidate instead. Recall preserved, precision preserved.

That "require independent corroboration before a destructive merge" instinct is
the part I'd carry to any other matching problem.

STRATEGY (cheap first, expensive only where it's needed)
--------------------------------------------------------
  1. EXACT (blocking): identical std_name + country -> certainly the same company.
  2. FUZZY: among the surviving candidates, look for a name-subset relationship
     AND a corroborating signal. Merge only when both hold.
  3. HIERARCHY: subset-name pairs WITHOUT a shared signal are flagged as
     parent/subsidiary candidates — linked in Phase 5, never merged.

Note on the final union-find step: I do it in plain Python because after blocking
the candidate set is tiny. At real scale this is a connected-components problem on
a graph and I'd reach for GraphFrames. Same logic, tool sized to the data.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# 0. WHERE TO READ AND WRITE — the one seam between local and cloud
# ---------------------------------------------------------------------------
# awsglue only exists inside the Glue runtime, so the import is guarded. Locally
# it throws, I swallow it, and MDM_BASE defaults to "." — the same relative paths
# then resolve against my disk instead of S3. One codebase, two environments.
import os, sys
try:
    from awsglue.utils import getResolvedOptions
    os.environ["MDM_BASE"] = getResolvedOptions(sys.argv, ["MDM_BASE"])["MDM_BASE"]
except Exception:
    pass  # not on Glue — running locally
MDM_BASE = os.environ.get("MDM_BASE", ".").rstrip("/")
import mdm_io


# This threshold is ONLY a candidate gate — it decides what's worth looking at
# more closely, not what gets merged. A pair scoring 100 still won't merge unless
# it also passes the subset test AND shares a signal. That's deliberate: I never
# want a single tunable number to be able to cause a bad merge on its own.
NAME_SIM_THRESHOLD = 85


def main():
    # On Glue I must not set a master — Glue provides its own Spark cluster.
    # Locally I do, so the same file runs in both places.
    _builder = SparkSession.builder.appName("mdm-phase3-match")
    if not MDM_BASE.startswith("s3://"):
        _builder = _builder.master("local[*]")
    spark = _builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(f"{MDM_BASE}/data/processed/standardized.csv",
                        header=True, inferSchema=True)

    # ==========================================================================
    # STEP 1 — EXACT MATCH (blocking)
    #
    # If two records agree on both the cleaned name AND the country, they're the
    # same company — no fuzziness needed. This is the cheap win, and it also
    # shrinks the problem: everything after this compares GROUPS, not records.
    # ==========================================================================
    df = df.withColumn("exact_key",
                       F.concat_ws("|", F.col("std_name"), F.col("country_std")))

    print("\n=== STEP 1: exact groups (std_name | country) ===")
    (df.groupBy("exact_key")
        .agg(F.count("*").alias("records"),
             F.collect_list("source_id").alias("ids"))
        .orderBy("exact_key").show(40, truncate=False))

    # ==========================================================================
    # STEP 2 — CANDIDATE ENTITIES AND THEIR SIGNALS
    #
    # Collapse to one row per exact_key, gathering the evidence I'll use to
    # confirm or reject a fuzzy merge: every email domain and every city that
    # candidate has been seen with. collect_set gives me the distinct non-null
    # values as an array — so a candidate carries ALL its known domains, not just
    # one, and a match on any of them counts.
    # ==========================================================================
    candidates = (
        df.groupBy("exact_key", "std_name", "country_std")
          .agg(F.collect_set("email_domain").alias("domains"),
               F.collect_set("city").alias("cities"))
    )

    # ==========================================================================
    # STEP 3 — PAIRWISE COMPARISON (self-join, blocked by country)
    #
    # Joining candidates to themselves gives me every pair to compare. Two things
    # keep this from exploding:
    #   - country equality: I never compare a German company to a US one. This is
    #     BLOCKING, and it's what stops the comparison being O(n^2) over everything.
    #   - a.exact_key < b.exact_key: stops a row pairing with itself, and stops me
    #     seeing (A,B) and (B,A) as two separate pairs.
    #
    # At real scale, country alone is too coarse a block (imagine 2M US companies).
    # I'd block on something finer — name prefix, or a phonetic key — but the
    # principle is identical: never compare things that can't possibly match.
    # ==========================================================================
    a = candidates.alias("a")
    b = candidates.alias("b")
    pairs = (
        a.join(b, (F.col("a.country_std") == F.col("b.country_std")) &
                  (F.col("a.exact_key") < F.col("b.exact_key")))
    )

    # A UDF lets me call rapidfuzz (ordinary Python) on Spark columns. UDFs are
    # slow compared to native Spark functions — they break out of the JVM for every
    # row — but after blocking there are only a handful of pairs, so the cost is
    # irrelevant here. On a big dataset I'd want to avoid a UDF in the hot path.
    #
    # token_set_ratio compares the SETS of words, so word order doesn't matter and
    # a shorter name scores highly against a longer one that contains it.
    sim_udf = F.udf(lambda x, y: int(fuzz.token_set_ratio(x or "", y or "")),
                    IntegerType())

    pairs = pairs.select(
        F.col("a.exact_key").alias("key_a"), F.col("a.std_name").alias("name_a"),
        F.col("b.exact_key").alias("key_b"), F.col("b.std_name").alias("name_b"),
        F.col("a.domains").alias("dom_a"), F.col("b.domains").alias("dom_b"),
        F.col("a.cities").alias("cit_a"),  F.col("b.cities").alias("cit_b"),
    ).withColumn("name_sim", sim_udf(F.col("name_a"), F.col("name_b")))

    # Everything below this bar isn't even worth a second look.
    pairs = pairs.filter(F.col("name_sim") >= NAME_SIM_THRESHOLD)

    def is_subset(name_a, name_b):
        """Is one name's word-set contained in the other's?

        {globex} is a subset of {globex, industries}. This is the shape that both
        an abbreviation AND a subsidiary take — which is exactly why this test
        alone can't decide anything, and why shares_signal() has to exist.
        """
        ta, tb = set((name_a or "").split()), set((name_b or "").split())
        return ta.issubset(tb) or tb.issubset(ta)

    def shares_signal(dom_a, dom_b, cit_a, cit_b):
        """Do these two candidates share ANY email domain or ANY city?

        This is the corroborating evidence — information that comes from somewhere
        other than the name. A shared domain is strong: two records at
        @globex.com are the same company. A shared city is weaker but still real.

        A subsidiary typically has its own domain and its own office, which is
        precisely why this test separates 'globex/globex industries' (merge) from
        'acme/acme cloud services' (don't).
        """
        da, db = set(dom_a or []), set(dom_b or [])
        ca, cb = set(cit_a or []), set(cit_b or [])
        return bool(da & db) or bool(ca & cb)

    # Declaring BooleanType matters: without it Spark assumes these UDFs return
    # strings, and the comparisons below silently become string comparisons
    # against "true". That works by accident and breaks the moment anyone touches it.
    subset_udf = F.udf(is_subset, BooleanType())
    signal_udf = F.udf(shares_signal, BooleanType())

    pairs = (pairs
        .withColumn("name_subset", subset_udf(F.col("name_a"), F.col("name_b")))
        .withColumn("shares_signal", signal_udf("dom_a", "dom_b", "cit_a", "cit_b")))

    # THE MERGE RULE — the heart of the whole pipeline.
    #
    #   subset AND signal  -> MERGE        (same company, one name abbreviated)
    #   subset, NO signal  -> HIERARCHY?   (parent/subsidiary — link, don't merge)
    #   no subset          -> keep separate (different companies that look alike)
    #
    # Note the middle case: a pair that looks mergeable by name but has no
    # corroboration isn't discarded as noise — it's evidence of a hierarchy. The
    # signal that PREVENTS a merge is the same signal that reveals the structure.
    pairs = pairs.withColumn(
        "decision",
        F.when(F.col("name_subset") & F.col("shares_signal"), F.lit("MERGE"))
         .when(F.col("name_subset"), F.lit("HIERARCHY?"))
         .otherwise(F.lit("keep separate"))
    )

    print("\n=== STEP 3: candidate pairs & decisions ===")
    pairs.select("name_a", "name_b", "name_sim",
                 "name_subset", "shares_signal", "decision").show(40, truncate=False)

    # ==========================================================================
    # STEP 4 — BUILD THE FINAL GROUPS (union-find)
    #
    # Merging is transitive: if A merges with B, and B merges with C, then A, B and
    # C are all one company — even though I never directly compared A to C. Union-
    # find handles that transitivity for me. Start every candidate in its own group,
    # then union each MERGE pair; whatever ends up connected is one company.
    #
    # This is connected-components on a graph. Doing it in Python is fine because
    # blocking left me with very few candidates; at scale it's a GraphFrames job.
    # ==========================================================================
    keys = [r["exact_key"] for r in candidates.select("exact_key").collect()]
    merge_pairs = [(r["key_a"], r["key_b"])
                   for r in pairs.filter(F.col("decision") == "MERGE")
                                 .select("key_a", "key_b").collect()]
    hierarchy_pairs = [(r["key_a"], r["key_b"])
                       for r in pairs.filter(F.col("decision") == "HIERARCHY?")
                                     .select("key_a", "key_b").collect()]

    parent = {k: k for k in keys}

    def find(x):
        # Path compression: point every node straight at its root as I walk up, so
        # repeated lookups get flatter and faster.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for x, y in merge_pairs:
        union(x, y)

    # One group id per connected component. Sorted so the ids are STABLE across
    # runs — if G003 became G007 just because rows arrived in a different order,
    # the ids would be meaningless. (Phase 5 makes this guarantee properly, with a
    # content-based hash rather than a counter.)
    roots = sorted(set(find(k) for k in keys))
    group_id = {root: f"G{idx+1:03d}" for idx, root in enumerate(roots)}
    key_to_group = {k: group_id[find(k)] for k in keys}

    # Push the group ids back onto the ORIGINAL 28 records — I never lost them,
    # I only grouped them. Survivorship needs every source record to choose between.
    mapping = spark.createDataFrame(
        [(k, g) for k, g in key_to_group.items()], ["exact_key", "match_group_id"])
    matched = df.join(mapping, on="exact_key", how="left")

    print("\n=== STEP 4: final match groups ===")
    (matched.groupBy("match_group_id")
            .agg(F.collect_list("raw_name").alias("names_merged"),
                 F.collect_set("source_system").alias("sources"))
            .orderBy("match_group_id").show(40, truncate=False))

    n_groups = matched.select("match_group_id").distinct().count()
    print(f"Distinct companies (match groups): {n_groups}   (expected: 12)")
    if hierarchy_pairs:
        print(f"Parent/subsidiary candidates flagged: {hierarchy_pairs}")

    # Keep exact_key out of the output — it was scaffolding for the blocking step,
    # not something Phase 4 should depend on.
    mdm_io.write_csv(matched.drop("exact_key").toPandas(), "data/processed/matched.csv")
    print("\nWritten: data/processed/matched.csv")

    spark.stop()


if __name__ == "__main__":
    main()