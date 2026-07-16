**Why this exists:**I wanted to build a real MDM pipeline end to end rather than just talk about one — the entity-matching problem in particular is more subtle than it looks, and the only way to understand it properly was to build it and watch where it broke.

# Client Account MDM — Golden Record Pipeline

A **Master Data Management** pipeline that consolidates client-company records from three
disagreeing internal source systems into a single trusted **golden record** per company —
using explicit survivorship rules, an auditable trail of every decision, and a fully
**deployed, event-driven AWS architecture**.

**Result: 28 conflicting records across 3 source systems → 12 trusted golden records**, with
parent/subsidiary hierarchy preserved and full provenance on every surviving field.

Built in **PySpark**, deployed on **AWS Glue / Step Functions / EventBridge / Lambda / SNS / SQS /
Athena**, provisioned entirely with **Terraform**.

---

## The problem it solves

The same client company appears in the CRM, the marketing/events system, and the billing
system — under different names, different IDs, different formats, with conflicting values and
duplicates. Left unmastered, this causes double-counting, missed account relationships, and
decisions made on wrong data.

The hard part isn't merging records. It's **merging the right ones and not the wrong ones**:

- `globex` and `globex industries` are the *same* company abbreviated → **must merge**
- `acme corp` and `acme cloud services` are *parent and subsidiary* → **must NOT merge** (but must be linked)
- `stark industries` and `stark manufacturing` are *different companies* → **must NOT merge**

Fuzzy name similarity alone scores the first two cases identically, so name matching by itself
cannot solve this. This pipeline resolves it by requiring a **second corroborating signal**
(shared email domain or city) before merging — preserving recall without sacrificing precision.

---

## Deployed AWS architecture

The pipeline runs as a fully event-driven workflow. Dropping a trigger file into S3 sets off
the entire chain — no cron, no manual orchestration:

```
  data lands in S3 (data/raw/)
          │
          ▼
    EventBridge  ── rule: object key ends in ".trigger"
          │
          ▼
      Lambda  ── starts the workflow
          │
          ▼
  Step Functions  ── orchestrates, waits for each stage, halts on failure
          │
          ├──▶ Glue job 1: standardize    (PySpark)
          ├──▶ Glue job 2: match          (PySpark)
          ├──▶ Glue job 3: survivorship   (PySpark)
          ├──▶ Glue job 4: finalize       (PySpark)
          │
          ▼
    SNS topic  ──┬──▶ email notification
                 └──▶ SQS queue (downstream consumer)
          │
          ▼
  S3 (golden/)  ──▶ Glue Crawler ──▶ Glue Data Catalog ──▶ Athena (SQL)
```

All 22 AWS resources are defined in **Terraform** (`terraform/main.tf`) — reproducible,
reviewable, and destroyable to $0.

---

## The four pipeline stages

| Stage | What it does |
|---|---|
| **1. Standardise** | Maps three different source schemas onto one common schema. Canonicalises company names (strips legal suffixes), countries (`USA`/`US`/`United States` → one value), industries (`Tech`/`Technology` → one label), parses three different date formats, and tags revenue with its currency instead of pretending USD and EUR are comparable. |
| **2. Match** | Layered strategy: exact blocking on `std_name + country` (cheap), then pairwise fuzzy comparison **within country** (blocking keeps it cheap). Merges only when names are subset-related **AND** a corroborating signal agrees. Subset-name pairs *without* a shared signal are flagged as parent/subsidiary candidates — linked, never merged. |
| **3. Survivorship** | Field-level source authority with recency tiebreak, implemented with **window functions**. Billing wins financial/legal fields (it's contractual and audited); CRM wins sales-owned fields; marketing is self-entered so it only fills gaps. Nulls are skipped, so the golden record is as complete as the data allows. Every winning value records **which source it came from**. |
| **4. Finalise** | Assigns deterministic, stable `golden_id`s (same across re-runs — critical for MDM), links subsidiaries to parents without merging them, and emits a data-quality summary + machine-readable run report. |

---

## Survivorship rules (and why)

Different source systems are trustworthy for different things, so authority is assigned
**per field**, not per record:

- **Billing** → legal name, revenue, VAT ID, billing address. It's contractual and audited, so it's the most accurate for money and identity.
- **CRM** → industry classification, contact/email domain. It owns the sales relationship.
- **Marketing** → self-entered at event signup, so least reliable. Used only to **fill gaps** where better sources are null.
- **Recency** breaks ties within an equally-trusted source, so stale values never beat fresh ones.

This is the standard enterprise MDM survivorship design: maximise trust (use each system for
what it's best at) and completeness, with a full audit trail.

---

## Portable code: same PySpark runs locally or on Glue

The pipeline logic is **identical** in both environments. The only difference is where files
are read and written, injected as configuration (`MDM_BASE`):

- Locally, the base path is `.` → reads `./data/raw/...`
- On Glue, `--MDM_BASE s3://bucket` is passed → the *same* relative paths resolve to S3

`src/mdm_io.py` is the one seam that handles this. There is no separate "cloud version" of
the matching or survivorship logic to drift out of sync.

---

## Running it

### Locally

```bash
pip install -r requirements.txt

python src/standardize.py     # 3 sources → one standardised table
python src/match.py           # → 12 match groups
python src/survivorship.py    # → 12 golden records + audit trail
python src/finalize.py        # → stable IDs, hierarchy, quality report
```

### On AWS

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # add your notification email
terraform init
terraform plan                                 # review before creating anything
terraform apply

# upload sources and fire the event-driven pipeline
aws s3 cp ../data/raw/ s3://<bucket>/data/raw/ --recursive --include "*.csv"
aws s3 cp <any-file> s3://<bucket>/data/raw/run.trigger

# after the run, catalog the output for Athena
aws glue start-crawler --name mdm-golden-crawler
```

Then query the golden records in Athena:

```sql
SELECT * FROM mdm.golden;
```

Tear everything down with `terraform destroy`.

---

## Repository layout

```
mdm-pipeline/
├── src/                       # pipeline code (PySpark)
│   ├── standardize.py
│   ├── match.py
│   ├── survivorship.py
│   ├── finalize.py
│   └── mdm_io.py              # local-or-S3 I/O seam
├── terraform/                 # the entire AWS stack as code
│   ├── main.tf                # S3, Glue, Step Functions, Lambda, SNS, SQS, EventBridge, Athena
│   ├── variables.tf
│   └── outputs.tf
├── lambda/                    # Lambda trigger (fires the pipeline)
├── sql/
│   ├── schema.sql             # Postgres: TEXT staging layer → typed core layer
│   └── analysis.sql           # analytical queries over the golden master
├── data/raw/                  # the three source systems
├── docs/
    ├── data_dictionary.md     # schemas, embedded conflicts, expected output
    └── data_quality_summary.md

```

---

## Postgres layer

`sql/schema.sql` implements a deliberate **two-layer pattern**:

1. A **staging layer** where every column is `TEXT`, so ingestion never fails on a type-coercion error.
2. A typed **core layer** (`golden_master`) where the database enforces types, keys, and referential integrity.

> *"Stage everything as TEXT so ingestion never fails, then cast into a typed core layer so the
> database enforces validity, maths and dates work, sorting is correct, and queries can be
> indexed and optimised."*

---

## Design decisions worth noting

- **Event-driven, not scheduled.** Master data arrives irregularly; a file landing triggers the run. No wasted cron cycles, no waiting for the next slot.
- **Orchestrated, not scripted.** Step Functions gives a visual audit trail of exactly which stage ran, how long it took, and where it failed — which matters enormously for *master* data, where trust is the product.
- **Idempotent.** `golden_id`s are deterministic hashes, so re-running produces identical IDs. A company's master ID must never change just because data was reprocessed.
- **Serverless and cost-aware.** Nothing runs (or bills) when idle; Glue clusters exist only for the minutes a job runs.
- **Auditable.** Every surviving field records its source system. You can always answer "where did this value come from?"

---