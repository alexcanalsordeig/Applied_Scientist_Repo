"""
mdm_io.py — the one seam that lets the pipeline run BOTH locally and on AWS Glue
--------------------------------------------------------------------------------
Every stage reads and writes under a single BASE path. Locally BASE is "." so
"data/processed/golden.csv" resolves to ./data/processed/golden.csv. On Glue,
Terraform passes --MDM_BASE s3://<bucket> and the SAME relative path resolves to
S3 instead.

That's the whole trick. It means there is no "cloud version" of the pipeline —
the matching and survivorship logic never learns whether it's on a laptop or a
Spark cluster, because the only thing that differs is where the bytes land.

Why this file exists at all: Spark can read from S3 natively, so the READS are
handled by spark.read.csv() with an s3:// path. But the final outputs go through
pandas (small, single-file results), and pandas cannot write to S3 without extra
libraries. Rather than add s3fs or awswrangler as a dependency I'd have to install
onto every Glue worker, I use boto3 — which is already present everywhere — and
keep the whole cloud/local decision in one small, testable place.
"""
import os
import io
import boto3
import pandas as pd

# Injected by Glue via --MDM_BASE. Defaults to "." so a local run needs no config.
BASE = os.environ.get("MDM_BASE", ".").rstrip("/")


def _full(rel_path: str) -> str:
    """Resolve a relative path against BASE — the single branch point."""
    return f"{BASE}/{rel_path.lstrip('/')}"


def _split_s3(uri: str):
    """s3://bucket/some/key -> ("bucket", "some/key")"""
    without = uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key


def write_csv(df: pd.DataFrame, rel_path: str, **kwargs) -> str:
    """Write a pandas DataFrame to CSV — to local disk or to S3, transparently."""
    kwargs.setdefault("index", False)
    path = _full(rel_path)

    if path.startswith("s3://"):
        # Serialise to an in-memory buffer and PUT it. No temp files, no extra deps.
        bucket, key = _split_s3(path)
        buf = io.StringIO()
        df.to_csv(buf, **kwargs)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        # Locally the directory may not exist yet (data/processed/golden/ etc).
        # S3 needs no equivalent — "folders" there are just key prefixes.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, **kwargs)

    return path


def write_text(text: str, rel_path: str) -> str:
    """Same idea for plain text — the run report and the data-quality summary."""
    path = _full(rel_path)

    if path.startswith("s3://"):
        bucket, key = _split_s3(path)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=text)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    return path