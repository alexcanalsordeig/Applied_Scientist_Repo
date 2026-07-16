###############################################################################
# MDM golden-record pipeline — AWS infrastructure
#
# The whole stack, declared as code:
#   S3 (data lake) -> EventBridge -> Lambda -> Step Functions
#                                                  |
#                                    4x Glue PySpark jobs, in sequence
#                                                  |
#                                        SNS -> email + SQS queue
#                                                  |
#                              Glue Crawler -> Data Catalog -> Athena (SQL)
#
# Terraform is declarative: this file describes the DESIRED state, and
# `terraform apply` reconciles reality to match it. Resource order is never
# specified — it is inferred from the references between blocks.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    random  = { source = "hashicorp/random", version = "~> 3.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.0" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "me" {}

# S3 bucket names are globally unique across all of AWS, so a random suffix
# avoids collisions. Nothing hardcodes the name — every resource references it.
resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  name   = var.project
  bucket = "${var.project}-${random_id.suffix.hex}"

  # The four pipeline stages, in run order. Each name must match a script in
  # ../src/ (e.g. standardize -> ../src/standardize.py). Defining them once
  # here means the Glue jobs and the S3 uploads below stay in sync automatically.
  stages = ["standardize", "match", "survivorship", "finalize"]
}

# ---------------------------------------------------------------------------
# S3 — the data lake
#   data/raw/       source CSVs + the .trigger file that starts a run
#   data/processed/ intermediate stage outputs
#   data/processed/golden/  the final golden master (what Athena queries)
#   scripts/        the PySpark files Glue executes
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "data" {
  bucket = local.bucket

  # Lets `terraform destroy` empty the bucket instead of failing on a non-empty
  # bucket. Convenient for a demo — I would never set this in production, where
  # accidental deletion of a data lake is exactly what you want to prevent.
  force_destroy = true
}

# Keeps old versions of overwritten objects. Cheap insurance for a demo; in
# production this needs a lifecycle rule to expire old versions, or storage
# costs creep up forever.
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

# The most load-bearing line in this file. Without it S3 stays silent, nothing
# fires, and the pipeline is manual-only. This is what makes it EVENT-DRIVEN.
resource "aws_s3_bucket_notification" "data" {
  bucket      = aws_s3_bucket.data.id
  eventbridge = true
}

# Glue runs scripts from S3, not from a laptop — so the pipeline code is
# uploaded here. mdm_io.py rides along because the jobs import it.
resource "aws_s3_object" "scripts" {
  for_each = toset(concat(local.stages, ["mdm_io"]))
  bucket   = aws_s3_bucket.data.id
  key      = "scripts/${each.key}.py"
  source   = "${path.module}/../src/${each.key}.py"

  # etag = file hash, so re-uploading only happens when the script actually
  # changes. Without this, Terraform can't tell that a script was edited.
  etag = filemd5("${path.module}/../src/${each.key}.py")
}

# ---------------------------------------------------------------------------
# IAM — one role per service, each scoped to only what that service does.
#
#   Glue           -> read/write S3
#   Step Functions -> start Glue jobs, publish to SNS
#   Lambda         -> start the Step Functions execution
#
# Separate roles rather than one shared admin role: least privilege by
# separation. If the Lambda were ever compromised, it can start a workflow —
# it cannot read the data lake.
# ---------------------------------------------------------------------------

# A trust policy says WHO may assume a role. Here: the AWS service itself.
# Generating all three from one block keeps them consistent.
data "aws_iam_policy_document" "assume" {
  for_each = toset(["glue.amazonaws.com", "states.amazonaws.com", "lambda.amazonaws.com"])
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = [each.key]
    }
  }
}

# ---- Glue role ----
resource "aws_iam_role" "glue" {
  name               = "${local.name}-glue"
  assume_role_policy = data.aws_iam_policy_document.assume["glue.amazonaws.com"].json
}

# AWS-managed baseline for Glue (CloudWatch logging, catalog access, etc).
resource "aws_iam_role_policy_attachment" "glue_managed" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

# Scoped to THIS bucket only — not s3:* on every bucket in the account.
resource "aws_iam_role_policy" "glue_s3" {
  name = "s3-access"
  role = aws_iam_role.glue.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    }]
  })
}

# ---- Step Functions role ----
resource "aws_iam_role" "sfn" {
  name               = "${local.name}-sfn"
  assume_role_policy = data.aws_iam_policy_document.assume["states.amazonaws.com"].json
}

# Step Functions needs to start/poll Glue jobs and publish the completion event.
# Note: glue:* is on "*" because the .sync integration polls job runs — tightening
# this to specific job ARNs is a known improvement.
resource "aws_iam_role_policy" "sfn_policy" {
  name = "orchestrate"
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["glue:StartJobRun", "glue:GetJobRun", "glue:GetJobRuns", "glue:BatchStopJobRun"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.done.arn
      }
    ]
  })
}

# ---- Lambda role ----
resource "aws_iam_role" "lambda" {
  name               = "${local.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume["lambda.amazonaws.com"].json
}

# The Lambda can do exactly two things: start THIS state machine, and write logs.
resource "aws_iam_role_policy" "lambda_policy" {
  name = "start-sfn"
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["states:StartExecution"]
        Resource = aws_sfn_state_machine.pipeline.arn
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Glue — the four PySpark stages
#
# command.name = "glueetl" means a real managed SPARK cluster, which is what
# PySpark requires. ("pythonshell" would be a single machine with pandas — the
# wrong tool for Spark code.) One job per stage, all sharing the same MDM_BASE
# so they read and write the same S3 paths.
# ---------------------------------------------------------------------------
resource "aws_glue_job" "stage" {
  for_each     = toset(local.stages)
  name         = "${local.name}-${each.key}"
  role_arn     = aws_iam_role.glue.arn
  glue_version = "4.0" # Glue 4.0 = Spark 3.3, Python 3.10

  # Smallest real Spark cluster available. The dataset is 28 rows, so anything
  # larger is paying for idle workers. Cluster exists only while the job runs.
  worker_type       = "G.1X"
  number_of_workers = 2

  # Fail fast and surface the error rather than silently retrying. In production
  # this would be 1-2 retries with a dead-letter path for genuine transients.
  max_retries = 0

  # One run at a time — the stages are sequential and share files in S3, so
  # concurrent runs of the same stage would race.
  execution_property { max_concurrent_runs = 1 }

  command {
    name            = "glueetl" # a real Spark ETL job (NOT python shell)
    python_version  = "3"
    script_location = "s3://${aws_s3_bucket.data.id}/scripts/${each.key}.py"
  }

  default_arguments = merge(
    {
      # This is the ONE seam between local and cloud. The scripts read MDM_BASE
      # and resolve their (unchanged) relative paths against it: "." on a laptop,
      # "s3://bucket" here. No separate cloud version of the pipeline logic.
      "--MDM_BASE" = "s3://${aws_s3_bucket.data.id}"

      # Spark workers start clean, so ship the shared I/O helper to the cluster.
      "--extra-py-files" = "s3://${aws_s3_bucket.data.id}/scripts/mdm_io.py"

      "--job-language" = "python"
      "--TempDir"      = "s3://${aws_s3_bucket.data.id}/glue-temp/" # Spark scratch space

      # Stream logs to CloudWatch — the only way to debug a failed cloud run.
      "--enable-continuous-cloudwatch-log" = "true"
    },

    # pip-installs extra libraries onto the workers (rapidfuzz, for the fuzzy
    # matching in match.py). Conditional so the argument is omitted entirely if
    # no extra modules are needed.
    var.glue_extra_modules == "" ? {} : { "--additional-python-modules" = var.glue_extra_modules }
  )

  # A job can't point at a script that hasn't been uploaded yet.
  depends_on = [aws_s3_object.scripts]
}

# ---------------------------------------------------------------------------
# Step Functions — the orchestrator
#
# Runs the four stages strictly in order, then publishes a completion event.
#
# The critical detail is ".sync" on the Glue integration: it makes Step Functions
# WAIT for each job to finish before starting the next. Without it all four
# would fire at once and match.py would try to read a file standardize.py had
# not written yet.
#
# If any stage fails, the chain halts there — the state machine will not run
# survivorship on a broken match output.
# ---------------------------------------------------------------------------
resource "aws_sfn_state_machine" "pipeline" {
  name     = "${local.name}-pipeline"
  role_arn = aws_iam_role.sfn.arn
  definition = jsonencode({
    Comment = "MDM golden-record pipeline: standardize -> match -> survivorship -> finalize -> notify"
    StartAt = "standardize"
    States = {
      standardize = {
        Type       = "Task"
        Resource   = "arn:aws:states:::glue:startJobRun.sync"
        Parameters = { JobName = aws_glue_job.stage["standardize"].name }
        Next       = "match"
      }
      match = {
        Type       = "Task"
        Resource   = "arn:aws:states:::glue:startJobRun.sync"
        Parameters = { JobName = aws_glue_job.stage["match"].name }
        Next       = "survivorship"
      }
      survivorship = {
        Type       = "Task"
        Resource   = "arn:aws:states:::glue:startJobRun.sync"
        Parameters = { JobName = aws_glue_job.stage["survivorship"].name }
        Next       = "finalize"
      }
      finalize = {
        Type       = "Task"
        Resource   = "arn:aws:states:::glue:startJobRun.sync"
        Parameters = { JobName = aws_glue_job.stage["finalize"].name }
        Next       = "notify"
      }
      notify = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.done.arn
          Subject  = "MDM pipeline complete"
          Message  = "The MDM pipeline finished. Golden master written to s3://${aws_s3_bucket.data.id}/data/processed/golden/"
        }
        End = true
      }
    }
  })
}

# ---------------------------------------------------------------------------
# SNS + SQS — completion event, fanned out
#
# Classic pub-sub fan-out: ONE published event reaches TWO subscribers that know
# nothing about each other —
#   * email  -> a human is told the run finished
#   * SQS    -> a downstream system (a CRM, a dashboard) can consume golden-record
#               updates at its own pace, and messages survive if it is offline
#
# The queue here stands in for that real downstream consumer. Adding a third
# subscriber later would require no change to the publisher.
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "done" {
  name = "${local.name}-complete"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.done.arn
  protocol  = "email"
  endpoint  = var.notification_email # must be confirmed via the email AWS sends
}

resource "aws_sqs_queue" "downstream" {
  name = "${local.name}-downstream"
}

resource "aws_sns_topic_subscription" "queue" {
  topic_arn = aws_sns_topic.done.arn
  protocol  = "sqs"
  endpoint  = aws_sqs_queue.downstream.arn
}

# A queue rejects messages by default. This grants ONLY this SNS topic the right
# to write to it (the ArnEquals condition), not the world.
resource "aws_sqs_queue_policy" "allow_sns" {
  queue_url = aws_sqs_queue.downstream.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sns.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.downstream.arn
      Condition = { ArnEquals = { "aws:SourceArn" = aws_sns_topic.done.arn } }
    }]
  })
}

# ---------------------------------------------------------------------------
# EventBridge + Lambda — the trigger
#
# Trade-off worth naming: EventBridge CAN target Step Functions directly, so this
# Lambda is technically an extra hop. I kept it as an explicit place to add
# pre-flight validation (are all three source files present? is the schema sane?)
# before committing to a multi-minute Spark run.
# ---------------------------------------------------------------------------

# Zips the handler at plan time — no build step, no committed .zip artifact.
data "archive_file" "trigger" {
  type        = "zip"
  source_file = "${path.module}/../lambda/trigger_lambda.py"
  output_path = "${path.module}/trigger_lambda.zip"
}

resource "aws_lambda_function" "trigger" {
  function_name = "${local.name}-trigger"
  role          = aws_iam_role.lambda.arn
  handler       = "trigger_lambda.handler"
  runtime       = "python3.12"
  filename      = data.archive_file.trigger.output_path

  # Redeploys the function whenever the source actually changes.
  source_code_hash = data.archive_file.trigger.output_base64sha256
  timeout          = 30

  # The Lambda doesn't hardcode the state machine — it's injected, so the code
  # stays environment-agnostic.
  environment {
    variables = { STATE_MACHINE_ARN = aws_sfn_state_machine.pipeline.arn }
  }
}

# Fires ONLY on files ending in ".trigger". This is deliberate: uploading the
# three source CSVs must NOT start three pipeline runs. The trigger file is an
# explicit "the data is ready, go" signal, which keeps control of WHEN a run
# happens while still being fully event-driven.
resource "aws_cloudwatch_event_rule" "on_upload" {
  name = "${local.name}-on-raw-upload"
  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.data.id] }
      object = { key = [{ suffix = ".trigger" }] }
    }
  })
}

resource "aws_cloudwatch_event_target" "to_lambda" {
  rule = aws_cloudwatch_event_rule.on_upload.name
  arn  = aws_lambda_function.trigger.arn
}

# EventBridge is not allowed to invoke a Lambda unless the Lambda says so.
# Easy to forget — the symptom is a rule that "fires" but nothing ever happens.
resource "aws_lambda_permission" "allow_events" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.on_upload.arn
}

# ---------------------------------------------------------------------------
# Glue Data Catalog + Crawler — making the output queryable
#
# The crawler reads the golden CSV, infers its schema, and registers it as a
# table. Athena then queries the file in place — no database server, no loading.
# ---------------------------------------------------------------------------
resource "aws_glue_catalog_database" "mdm" {
  name = "mdm"
}

resource "aws_glue_crawler" "golden" {
  name          = "${local.name}-golden-crawler"
  role          = aws_iam_role.glue.arn
  database_name = aws_glue_catalog_database.mdm.name

  # Points at golden/ specifically, NOT all of processed/. Otherwise the crawler
  # would also catalogue every intermediate file and Athena would show a mess of
  # tables instead of one clean golden table.
  s3_target { path = "s3://${aws_s3_bucket.data.id}/data/processed/golden/" }
}

# ---------------------------------------------------------------------------
# Athena — the SQL query layer
#
# Serverless: queries run directly against the CSV in S3 and bill per byte
# scanned. Chosen over RDS for a demo because it costs nothing at rest; a
# production system would likely land the golden master in Postgres instead
# (see sql/schema.sql for that design).
#
# A workgroup must have a results location, or every query fails with
# "No output location provided".
# ---------------------------------------------------------------------------
resource "aws_athena_workgroup" "mdm" {
  name          = local.name
  force_destroy = true
  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.data.id}/athena-results/"
    }
  }
}
