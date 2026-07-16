output "bucket" {
  description = "Your data-lake bucket."
  value       = aws_s3_bucket.data.id
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.pipeline.arn
}

output "sns_topic" {
  value = aws_sns_topic.done.arn
}

output "downstream_queue_url" {
  value = aws_sqs_queue.downstream.id
}

output "athena_workgroup" {
  value = aws_athena_workgroup.mdm.name
}

output "STEP_1_upload_sources" {
  description = "Upload your three raw source files here."
  value       = "aws s3 cp ./data/raw/ s3://${aws_s3_bucket.data.id}/data/raw/ --recursive --exclude '*' --include '*.csv'"
}

output "STEP_2_launch_pipeline" {
  description = "Drop a trigger file to fire EventBridge -> Lambda -> Step Functions."
  value       = "aws s3 cp /dev/null s3://${aws_s3_bucket.data.id}/data/raw/run.trigger"
}

output "STEP_3_catalog_golden" {
  description = "After the run finishes, catalog the output so Athena can see it."
  value       = "aws glue start-crawler --name ${aws_glue_crawler.golden.name}"
}
