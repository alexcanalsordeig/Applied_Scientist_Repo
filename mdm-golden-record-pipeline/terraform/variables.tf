variable "region" {
  description = "AWS region to deploy into. eu-west-1 (Ireland) is close to NL."
  type        = string
  default     = "eu-west-1"
}

variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "mdm"
}

variable "notification_email" {
  description = "Email that receives the 'pipeline complete' SNS notification. You must click the confirmation link AWS emails you."
  type        = string
}

variable "glue_extra_modules" {
  description = "Comma-separated pip modules to install into the Glue jobs (e.g. your fuzzy-matching lib). Match this to your requirements.txt. Set to \"\" if you need none."
  type        = string
  default     = "rapidfuzz"
}
