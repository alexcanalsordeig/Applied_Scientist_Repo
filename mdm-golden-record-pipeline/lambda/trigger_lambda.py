"""
trigger_lambda.py  --  EventBridge -> this Lambda -> Step Functions
-------------------------------------------------------------------
Fires when a '*.trigger' object lands in data/raw/ on the bucket.
It starts one Step Functions execution (the MDM pipeline). Tiny on purpose:
its job is to be the glue between the event and the orchestrator, and to give
you an honest "yes, the pipeline uses Lambda" line for the interview.
"""
import os
import json
import boto3

sfn = boto3.client("stepfunctions")


def handler(event, context):
    key = (
        event.get("detail", {})
        .get("object", {})
        .get("key", "manual")
    )
    resp = sfn.start_execution(
        stateMachineArn=os.environ["STATE_MACHINE_ARN"],
        input=json.dumps({"triggered_by": key}),
    )
    print(f"Started execution: {resp['executionArn']} (trigger key: {key})")
    return {"started": True, "executionArn": resp["executionArn"]}
