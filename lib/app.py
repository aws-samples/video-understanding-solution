#!/usr/bin/env python3
import os

from aws_cdk import ( 
    App, Environment, Tags
)

from video_understanding_solution_stack import VideoUnderstandingSolutionStack


account=os.environ["CDK_DEPLOY_ACCOUNT"]
region=os.environ["CDK_DEPLOY_REGION"]
allowed_regions = ["us-east-1", "us-west-2"]

print(f"Account ID: {account}")
print(f"Region: {region}")

try:
    if region not in allowed_regions:
        raise AssertionError
except AssertionError:
    print(f"Selected region is {region}. Please use only one of these regions {str(allowed_regions)}")


app = App()
vus_main_stack = VideoUnderstandingSolutionStack(app, "VideoUnderstandingStack", env=Environment(
    account=account,
    region=region
))
Tags.of(vus_main_stack).add("Application", "VideoUnderstandingSolution")

app.synth()