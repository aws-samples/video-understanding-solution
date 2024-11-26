import boto3
import os
import json

amplify = boto3.client('amplify')
codepipeline = boto3.client('codepipeline')
app_id = os.environ['AMPLIFY_APP_ID']
branch_name = os.environ['BRANCH_NAME']

# Notify CodePipeline of a successful job
def put_job_success(job_id, message):
    codepipeline.put_job_success_result(jobId=job_id)

# Notify CodePipeline of a failed job
def put_job_failure(job_id, message):
    codepipeline.put_job_failure_result(
        jobId=job_id,
        failureDetails={
            'message': json.dumps(message),
            'type': 'JobFailed',
        }
    )

def handler(event, context):
    job_id = event['CodePipeline.job']['id']
    try:
        user_parameters = json.loads(event['CodePipeline.job']['data']['actionConfiguration']['configuration']['UserParameters'])
        bucket_name = user_parameters['bucket_name']
        object_key = user_parameters['prefix']
        
        # Then start the deployment with the created job
        start_response = amplify.start_deployment(
            appId=app_id,
            branchName=branch_name,
            sourceUrl=f"s3://{bucket_name}/{object_key}/",
            sourceUrlType="BUCKET_PREFIX"
        )
        
        put_job_success(job_id, "Deployment is successful")

        return {
            'statusCode': 200,
            'body': json.dumps('Successfully triggered Amplify deployment')
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        put_job_failure(job_id, str(e))
        raise e