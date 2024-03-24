# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import time
import boto3

client = boto3.client("cognito-idp")

def on_event(event, context):
    print(event)
    request_type = event["RequestType"].lower()
    if request_type == "create":
        return on_create(event)
    if request_type == "update":
        return on_update(event)
    if request_type == "delete":
        return on_delete(event)
    raise Exception(f"Invalid request type: {request_type}")


def on_create(event):
    props = event["ResourceProperties"]
    print(f"create new resource with {props=}")
    
    user_pool = client.describe_user_pool(UserPoolId=props['user_pool_id'])['UserPool']
    
    user_pool['AdminCreateUserConfig'] = {}

    user_pool['AdminCreateUserConfig']['AllowAdminCreateUserOnly'] = True
    
    user_pool['AdminCreateUserConfig']['InviteMessageTemplate']= {}
    user_pool['AdminCreateUserConfig']['InviteMessageTemplate']['SMSMessage'] = f"Hello {{username}}. Temporary password is {{####}}  The video understanding portal will be at {props['url']}"
    user_pool['AdminCreateUserConfig']['InviteMessageTemplate']['EmailMessage'] = f"Hello {{username}}, welcome to video understanding solution. Your temporary password is {{####}}  The video understanding portal will be at {props['url']}"
    user_pool['AdminCreateUserConfig']['InviteMessageTemplate']['EmailSubject'] = 'Welcome to the video understanding solution'
    
    response = client.update_user_pool(
        UserPoolId=props['user_pool_id'],
        Policies=user_pool["Policies"],
        DeletionProtection=user_pool["DeletionProtection"],
        MfaConfiguration=user_pool["MfaConfiguration"],
        AdminCreateUserConfig=user_pool['AdminCreateUserConfig']
    )
    
    response = client.admin_create_user(
        UserPoolId=props['user_pool_id'],
        Username=props['admin_email_address'],
        UserAttributes=[
            {
                'Name': 'email',
                'Value': props['admin_email_address']
            },
        ],
        DesiredDeliveryMediums=['EMAIL'],
    )
    
    physical_id = "admincognitouser"
    
    return {"PhysicalResourceId": physical_id}


def on_update(event):
    props = event["ResourceProperties"]
    print(f"no op")

    return {"PhysicalResourceId": "admincognitouser"}

def on_delete(event):
    props = event["ResourceProperties"]
    print(f"delete resource with {props=}")
    
    response = client.admin_delete_user(
        UserPoolId=props['user_pool_id'],
        Username=props['admin_email_address']
    )

    return {"PhysicalResourceId": None}
