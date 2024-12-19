import boto3
import json
import base64

def on_event(event, context):
    print(event)
    request_type = event['RequestType'].lower()
    
    if request_type == 'create' or request_type == 'update':
        return on_create(event)
    if request_type == 'delete':
        return on_delete(event)
    raise Exception(f"Invalid request type: {request_type}")

def on_create(event):
    props = event["ResourceProperties"]
    bedrock = boto3.client('bedrock-agent')
    ssm = boto3.client('ssm')
    
    # Base64 encode the prompts to safely handle special characters
    system_prompt = base64.b64decode(props['visual_extraction_system_prompt_text']).decode('utf-8')
    task_prompt = base64.b64decode(props['visual_extraction_task_prompt_text']).decode('utf-8')
    
    # Create prompt
    prompt = bedrock.create_prompt(
        name=props['visual_extraction_prompt_name'],
        description=props['visual_extraction_prompt_description'],
        defaultVariant=props['visual_extraction_prompt_variant_name'],
        variants=[{
            'name': props['visual_extraction_prompt_variant_name'],
            'templateConfiguration': {
                'chat': {
                    'messages': [
                        {
                            'content': [
                                {
                                    'text': task_prompt
                                },
                            ],
                            'role': 'user'
                        },
                    ],
                    'system': [
                        {
                            'text': system_prompt
                        },
                    ]
                }
            },
            'templateType': 'CHAT'
        }]
    )
    
    # Create prompt version
    prompt_version = bedrock.create_prompt_version(
        promptIdentifier=prompt['id'],
        description=props['visual_extraction_prompt_version_description']
    )['version']
    
    # Get current parameter store value
    try:
        parameter = ssm.get_parameter(Name=props['configuration_parameter_name'])
        config = json.loads(parameter['Parameter']['Value'])
    except ssm.exceptions.ParameterNotFound:
        config = {}
    
    # Update prompt configuration
    config['visual_extraction_prompt'] = {
        "prompt_id": prompt['id'],
        "variant_name": props['visual_extraction_prompt_variant_name'],
        "version_id": prompt_version
    }
    
    # Update parameter store
    ssm.put_parameter(
        Name=props['configuration_parameter_name'],
        Value=json.dumps(config),
        Type='String',
        Overwrite=True
    )
    
    return {
        'PhysicalResourceId': f"{prompt['id']}|{prompt_version}",
        'Data': {
            'PromptId': prompt['id'],
            'VersionId': prompt_version
        }
    }

def on_delete(event):
    physical_id = event['PhysicalResourceId']
    prompt_id, version_id = physical_id.split('|')
    bedrock = boto3.client('bedrock-agent')

    try:
        bedrock.delete_prompt(promptIdentifier=prompt_id)
    except:
        pass
    
    return {
        'PhysicalResourceId': physical_id
    }