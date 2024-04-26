import os, json
import boto3

bedrock = boto3.client("bedrock-runtime")

def handler(event, context):
    print(event)

    mode = "rest"
    if ('requestContext' in event) and ('routeKey' in event['requestContext']): mode = "websocket"
    
    event_body = event['body']
    if len(event_body) > 200000:
        return {
            "statusCode": 400,
            'body': 'Event body is too large'
        }
    # Disabling semgrep rule for checking data size to be loaded to JSON as the check is already done right above.
    # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
    event_body = json.loads(event_body)
    input_text = event_body['text']
    
    llm_parameters = {} #TODO
    # Merge prompt with the LLM parameters
    llm_parameters['prompt'] = prompt
    body = json.dumps(llm_parameters)
    
    # Get the recommended item text from LLM
    response = bedrock.invoke_model(body=body, modelId=ssm_recommendation_parameters['model_id'])
    
    # Post-process suggested item types where it can be more than 1.
    # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
    # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
    recommended_item_types = json.loads(response.get("body").read())["completion"]
    recommended_item_types = recommended_item_types.split("###") if "\n###" in recommended_item_types else [recommended_item_types]
    recommended_item_types = list(filter(lambda x: x != '' and not x.isspace(), recommended_item_types))
    
    recommended_item_embeddings = []
    
    # Call the text-to-embedding model to get the embedding for each of the suggested item types.
    for item_type in recommended_item_types:
        # Get the embedding of the recommended item text
        body = json.dumps(
            {
                "inputText": item_type,
            }
        )

        response = bedrock.invoke_model(body=body, modelId="amazon.titan-embed-text-v1")
        # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
        # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
        recommended_item_embeddings.append(json.loads(response.get("body").read())["embedding"])

    recommended_items = []
    
    # Do search on vector database 
    try:
        for embedding in recommended_item_embeddings:
            recommended_items = recommended_items + db.search(query_template, 
                                          embedding, 
                                          num_items=ssm_recommendation_parameters['num_items'])
    except Exception as e:
        print("An exception happened when doing the search on the vector database")
        print(e)
    finally: 
        db.close_connection()

    # Deduplicate
    final_recommended_items = {}
    for item in sorted(recommended_items, key = lambda k: k["distance"]):
        if item['id'] not in final_recommended_items: final_recommended_items[item['id']] = item

    final_recommended_items = list({'id': v[1]['id'], 'distance': v[1]['distance'], 'description': v[1]['description']} for v in final_recommended_items.items())
    
    if mode == "websocket":
        domain = event['requestContext']['domainName']
        stage = event['requestContext']['stage']
        connection_id = event['requestContext']['connectionId']
        callback_url = f"https://{domain}/{stage}"
        apigw = boto3.client('apigatewaymanagementapi', endpoint_url= callback_url)

        response = apigw.post_to_connection(
            Data=bytes(json.dumps({
                "items": final_recommended_items
            }), "utf-8"),
            ConnectionId=connection_id
        )
        return {
            "statusCode": 200
        }
    
    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "items": final_recommended_items
        })
    }

    return response