import os, json, time, base64
from aws_cdk import (
    # Duration,
    Stack,
    CfnOutput,
    aws_iam as _iam,
    aws_s3 as _s3,
    aws_s3_assets as _s3_assets,
    aws_ec2 as _ec2,
    aws_rds as _rds,
    aws_events as _events,
    aws_events_targets as _events_targets,
    aws_lambda as _lambda,
    aws_apigateway as _apigw,
    aws_apigatewayv2 as _apigw2,
    aws_ecs as _ecs,
    aws_logs as _logs,
    aws_ssm as _ssm,
    aws_kms as _kms,
    aws_stepfunctions as _sfn,
    aws_stepfunctions_tasks as _sfn_tasks,
    aws_codebuild as _codebuild,
    aws_codepipeline as _codepipeline,
    aws_codepipeline_actions as _codepipeline_actions,
    aws_amplify_alpha as _amplify,
    aws_cognito as _cognito,
    aws_bedrock as _bedrock,
    aws_secretsmanager as _secretsmanager,
    custom_resources as _custom_resources,
    Duration, CfnOutput, BundlingOptions, RemovalPolicy, CustomResource, Aspects, Size
)
from constructs import Construct
from aws_cdk.custom_resources import Provider
from aws_cdk.aws_ecr_assets import DockerImageAsset
from cdk_nag import AwsSolutionsChecks, NagSuppressions



model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
vqa_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
frame_interval = "1000" # milliseconds
fast_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
balanced_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
embedding_model_id = "cohere.embed-multilingual-v3"
raw_folder = "source"
summary_folder = "summary"
video_script_folder = "video_timeline"
video_caption_folder = "captions"
transcription_root_folder = "audio_transcript"
transcription_folder = f"{transcription_root_folder}/{raw_folder}"
entity_sentiment_folder = "entities"
database_name = "videos"
video_table_name = "videos"
entities_table_name = "entities"
content_table_name = "content"
embedding_dimension = 1024
video_search_by_summary_acceptable_embedding_distance = 0.50 # Using cosine distance
videos_api_resource = "videos"
visual_objects_detection_confidence_threshold = 30.0
visual_extraction_prompt_name = "vus-visual-extraction-prompt"
visual_extraction_prompt_variant_name = "claude3"
visual_extraction_prompt_version_description = "Default version"

BASE_DIR = os.getcwd()
CONFIG_LABEL_DETECTION_ENABLED = "label_detection_enabled" # Value is "1" or "0"
CONFIG_TRANSCRIPTION_ENABLED = "transcription_enabled" # Value is "1" or "0"
CONFIG_VISUAL_EXTRACTION_PROMPT = "visual_extraction_prompt" # Value is a JSON

with open('./lib//main_analyzer/default_visual_extraction_system_prompt.txt', 'r') as file:
    default_visual_extraction_system_prompt = file.read()
with open('./lib/main_analyzer/default_visual_extraction_task_prompt.txt', 'r') as file:
    default_visual_extraction_task_prompt = file.read()

class VideoUnderstandingSolutionStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        aws_region = Stack.of(self).region
        aws_account_id = Stack.of(self).account
        admin_email_address = self.node.try_get_context("email") 

        Aspects.of(self).add(AwsSolutionsChecks(verbose=True))

        # VPC
        vpc = _ec2.Vpc(self, f"Vpc",
          ip_addresses         = _ec2.IpAddresses.cidr("10.120.0.0/16"),
          max_azs              = 3,
          enable_dns_support   = True,
          enable_dns_hostnames = True,
          nat_gateways=1,
          vpc_name=f"{construct_id}-VPC",
          subnet_configuration = [
            _ec2.SubnetConfiguration(
              cidr_mask   = 24,
              name        = 'public',
              subnet_type = _ec2.SubnetType.PUBLIC,
            ),
            _ec2.SubnetConfiguration(
              cidr_mask   = 24,
              name        = 'private_with_egress',
              subnet_type = _ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
            _ec2.SubnetConfiguration(
              cidr_mask   = 24,
              name        = 'private',
              subnet_type = _ec2.SubnetType.PRIVATE_ISOLATED,
            )
          ]
        )
        vpc.add_flow_log("FlowLogS3")
        private_subnets =  _ec2.SubnetSelection(subnet_type=_ec2.SubnetType.PRIVATE_ISOLATED)
        private_with_egress_subnets = _ec2.SubnetSelection(subnet_type=_ec2.SubnetType.PRIVATE_WITH_EGRESS)

        # Parameter store for storing application configuration
        default_configuration_parameters = {
            CONFIG_LABEL_DETECTION_ENABLED: "1",
            CONFIG_TRANSCRIPTION_ENABLED: "1",
            CONFIG_VISUAL_EXTRACTION_PROMPT: {
                "prompt_id": "",  # Will be updated by custom resource
                "variant_name": "claude3",  # Default variant
                "version_id": ""  # Will be updated by custom resource
            }
        }
        default_configuration_parameters_json = json.dumps(default_configuration_parameters)
        configuration_parameters_ssm = _ssm.StringParameter(self, f"{construct_id}-configuration-parameters",
            parameter_name = f"{construct_id}-configuration",
            string_value = default_configuration_parameters_json
        )

        # S3 - Video bucket
        video_bucket_s3 = _s3.Bucket(
            self, 
            "video-understanding", 
            event_bridge_enabled=True,
            block_public_access=_s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            #auto_delete_objects=True,
            cors=[_s3.CorsRule(
                allowed_headers=["*"],
                allowed_methods=[_s3.HttpMethods.PUT, _s3.HttpMethods.GET, _s3.HttpMethods.HEAD, _s3.HttpMethods.POST, _s3.HttpMethods.DELETE],
                allowed_origins=["*"],
                exposed_headers=["x-amz-server-side-encryption",
                  "x-amz-request-id",
                  "x-amz-id-2",
                  "ETag",
                  "x-amz-meta-foo"],
                max_age=3000
            )],
            server_access_logs_prefix="access_logs/",
            enforce_ssl=True     
        )

        # Add suppressions for using AWSLambdaBasicExecutionRole service managed role and for using * in the policy for bucket notification as both are managed by CDK
        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role", [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow the bucket notification to use AWSLambdaBasicExecutionRole service managed role'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow using * in the policy for bucket notification as this is managed by CDK. '}
        ], True)

        # EventBridge - Rule
        new_video_uploaded_rule = _events.Rule(
            self,
            f"{construct_id}-new-video-uploaded",
            event_pattern=_events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": _events.Match.exact_string(video_bucket_s3.bucket_name)
                    },
                    "object": {
                        "key": [{ "wildcard": f"{raw_folder}/*"}]
                    },
                },
            ),
        )

        # Suppress cdk_nag rule to allow for using AWSLambdaBasicExecutionRole managed role/policy and for using <arn>* in the IAM policy since video file names can vary
        NagSuppressions.add_resource_suppressions(new_video_uploaded_rule, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWSLambdaBasicExecutionRole managed role/policy'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow using <arn>* in the policy since video file names can vary'}
        ], True)

        # Prompt storage infrastructure
        prompt_setup_role = _iam.Role(
            id="PromptSetupRole",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-prompt-setup",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "PromptSetupPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "bedrock:CreatePrompt",
                                "bedrock:DeletePrompt",
                                "bedrock:CreatePromptVersion"
                            ],
                            resources=["*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=[
                                "ssm:GetParameter",
                                "ssm:PutParameter"
                            ],
                            resources=[configuration_parameters_ssm.parameter_arn],
                            effect=_iam.Effect.ALLOW,
                        )
                    ]
                )
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        NagSuppressions.add_resource_suppressions(prompt_setup_role, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWSLambdaBasicExecutionRole managed role'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow using * for Bedrock prompt management APIs'}
        ], True)

        prompt_setup_lambda = _lambda.Function(
            scope=self,
            id="PromptSetupLambda",
            function_name=f"{construct_id}-prompt-setup",
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset("./lib/prompt_setup_lambda",
                bundling= BundlingOptions(
                    image= _lambda.Runtime.PYTHON_3_13.bundling_image,
                    command= [
                        'bash',
                        '-c',
                        'pip install --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt -t /asset-output && cp -au . /asset-output',
                    ],
                )
            ),
            handler="index.on_event",
            role=prompt_setup_role,
            timeout=Duration.minutes(5),
        )

        prompt_setup_provider = Provider(
            scope=self,
            id="PromptSetupProvider",
            on_event_handler=prompt_setup_lambda,
        )

        NagSuppressions.add_resource_suppressions(prompt_setup_provider, [
            { "id": 'AwsSolutions-L1', "reason": "The Lambda function's runtime is managed by CDK Provider. Solution is to update CDK version."},
            { "id": 'AwsSolutions-IAM4', "reason": 'The Lambda function is managed by Provider'},
            { "id": 'AwsSolutions-IAM5', "reason": 'The Lambda function is managed by Provider.'}
        ], True)

        prompt_setup_custom_resource = CustomResource(
            scope=self,
            id="PromptSetup",
            service_token=prompt_setup_provider.service_token,
            properties={
                "visual_extraction_prompt_name": visual_extraction_prompt_name,
                "visual_extraction_prompt_version_description": visual_extraction_prompt_version_description,
                "visual_extraction_prompt_description": "The system prompt used to instruct the model to extract information from the video frames",
                "visual_extraction_prompt_variant_name": visual_extraction_prompt_variant_name,
                "visual_extraction_system_prompt_text": base64.b64encode(default_visual_extraction_system_prompt.encode('utf-8')).decode('utf-8'),
                "visual_extraction_task_prompt_text": base64.b64encode(default_visual_extraction_task_prompt.encode('utf-8')).decode('utf-8'),
                "configuration_parameter_name": configuration_parameters_ssm.parameter_name
            }
        )

        # Subnet group
        db_subnet_group = _rds.SubnetGroup(self, f"DBSubnetGroup",
           vpc = vpc,
           description = "Aurora subnet group",
           vpc_subnets = private_subnets,
           subnet_group_name= "Aurora subnet group"
        )

        # Creating security group to be used by Lambda function that rotates DB secret
        secret_rotation_security_group = _ec2.SecurityGroup(self, 'RotateDBSecurityGroup',
            security_group_name = f"{construct_id}-vectorDB-secret-rotator",
            vpc = vpc,
            allow_all_outbound = True,
            description = "Security group for Lambda function that rotates DB secret",
        )

        # Creating the database security group
        db_security_group = _ec2.SecurityGroup(self, "VectorDBSecurityGroup",
            security_group_name = f"{construct_id}-vectorDB",
            vpc = vpc,
            allow_all_outbound = True,
            description = "Security group for Aurora Serverless PostgreSQL",
        )

        db_security_group.add_ingress_rule(
            peer =db_security_group,
            connection =_ec2.Port(protocol=_ec2.Protocol("ALL"), string_representation="ALL"),
            description="Any in connection from self"
        )
        
        db_security_group.add_ingress_rule(
            peer =_ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection =_ec2.Port(protocol=_ec2.Protocol("TCP"), from_port=5432, to_port=5432, string_representation="tcp5432 PostgreSQL"),
            description="Postgresql port in from within the VPC"
        )

        db_security_group.add_ingress_rule(
          peer= _ec2.Peer.security_group_id(secret_rotation_security_group.security_group_id),
          connection=_ec2.Port.tcp(5432),
          description="Allow DB access from Lambda Functions that rotate Secrets"
        )

        # Create secret for use with Aurora Serverless
        aurora_cluster_username="clusteradmin"
        aurora_cluster_secret = _secretsmanager.Secret(self, "AuroraClusterCredentials",
          secret_name = f"{construct_id}-vectorDB-creds",
          description = "Aurora Cluster Credentials",
          generate_secret_string=_secretsmanager.SecretStringGenerator(
            exclude_characters ="\"@/\\ '",
            generate_string_key ="password",
            password_length =30,
            secret_string_template=json.dumps(
              {
                "username": aurora_cluster_username,
                "engine": "postgres"
              })),
        )
        aurora_cluster_secret.add_rotation_schedule(
          id="1",
          automatically_after=Duration.days(30),
          hosted_rotation=_secretsmanager.HostedRotation.postgre_sql_single_user(
            security_groups=[secret_rotation_security_group],
            vpc=vpc,
            vpc_subnets=private_with_egress_subnets
          )
        )
        self.db_secret_arn = aurora_cluster_secret.secret_full_arn
        self.db_secret_name = aurora_cluster_secret.secret_name
        aurora_cluster_credentials = _rds.Credentials.from_secret(aurora_cluster_secret, aurora_cluster_username)
        
        # Provisioning the Aurora Serverless database
        aurora_cluster = _rds.DatabaseCluster(self, f"{construct_id}AuroraDatabase",
          credentials= aurora_cluster_credentials,
          iam_authentication=True,
          engine= _rds.DatabaseClusterEngine.aurora_postgres(version=_rds.AuroraPostgresEngineVersion.VER_15_5),
          writer=_rds.ClusterInstance.serverless_v2("writer"),
          readers=[
            _rds.ClusterInstance.serverless_v2("reader1",  scale_with_writer=True),
          ],
          serverless_v2_min_capacity=0.5,
          serverless_v2_max_capacity=1,
          default_database_name=database_name,
          security_groups=[db_security_group],
          vpc=vpc,
          subnet_group=db_subnet_group,
          storage_encrypted=True,
          deletion_protection=True,
        )
        
        self.db_writer_endpoint = aurora_cluster.cluster_endpoint
        self.db_reader_endpoint = aurora_cluster.cluster_read_endpoint
        
        # Lambda for setting up database
        db_setup_event_handler = _lambda.Function(self, "DatabaseSetupHandler",
            function_name=f"{construct_id}-DatabaseSetupHandler",
            runtime=_lambda.Runtime.PYTHON_3_13,
            timeout=Duration.minutes(15),
            code=_lambda.Code.from_asset('./lib/db_setup_lambda',
                bundling= BundlingOptions(
                  image= _lambda.Runtime.PYTHON_3_13.bundling_image,
                  command= [
                    'bash',
                    '-c',
                    'pip install --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt -t /asset-output && cp -au . /asset-output',
                  ],
                )
            ),
            # architecture = _lambda.Architecture.ARM_64,
            environment = {
                'DB_WRITER_ENDPOINT': self.db_writer_endpoint.hostname,
                'DATABASE_NAME': database_name,
                'VIDEO_TABLE_NAME': video_table_name,
                'ENTITIES_TABLE_NAME': entities_table_name,
                'CONTENT_TABLE_NAME': content_table_name,
                'SECRET_NAME': self.db_secret_name,
                "EMBEDDING_DIMENSION": str(embedding_dimension)
            },
            vpc=vpc,
            vpc_subnets=private_with_egress_subnets,
            handler='index.on_event'
        )

        # Suppress CDK nag rule to allow the use of AWS managed policies/roles AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole
        NagSuppressions.add_resource_suppressions(db_setup_event_handler, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow the use of AWS managed policies/roles AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole'},
        ], True)
        
        # IAM Policy statement for the Lambda function that configures the database
        statement = _iam.PolicyStatement()
        statement.add_actions("secretsmanager:GetSecretValue")
        statement.add_resources(aurora_cluster_secret.secret_full_arn)
        db_setup_event_handler.add_to_role_policy(statement)
 
 
        provider = Provider(self, f'{construct_id}DatabaseSetupProvider', 
                    on_event_handler=db_setup_event_handler)

        
        # Suppress cdk_nag rule for using not the latest runtime for non container Lambda, as this is managed by CDK Provider.
        # Also suppress it for using * in IAM policy and for using managed policy, as this is managed by CDK Provider.
        NagSuppressions.add_resource_suppressions(provider, [
            { "id": 'AwsSolutions-L1', "reason": "The Lambda function's runtime is managed by CDK Provider. Solution is to update CDK version."},
            { "id": 'AwsSolutions-IAM4', "reason": 'The Lambda function is managed by Provider'},
            { "id": 'AwsSolutions-IAM5', "reason": 'The Lambda function is managed by Provider.'}
        ], True)


        db_setup_custom_resource = CustomResource(
            scope=self,
            id='DatabaseSetup',
            service_token=provider.service_token,
            removal_policy=RemovalPolicy.DESTROY,
            resource_type="Custom::DatabaseSetupCustomResource"
        )

        db_setup_custom_resource.node.add_dependency(aurora_cluster) 

        # Role for the preprocessing Lambda
        preprocessing_lambda_role = _iam.Role(
            id="PreprocessingLambdaRole",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-preprocessing-lambda",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "PreprocessingLambdaPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["secretsmanager:GetSecretValue"],
                            resources=[aurora_cluster_secret.secret_full_arn],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["ssm:GetParameter"],
                            resources=[configuration_parameters_ssm.parameter_arn],
                            effect=_iam.Effect.ALLOW
                        )
                    ]
                ),
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                _iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
            ],
        )


        # Suppress cdk_nag it for using * in IAM policy as reasonable in the resources and for using AWSLambdaBasicExecutionRole  and AWSLambdaVPCAccessExecutionRole managed role by AWS.
        NagSuppressions.add_resource_suppressions(preprocessing_lambda_role, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow to use AWSLambdaBasicExecutionRole and AWSLambdaVPCAccessExecutionRole AWS managed service role'},
        ], True)
        

        # Lambda function to run the video preprocessing task
        preprocessing_lambda = _lambda.Function(self, "PreprocessingLambda",
            function_name=f"{construct_id}-preprocessing",
            handler='index.handler',
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset('./lib/preprocessing_lambda', 
                bundling= BundlingOptions(
                    image= _lambda.Runtime.PYTHON_3_13.bundling_image,
                    command= [
                    'bash',
                    '-c',
                    'pip install --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt -t /asset-output && cp -au . /asset-output',
                    ],
                )),
            role=preprocessing_lambda_role,                                    
            timeout=Duration.minutes(5),
            memory_size=128,
            vpc=vpc,
            vpc_subnets=private_with_egress_subnets,
            environment = {
                'DB_WRITER_ENDPOINT': self.db_writer_endpoint.hostname,
                'DATABASE_NAME': database_name,
                'VIDEO_TABLE_NAME': video_table_name,
                'SECRET_NAME': self.db_secret_name,
                'CONFIGURATION_PARAMETER_NAME': configuration_parameters_ssm.parameter_name,
                'EMBEDDING_DIMENSION': str(embedding_dimension)
            }
        )
        # Add provisioned concurrency configuration
        preprocessing_lambda_version = preprocessing_lambda.current_version
        preprocessing_lambda_alias = _lambda.Alias(
            self,
            "PreprocessingLambdaAlias",
            alias_name="dev",
            version=preprocessing_lambda_version
        )
        # Configure provisioned concurrency
        preprocessing_lambda_alias.add_auto_scaling(
            min_capacity=1,
            max_capacity=1
        )

        preprocessing_task = _sfn_tasks.LambdaInvoke(self, "CallPreprocessingLambda",
            lambda_function=preprocessing_lambda,
            result_path="$.preprocessingResult",
        )

        preprocessing_task.add_retry(max_attempts=50, backoff_rate=3, interval=Duration.seconds(5))

        # Step function task to start the Rekognition label detection task to detect visual scenes
        start_rekognition_label_detection_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "StartRekognitionLabelDetectionSfnTask",
            service="rekognition",
            action="startLabelDetection",
            parameters={
                "Video": {
                    "S3Object": {
                        "Bucket": video_bucket_s3.bucket_name,
                        "Name.$": "$.videoS3Path",
                    }
                },
                "MinConfidence": visual_objects_detection_confidence_threshold,
            },
            result_path="$.startLabelDetectionResult",
            iam_resources=["*"],
            additional_iam_statements=[
                _iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"], 
                    resources=[
                        f"arn:aws:s3:::{video_bucket_s3.bucket_name}", 
                        f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{raw_folder}/*"
                    ]
                )
            ],
        )

        # Step function task to check the status of the Rekognition label detection task to detect visual scenes
        get_rekognition_label_detection_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "GetRekognitionLabelDetectionSfnTask",
            service="rekognition",
            action="getLabelDetection",
            parameters={
                "JobId.$": "$.startLabelDetectionResult.JobId",
                "MaxResults": 1
            },
            result_path="$.labelDetectionResult",
            result_selector={
                "JobId.$": "$.JobId",
                "JobStatus.$": "$.JobStatus"
            },
            iam_resources=["*"],
        )

        label_detection_success = _sfn.Succeed(self, "Label detection is successful")
        label_detection_skipped = _sfn.Pass(self, "Label detection is skipped", result_path="$.labelDetectionResult", result=_sfn.Result.from_object({"JobId": ""}))
        label_detection_failure = _sfn.Pass(self, "Label detection is failed, but continuing anyway.")
        label_detection_choice = _sfn.Choice(self, "Label detection choice")
        label_detection_success_condition = _sfn.Condition.string_equals("$.labelDetectionResult.JobStatus", "SUCCEEDED")
        label_detection_failure_condition = _sfn.Condition.string_equals("$.labelDetectionResult.JobStatus", "FAILED")
        label_detection_wait = _sfn.Wait(self, "Label detection wait",time=_sfn.WaitTime.duration(Duration.seconds(30))).next(get_rekognition_label_detection_sfn_task)

        # Build the flow
        start_rekognition_label_detection_sfn_task.next(get_rekognition_label_detection_sfn_task).next(label_detection_choice)
        label_detection_choice.when(label_detection_success_condition, label_detection_success).when(label_detection_failure_condition,label_detection_failure).otherwise(label_detection_wait)

        # Step function task to start the Transcribe transcription task to extract human voice and transcribe it
        start_transcription_job_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "StartTranscriptionJobSfnTask",
            service="transcribe",
            action="startTranscriptionJob",
            parameters={
                "Media": {
                    "MediaFileUri.$": f"States.Format('s3://{video_bucket_s3.bucket_name}/{{}}', $.videoS3Path)"
                },
                "TranscriptionJobName.$": "$.eventId",
                "MediaFormat": "mp4",
                "OutputBucketName": video_bucket_s3.bucket_name,
                "OutputKey.$": f"States.Format('{transcription_root_folder}/{{}}.txt', $.videoS3Path)",
                "Settings": {
                    "ShowSpeakerLabels": True,    
                    "MaxSpeakerLabels": 10
                },
                "IdentifyMultipleLanguages": True
            },
            result_path="$.startTranscriptionResult",
            iam_resources=["*"],
            additional_iam_statements=[
                _iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"], 
                    resources=[
                        f"arn:aws:s3:::{video_bucket_s3.bucket_name}", 
                        f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{raw_folder}/*"
                    ]
                ),
                _iam.PolicyStatement(
                    actions=["s3:PutObject"], resources=[f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{transcription_folder}/*"]
                )
            ],
        )

        # Step function task to check the status of the Transcribe job to transcribe human voice
        get_transcription_job_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "GetTranscriptionJobSfnTask",
            service="transcribe",
            action="getTranscriptionJob",
            parameters={
                "TranscriptionJobName.$": "$.startTranscriptionResult.TranscriptionJob.TranscriptionJobName"
            },
            result_path="$.transcriptionResult",
            result_selector={
                "TranscriptionJobName.$": "$.TranscriptionJob.TranscriptionJobName",
                "TranscriptionJobStatus.$": "$.TranscriptionJob.TranscriptionJobStatus"
            },
            iam_resources=[f"arn:aws:transcribe:{aws_region}:{aws_account_id}:transcription-job/*"],
        )

        transcription_success = _sfn.Succeed(self, "Transcription is successful")
        transcription_failure = _sfn.Pass(self, "Transcription is failed, but continuing anyway.")
        transcription_skipped = _sfn.Pass(self, "Transcription is skipped", result_path="$.transcriptionResult", result=_sfn.Result.from_object({"TranscriptionJobName": ""}))
        transcription_choice = _sfn.Choice(self, "Transcription choice")
        transcription_success_condition = _sfn.Condition.string_equals("$.transcriptionResult.TranscriptionJobStatus", "COMPLETED")
        transcription_failure_condition = _sfn.Condition.string_equals("$.transcriptionResult.TranscriptionJobStatus", "FAILED")
        transcription_wait = _sfn.Wait(self, "Transcription wait",time=_sfn.WaitTime.duration(Duration.seconds(30))).next(get_transcription_job_sfn_task)

        # Build the flow
        start_transcription_job_sfn_task.next(get_transcription_job_sfn_task).next(transcription_choice)
        transcription_choice.when(transcription_success_condition, transcription_success).when(transcription_failure_condition, transcription_failure).otherwise(transcription_wait)

        # Define the parallel tasks for Rekognition and Transcribe.
        parallel_sfn = _sfn.Parallel(self, "StartVideoAnalysisParallelSfn")

        start_label_detection_choice = _sfn.Choice(self, "StartLabelDetectionChoice")
        start_label_detection_choice.when(_sfn.Condition.string_equals(f"$.preprocessingResult.Payload.body.{CONFIG_LABEL_DETECTION_ENABLED}", "1"), start_rekognition_label_detection_sfn_task).otherwise(label_detection_skipped)
        parallel_sfn = parallel_sfn.branch(start_label_detection_choice)

        start_transcription_choice = _sfn.Choice(self, "StartTranscriptionChoice")
        start_transcription_choice.when(_sfn.Condition.string_equals(f"$.preprocessingResult.Payload.body.{CONFIG_TRANSCRIPTION_ENABLED}", "1"), start_transcription_job_sfn_task).otherwise(transcription_skipped)
        parallel_sfn = parallel_sfn.branch(start_transcription_choice)

        # Chain parallel task after preprocessing lambda
        preprocessing_task.next(parallel_sfn)

        # Role for the main video analysis ECS task execution
        main_analyzer_execution_role = _iam.Role(
            id="AnalyzerExecutionRole",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-main-analyzer-execution",
            assumed_by=_iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            inline_policies={
                "AllowECRAndLogsAccessPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "logs:CreateLogStream", "logs:PutLogEvents"],
                            resources=["*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                    ]
                )
            },
        )

        # Role for the main video analysis
        main_analyzer_role = _iam.Role(
            id="MainAnalyzerRole",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-main-analyzer",
            assumed_by=_iam.ServicePrincipal("ecs-tasks.amazonaws.com"), #_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "MainAnalyzerPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["bedrock:InvokeModel"],
                            resources=[f"arn:aws:bedrock:{aws_region}::foundation-model/*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["bedrock:GetPrompt"],
                            resources=[f"*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["rekognition:GetLabelDetection", "rekognition:GetTextDetection", "rekognition:RecognizeCelebrities", "rekognition:DetectFaces"],
                            resources=["*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["transcribe:GetTranscriptionJob"],
                            resources=[f"arn:aws:transcribe:{aws_region}:{aws_account_id}:transcription-job/*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:GetObject", "s3:ListBucket"],
                            resources=[
                                video_bucket_s3.bucket_arn, 
                                video_bucket_s3.arn_for_objects(f"{transcription_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{raw_folder}/*")
                            ],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:PutObject"],
                            resources=[
                                video_bucket_s3.arn_for_objects(f"{summary_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{video_script_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{video_caption_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{entity_sentiment_folder}/*")
                            ],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["secretsmanager:GetSecretValue"],
                            resources=[aurora_cluster_secret.secret_full_arn],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["ssm:GetParameter"],
                            resources=[configuration_parameters_ssm.parameter_arn],
                            effect=_iam.Effect.ALLOW,
                        )
                    ]
                )
            },
        )


        # Suppress cdk_nag it for using * in IAM policy as reasonable in the resources and for using AmazonECSTaskExecutionRolePolicy managed role by AWS.
        NagSuppressions.add_resource_suppressions(main_analyzer_role, [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow to use * for Rekognition read APIs which resources have to be *, and to use <arn>/* for Transcribe GetTranscriptionJob as the job name can vary'}
        ], True)

        # ECS cluster for main analyzer
        ecs_cluster = _ecs.Cluster(
            self,
            "ECSCluster",
            cluster_name=f"{construct_id}-ecs-cluster",
            enable_fargate_capacity_providers=True,
            vpc=vpc,
            container_insights=True
        )

        # Task definition for main analyzer
        analyzer_task_definition = _ecs.FargateTaskDefinition(self, "TaskDefinition",
            cpu=4096,
            memory_limit_mib=8192,
            task_role= main_analyzer_role,
            execution_role= main_analyzer_execution_role
        )

        # Log group for the container
        main_analyzer_log_group = _logs.LogGroup(self, "AnalyzerAccessLogGroup", 
            log_group_name=f"{construct_id}-analyzer",
            removal_policy=RemovalPolicy.DESTROY,
        )

        analyzer_container_definition = analyzer_task_definition.add_container("analyzer",
            image=_ecs.ContainerImage.from_docker_image_asset(
                DockerImageAsset(self, "AnalyzerImageBuild",
                    directory=f"{BASE_DIR}/lib/main_analyzer/"
                ),
            ),
            memory_limit_mib=8192,
            logging=_ecs.LogDrivers.aws_logs(
                log_group=main_analyzer_log_group,
                stream_prefix="main",
                mode=_ecs.AwsLogDriverMode.NON_BLOCKING,
                max_buffer_size=Size.mebibytes(25)
            )

        )

        main_analyzer_task = _sfn_tasks.EcsRunTask(self, "CallMainAnalyzer",
            integration_pattern=_sfn.IntegrationPattern.RUN_JOB,
            cluster=ecs_cluster,
            task_definition=analyzer_task_definition,
            assign_public_ip=True,
            launch_target=_sfn_tasks.EcsFargateLaunchTarget(
                platform_version=_ecs.FargatePlatformVersion.VERSION1_4
            ),
            subnets=private_with_egress_subnets,
            container_overrides=[_sfn_tasks.ContainerOverride(
                container_definition=analyzer_container_definition,
                environment=[
                    _sfn_tasks.TaskEnvironmentVariable(name="VIDEO_S3_PATH", value=_sfn.JsonPath.string_at("$[0].videoS3Path")),
                    _sfn_tasks.TaskEnvironmentVariable(name="LABEL_DETECTION_JOB_ID", value=_sfn.JsonPath.string_at("$[0].labelDetectionResult.JobId")),
                    _sfn_tasks.TaskEnvironmentVariable(name="TRANSCRIPTION_JOB_NAME", value=_sfn.JsonPath.string_at("$[1].transcriptionResult.TranscriptionJobName")),
                    _sfn_tasks.TaskEnvironmentVariable(name=CONFIG_LABEL_DETECTION_ENABLED, value=_sfn.JsonPath.string_at(f"$[0].preprocessingResult.Payload.body.{CONFIG_LABEL_DETECTION_ENABLED}")),
                    _sfn_tasks.TaskEnvironmentVariable(name=CONFIG_TRANSCRIPTION_ENABLED, value=_sfn.JsonPath.string_at(f"$[0].preprocessingResult.Payload.body.{CONFIG_TRANSCRIPTION_ENABLED}")),
                    _sfn_tasks.TaskEnvironmentVariable(name='CONFIG_PARAMETER_NAME', value= configuration_parameters_ssm.parameter_name),
                    _sfn_tasks.TaskEnvironmentVariable(name='DATABASE_NAME', value= database_name),
                    _sfn_tasks.TaskEnvironmentVariable(name='VIDEO_TABLE_NAME', value= video_table_name),
                    _sfn_tasks.TaskEnvironmentVariable(name='ENTITIES_TABLE_NAME', value= entities_table_name),
                    _sfn_tasks.TaskEnvironmentVariable(name='CONTENT_TABLE_NAME', value= content_table_name),
                    _sfn_tasks.TaskEnvironmentVariable(name='SECRET_NAME', value= self.db_secret_name),
                    _sfn_tasks.TaskEnvironmentVariable(name="EMBEDDING_DIMENSION", value=str(embedding_dimension)),
                    _sfn_tasks.TaskEnvironmentVariable(name='DB_WRITER_ENDPOINT', value= self.db_writer_endpoint.hostname),
                    _sfn_tasks.TaskEnvironmentVariable(name="EMBEDDING_MODEL_ID", value= embedding_model_id),
                    _sfn_tasks.TaskEnvironmentVariable(name="MODEL_ID", value= model_id),
                    _sfn_tasks.TaskEnvironmentVariable(name='VQA_MODEL_ID', value= vqa_model_id),
                    _sfn_tasks.TaskEnvironmentVariable(name="BUCKET_NAME", value= video_bucket_s3.bucket_name),
                    _sfn_tasks.TaskEnvironmentVariable(name="RAW_FOLDER", value= raw_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name="VIDEO_SCRIPT_FOLDER", value= video_script_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name="TRANSCRIPTION_FOLDER", value= transcription_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name="ENTITY_SENTIMENT_FOLDER", value= entity_sentiment_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name="SUMMARY_FOLDER", value= summary_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name="VIDEO_CAPTION_FOLDER", value= video_caption_folder),
                    _sfn_tasks.TaskEnvironmentVariable(name='FRAME_INTERVAL', value= frame_interval)
                ]
            )],
        )
        # Chain the analysis step after the parallel step
        parallel_sfn.next(main_analyzer_task)

        # CloudWatch Log Group for the Step Functions
        sfn_log_group = _logs.LogGroup(self, "SFNLogGroup")
        
        # The video analysis Step Function
        video_analysis_sfn = _sfn.StateMachine(
            self,
            "StartVideoAnalysisSfn",
            definition_body=_sfn.DefinitionBody.from_chainable(preprocessing_task),
            logs=_sfn.LogOptions(
                destination=sfn_log_group,
                level=_sfn.LogLevel.ALL
            ),
            tracing_enabled=True
        )

        # The below added policy is needed because the default policy auto-created does not have the task revision portion in the resource ARN
        ecs_run_task_policy = _iam.PolicyStatement(
            actions=["ecs:RunTask"],
            resources=[analyzer_task_definition.task_definition_arn],
            effect=_iam.Effect.ALLOW,
        )
        video_analysis_sfn.add_to_role_policy(ecs_run_task_policy)

        # Suppress cdk_nag rule for using * in IAM since the file name of the videos on S3 can be various.
        NagSuppressions.add_resource_suppressions(video_analysis_sfn, [
            { "id": 'AwsSolutions-IAM5', "reason": 'This is providing access to videos on S3 for AWS AI services and since the file name can vary, we need <Arn>/* in the IAM policy.'}
        ], True)
        
        # IAM Role for EventBridge to call Step Function
        event_bridge_role = _iam.Role(self, "EventBridgeRole",
            role_name=f"{construct_id}-{aws_region}-event-bridge",
            assumed_by=_iam.ServicePrincipal("events.amazonaws.com"),
            inline_policies={
                "EventBridgeTriggersStepFunctionPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "states:StartExecution",
                            ],
                            resources=[
                                f"arn:aws:states:::execution/{video_analysis_sfn.state_machine_name}"
                            ],
                            effect=_iam.Effect.ALLOW,
                        )
                    ]
                )
            },
        )
        
        # Add target for EventBridge to trigger Step Function for videos
        new_video_uploaded_rule.add_target(
          _events_targets.SfnStateMachine(video_analysis_sfn,
            input= _events.RuleTargetInput.from_object({
              'detailType': _events.EventField.detail_type,
              'eventId': _events.EventField.from_path('$.id'),
              'videoS3Path': _events.EventField.from_path('$.detail.object.key'),
              'videoS3BucketName': _events.EventField.from_path('$.detail.bucket.name')
            }),
            role=event_bridge_role)
        )
        # Cognito User Pool
        user_pool = _cognito.UserPool(self, "UserPool",
            user_pool_name="video-understanding-user-pool",
            standard_attributes=_cognito.StandardAttributes(
                email=_cognito.StandardAttribute(
                    required=True,
                    mutable=True
                ),
            ),
            self_sign_up_enabled=False,
            sign_in_aliases= _cognito.SignInAliases(email=True),     
            removal_policy=RemovalPolicy.DESTROY,
            mfa=_cognito.Mfa.REQUIRED,
            mfa_second_factor=_cognito.MfaSecondFactor(
                sms=False,
                otp=True
            ),
            password_policy=_cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
                temp_password_validity=Duration.days(3)
            ),
             advanced_security_mode=_cognito.AdvancedSecurityMode.ENFORCED
        )

        # Cognito User Pool Client
        user_pool_client = _cognito.UserPoolClient(self, 'UserPoolClient',
            user_pool=user_pool,
            supported_identity_providers= [
                _cognito.UserPoolClientIdentityProvider.COGNITO,
            ],
        )

        # Cognito Identity Pool
        identity_pool = _cognito.CfnIdentityPool(self, 'IdentityPool',
            identity_pool_name= 'video-understanding-identity-pool',
            allow_unauthenticated_identities= False,
            cognito_identity_providers= [_cognito.CfnIdentityPool.CognitoIdentityProviderProperty(
                client_id= user_pool_client.user_pool_client_id,
                provider_name= user_pool.user_pool_provider_name,
            )],
        )

        # Cognito Role
        auth_role = _iam.Role(self, 'CognitoAuthRole',
            role_name=f"{construct_id}-{aws_region}-cognito-auth",
            assumed_by= _iam.FederatedPrincipal(
                'cognito-identity.amazonaws.com',
                {
                  'StringEquals': {
                    'cognito-identity.amazonaws.com:aud': identity_pool.ref,
                  },
                  'ForAnyValue:StringLike': {
                    'cognito-identity.amazonaws.com:amr': 'authenticated',
                  },
                },
                'sts:AssumeRoleWithWebIdentity',
             ),
             inline_policies={
                "WebUIPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["bedrock:InvokeModelWithResponseStream", "bedrock:InvokeModel"],
                            resources=[f"arn:aws:bedrock:{aws_region}::foundation-model/*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:GetObject", "s3:ListBucket"], 
                            resources=[
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}", 
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{raw_folder}/*",
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{video_script_folder}/*",
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{video_caption_folder}/*",
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{summary_folder}/*",
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{entity_sentiment_folder}/*"
                            ]
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"], 
                            resources=[
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{raw_folder}/*",
                            ]
                        ),
                    ]
                )
            },
        )

        # Identity pool auth role attachment
        auth_role_attachment = _cognito.CfnIdentityPoolRoleAttachment(
            self,
            'IdentityPoolAuthRoleAttachment',
            identity_pool_id= identity_pool.ref,
            roles= {
                "authenticated": auth_role.role_arn,
            },
            role_mappings= {
                "mapping": _cognito.CfnIdentityPoolRoleAttachment.RoleMappingProperty(
                    type= 'Token',
                    ambiguous_role_resolution= 'AuthenticatedRole',
                    identity_provider= f"cognito-idp.{aws_region}.amazonaws.com/{user_pool.user_pool_id}:{user_pool_client.user_pool_client_id}",
                ),
            },
        )

        # Log group for API Gateway REST API
        rest_access_log_group = _logs.LogGroup(self, "RestAPIAccessLogGroup", 
            log_group_name=f"{construct_id}-rest-api",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # API Gateway REST API
        rest_api = _apigw.RestApi(self, 
                            'RestAPI', 
                            rest_api_name=f'{construct_id}RestAPI',
                            deploy_options=_apigw.StageOptions(
                                logging_level=_apigw.MethodLoggingLevel.ERROR,
                                access_log_destination=_apigw.LogGroupLogDestination(rest_access_log_group),
                                access_log_format=_apigw.AccessLogFormat.clf()
                            ),
                            cloud_watch_role=True
                            )
        
        self.rest_api = rest_api
        self.rest_api_url = rest_api.url

        # Suppress CDK rule to allow using AWS managed policy AmazonAPIGatewayPushToCloudWatchLogs
        NagSuppressions.add_resource_suppressions(rest_api, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWS managed policy AmazonAPIGatewayPushToCloudWatchLogs' },
        ], True)

        # Add "videos" resource in the API
        api_resource_videos = rest_api.root.add_resource(
            videos_api_resource,
            default_cors_preflight_options=_apigw.CorsOptions(
                allow_methods=['GET', 'OPTIONS'],
                allow_origins=_apigw.Cors.ALL_ORIGINS)
        )

        # Suppress CDK rule to allow OPTIONS be called without auth header
        # Suppress CDK rule to allow OPTIONS be called without auth header
        NagSuppressions.add_resource_suppressions(api_resource_videos, [
            { "id": 'AwsSolutions-COG4', "reason": 'Allow OPTIONS to be called without auth header' },
            { "id": 'AwsSolutions-APIG4', "reason": 'Allow OPTIONS to be called without auth header' }
        ], True)

        # Role for the main video analysis
        video_search_role = _iam.Role(
            id="VideoSearchRole",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-video-search",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "VideoSearchPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["bedrock:InvokeModel"],
                            resources=[f"arn:aws:bedrock:{aws_region}::foundation-model/*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["secretsmanager:GetSecretValue"],
                            resources=[aurora_cluster_secret.secret_full_arn],
                            effect=_iam.Effect.ALLOW,
                        ),
                    ]
                )
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                _iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
            ]
        )


        # Suppress CDK nag rule for using * in IAM policy since for flexibility in choosing Bedrock model.
        # Suppress CDK nag rule for using managed policy AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole
        NagSuppressions.add_resource_suppressions(video_search_role, [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow  * in the resources string for flexibility in choosing Bedrock model' },
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow  using managed policy AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole' }
        ], True)

        # Define the Lambda function for video search
        videos_search_function = _lambda.Function(self, f'SearchLambda',
            role=video_search_role,
            function_name=f"{construct_id}-videos-search",
            handler='lambda-handler.handler',
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset('./lib/videos_search',
                bundling= BundlingOptions(
                image= _lambda.Runtime.PYTHON_3_13.bundling_image,
                command= [
                    'bash',
                    '-c',
                    'pip install --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt -t /asset-output && cp -au . /asset-output',
                ],
                )
            ),  
            vpc=vpc,
            vpc_subnets=private_with_egress_subnets,
            timeout=Duration.minutes(2),
            memory_size=128,
            environment = {
                    'DB_READER_ENDPOINT': self.db_reader_endpoint.hostname,
                    'SECRET_NAME': self.db_secret_name,
                    'DATABASE_NAME': database_name,
                    'VIDEO_TABLE_NAME': video_table_name,
                    'EMBEDDING_MODEL_ID': embedding_model_id,
                    'EMBEDDING_DIMENSION': str(embedding_dimension),
                    'ACCEPTABLE_EMBEDDING_DISTANCE': str(video_search_by_summary_acceptable_embedding_distance),
                    'DISPLAY_PAGE_SIZE': str(25)
            }
        )
        # Add provisioned concurrency configuration
        videos_search_version = videos_search_function.current_version
        videos_search_alias = _lambda.Alias(
            self,
            "VideoSearchLambdaAlias",
            alias_name="dev",
            version=videos_search_version
        )
        # Configure provisioned concurrency
        videos_search_alias.add_auto_scaling(
            min_capacity=1,
            max_capacity=1
        )

        # Suppress CDK nag rule for using * in IAM policy since for flexibility in choosing Bedrock model and for calling routes in API Gateway WebSocket APIs
        # Suppress CDK nag rule for using managed policy AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole
        NagSuppressions.add_resource_suppressions(videos_search_function, [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow  * in the resources string for flexibility in choosing Bedrock model' },
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow  using managed policy AWSLambdaVPCAccessExecutionRole and AWSLambdaBasicExecutionRole' }
        ], True)
        
        # API Gateway - Lambda integration
        videos_search_function_integration = _apigw.LambdaIntegration(
            videos_search_function,
            integration_responses=[
                _apigw.IntegrationResponse(
                    status_code="200",
                    response_parameters={
                        'method.response.header.Access-Control-Allow-Origin': "'*'"
                    }
                )
            ]
        )

        # Request validator for the "GET"" method
        videos_search_request_validator = _apigw.RequestValidator(self, "VideosSearchRequestValidator",
            rest_api=rest_api,
            request_validator_name="videos-search-request-validator",
            validate_request_body=False,
            validate_request_parameters=True
        )

        # Cognito auth for the API Gateway REST API
        auth = _apigw.CognitoUserPoolsAuthorizer(self, "VideosAuthorizer",
            cognito_user_pools=[user_pool]
        )
        
        # Add "GET" method to the API "videos" resource
        api_resource_videos.add_method(
            'GET', videos_search_function_integration,
            authorizer=auth,
            authorization_type=_apigw.AuthorizationType.COGNITO,
            method_responses=[
                _apigw.MethodResponse(
                    status_code="200",
                    response_parameters={
                        'method.response.header.Access-Control-Allow-Origin': True,
                        'method.response.header.Access-Control-Allow-Headers': True,
                        'method.response.header.Access-Control-Allow-Methods': True
                    }
                )
            ],
            request_parameters={
                "method.request.querystring.page": True,
                "method.request.querystring.videoNameContains": False,
                "method.request.querystring.uploadedBetween": False,
                "method.request.querystring.about": False,
            },
            request_validator=videos_search_request_validator
        )

        """
        # API Gateway WebSocket
        ws_api = _apigw2.CfnApi(self, "WebSocketAPI",
            name=f"{construct_id}-ws-api",
            protocol_type="WEBSOCKET",
            route_selection_expression="$request.body.action"
        )
        self.ws_api = ws_api
        self.ws_api_endpoint = ws_api.attr_api_endpoint
        
        connect_route_key = "$connect"
        
        ws_connect_route = _apigw2.CfnRoute(self, "ConnectRoute",
            api_id=ws_api.attr_api_id,
            route_key=connect_route_key,
            authorization_type="AWS_IAM",
            operation_name="ConnectRoute"
        )
        """



        branch_name = "main"
        webui_path = f"{BASE_DIR}/webui/"
        webui_zip_file_name = [i for i in os.listdir(webui_path) if os.path.isfile(os.path.join(webui_path,i)) and i.startswith("ui_repo")][0]
        
        ui_asset = _s3_assets.Asset(self, "UIAsset", 
            path= f"{webui_path}{webui_zip_file_name}",
        )

        ui_artifacts_bucket = _s3.Bucket(self, "PipelineArtifactsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=_s3.BucketEncryption.S3_MANAGED,
            server_access_logs_prefix="access_logs/",
            enforce_ssl=True,
            block_public_access=_s3.BlockPublicAccess.BLOCK_ALL,
        )

        ui_artifacts_bucket.add_to_resource_policy(
            _iam.PolicyStatement(
                actions=["s3:*"],
                resources=[
                    ui_artifacts_bucket.bucket_arn,
                    f"{ui_artifacts_bucket.bucket_arn}/*"
                ],
                principals=[_iam.ServicePrincipal("amplify.amazonaws.com")]
            )
        )

        build_project = _codebuild.Project(self, "UIBuild",
            environment=_codebuild.BuildEnvironment(
                build_image=_codebuild.LinuxBuildImage.STANDARD_7_0
            ),
            build_spec = _codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "pre_build": {
                        "commands": [
                            'node -e "console.log(\'Currently running Node.js \' + process.version)"',
                            "npm ci"
                        ]
                    },
                    "build": {
                        "commands": [
                            'sed -i "s/PLACEHOLDER_REGION/$REGION/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_COGNITO_REGION/$REGION/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_IDENTITY_POOL_ID/$AMPLIFY_IDENTITYPOOL_ID/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_USER_POOL_ID/$AMPLIFY_USERPOOL_ID/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_USER_POOL_WEB_CLIENT_ID/$AMPLIFY_WEBCLIENT_ID/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_BUCKET_NAME/$BUCKET_NAME/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_FAST_MODEL_ID/$FAST_MODEL_ID/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_BALANCED_MODEL_ID/$BALANCED_MODEL_ID/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_RAW_FOLDER/$RAW_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_VIDEO_SCRIPT_FOLDER/$VIDEO_SCRIPT_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_VIDEO_CAPTION_FOLDER/$VIDEO_CAPTION_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_TRANSCRIPTION_FOLDER/$TRANSCRIPTION_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_ENTITY_SENTIMENT_FOLDER/$ENTITY_SENTIMENT_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_SUMMARY_FOLDER/$SUMMARY_FOLDER/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_REST_API_URL/$REST_API_URL/g" ./src/aws-exports.js',
                            'sed -i "s/PLACEHOLDER_VIDEOS_API_RESOURCE/$VIDEOS_API_RESOURCE/g" ./src/aws-exports.js',
                            "npm run build"
                        ]
                    }
                },
                "artifacts": {
                    "base-directory": "build",
                    "files": [
                        "**/*"
                    ]
                }
            }),
            environment_variables={
                "AMPLIFY_USERPOOL_ID": _codebuild.BuildEnvironmentVariable(value=user_pool.user_pool_id),
                "AMPLIFY_WEBCLIENT_ID": _codebuild.BuildEnvironmentVariable(value=user_pool_client.user_pool_client_id),
                "REGION": _codebuild.BuildEnvironmentVariable(value=aws_region),
                "AMPLIFY_IDENTITYPOOL_ID": _codebuild.BuildEnvironmentVariable(value=identity_pool.ref),
                "BUCKET_NAME": _codebuild.BuildEnvironmentVariable(value=video_bucket_s3.bucket_name),
                "FAST_MODEL_ID": _codebuild.BuildEnvironmentVariable(value=fast_model_id),
                "BALANCED_MODEL_ID": _codebuild.BuildEnvironmentVariable(value=balanced_model_id),
                "RAW_FOLDER": _codebuild.BuildEnvironmentVariable(value=raw_folder),
                "VIDEO_SCRIPT_FOLDER": _codebuild.BuildEnvironmentVariable(value=video_script_folder),
                "VIDEO_CAPTION_FOLDER": _codebuild.BuildEnvironmentVariable(value=video_caption_folder),
                "TRANSCRIPTION_FOLDER": _codebuild.BuildEnvironmentVariable(value=transcription_folder.replace("/","\/")),
                "ENTITY_SENTIMENT_FOLDER": _codebuild.BuildEnvironmentVariable(value=entity_sentiment_folder),
                "SUMMARY_FOLDER": _codebuild.BuildEnvironmentVariable(value=summary_folder),
                "REST_API_URL": _codebuild.BuildEnvironmentVariable(value=self.rest_api_url.replace("/","\/")),
                "VIDEOS_API_RESOURCE": _codebuild.BuildEnvironmentVariable(value=videos_api_resource)
            }
        )

        # Suppress cdk_nag rule to allow * in the CodeBuild role
        NagSuppressions.add_resource_suppressions(build_project, [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow to use * in the IAM policies for the default CodeBuild role.'}
        ], True)
         # Suppress cdk_nag rule to allow using default S3 encryption on the build arfifact rather than using custom KMS key
        NagSuppressions.add_resource_suppressions(build_project, [
            { "id": 'AwsSolutions-CB4', "reason": 'allow using default S3 encryption on the build arfifact rather than using custom KMS key.'}
        ], True)

        ui_amplify_app = _amplify.App(self, "VideoUnderstandingSolutionUIApp")
        master_branch = ui_amplify_app.add_branch(branch_name)

        amplify_deploy_role = _iam.Role(
            self, "AmplifyDeployRole",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "AmplifyDeployPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "amplify:StartDeployment",
                            ],
                            resources=[
                                f"arn:aws:amplify:{aws_region}:{aws_account_id}:apps/{ui_amplify_app.app_id}/branches/{master_branch.branch_name}/*",
                            ]
                        )
                    ]
                ),
                "AmplifyDeployAccessS3": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "s3:PutObject",
                                "s3:GetObjectAcl",
                                "s3:GetObject",
                                "s3:PutObjectVersionAcl",
                                "s3:ListBucket",
                                "s3:DeleteObject",
                                "s3:PutObjectAcl"
                            ],
                            resources=[
                                f"{ui_artifacts_bucket.bucket_arn}/*",
                                f"{ui_artifacts_bucket.bucket_arn}"
                            ]
                        )
                    ]
                ),
                "CallCodePipeline": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "codepipeline:PutJobSuccessResult",
                                "codepipeline:PutJobFailureResult"
                            ],
                            resources=["*"]
                        )
                    ]
                ),
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        NagSuppressions.add_resource_suppressions(amplify_deploy_role, [
            {
                "id": "AwsSolutions-IAM5", 
                "reason": "Amplify deploy requires access to all branches",
            }
        ])

        amplify_deploy_function = _lambda.Function(
            self, "AmplifyDeployFunction",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="index.handler",
            code=_lambda.Code.from_asset('./lib/amplify_deploy_lambda',
                bundling= BundlingOptions(
                image= _lambda.Runtime.PYTHON_3_13.bundling_image,
                command= [
                    'bash',
                    '-c',
                    'pip install --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt -t /asset-output && cp -au . /asset-output',
                ],
                )
            ),  
            role=amplify_deploy_role,
            environment={
                "AMPLIFY_APP_ID": ui_amplify_app.app_id,
                "BRANCH_NAME": branch_name
            }
        )

        pipeline = _codepipeline.Pipeline(self, "UIDeployPipeline",
            artifact_bucket=ui_artifacts_bucket,
            stages=[
                _codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        _codepipeline_actions.S3SourceAction(
                            action_name="S3Source",
                            bucket=ui_asset.bucket,
                            bucket_key=ui_asset.s3_object_key,
                            output=_codepipeline.Artifact("SourceOutput")
                        )
                    ]
                ),
                _codepipeline.StageProps(
                    stage_name="Build",
                    actions=[
                        _codepipeline_actions.CodeBuildAction(
                            action_name="BuildAction",
                            project=build_project,
                            input=_codepipeline.Artifact("SourceOutput"),
                            outputs=[_codepipeline.Artifact("BuildOutput")]
                        )
                    ]
                ),
                _codepipeline.StageProps(
                    stage_name="DeployToS3",
                    actions=[
                        _codepipeline_actions.S3DeployAction(
                            action_name="S3Deploy",
                            input=_codepipeline.Artifact("BuildOutput"),
                            bucket=ui_artifacts_bucket,
                            object_key="live",
                            extract=True
                        )
                    ]
                ),
                _codepipeline.StageProps(
                    stage_name="DeployToAmplify",
                    actions=[
                        _codepipeline_actions.LambdaInvokeAction(
                            action_name="InvokeAmplifyDeploy",
                            lambda_=amplify_deploy_function,
                            user_parameters={
                                "bucket_name": ui_artifacts_bucket.bucket_name,
                                "prefix": "live"
                            }
                        )
                    ]
                )
            ]
        )

        # Suppress cdk_nag rule to allow <Arn>* in the IAM policy as the video file names can vary
        NagSuppressions.add_resource_suppressions(auth_role, [
            { "id": 'AwsSolutions-IAM5', "reason": 'The <arn>* is needed in the IAM policy to allow variety of file names in S3 bucket.'}
        ], True)

  

        NagSuppressions.add_resource_suppressions(amplify_deploy_role, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWSLambdaBasicExecutionRole managed role'},
        ], True)

        # Role for custom resource lambda to create Cognito admin user
        cognito_user_setup_role = _iam.Role(
            id="CognitoUserSetupLambdaCR",
            scope=self,
            role_name=f"{construct_id}-{aws_region}-cognito-setup",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "CognitoUserSetupPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=[
                                "cognito-idp:AdminCreateUser",
                                "cognito-idp:DescribeUserPool",
                                "cognito-idp:UpdateUserPool",
                                "cognito-idp:AdminDeleteUser"
                            ],
                            resources=[
                                f"arn:aws:cognito-idp:{aws_region}:{aws_account_id}:userpool/{user_pool.user_pool_id}",
                                #f"arn:aws:cognito-idp:{aws_region}:{aws_account_id}:userpool/{user_pool.user_pool_id/*"
                            ],
                            effect=_iam.Effect.ALLOW,
                        )
                    ]
                )
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Suppress cdk_nag rule to allow using AWSLambdaBasicExecutionRole service managed role.
        NagSuppressions.add_resource_suppressions(cognito_user_setup_role, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allowing the use of AWSLambdaBasicExecutionRole service managed role'},
        ], True)

        cognito_user_setup_lambda = _lambda.Function(
            scope=self,
            id="CognitoUserSetupLambda",
            function_name=f"CognitoUserSetupLambda",
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset("./lib/cognito_user_setup"),
            handler="index.on_event",
            role=cognito_user_setup_role,
            timeout=Duration.minutes(5),
        )

        cognito_user_setup_provider = Provider(
            scope=self,
            id="CognitoUserSetupCRProvider",
            provider_function_name=f"CognitoUserSetupCRProvider",
            on_event_handler=cognito_user_setup_lambda,
        )

        # Suppress cdk_nag rule for using not the latest runtime for non container Lambda, as this is managed by CDK Provider.
        # Also suppress it for using * in IAM policy and for using managed policy, as this is managed by CDK Provider.
        NagSuppressions.add_resource_suppressions(cognito_user_setup_provider, [
            { "id": 'AwsSolutions-L1', "reason": "The Lambda function's runtime is managed by CDK Provider. Solution is to update CDK version."},
            { "id": 'AwsSolutions-IAM4', "reason": 'The Lambda function is managed by Provider'},
            { "id": 'AwsSolutions-IAM5', "reason": 'The Lambda function is managed by Provider.'}
        ], True)

        cognito_user_setup_custom_resource = CustomResource(
            scope=self,
            id=f"CognitoUserSetupCR",
            service_token=cognito_user_setup_provider.service_token,
            removal_policy=RemovalPolicy.DESTROY,
            resource_type="Custom::CognitoAdminUser",
            properties={
                "user_pool_id": user_pool.user_pool_id,
                "admin_email_address": admin_email_address,
                "url": f"https://{branch_name}.{ui_amplify_app.default_domain}"
            }
        )

        # Suppress CDK rule on request validator. This resource contains GET request which does not have payload. Query string validator is in place. However, the rule is activated if payload validator is not there (link to implementation https://github.com/cdklabs/cdk-nag/blob/main/src/rules/apigw/APIGWRequestValidation.ts). Hence disabling the rule.
        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/RestAPI/Resource", [
            { "id": 'AwsSolutions-APIG2', "reason": 'This resource contains GET request which does not have payload. Query string validator is in place. However, the rule is activated if payload validator is not there (link to implementation https://github.com/cdklabs/cdk-nag/blob/main/src/rules/apigw/APIGWRequestValidation.ts). Hence disabling the rule.'},
        ], True)

        # Suppress cdk_nag rule to allow using * in IAM policy for ECR and logs access
        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/AnalyzerExecutionRole/Resource", [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow to use * for ECR and logs access'}
        ], True)

        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/UIDeployPipeline/Role/DefaultPolicy/Resource", [
            { "id": "AwsSolutions-IAM5", "reason": "Pipeline requires these permissions for artifact management"}
        ])

        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/UIDeployPipeline/Source/S3Source/CodePipelineActionRole/DefaultPolicy/Resource", [
            {"id": "AwsSolutions-IAM5", "reason": "S3 Source action requires these permissions for artifact management"}
        ])

        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/UIDeployPipeline/DeployToAmplify/InvokeAmplifyDeploy/CodePipelineActionRole/DefaultPolicy/Resource", [
            {"id": "AwsSolutions-IAM5", "reason": "Amplify Deploy action requires these permissions for artifact management"}
        ])

        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/UIDeployPipeline/DeployToS3/S3Deploy/CodePipelineActionRole/DefaultPolicy/Resource", [
            {"id": "AwsSolutions-IAM5", "reason": "S3 Deploy action requires these permissions for artifact management"}
        ])

        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/AmplifyDeployRole/DefaultPolicy/Resource", [
            {"id": "AwsSolutions-IAM5", "reason": "Allowing S3 access for Lambda to deploy to Amplify with the wildcard use"}
        ])

        CfnOutput(self, "bucket_name", value=video_bucket_s3.bucket_name)
        CfnOutput(self, "web_portal_url", value=f"https://{branch_name}.{ui_amplify_app.default_domain}")