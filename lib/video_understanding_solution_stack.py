import os
from aws_cdk import (
    # Duration,
    Stack,
    CfnOutput,
    aws_iam as _iam,
    aws_s3 as _s3,
    aws_ec2 as _ec2,
    aws_rds as _rds,
    aws_events as _events,
    aws_events_targets as _events_targets,
    aws_lambda as _lambda,
    aws_logs as _logs,
    aws_stepfunctions as _sfn,
    aws_stepfunctions_tasks as _sfn_tasks,
    aws_codecommit as _codecommit,
    aws_amplify_alpha as _amplify,
    aws_cognito as _cognito,
    aws_secretsmanager as _secretsmanager,
    custom_resources as _custom_resources,
    Duration, CfnOutput, BundlingOptions, RemovalPolicy, CustomResource, Aspects
)
from constructs import Construct
from aws_cdk.custom_resources import Provider
from cdk_nag import AwsSolutionsChecks, NagSuppressions


#model_id = "anthropic.claude-v2:1"
model_id = "anthropic.claude-instant-v1"
visual_scene_detection_confidence_threshold = 98.0
visual_text_detection_confidence_threshold = 98.0
raw_folder = "source"
summary_folder = "summary"
video_script_folder = "video_script"
transcription_root_folder = "audio_transcript"
transcription_folder = f"{transcription_root_folder}/{raw_folder}"
entity_sentiment_folder = "sentiment"
database_name = "videos"
video_table_name = "videos"
entities_table_name = "entities"
embedding_dimension = 1536

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

        # EventBridge - Rule mp4
        new_mp4_video_uploaded_rule = _events.Rule(
            self,
            "New mp4 video uploaded rule",
            event_pattern=_events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": _events.Match.exact_string(video_bucket_s3.bucket_name)
                    },
                    "object": {
                        "key": [{ "wildcard": f"{raw_folder}/*.mp4"}]
                    },
                },
            ),
        )
        # EventBridge - Rule MP4
        new_MP4_video_uploaded_rule = _events.Rule(
            self,
            "New MP4 video uploaded rule",
            event_pattern=_events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": _events.Match.exact_string(video_bucket_s3.bucket_name)
                    },
                    "object": {
                        "key": [{ "wildcard": f"{raw_folder}/*.MP4"}]
                    },
                },
            ),
        )
        # Suppress cdk_nag rule to allow for using AWSLambdaBasicExecutionRole managed role/policy and for using <arn>* in the IAM policy since video file names can vary
        NagSuppressions.add_resource_suppressions(new_mp4_video_uploaded_rule, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWSLambdaBasicExecutionRole managed role/policy'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow using <arn>* in the policy since video file names can vary'}
        ], True)

        # Suppress cdk_nag rule to allow for using AWSLambdaBasicExecutionRole managed role/policy and for using <arn>* in the IAM policy since video file names can vary
        NagSuppressions.add_resource_suppressions(new_MP4_video_uploaded_rule, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow using AWSLambdaBasicExecutionRole managed role/policy'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow using <arn>* in the policy since video file names can vary'}
        ], True)

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
                "MinConfidence": visual_scene_detection_confidence_threshold,
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
        label_detection_failure = _sfn.Fail(self, "Label detection is failed")
        label_detection_choice = _sfn.Choice(self, "Label detection choice")
        label_detection_success_condition = _sfn.Condition.string_equals("$.labelDetectionResult.JobStatus", "SUCCEEDED")
        label_detection_failure_condition = _sfn.Condition.string_equals("$.labelDetectionResult.JobStatus", "FAILED")
        label_detection_wait = _sfn.Wait(self, "Label detection wait",time=_sfn.WaitTime.duration(Duration.seconds(30))).next(get_rekognition_label_detection_sfn_task)

        # Build the flow
        start_rekognition_label_detection_sfn_task.next(get_rekognition_label_detection_sfn_task).next(label_detection_choice)
        label_detection_choice.when(label_detection_success_condition, label_detection_success).when(label_detection_failure_condition,label_detection_failure).otherwise(label_detection_wait)

        # Step function task to start the Rekognition text detection task to detect visual texts
        start_rekognition_text_detection_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "StartRekognitionTextDetectionSfnTask",
            service="rekognition",
            action="startTextDetection",
            parameters={
                "Video": {
                    "S3Object": {
                        "Bucket": video_bucket_s3.bucket_name,
                        "Name.$": "$.videoS3Path",
                    }
                },
                "Filters": {"WordFilter": {"MinConfidence": visual_text_detection_confidence_threshold}},
            },
            result_path="$.startTextDetectionResult",
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

        # Step function task to check the status of the Rekognition text detection task to detect visual texts
        get_rekognition_text_detection_sfn_task = _sfn_tasks.CallAwsService(
            self,
            "GetRekognitionTextDetectionSfnTask",
            service="rekognition",
            action="getTextDetection",
            parameters={
                "JobId.$": "$.startTextDetectionResult.JobId",
                "MaxResults": 1
            },
            result_path="$.textDetectionResult",
            result_selector={
                "JobId.$": "$.JobId",
                "JobStatus.$": "$.JobStatus"
            },
            iam_resources=["*"],
        )

        text_detection_success = _sfn.Succeed(self, "Text detection is successful")
        text_detection_failure = _sfn.Fail(self, "Text detection is failed")
        text_detection_choice = _sfn.Choice(self, "Text detection choice")
        text_detection_success_condition = _sfn.Condition.string_equals("$.textDetectionResult.JobStatus", "SUCCEEDED")
        text_detection_failure_condition = _sfn.Condition.string_equals("$.textDetectionResult.JobStatus", "FAILED")
        text_detection_wait = _sfn.Wait(self, "Text detection wait",time=_sfn.WaitTime.duration(Duration.seconds(30))).next(get_rekognition_text_detection_sfn_task)

        # Build the flow
        start_rekognition_text_detection_sfn_task.next(get_rekognition_text_detection_sfn_task).next(text_detection_choice)
        text_detection_choice.when(text_detection_success_condition, text_detection_success).when(text_detection_failure_condition, text_detection_failure).otherwise(text_detection_wait)

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
                "LanguageCode": "en-US",
                "MediaFormat": "mp4",
                "OutputBucketName": video_bucket_s3.bucket_name,
                "OutputKey.$": f"States.Format('{transcription_root_folder}/{{}}.txt', $.videoS3Path)",
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
        transcription_failure = _sfn.Fail(self, "Transcription is failed")
        transcription_choice = _sfn.Choice(self, "Transcription choice")
        transcription_success_condition = _sfn.Condition.string_equals("$.transcriptionResult.TranscriptionJobStatus", "COMPLETED")
        transcription_failure_condition = _sfn.Condition.string_equals("$.transcriptionResult.TranscriptionJobStatus", "FAILED")
        transcription_wait = _sfn.Wait(self, "Transcription wait",time=_sfn.WaitTime.duration(Duration.seconds(30))).next(get_transcription_job_sfn_task)

        # Build the flow
        start_transcription_job_sfn_task.next(get_transcription_job_sfn_task).next(transcription_choice)
        transcription_choice.when(transcription_success_condition, transcription_success).when(transcription_failure_condition, transcription_failure).otherwise(transcription_wait)

        parallel_sfn = _sfn.Parallel(self, "StartVideoAnalysisParallelSfn")
        parallel_sfn = parallel_sfn.branch(
            start_rekognition_label_detection_sfn_task,
            start_rekognition_text_detection_sfn_task,
            start_transcription_job_sfn_task,
        )
        
        # Role for the main video analysis lambda
        main_analyzer_lambda_role = _iam.Role(
            id="MainAnalyzerLambdaRole",
            scope=self,
            role_name=f"MainAnalyzerLambdaRole",
            assumed_by=_iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "MainAnalyzerLambdaPolicy": _iam.PolicyDocument(
                    statements=[
                        _iam.PolicyStatement(
                            actions=["bedrock:InvokeModel"],
                            resources=[f"arn:aws:bedrock:{aws_region}::foundation-model/*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["rekognition:GetLabelDetection"],
                            resources=["*"],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["rekognition:GetTextDetection"],
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
                            resources=[video_bucket_s3.bucket_arn, video_bucket_s3.arn_for_objects(f"{transcription_folder}/*")],
                            effect=_iam.Effect.ALLOW,
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:PutObject"],
                            resources=[
                                video_bucket_s3.arn_for_objects(f"{summary_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{video_script_folder}/*"),
                                video_bucket_s3.arn_for_objects(f"{entity_sentiment_folder}/*")
                            ],
                            effect=_iam.Effect.ALLOW,
                        ),
                    ]
                )
            },
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )


        # Suppress cdk_nag it for using * in IAM policy as reasonable in the resources and for using AWSLambdaBasicExecutionRole managed role by AWS.
        NagSuppressions.add_resource_suppressions(main_analyzer_lambda_role, [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow to use AWSLambdaBasicExecutionRole AWS managed service role'},
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow to use * for Rekognition read APIs which resources have to be *, and to use <arn>/* for Transcribe GetTranscriptionJob as the job name can vary'}
        ], True)
        

        # Lambda function to run the main video analysis
        main_analyzer_lambda = _lambda.Function(self, f'MainAnalyzerLambda',
           handler='index.handler',
           runtime=_lambda.Runtime.PYTHON_3_12,
           code=_lambda.Code.from_asset('./lib/main_analyzer', 
               bundling= BundlingOptions(
                  image= _lambda.Runtime.PYTHON_3_12.bundling_image,
                  command= [
                    'bash',
                    '-c',
                    'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output',
                  ],
                )),
           role=main_analyzer_lambda_role,                                    
           timeout=Duration.minutes(15),
           memory_size=2048,
           environment = {
                'MODEL_ID': model_id,
                'BUCKET_NAME': video_bucket_s3.bucket_name,
                "MODEL_ID": model_id,
                "RAW_FOLDER": raw_folder,
                "VIDEO_SCRIPT_FOLDER": video_script_folder,
                "TRANSCRIPTION_FOLDER": transcription_folder,
                "ENTITY_SENTIMENT_FOLDER": entity_sentiment_folder,
                "SUMMARY_FOLDER": summary_folder
           }
        )

        main_analyzer_task = _sfn_tasks.LambdaInvoke(self, "CallMainAnalyzerLambda",
            lambda_function=main_analyzer_lambda,
        )
        
        # Chain the analysis step after the parallel step
        parallel_sfn.next(main_analyzer_task)

        # CloudWatch Log Group for the Step Functions
        sfn_log_group = _logs.LogGroup(self, "SFNLogGroup")
        
        # The video analysis Step Function
        video_analysis_sfn = _sfn.StateMachine(
            self,
            "StartVideoAnalysisSfn",
            definition_body=_sfn.DefinitionBody.from_chainable(parallel_sfn),
            logs=_sfn.LogOptions(
                destination=sfn_log_group,
                level=_sfn.LogLevel.ALL
            ),
            tracing_enabled=True
        )

        # Suppress cdk_nag rule for using * in IAM since the file name of the videos on S3 can be various.
        NagSuppressions.add_resource_suppressions(video_analysis_sfn, [
            { "id": 'AwsSolutions-IAM5', "reason": 'This is providing access to videos on S3 for AWS AI services and since the file name can vary, we need <Arn>/* in the IAM policy.'}
        ], True)
        
        # IAM Role for EventBridge to call Step Function
        event_bridge_role = _iam.Role(self, "EventBridgeRole",
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
        
        # Add target for EventBridge to trigger Step Function for mp4 videos
        new_mp4_video_uploaded_rule.add_target(
          _events_targets.SfnStateMachine(video_analysis_sfn,
            input= _events.RuleTargetInput.from_object({
              'detailType': _events.EventField.detail_type,
              'eventId': _events.EventField.from_path('$.id'),
              'videoS3Path': _events.EventField.from_path('$.detail.object.key'),
              'videoS3BucketName': _events.EventField.from_path('$.detail.bucket.name')
            }),
            role=event_bridge_role)
        )

        # Add target for EventBridge to trigger Step Function for MP4 videos
        new_MP4_video_uploaded_rule.add_target(
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
            sign_in_aliases= _cognito.SignInAliases(
                email=True,
            ),                         
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
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{summary_folder}/*",
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{entity_sentiment_folder}/*"
                            ]
                        ),
                        _iam.PolicyStatement(
                            actions=["s3:PutObject"], 
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

        # CodeCommit repo
        BASE_DIR = os.getcwd()
        branch_name = "main"
        repo = _codecommit.Repository(
          self,
          "VideoUnderstandingRepo",
          repository_name="video-understanding-repo",
          description="CodeCommit repository for the video understanding solution's UI",
          code=_codecommit.Code.from_zip_file(
                f"{BASE_DIR}/webui/ui_repo.zip",
                branch=branch_name,
            ),
        )

        # Suppress cdk_nag rule to allow <Arn>* in the IAM policy as the video file names can vary
        NagSuppressions.add_resource_suppressions(auth_role, [
            { "id": 'AwsSolutions-IAM5', "reason": 'The <arn>* is needed in the IAM policy to allow variety of file names in S3 bucket.'}
        ], True)

        ui_amplify_app = _amplify.App(self, "VideoUnderstandingSolutionUIApp", 
            source_code_provider=_amplify.CodeCommitSourceCodeProvider(
                repository=repo,
            ),
            environment_variables={
                "AMPLIFY_USERPOOL_ID": user_pool.user_pool_id,
                "AMPLIFY_WEBCLIENT_ID": user_pool_client.user_pool_client_id,
                "REGION": aws_region,
                "AMPLIFY_IDENTITYPOOL_ID": identity_pool.ref,
                "BUCKET_NAME": video_bucket_s3.bucket_name,
                "MODEL_ID": model_id,
                "RAW_FOLDER": raw_folder,
                "VIDEO_SCRIPT_FOLDER": video_script_folder,
                "TRANSCRIPTION_FOLDER": transcription_folder.replace("/","\/"),
                "ENTITY_SENTIMENT_FOLDER": entity_sentiment_folder,
                "SUMMARY_FOLDER": summary_folder,
            }                          
        )
        master_branch = ui_amplify_app.add_branch(branch_name)

        # Role for custom resource lambda to create Cognito admin user
        cognito_user_setup_role = _iam.Role(
            id="CognitoUserSetupLambdaCR",
            scope=self,
            role_name=f"CognitoUserSetupLambdaCRRole",
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
            runtime=_lambda.Runtime.PYTHON_3_12,
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
        
        amplify_build_trigger = _custom_resources.AwsCustomResource(self, 'TriggerAmplifyBuild',
            policy= _custom_resources.AwsCustomResourcePolicy.from_statements([
                    _iam.PolicyStatement(
                        actions=["amplify:StartJob"],
                        resources=[f"arn:aws:amplify:{aws_region}:{aws_account_id}:apps/{ui_amplify_app.app_id}/branches/{master_branch.branch_name}/jobs/*"],
                        effect=_iam.Effect.ALLOW,
                    ),
                ],
            ),
            on_create=_custom_resources.AwsSdkCall(
                service='Amplify',
                action='startJob',
                physical_resource_id= _custom_resources.PhysicalResourceId.of('trigger-amplify-build'),
                parameters= {
                    "appId": ui_amplify_app.app_id,
                    "branchName": branch_name,
                    "jobType": 'RELEASE',
                    "jobReason": 'Trigger app build on create',
                }
            ),
            on_update=_custom_resources.AwsSdkCall(
                service='Amplify',
                action='startJob',
                physical_resource_id= _custom_resources.PhysicalResourceId.of('trigger-amplify-build'),
                parameters= {
                    "appId": ui_amplify_app.app_id,
                    "branchName": branch_name,
                    "jobType": 'RELEASE',
                    "jobReason": 'Trigger app build on update',
                }
            ),
        )

        # Suppress cdk_nag rule to allow the custom resource triggering the Amplify App build to use <arn>* to specify the job ID
        NagSuppressions.add_resource_suppressions(amplify_build_trigger, [
            { "id": 'AwsSolutions-IAM5', "reason": 'Allow the custom resource triggering the Amplify App build to use <arn>* to specify the job ID'},
        ], True)


        # Add suppressions for using AWSLambdaBasicExecutionRole service managed role and for using the non-latest Lambda runtime as both are managed by CDK
        NagSuppressions.add_resource_suppressions_by_path(self, "/VideoUnderstandingStack/AWS679f53fac002430cb0da5b7982bd2287", [
            { "id": 'AwsSolutions-IAM4', "reason": 'Allow to use AWSLambdaBasicExecutionRole service managed role as this is managed by CDK'},
            { "id": 'AwsSolutions-L1', "reason": 'As this is managed by CDK, allowing using the non-latest Lambda runtime.'}
        ], True)

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
            connection =_ec2.Port(protocol=ec2.Protocol("ALL"), string_representation="ALL"),
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
            runtime=_lambda.Runtime.PYTHON_3_12,
            timeout=Duration.minutes(1),
            code=_lambda.Code.from_asset('./lib/db_setup_lambda',
                bundling= BundlingOptions(
                  image= _lambda.Runtime.PYTHON_3_12.bundling_image,
                  command= [
                    'bash',
                    '-c',
                    'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output',
                  ],
                )
            ),
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
        statement = iam.PolicyStatement()
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

        CfnOutput(self, "bucket_name", value=video_bucket_s3.bucket_name)
        CfnOutput(self, "web_portal_url", value=f"https://{branch_name}.{ui_amplify_app.default_domain}")
        

