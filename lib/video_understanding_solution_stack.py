import os
from aws_cdk import (
    # Duration,
    Stack,
    CfnOutput,
    aws_iam as _iam,
    aws_s3 as _s3,
    aws_events as _events,
    aws_events_targets as _events_targets,
    aws_lambda as _lambda,
    aws_logs as _logs,
    aws_stepfunctions as _sfn,
    aws_stepfunctions_tasks as _sfn_tasks,
    aws_codecommit as _codecommit,
    aws_amplify_alpha as _amplify,
    aws_cognito as _cognito,
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
raw_folder = "raw"
summary_folder = "summary"
video_script_folder = "video_script"
transcription_root_folder = "audio_transcript"
transcription_folder = f"{transcription_root_folder}/{raw_folder}"
entity_sentiment_folder = "sentiment"

class VideoUnderstandingSolutionStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        aws_region = Stack.of(self).region
        aws_account_id = Stack.of(self).account
        admin_email_address = self.node.try_get_context("email") 

        Aspects.of(self).add(AwsSolutionsChecks(verbose=True))

        ##
        ## Video Analysis
        ##

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
            "New video uploaded rule",
            event_pattern=_events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": _events.Match.exact_string(video_bucket_s3.bucket_name)
                    },
                    "object": {
                        "key": [{ "wildcard": f"{raw_folder}/*.mp4"}]
                                                    # _events.Match.prefix(f"{raw_folder}/")
                                                    #_events.Match.all_of(_events.Match.prefix("raw/")) #,
                                                    #_events.Match.any_of(_events.Match.suffix(".mp4"),
                                                    #                     _events.Match.suffix(".MP4"),
                                                    #                     _events.Match.suffix(".mov"),
                                                    #                     _events.Match.suffix(".MOV")))
                    },
                },
            ),
        )
        # Suppress cdk_nag rule to allow for using AWSLambdaBasicExecutionRole managed role/policy and for using <arn>* in the IAM policy since video file names can vary
        NagSuppressions.add_resource_suppressions(new_video_uploaded_rule, [
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
                                video_bucket_s3.arn_for_objects(f"{video_script_folder}/*")
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
        
        # Add target for EventBridge to trigger Step Function
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
                                f"arn:aws:s3:::{video_bucket_s3.bucket_name}/{summary_folder}/*"
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

        ui_amplify_app = _amplify.App(self, "UiAmplifyApp", 
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


        CfnOutput(self, "bucket_name", value=video_bucket_s3.bucket_name)
        CfnOutput(self, "web_portal_url", value=f"https://{branch_name}.{ui_amplify_app.default_domain}")
        
        # Permission

        # video_bucket_s3.grant_read(start_rekognition_label_detection_sfn_task)
        # video_bucket_s3.grant_read(start_rekognition_text_detection_sfn_task)
