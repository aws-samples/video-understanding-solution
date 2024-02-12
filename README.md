# Video Understanding Solution

## Introduction

This is a deployable solution which can help save your time in understanding the videos you have without having to watch every video. This solution automatically generate AI-powered summary of each video uploaded to your Amazon Simple Storage Service (S3) bucket. Not only that, it also allows you to ask questions about the video in a UI chatbot experience, like "How many people are in this video", "At which seconds does it talk about company's vision?". It can extract information from visual scenes, audio, and visible text in the video

You can upload videos to your S3 bucket by using AWS console, CLI, SDK, or other means (e.g. via AWS Transfer Family). This solution will automatically trigger processes by Amazon Rekognition and Amazon Transcribe to extract the visual scenes and texts and the audio transcription. This combined information is used to generate the summary as powered by generative AI with Amazon Bedrock. The UI chatbot also uses Amazon Bedrock to allows you to ask questions about the videos. Refer to the diagram below on the architecture.

![Architecture](./assets/architecture.jpg)

Refer to the Deployment section below on how to deploy it to your own AWS account. Refer to the Use section on how to use it.

## Deployment

### Prerequisites

Here are prerequisites for deploying this solution with AWS CDK:

1. AWS account with Amazon Bedrock model access enabled for Claude and Titan Embeddings G1 - Text. Follow the steps in https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#add-model-access to enable those models and make sure the region is correct.
2. Make 3.82 or above in your environment
3. AWS IAM credentials with sufficient permission for deploying this solution configured in your environment. See https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html for credentials setup.
4. AWS CLI 2.15.5 or above / 1.29.80 or above. Follow https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html for installation.
5. Python 3.8 or above, pip, and virtualenv.
6. Docker 20.10.25 or above
7. NodeJS 16 with version 16.20.2 above, or NodeJS 20 with version 20.10.0 or above, along with NPM.
8. jq, zip, unzip
9. CDK Toolkit 2.122.0 or above. Follow https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_install for installation.
10. Python CDK lib 2.122.0 or above with @aws-cdk/aws-amplify-alpha. Follow https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_concepts
11. CDK nag 2.28.16 or above.

To automatically install item 4-11 of those prerequisites run. `make prepare`

The `make prepare` utility is currently supported only for MacOS Ventura, Amazon Linux 2, and Amazon Linux 2023 operating systems.

NOTE: Currently those packages need to be installed inside virtual environment with name "venv" in the folder where this repository resides. 

### Deploy

**IMPORTANT**

At the moment,this solution can only be used in AWS regions where Amazon Bedrock is available.
At the time of the writing of this document, you can deploy and use this in us-east-1 (N. Virginia) and us-west-2 (Oregon) regions.

Follow the steps below for deploying this solution from your current environment

1. Make sure that all prerequisites are met. Please refer to the [Prerequisites](#prerequisites) section above.

2. Run `make bootstrap_and_deploy` and enter an email address and the region to deploy when prompted. 

3. [Optional] For second or later redeployment, run `make deploy` instead.


## Use

After a successful deployment, the email address that you inputted during deployment should receive an email with temporary password and the web UI portal URL. Make sure the deployment is completed before going to the URL. Visit the URL, use the email address as username, and input the temporary password. It should lead you to password reset dialog. After a successful sign-in, you should be able to see the page that will display your videos.

You can upload videos into the S3 bucket that this solution deployed. Upload the videos into the "raw" folder (create one if not already exist). It will automatically trigger asynchronous task. Wait for few minutes for the task to finish. The resulted analysis will be displayed in the web UI portal.

In the Web UI portal, you can search for videos in your S3 bucket. For each video found, you can view the video, see its summary, see the extracted information by the seconds from Amazon Rekognition and Amazon Transcribe, and ask the chatbot about the video e.g. "How many people are in this video". The chatbot is equipped with memory for the current conversation, so the user can have a context-aware conversation with the bot.

## Limitations

1. Currently only .mp4 video files are supported
2. This solution is only tested on English videos
3. Only videos less than 15 minutes are supported for now.

## Removal
To remove the solution from your AWS account, run `make destroy` and specify the region.

## Security

Run `make scan` before submitting any pull request to make sure that the introduced changes do not open a vulnerability. Make sure the generated banditreport.txt, semgrepreport.txt, semgreplog.txt, npmauditreport.txt, and the cdk_nag files in cdk.out folders all show **no high or critical finding**.

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.