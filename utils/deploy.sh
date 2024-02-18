#!/bin/bash

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm use 20.10.0 || nvm use 16.20.2
node -e "console.log('Currently running Node.js ' + process.version)"

account=$(aws sts get-caller-identity | jq -r '.Account')
export CDK_DEPLOY_ACCOUNT=$account

default_region="us-west-2" # This is a default value which is expected to be overridden by user input.
if [ -f .DEFAULT_REGION ]
then
    default_region=$(cat .DEFAULT_REGION)
fi
echo ""
email="" # This is a default value
read -p "The admin email for the video management: " email
email=${email:-$email} 

read -p "Which AWS region to deploy this solution to? (default: $default_region) : " region
region=${region:-$default_region} 
export CDK_DEPLOY_REGION=$region

npm --prefix ./webui  install ./webui

cd webui && find  . -name 'ui_repo*.zip' -exec rm {} \; && zip -r "ui_repo$(date +%s).zip" src package.json package-lock.json amplify.yml public && cd ..

cdk deploy --outputs-file ./deployment-output.json --context email=$email

if [ -d "venv" ]; then
    deactivate
fi
