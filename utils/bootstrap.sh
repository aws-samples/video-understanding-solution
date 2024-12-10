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

environment="dev" # This is a default value
echo "Bootstrap is using default value for environment with value = dev"

default_region="us-west-2" # This is a default value which is expected to be overridden by user input.
echo ""
read -p "Which AWS region to deploy this solution to? (default: $default_region) : " region
region=${region:-$default_region} 
export CDK_DEPLOY_REGION=$region
echo $region > .DEFAULT_REGION

npm --prefix ./webui  install ./webui

cd webui && find  . -name 'ui_repo*.zip' -exec rm {} \; && zip -r "ui_repo$(date +%s).zip" src package.json package-lock.json amplify.yml public && cd ..

cdk bootstrap aws://${account}/${region} --context environment=$environment

if [ -d "venv" ]; then
    deactivate
fi