#!/bin/bash

# Install brew if not there
if ! command -v brew &> /dev/null
then
    curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o /tmp/brew_install.sh
    /bin/bash /tmp/brew_install.sh
    brew update
    rm /tmp/brew_install.sh
fi

# Install Python if not installed or if the version is below the required one
needPython=false
version=$(python3 --version | sed 's/.* \([0-9\.]*\).*/\1/')
parsedVersion=$(echo "${version//./}")
if [[ -z "$version" ]]
then
    needPython=true
else
    if { echo $version; echo "3.8.0"; } | sort --version-sort --check
    then 
        needPython=true
    fi
fi

if [[ "$needPython" = true ]]
then
    brew install python@3.9
fi

# Install pip
python3 -m ensurepip

# Install VirtualEnv if not installed
isVirtualEnvInstalled=$(python3 -m pip list | grep "virtualenv")
if [[ -z "$isVirtualEnvInstalled" ]]
then
    python3 -m pip install --user virtualenv
fi

if [[ "$needPython" = true ]]
then
    virtualenv -p $(which python3.9) venv
else
    virtualenv -p $(which python3) venv
fi

# Activate virtual environment
source venv/bin/activate

# Install zip if not installed
if ! command -v zip &> /dev/null
then
    brew install zip
fi

# Install unzip if not installed
if ! command -v unzip &> /dev/null
then
    brew install unzip
fi

# Install NPM if not installed
brew install nvm 

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm install v20.10.0
nvm use 20.10.0

node -e "console.log('Currently running Node.js ' + process.version)"
sudo npm install -g aws-cdk@">=2.122.0"

python3 -m pip install "aws-cdk-lib>=2.122.0"
python3 -m pip install "aws-cdk.aws-amplify-alpha"
python3 -m pip install "cdk-nag>=2.28.16"

# Install Docker if not installed. It will be needed to build the code for the AWS Lambda functions during CDK deployment.
if ! command -v docker &> /dev/null
then
    brew install --cask docker
fi

# Install JQ if not installed
if ! command -v jq &> /dev/null
then
    brew install jq
fi

# Install AWS CLI if not installed
if ! command -v aws &> /dev/null
then
    curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
    sudo installer -pkg AWSCLIV2.pkg -target /
fi

# Open docker
open /Applications/Docker.app

# Deactivate virtual environment
deactivate
