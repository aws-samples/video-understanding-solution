#!/bin/bash

# Install Python if not installed or if the version is below the required one
needPython=false
version=$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)')
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
    sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel
    mkdir packages
    cd packages
    curl https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz -o Python-3.9.18.tgz
    tar xzf Python-3.9.18.tgz && cd Python-3.9.18
    ./configure --enable-optimizations
    sudo make altinstall
    cd .. && sudo rm ./Python-3.9.18.tgz && sudo rm -rf ./Python-3.9.18
    cd .. && rm -rf ./packages
fi

# Install pip
python3 -m ensurepip

# Install VirtualEnv if not installed
isVirtualEnvInstalled=$(python3 -m pip list | grep "virtualenv")
if [[ -z "$isVirtualEnvInstalled" ]]
then
    sudo python3 -m pip install virtualenv
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
    sudo yum update -y
    sudo yum install zip -y
fi

# Install unzip if not installed
if ! command -v unzip &> /dev/null
then
    sudo yum update -y
    sudo yum install unzip -y
fi

# Install NPM
curl -o /tmp/nvm_install.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh
/bin/bash /tmp/nvm_install.sh
. ~/.nvm/nvm.sh

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm install v16.20.2
nvm use 16.20.2
rm /tmp/nvm_install.sh

node -e "console.log('Currently running Node.js ' + process.version)"
npm install -g aws-cdk@">=2.122.0"

# Install python packages
python3 -m pip install "aws-cdk-lib>=2.122.0"
python3 -m pip install "aws-cdk.aws-amplify-alpha"
python3 -m pip install "cdk-nag>=2.28.16"

# Install Docker if not installed. It will be needed to build the code for the AWS Lambda functions during CDK deployment.
if ! command -v docker &> /dev/null
then
    sudo yum update -y
    sudo yum install -y docker
fi

# Give users permissions to access docker
sudo groupadd docker
sudo usermod -a -G docker ec2-user
sudo usermod -a -G docker ssm-user
# IMPORTANT #
# If you are using OS user other than ec2-user and ssm-user, also add them to the `docker` group below. Edit appropriately and uncomment below line. You can run `whoami` to get your OS user name.
# sudo usermod -a -G docker <your OS user name>
# --------- #

# Start Docker service
sudo service docker start

# Install JQ if not installed
if ! command -v jq &> /dev/null
then
    sudo yum update -y
    sudo yum install jq -y
fi

# Install AWS CLI if not installed
if ! command -v aws &> /dev/null
then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install && sudo rm awscliv2.zip
fi

# Deactivate virtual environment
deactivate