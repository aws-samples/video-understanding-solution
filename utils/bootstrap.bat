@echo off
setlocal enabledelayedexpansion

:: Activate virtual environment if it exists
if exist "venv" (
    call venv\Scripts\activate.bat
)

:: NVM setup and Node.js version check
call nvm use 20.0.0 2>nul || call nvm use 16.0.0 2>nul
node -e "console.log('Currently running Node.js ' + process.version)"

:: Get AWS account ID using AWS CLI and store it in a variable
for /f "tokens=* USEBACKQ" %%F in (`aws sts get-caller-identity --query "Account" --output text`) do (
    set account=%%F
)
set CDK_DEPLOY_ACCOUNT=%account%

:: Set default environment
set environment=dev
echo Bootstrap is using default value for environment with value = dev

:: Set default region and prompt for user input
set default_region=us-east-1
echo.
set /p "region=Which AWS region to deploy this solution to? (default: %default_region%) : "
if "!region!"=="" set "region=%default_region%"
set CDK_DEPLOY_REGION=!region!
echo !region!> .DEFAULT_REGION

:: Run CDK bootstrap
call cdk bootstrap aws://%account%/%region% --context environment=%environment%

:: Deactivate virtual environment if it exists
if exist "venv" (
    call venv\Scripts\deactivate.bat
)

endlocal 