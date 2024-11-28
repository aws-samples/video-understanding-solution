@echo on
setlocal enabledelayedexpansion

:: Activate venv if it exists
if exist venv (
    call venv\Scripts\activate.bat
)

:: NVM setup and Node.js version check
call nvm use 20.10.0 2>nul || call nvm use 16.20.2
node -e "console.log('Currently running Node.js ' + process.version)"

:: Get AWS account ID
for /f "tokens=* usebackq" %%a in (`aws sts get-caller-identity ^| findstr Account`) do (
    for /f "tokens=2 delims=:" %%b in ("%%a") do (
        set "account=%%~b"
        set "account=!account:"=!"
        set "account=!account:,=!"
        set "account=!account: =!"
    )
)
set CDK_DEPLOY_ACCOUNT=%account%

:: Set default region
set "default_region=us-east-1"
if exist .DEFAULT_REGION (
    set /p default_region=<.DEFAULT_REGION
)

echo.
set "email="
set /p "email=The admin email for the video management: "

set /p "region=Which AWS region to deploy this solution to? (default: %default_region%) : "
if "!region!"=="" set "region=%default_region%"
set CDK_DEPLOY_REGION=!region!

:: Create UI zip file
cd webui
del /f /q ui_repo*.zip 2>nul
npm install
set timestamp=%date:~10,4%%date:~4,2%%date:~7,2%%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
powershell Compress-Archive -Path src,package.json,package-lock.json,amplify.yml,public -DestinationPath "ui_repo%timestamp%.zip" -Force
cd ..

:: Deploy with CDK
call cdk deploy --outputs-file .\deployment-output.json --context email=%email%

:: Deactivate venv if it was activated
if exist venv (
    call venv\Scripts\deactivate.bat
)

endlocal 