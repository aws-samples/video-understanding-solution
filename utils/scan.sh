#!/bin/bash

python3 -m pip install bandit
bandit -r . -o banditreport.txt -f txt --exclude "./cdk.out/,./webui/node_modules/,./venv/"

python3 -m pip install semgrep
semgrep login
semgrep scan --exclude "./cdk.out/" -o semgrepreport.txt &> semgreplog.txt

cd webui && npm i --package-lock-only && npm audit --json > ../npmauditreport.json
cd ..