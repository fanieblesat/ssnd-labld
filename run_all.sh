#!/bin/bash
# Runs phase 1 and then phase 2; everything is resumable, so after an interruption just start it again.
cd "$(dirname "$0")"
mkdir -p logs
bash phase1.sh >> logs/phase1.log 2>&1 && bash phase2.sh >> logs/phase2.log 2>&1
