#!/bin/bash
# Phase 2 (about 15-18 hours): train the policy on 60-scenario instances and test it at 240.
set -e
cd "$(dirname "$0")"
source ~/ssnd_labld/.venv/bin/activate
mkdir -p models results logs data
COMMON240="--terminals 6 --periods 6 --commodities 10 --solver HiGHS --parallel 6"
STUDYA="--seeds 1-5 --terminals 5 --periods 5 --commodities 8 --scenarios 20 --bundles 5 --solver HiGHS --parallel 6 --de-csv ../results/studyA_ryzen/de.csv --skip-bundling"

echo "=== 2.1 training data: 20 instances with 60 scenarios, m = 12, seeds 301-320  $(date)"
python run_learning.py collect --seeds 301-320 $COMMON240 --scenarios 60 --bundles 12 \
    --bundle-time-limit 300 --skip-pool --mult-parts stratified --iters-long 60 --out data/train_s60

echo "=== 2.2 policy search, starting from the rescaled small policy  $(date)"
python run_learning.py train --data data/train_s60 --out models/s60.pkl \
    --init-from models/small_bundle.pkl --scorer-from models/small_bundle.pkl \
    --max-evals 60 --parallel 6 | tee logs/train_s60.txt

echo "=== 2.3 quick check at 240 scenarios  $(date)"
python check_start.py --models models/s60.pkl --seeds 1 2 3 4 5 --scenarios 240 $COMMON240 \
    --alphas 1 0.5 2 --out results/check240_s60.csv

echo "=== 2.4 scaling at 240 scenarios with the 60-scenario policy  $(date)"
python run_scaling.py --scenarios 240 --seeds 1 2 3 4 5 $COMMON240 --bundle-size 5 --time-limit 1800 \
    --polish-time 0 --skip-de --models models/s60.pkl --out results/scaling_ml_s60

echo "=== 2.5 Study A with the 60-scenario policy  $(date)"
python run_learning.py eval --models models/s60.pkl $STUDYA --starts ml \
    --ws-partition stratified --out results/learnA_s60
python run_learning.py eval --models models/s60.pkl $STUDYA --starts ml \
    --ws-partition learned --out results/learnA_labld_s60

echo "=== phase 2 done  $(date)"
