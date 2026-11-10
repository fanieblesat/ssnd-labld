#!/bin/bash
# Phase 1 (about 3-4 hours): the v5-v8 policy with the scale fix, no new training.
set -e
cd "$(dirname "$0")"
source ~/ssnd_labld/.venv/bin/activate
mkdir -p models results logs
COMMON240="--terminals 6 --periods 6 --commodities 10 --solver HiGHS --parallel 6"
STUDYA="--seeds 1-5 --terminals 5 --periods 5 --commodities 8 --scenarios 20 --bundles 5 --solver HiGHS --parallel 6 --de-csv ../results/studyA_ryzen/de.csv --skip-bundling"

echo "=== 1.0 rescale the trained policy (exact at its training size m = 4)  $(date)"
python run_learning.py rescale --models ../ssnd_v5/models/small.pkl --m-train 4 --out models/small_bundle.pkl

echo "=== 1.1 quick check at 240 scenarios: is the start accepted?  $(date)"
python check_start.py --models models/small_bundle.pkl --seeds 1 2 3 4 5 --scenarios 240 $COMMON240 \
    --alphas 1 0.5 2 --out results/check240_small.csv

echo "=== 1.2 scaling at 240 scenarios with the rescaled start (DE results of results/scaling reused)  $(date)"
python run_scaling.py --scenarios 240 --seeds 1 2 3 4 5 $COMMON240 --bundle-size 5 --time-limit 1800 \
    --polish-time 0 --skip-de --models models/small_bundle.pkl --out results/scaling_ml_bundle

echo "=== 1.3 Study A, stratified partition: zero vs rescaled start  $(date)"
python run_learning.py eval --models models/small_bundle.pkl $STUDYA --starts zero,ml \
    --ws-partition stratified --out results/learnA_bundle

echo "=== 1.4 Study A, learned partition: zero vs rescaled start (LABLD)  $(date)"
python run_learning.py eval --models models/small_bundle.pkl $STUDYA --starts zero,ml \
    --ws-partition learned --out results/learnA_labld_bundle

echo "=== phase 1 done  $(date)"
