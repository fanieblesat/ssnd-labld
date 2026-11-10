#!/bin/bash
# Phase 3 (about 4-5 hours): smaller first step after an accepted learned start (theta_warm).
# 3.1 chooses theta_warm on three TRAINING instances (60 scenarios); 3.2-3.3 apply the chosen value to the
# 240-scenario instances and to Study A with the policy P60 (models/s60.pkl).
set -e
cd "$(dirname "$0")"
source ~/ssnd_labld/.venv/bin/activate
mkdir -p models results logs
[ -f models/s60.pkl ] || cp ../ssnd_v9/models/s60.pkl models/
COMMON240="--terminals 6 --periods 6 --commodities 10 --solver HiGHS --parallel 6"
STUDYA="--seeds 1-5 --terminals 5 --periods 5 --commodities 8 --scenarios 20 --bundles 5 --solver HiGHS --parallel 6 --de-csv ../results/studyA_ryzen/de.csv --skip-bundling"

echo "=== 3.1 choose theta_warm on training instances 301-303 (60 scenarios)  $(date)"
python tune_theta.py --models models/s60.pkl --seeds 301 302 303 --scenarios 60 --bundles 12 \
    $COMMON240 --thetas 1 0.5 0.25 --iters 20 --out results/tune_theta
TW=$(cat results/tune_theta/chosen_theta.txt)
echo "theta_warm = $TW"
if [ "$TW" = "1.0" ]; then echo "=== theta_warm = 1 is best: nothing to rerun  $(date)"; exit 0; fi

echo "=== 3.2 scaling at 240 scenarios, P60 with theta_warm = $TW  $(date)"
python run_scaling.py --scenarios 240 --seeds 1 2 3 4 5 $COMMON240 --bundle-size 5 --time-limit 1800 \
    --polish-time 0 --skip-de --models models/s60.pkl --theta-warm $TW --out results/scaling_ml_s60_tw

echo "=== 3.3 Study A, P60 with theta_warm = $TW  $(date)"
python run_learning.py eval --models models/s60.pkl $STUDYA --starts ml --theta-warm $TW \
    --ws-partition stratified --out results/learnA_s60_tw
python run_learning.py eval --models models/s60.pkl $STUDYA --starts ml --theta-warm $TW \
    --ws-partition learned --out results/learnA_labld_s60_tw

echo "=== phase 3 done  $(date)"
