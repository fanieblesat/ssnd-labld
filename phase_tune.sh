#!/bin/sh
# Pre-registered tuning on seeds 3 and 4 (tuning seeds; never used for evaluation).
# Instances per seed: base CV 0.3 with N = 500 and 2,000; stressed CV 0.8 with N = 500.
# Settings (one factor at a time from the default):
#   LD  default: bundle size 10, bundle MIP gap 1e-4, bundle cap 300 s, theta 1
#       t05: theta 0.5   cap60: bundle cap 60 s   b20: bundle size 20
#   BD  default: mean-demand block, phase-1 share 0.25
#       noev: without the mean-demand block   p10: phase-1 share 0.10
# 7 settings x 6 instances x 30 min = 21 h. Split over two machines with PART=1 and PART=2
# (PART=1: seed 3, PART=2: seed 4); without PART both seeds run.
# Usage: X=~/ssnd_labld/snd_extra PART=1 nohup sh phase_tune.sh > logs/tune1.log 2>&1 &
X=${X:-$HOME/ssnd_labld/snd_extra}
case "${PART:-all}" in 1) SEEDS="3";; 2) SEEDS="4";; *) SEEDS="3 4";; esac
T=1800; P=${P:-10}
pick() { find "$X" -path "*$1*" -name "snd_10_60_10_$2_seed$3.txt" | head -1; }
for s in $SEEDS; do
  for spec in "scaling 500" "scaling 2000" "main 500"; do
    set -- $spec
    F=$(pick $1 $2 $s)
    [ -z "$F" ] && { echo "missing instance: $1 N=$2 seed $s"; continue; }
    O=results/tune
    C="--instances $F --time-limit $T --parallel $P --out $O"
    python run_snd.py $C --methods ld --bundle-size 10 --bundle-gap 1e-4 --tag def
    python run_snd.py $C --methods ld --bundle-size 10 --bundle-gap 1e-4 --theta 0.5 --tag t05
    python run_snd.py $C --methods ld --bundle-size 10 --bundle-gap 1e-4 --bundle-time-limit 60 --tag cap60
    python run_snd.py $C --methods ld --bundle-size 20 --bundle-gap 1e-4 --tag b20
    python run_snd.py $C --methods bd --tag def
    python run_snd.py $C --methods bd --bd-no-ev --tag noev
    python run_snd.py $C --methods bd --bd-phase1-share 0.10 --tag p10
  done
done
