#!/bin/sh
# Evaluation on the held-out seeds with the tuned settings (never run on tuning seeds 3-5).
#   base CV 0.3:      seeds 1, 6-10 at N = 500, 1,000, 2,000, 5,000   (24 instances)
#   stressed CV 0.8:  seeds 2, 6-10 at N = 500                         (6 instances)
#   methods: DE (Gurobi defaults), LD (bundle size LD_BS, bundle gap 1e-4, cap 300 s, theta 1),
#            BD (without the mean-demand block), 1,800 s each, 10 threads
# Worst case 30 x 3 x 30 min = 45 h; converged runs stop early. Split over machines with
# PART=1 (seeds 1/2, 6, 7) and PART=2 (seeds 8, 9, 10); without PART everything runs.
# Usage: LD_BS=20 OUT=results/eval2 nohup sh phase_eval.sh > logs/eval2.log 2>&1 &
# (DE rows go to OUT_DE, default results/eval; finished runs are skipped in both folders)
X=${X:-$HOME/ssnd_labld/snd_extra}          # Wang's additional seeds 3-10
I=${I:-../idsr/instances/snd}               # IDSR repository (seed 1 base, seed 2 stressed)
BS=${LD_BS:-20}; T=1800; P=${P:-10}
O_DE=${OUT_DE:-results/eval}; O=${OUT:-results/eval}   # DE and LD/BD output folders
case "${PART:-all}" in 1) SEEDS="1 6 7";; 2) SEEDS="8 9 10";; *) SEEDS="1 6 7 8 9 10";; esac
pick() { find "$X" "$I" -path "*$1*" -name "snd_10_60_10_$2_seed$3.txt" 2>/dev/null | head -1; }
run() {
  F=$1
  [ -z "$F" ] && { echo "missing instance"; return; }
  C="--instances $F --time-limit $T --parallel $P"
  python run_snd.py $C --out $O_DE --methods de
  python run_snd.py $C --out $O --methods ld --bundle-size $BS --bundle-gap 1e-4 --tag b$BS
  python run_snd.py $C --out $O --methods bd --bd-no-ev --tag noev
}
for s in $SEEDS; do
  for n in 500 1000 2000 5000; do run "$(pick scaling $n $s)"; done
  st=$s; [ "$s" = "1" ] && st=2               # the original stressed instance is seed 2
  run "$(pick main 500 $st)"
done
