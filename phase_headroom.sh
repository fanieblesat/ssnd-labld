#!/bin/sh
# Headroom check for a warm start at N = 5,000 (tuning seed 3, base configuration), about 4 hours.
#   C: LD from zero multipliers, 1,800 s (reference)
#   A: LD from zero multipliers, 3 hours; its best multipliers are saved
#   B: LD from A's multipliers, 1,800 s, no zero guard (oracle start: the best a warm start could do)
# If B is not clearly better than C, a learned start cannot help much at this size.
X=${X:-$HOME/ssnd_labld/snd_extra}
F=$(find "$X" -path "*scaling*" -name "snd_10_60_10_5000_seed3.txt" | head -1)
O=results/headroom; mkdir -p $O
C="--instances $F --methods ld --parallel ${P:-10} --bundle-size 20 --bundle-gap 1e-4 --out $O"
python run_snd.py $C --time-limit 1800 --tag zero
python run_snd.py $C --time-limit 10800 --tag long --save-lambda $O/lam_long.npy
python run_snd.py $C --time-limit 1800 --tag oracle --start-lambda $O/lam_long.npy --no-zero-guard
