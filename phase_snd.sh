#!/bin/sh
# Diagnosis on the main IDSR SND instance (seed 2), about 2.5 hours in total.
# Usage: nohup sh phase_snd.sh > logs/snd_diag.log 2>&1 &
I=${I:-../idsr/instances/snd}
X=$I/main/snd_10_60_10_500_seed2.txt
O=results/snd_diag
python run_snd.py --instances $X --methods ld --time-limit 1800 --parallel 10 --bundle-size 10 --bundle-gap 1e-4 --tag g4 --out $O
python run_snd.py --instances $X --methods ld --time-limit 1800 --parallel 10 --bundle-size 10 --bundle-gap 1e-4 --theta 0.5 --tag g4t05 --out $O
python run_snd.py --instances $X --methods ld --time-limit 1800 --parallel 10 --bundle-size 5 --bundle-gap 1e-4 --tag b5g4 --out $O
python run_snd.py --instances $X --methods de --time-limit 3600 --parallel 10 --tag 3600 --out $O
