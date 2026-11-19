#!/bin/sh
# Benders on the five IDSR SND instances, 1,800 s each (about 2.5 hours).
# Usage: nohup sh phase_bd.sh > logs/snd_bd.log 2>&1 &
I=${I:-../idsr/instances/snd}
for X in $I/scaling/snd_10_60_10_500_seed1.txt $I/main/snd_10_60_10_500_seed2.txt \
         $I/scaling/snd_10_60_10_1000_seed1.txt $I/scaling/snd_10_60_10_2000_seed1.txt \
         $I/scaling/snd_10_60_10_5000_seed1.txt; do
  python run_snd.py --instances $X --methods bd --time-limit 1800 --parallel 10 --out results/snd_pilot
done
