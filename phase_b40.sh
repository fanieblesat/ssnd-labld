#!/bin/sh
# Grid extension: bundle size 20 was the best and the largest value in the tuning grid, so the grid is
# extended by one step (bundle size 40) on the same six tuning instances, with the same rule.
X=${X:-$HOME/ssnd_labld/snd_extra}
pick() { find "$X" -path "*$1*" -name "snd_10_60_10_$2_seed$3.txt" | head -1; }
for s in 3 4; do
  for spec in "scaling 500" "scaling 2000" "main 500"; do
    set -- $spec
    python run_snd.py --instances $(pick $1 $2 $s) --time-limit 1800 --parallel ${P:-10} --out results/tune \
      --methods ld --bundle-size 40 --bundle-gap 1e-4 --tag b40
  done
done
