#!/bin/sh
# Learning experiments on the SND benchmark, in order of importance (about 45 h in total; every step is
# resumable, so after an interruption the same command continues where it stopped).
#   STAGE=all (default) or one of: long, train, ws_large, bundling, ws_small, ohc
#   1 long      (v19, done) long LD runs on base N = 500, seeds 3-5; kept, not used for training
#   2 train     fit the dual start policy on tuning instances with headroom            -> models/snd_policy_v2.pkl
#   3 ws_large  learned start, guarded and unguarded, base N = 2,000 and 5,000, evaluation seeds
#   4 bundling  bound at zero multipliers for seven partition rules (base N = 500, 2,000; stressed N = 500)
#   5 ws_small  learned start, guarded, base N = 500 and 1,000 and stressed N = 500, evaluation seeds
#   6 ohc       oracle start for the OHC diagnostic, base N = 1,000: long run (1 h) + oracle start
# Usage: nohup sh phase_learn.sh > logs/learn.log 2>&1 &
X=${X:-$HOME/ssnd_labld/snd_extra}; I=${I:-../idsr/instances/snd}
O=results/learn; P=${P:-10}; mkdir -p $O models
pick() { find "$X" "$I" -path "*$1*" -name "snd_10_60_10_$2_seed$3.txt" 2>/dev/null | head -1; }
LD="--methods ld --parallel $P --bundle-size 20 --bundle-gap 1e-4 --out $O"
st() { [ "${STAGE:-all}" = all ] || [ "${STAGE}" = "$1" ]; }

if st long; then for s in 3 4 5; do F=$(pick scaling 500 $s)
  python run_snd.py --instances $F $LD --time-limit 3600 --tag long --save-lambda $O/lam_base500_s$s.npy; done; fi

# Training cases must have headroom (LB* > L(0)): base N = 500 has none (LD closes it at lambda = 0), so the
# policy is trained on the tuning instances with headroom, with LB* from the tuning runs (results/tune) and
# a 60 s cap per bundle problem during training (L(0) re-evaluated with the same cap).
if st train; then [ -f models/snd_policy_v2.pkl ] || python snd_learning.py train \
  --instances $(pick scaling 2000 3) $(pick scaling 2000 4) $(pick main 500 3) $(pick main 500 4) \
  --long-csv results/tune/snd.csv --long-method ld_b20 --bundle-time-limit 60 --max-evals 24 \
  --parallel $P --out models/snd_policy_v2.pkl; fi

if st ws_large; then for n in 5000 2000; do for s in 1 6 7 8 9 10; do F=$(pick scaling $n $s)
  python run_snd.py --instances $F $LD --time-limit 1800 --tag ws2 --start-policy models/snd_policy_v2.pkl --theta-warm 0.5
  python run_snd.py --instances $F $LD --time-limit 1800 --tag wsng2 --start-policy models/snd_policy_v2.pkl --theta-warm 0.5 --no-zero-guard
done; done; fi

if st bundling; then
  python snd_learning.py bundling --parallel $P --out $O --instances \
    $(for s in 1 3 4 5 6 7 8 9 10; do pick scaling 500 $s; done) \
    $(for s in 1 6 7 8 9 10; do pick scaling 2000 $s; done) \
    $(for s in 2 6 7 8 9 10; do pick main 500 $s; done)
fi

if st ws_small; then
  for n in 500 1000; do for s in 1 6 7 8 9 10; do
    python run_snd.py --instances $(pick scaling $n $s) $LD --time-limit 1800 --tag ws2 --start-policy models/snd_policy_v2.pkl --theta-warm 0.5; done; done
  for s in 2 6 7 8 9 10; do
    python run_snd.py --instances $(pick main 500 $s) $LD --time-limit 1800 --tag ws2 --start-policy models/snd_policy_v2.pkl --theta-warm 0.5; done
fi

if st ohc; then for s in 1 6 7 8 9 10; do F=$(pick scaling 1000 $s)
  python run_snd.py --instances $F $LD --time-limit 3600 --tag long --save-lambda $O/lam_base1000_s$s.npy
  python run_snd.py --instances $F $LD --time-limit 1800 --tag oracle --start-lambda $O/lam_base1000_s$s.npy --no-zero-guard
done; fi
