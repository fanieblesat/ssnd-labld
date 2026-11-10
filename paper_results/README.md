# Result files behind the paper's tables and figures

Every figure of Section 6 is produced by running the scripts in this folder (`python plot_<name>.py`).

| File(s) | Content | Produced by |
|---|---|---|
| `A.csv`, `Ab.csv`, `hA.csv`, `hAb.csv`, `deA.csv` | Study A (20 scenarios, m = 5): bundling results, balanced controls, trajectories, deterministic equivalent | `run_study.py` |
| `B.csv`, `Bb.csv`, `hB.csv`, `hBb.csv`, `deB.csv` | Study B (30 scenarios, m = 6), same layout | `run_study.py` |
| `partitions.csv` | 1,096 candidate partitions of the 60 small training instances (diversity analysis) | `run_learning.py collect` |
| `scaling.csv`, `hS.csv` | Scaling study, zero start, deterministic equivalent vs decomposition | `run_scaling.py` |
| `check240_small.csv`, `check240_s60.csv` | Acceptance check of the learned starts at 240 scenarios | `check_start.py` |
| `scaling_mlb.csv`, `hS_mlb.csv` | 240 scenarios, learned start P12 | `run_scaling.py --models` |
| `scaling_s60.csv`, `hS_s60.csv` | 240 scenarios, learned start P60 | `run_scaling.py --models` |
| `hW.csv` | Study A warm starts: zero, LP dual, transfer, oracle (first run) | `run_learning.py eval` |
| `wsA_bundle.csv`, `hW_bundle.csv` | Study A, stratified partition: zero and P12 | `run_learning.py eval` |
| `wsA_s60.csv`, `hW_s60.csv` | Study A, stratified partition: P60 | `run_learning.py eval` |
| `wsLABLD_bundle.csv`, `hL_bundle.csv` | Study A, learned partition: zero and P12 (LABLD) | `run_learning.py eval --ws-partition learned` |
| `wsL_s60.csv`, `hL_s60.csv` | Study A, learned partition: P60 (LABLD) | `run_learning.py eval --ws-partition learned` |
| `sus.csv` | Sustainability study (out of sample, 5 seeds x 2 settings x 5 carbon prices) | `run_sustainability.py` |
| `tune_theta.csv` | Choice of the first-step factor theta_w on training instances 301-303 (60 scenarios) | `tune_theta.py` |
| `scaling_s60tw.csv`, `hS_s60tw.csv` | 240 scenarios, learned start P60 with theta_w = 0.5 | `run_scaling.py --models --theta-warm 0.5` |
| `wsA_s60tw.csv`, `hW_s60tw.csv` | Study A, stratified partition: P60 with theta_w = 0.5 | `run_learning.py eval --theta-warm 0.5` |
| `wsL_s60tw.csv`, `hL_s60tw.csv` | Study A, learned partition: P60 with theta_w = 0.5 (LABLD) | `run_learning.py eval --ws-partition learned --theta-warm 0.5` |
