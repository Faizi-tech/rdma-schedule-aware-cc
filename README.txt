Phase 3 code. Run order (python3, numpy, numba, scipy, scikit-learn, matplotlib):
  python3 train.py mlp                      # train the burst forecaster
  python3 train.py full_mlp sched nopred ma oracle fct_reward nophase   # ES training (about 1 min each)
  python3 experiments.py                    # all experiments, results.json, figs/
  python3 mkreport.py && pdflatex report.tex   # builds the report from results.json
Files: cnsim.py (simulator and controllers), scen.py (scenarios, forecasters), train.py, experiments.py
