# Mathematical formulation (draft; to be reconciled with `run_fl` before any proof)

Round t: honest client k returns a delta d_{k,t} (local model minus global model); the server applies the aggregator A to the stacked deltas with sample-size weights p_k and sets w_{t+1} = w_t + eta_s A(d_{1,t}, ..., d_{K,t}; p) for a server step eta_s. The code aggregates **deltas**, so the prompt's w_{t+1} = w_t + A(u) matches with u = delta (verified in `aggregators/mean.py`: weighted sum of deltas). Whether eta_s = 1 and how SCAFFOLD/FedNova rescale must be checked against `training/` before stating it formally.

Untargeted attacker C at round t: simulate the deployed rule on every candidate M in Q (34 candidates, rows identical across colluders) and pick the candidate whose simulated aggregate A(D_honest, M) has the smallest inner product with the benign direction s_t (the code minimises `agg @ score_dir`; see `agr_tailored_choose`). A learned attacker replaces Q by a continuous parameter theta and optimises the same score by search; the objective over the horizon is max_theta L_val(w_R(theta)).

Backdoor: maximise ASR = P_{x in D_trigger}[ f_{w_R}(x) = y_target ] subject to per-round norm bound.

Defender multi-objective: minimise (damage, ASR, clean-accuracy loss, honest-rejection FPR, calibration error, worst-client accuracy loss, overhead). We report the vector, not one scalar; Pareto plots are planned only where at least three rules differ on two axes.

No formal guarantee is proven here. `docs/threat_analysis.md` contains two elementary statements for the committee only.
