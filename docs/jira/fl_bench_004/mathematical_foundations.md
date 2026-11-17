# Mathematical foundations — fl_bench_004

## FL-BENCH-004.1 FedAvg baseline
w_{t+1} = Σ_k p_k w_k^{t+1}, p_k = n_k/Σ n_j; single client ⇒ local SGD.

## FL-BENCH-004.2 FedProx
min_w F_k(w) + (μ/2)||w − w_t||²; gradient + μ(w − w_t); μ=0 ⇒ FedAvg.

## FL-BENCH-004.3 SCAFFOLD
y ← y − η(g − c_i + c); c_i⁺ = c_i − c + (x − y)/(Kη); invariant c = mean(c_i) under full participation.

## FL-BENCH-004.4 FedNova
w ← w + τ_eff Σ p_k Δ_k/τ_k; equal τ ⇒ FedAvg.

## FL-BENCH-004.5 Four-algorithm comparison grid
—

## FL-BENCH-004.6 Statistical analysis
Wilcoxon minimum attainable p = 2/2ⁿ ⇒ n≥6 needed for p<0.05; 10 seeds used in main conditions; Cohen d_z = mean(d)/sd(d).

## FL-BENCH-004.7 Leaderboard
mean rank = average over criteria of rank (1 = best).

## FL-BENCH-004.10 Natural multi-site benchmark: UCI heart disease (4 collection sites as clients)
Protocol A: per site 70/30 stratified; 4 clients = 4 sites; nested tuning on 25% validation per split; 10 splits. Protocol B: train on 3 sites, test on all rows of the 4th; tuned on 25% validation of the training sites; 10 initialisation seeds (data fixed). Balanced accuracy = (recall + specificity)/2.

## FL-BENCH-004.11 Broader tuning: six-parameter random search vs lr/mu grid
Same 24 draws for every algorithm; selection by mean validation F1 (tie: lower validation loss) on 25% of TRAIN; three settings compared on the untouched test set: default, grid (lr[,mu]), random (6 parameters). Paired Wilcoxon / t by seed (synthetic, 10) or split (heart, 5).

## FL-BENCH-004.8 Validation-based tuning of the four FL baselines
theta* = argmax_theta mean_s F1_val(theta, seed s); grid: lr in {0.01,0.03,0.05,0.1,0.2,0.4,0.8}, mu in {0.001,0.01,0.05,0.1,0.5,1} (FedProx). Paired effect = F1_tuned - F1_default per seed.

## FL-BENCH-004.9 Benchmark on real public de-identified medical tables
Nested protocol per outer split s: tune on 25% validation of TRAIN_s; evaluate on TEST_s; 10 splits per dataset; Wilcoxon on per-split F1; Friedman across algorithms.
