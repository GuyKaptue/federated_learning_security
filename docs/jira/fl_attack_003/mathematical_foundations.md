# Mathematical foundations — fl_attack_003

## FL-ATTACK-003.1 Backdoor attack and defenses
ASR = P(f(x⊕t)=y_target | y≠y_target); poisoned local objective L_clean + L_poison; boosting Δ' = γΔ.

## FL-ATTACK-003.2 Byzantine attacks (random, sign-flip, scale, drift)
Krum score s_i = Σ_{j∈N_i} ||Δ_i−Δ_j||², n ≥ 2f+3; median per coordinate; trimmed mean removes β fraction each side.

## FL-ATTACK-003.3 Membership inference and privacy trade-off
AUC of score s(x); advantage = max_τ(TPR−FPR); ε from RDP accountant (Opacus).

## FL-ATTACK-003.4 Gradient leakage (DLG/iDLG, analytic inversion)
min_x ||∇_W L(f(x),y) − g||²; analytic: x = (∂L/∂W₁ row)/(∂L/∂b₁ entry). Success = relative L2 error ≤ 0.1 after Hungarian matching.

## FL-ATTACK-003.5 Countermeasure evaluation (before/after)
Defence effect = metric_after − metric_before with CIs.

## FL-ATTACK-003.6 Risk matrix and KPI verification
KPI check: measured vs threshold.

## FL-ATTACK-003.7 Adaptive (defence-aware, omniscient) attacks
ALIE: v = mu - z*sigma, z = Phi^{-1}((n-s)/n), s = floor(n/2+1) - f. MIN-MAX: v = mu + gamma p with gamma = max{g: max_i ||mu + g p - b_i|| <= max_{i,j} ||b_i - b_j||}, p = -mu/||mu||. Norm-matched backdoor: d' = d * median_i||b_i|| / ||d||. MIN-MAX backdoor: p = d_m - mu.

## FL-ATTACK-003.8 Stronger adaptive attacks: rule-aware (AGR-tailored), MIN-SUM, directed deviation, AGR-backdoor
AGR-tailored: v* = argmin_c <agg(D with malicious rows = c), mu/||mu||> over candidates c = mu + g*||mu||*d (3 directions x 10 gammas) and ALIE/MIN-MAX/MIN-SUM/FANG. MIN-SUM: gamma = max{g: sum_i ||mu + g p - b_i||^2 <= max_i sum_j ||b_i - b_j||^2}.
