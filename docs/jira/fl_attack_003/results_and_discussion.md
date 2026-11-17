# Results and discussion — fl_attack_003: Attack simulation and defences

All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).

## Backdoor (task 3.1)

| level | asr | clean_acc | kpi_asr_lt_5pct |
|---|---|---|---|
| none(clean FL) | 0.046 | 0.7796 | True |
| low | 0.741 | 0.7721 | False |
| medium | 0.9972 | 0.7415 | False |
| high | 0.9993 | 0.6902 | False |

## Membership inference (task 3.3)

| config | eps | auc | auc_loss_baseline | tpr_at_fpr_1pct | utility_acc |
|---|---|---|---|---|---|
| none | nan | 0.5772 | 0.5892 | 0.0138 | 0.8414 |
| clip_only(C=0.5) | nan | 0.5778 | 0.5892 | 0.0109 | 0.8414 |
| dp_sigma=0.1 | 2311.7783 | 0.5809 | 0.5863 | 0.0148 | 0.8394 |
| dp_sigma=0.3 | 321.5279 | 0.5642 | 0.5719 | 0.0146 | 0.8267 |
| dp_sigma=1.0 | 48.8017 | 0.5211 | 0.5408 | 0.0132 | 0.7539 |

## KPI verification (task 3.6)

| kpi | measured | met | note |
|---|---|---|---|
| Backdoor success rate < 5% | 0.999 (high, undefended); best defended 0.054 | False | clean-FL ASR of the trigger = 0.046 |
| Gradient leakage success < 10% | B=1 undefended 1.00; secagg K=10 (B_eff=20) 0.00 | False | NOT met for plaintext small-batch updates; met only when updates are aggregated/noised (see task 3.4) |
| Membership inference AUC < 0.60 | undefended 0.577; lowest 0.521 | True | evaluated in an overfit regime on synthetic data |

## Risk matrix (task 3.6)

| attack | attacker | objective | assumptions | component | observed_impact | detection | mitigation | residual_risk | evidence |
|---|---|---|---|---|---|---|---|---|---|
| Backdoor (data poisoning + boosting) | compromised hospital client(s) (20-30%) | trigger-activated misclassification of attacks as benign | attacker controls local data & update; persistent participation; synthetic trigger | global model / aggregation | ASR 1.00 at 'high' vs 0.05 for clean FL (trigger-natural ASR) | norm/cosine anomaly filter; SMPC norm bound | best measured: anomaly_filter -> ASR 0.05 | HIGH | task_3_1_backdoor.json, task_3_5_countermeasures.json |
| Byzantine update manipulation | compromised/colluding clients (10-30%) | degrade or steer the global model | attacker sends arbitrary vectors; scale 5 | aggregation rule / availability | sign-flip 30%: mean aggregator acc 0.47 | median/trimmed mean/Krum, anomaly filter | best at 30% sign-flip: multi_krum acc 0.78 | MEDIUM (rules rely on honest majority; not valid under secagg/HE) | task_3_2_byzantine.json |
| Membership inference (shadow-model) | black-box observer / honest-but-curious server with the global model | infer a patient record's training membership | attacker has same-distribution shadow data; overfit regime (small training set, 128 hidden) | global model | AUC 0.577 undefended (generalisation gap 0.12) | none in protocol | lowest AUC 0.521 with dp_sigma=1.0 (utility 0.75; eps=48.80169282486775) | MEDIUM | task_3_3_membership_inference.json |
| Gradient leakage (DLG / analytic) | honest-but-curious server seeing individual plaintext updates | reconstruct clinical feature vectors | small batches, known labels, smooth activation (DLG), first-layer bias (analytic) | client->server channel | B=1 success 1.00; successful (>10%) in 10 of 15 batch/arch settings | none | secure aggregation / HE / SMPC hide individual updates from the server; noise degrades inversion (task 3.4) | LOW vs server when secagg/HE/SMPC used; HIGH for plaintext updates | task_3_4_gradient_leakage.json |

## Adaptive untargeted attacks, synthetic data, K=20, f=4 (task 3.7; test accuracy)

| defense | attack | acc | acc_ci | f1 |
|---|---|---|---|---|
| mean | none | 0.7772 | [0.772, 0.783] | 0.6996 |
| mean | alie | 0.7772 | [0.770, 0.785] | 0.6994 |
| mean | minmax | 0.756 | [0.749, 0.763] | 0.6623 |
| median | none | 0.7767 | [0.770, 0.783] | 0.6992 |
| median | alie | 0.7751 | [0.768, 0.782] | 0.6947 |
| median | minmax | 0.7582 | [0.749, 0.767] | 0.6688 |
| trimmed_mean | none | 0.7756 | [0.768, 0.783] | 0.6976 |
| trimmed_mean | alie | 0.7762 | [0.769, 0.784] | 0.6972 |
| trimmed_mean | minmax | 0.7578 | [0.748, 0.767] | 0.6678 |
| krum | none | 0.7681 | [0.762, 0.774] | 0.6889 |
| krum | alie | 0.7736 | [0.767, 0.780] | 0.6912 |
| krum | minmax | 0.7491 | [0.723, 0.775] | 0.6524 |
| multi_krum | none | 0.7751 | [0.770, 0.780] | 0.6977 |
| multi_krum | alie | 0.7764 | [0.770, 0.783] | 0.6986 |
| multi_krum | minmax | 0.7574 | [0.750, 0.765] | 0.6634 |
| clip_median_norm | none | 0.7744 | [0.768, 0.781] | 0.6954 |
| clip_median_norm | alie | 0.7747 | [0.767, 0.782] | 0.6952 |
| clip_median_norm | minmax | 0.7534 | [0.747, 0.760] | 0.6567 |
| anomaly_filter | none | 0.7769 | [0.772, 0.782] | 0.6989 |
| anomaly_filter | alie | 0.7768 | [0.770, 0.784] | 0.6989 |
| anomaly_filter | minmax | 0.7773 | [0.771, 0.783] | 0.6996 |

## Adaptive untargeted attacks, Cleveland heart disease, K=10, f=2 (task 3.7)

| defense | attack | acc | acc_ci | f1 |
|---|---|---|---|---|
| mean | none | 0.8462 | [0.827, 0.865] | 0.8614 |
| mean | alie | 0.833 | [0.802, 0.864] | 0.8474 |
| mean | minmax | 0.8154 | [0.771, 0.860] | 0.8312 |
| median | none | 0.8484 | [0.831, 0.866] | 0.8641 |
| median | alie | 0.8374 | [0.801, 0.874] | 0.8518 |
| median | minmax | 0.8308 | [0.795, 0.866] | 0.8445 |
| trimmed_mean | none | 0.8462 | [0.827, 0.865] | 0.8618 |
| trimmed_mean | alie | 0.844 | [0.814, 0.874] | 0.8577 |
| trimmed_mean | minmax | 0.8198 | [0.797, 0.843] | 0.837 |
| krum | none | 0.8352 | [0.816, 0.854] | 0.8519 |
| krum | alie | 0.8352 | [0.810, 0.861] | 0.8499 |
| krum | minmax | 0.778 | [0.683, 0.873] | 0.787 |
| multi_krum | none | 0.8418 | [0.817, 0.867] | 0.8585 |
| multi_krum | alie | 0.8352 | [0.805, 0.866] | 0.8506 |
| multi_krum | minmax | 0.822 | [0.796, 0.848] | 0.8389 |
| clip_median_norm | none | 0.8462 | [0.833, 0.860] | 0.862 |
| clip_median_norm | alie | 0.833 | [0.802, 0.864] | 0.8474 |
| clip_median_norm | minmax | 0.811 | [0.781, 0.841] | 0.8276 |
| anomaly_filter | none | 0.8484 | [0.824, 0.873] | 0.8635 |
| anomaly_filter | alie | 0.8352 | [0.800, 0.870] | 0.8495 |
| anomaly_filter | minmax | 0.8352 | [0.800, 0.870] | 0.8484 |

## Static vs adaptive backdoor (task 3.7, synthetic)

| defense | attack | asr | asr_ci | clean_acc |
|---|---|---|---|---|
| mean | none_clean | 0.1064 | [-0.013, 0.226] | 0.7772 |
| mean | static_boost | 0.9978 | [0.997, 0.999] | 0.7378 |
| mean | norm_matched | 0.7532 | [0.577, 0.929] | 0.7607 |
| mean | minmax_backdoor | 0.6363 | [0.381, 0.892] | 0.7704 |
| median | none_clean | 0.1133 | [-0.014, 0.240] | 0.7767 |
| median | static_boost | 0.2606 | [0.015, 0.506] | 0.7742 |
| median | norm_matched | 0.2568 | [0.037, 0.477] | 0.7658 |
| median | minmax_backdoor | 0.2662 | [0.029, 0.503] | 0.7706 |
| trimmed_mean | none_clean | 0.1138 | [-0.014, 0.241] | 0.7756 |
| trimmed_mean | static_boost | 0.2883 | [0.032, 0.545] | 0.7725 |
| trimmed_mean | norm_matched | 0.2896 | [0.061, 0.519] | 0.7658 |
| trimmed_mean | minmax_backdoor | 0.311 | [0.063, 0.559] | 0.77 |
| krum | none_clean | 0.0968 | [-0.001, 0.194] | 0.7681 |
| krum | static_boost | 0.1204 | [-0.018, 0.259] | 0.7697 |
| krum | norm_matched | 0.1837 | [-0.062, 0.429] | 0.7658 |
| krum | minmax_backdoor | 0.9961 | [0.995, 0.997] | 0.7343 |
| multi_krum | none_clean | 0.104 | [-0.010, 0.218] | 0.7751 |
| multi_krum | static_boost | 0.1061 | [-0.010, 0.223] | 0.7768 |
| multi_krum | norm_matched | 0.5566 | [0.256, 0.858] | 0.7648 |
| multi_krum | minmax_backdoor | 0.763 | [0.590, 0.936] | 0.7672 |
| clip_median_norm | none_clean | 0.111 | [-0.011, 0.234] | 0.7744 |
| clip_median_norm | static_boost | 0.7881 | [0.629, 0.947] | 0.7586 |
| clip_median_norm | norm_matched | 0.775 | [0.612, 0.938] | 0.7583 |
| clip_median_norm | minmax_backdoor | 0.6391 | [0.389, 0.889] | 0.7671 |
| anomaly_filter | none_clean | 0.1069 | [-0.012, 0.225] | 0.7769 |
| anomaly_filter | static_boost | 0.1062 | [-0.010, 0.223] | 0.7771 |
| anomaly_filter | norm_matched | 0.4981 | [0.215, 0.781] | 0.7668 |
| anomaly_filter | minmax_backdoor | 0.6492 | [0.408, 0.890] | 0.7699 |

## Stronger adaptive attacks, worst case per rule (task 3.8)

| dataset | defense | clean_acc | worst_attack | worst_acc | drop | agr_tailored_acc |
|---|---|---|---|---|---|---|
| syn | mean | 0.7772 | agr_tailored | 0.3664 | 0.4108 | 0.3664 |
| syn | median | 0.7767 | fang | 0.7564 | 0.0203 | 0.7564 |
| syn | trimmed_mean | 0.7756 | fang | 0.7553 | 0.0203 | 0.7553 |
| syn | krum | 0.7681 | agr_tailored | 0.7039 | 0.0642 | 0.7039 |
| syn | multi_krum | 0.7751 | agr_tailored | 0.7532 | 0.0219 | 0.7532 |
| syn | clip_median_norm | 0.7744 | agr_tailored | 0.7436 | 0.0308 | 0.7436 |
| syn | anomaly_filter | 0.7769 | agr_tailored | 0.7607 | 0.0162 | 0.7607 |
| syn_f2 | mean | 0.7772 | agr_tailored | 0.3883 | 0.3888 | 0.3883 |
| syn_f2 | median | 0.7767 | minmax | 0.7675 | 0.0092 | 0.7676 |
| syn_f2 | trimmed_mean | 0.7758 | agr_tailored | 0.7661 | 0.0098 | 0.7661 |
| syn_f2 | krum | 0.7704 | agr_tailored | 0.7337 | 0.0367 | 0.7337 |
| syn_f2 | multi_krum | 0.7767 | agr_tailored | 0.7647 | 0.012 | 0.7647 |
| syn_f2 | clip_median_norm | 0.7744 | agr_tailored | 0.7587 | 0.0157 | 0.7587 |
| syn_f2 | anomaly_filter | 0.7769 | agr_tailored | 0.7734 | 0.0036 | 0.7734 |
| syn_f8 | mean | 0.7772 | agr_tailored | 0.4688 | 0.3084 | 0.4688 |
| syn_f8 | median | 0.7767 | agr_tailored | 0.7152 | 0.0615 | 0.7152 |
| syn_f8 | trimmed_mean | 0.7767 | agr_tailored | 0.7152 | 0.0615 | 0.7152 |
| syn_f8 | krum | 0.7681 | agr_tailored | 0.4972 | 0.2709 | 0.4972 |
| syn_f8 | multi_krum | 0.7749 | agr_tailored | 0.6165 | 0.1584 | 0.6165 |
| syn_f8 | clip_median_norm | 0.7744 | agr_tailored | 0.6144 | 0.1601 | 0.6144 |
| syn_f8 | anomaly_filter | 0.7769 | agr_tailored | 0.7188 | 0.0581 | 0.7188 |
| wdbc | mean | 0.9404 | agr_tailored | 0.3743 | 0.5661 | 0.3743 |
| wdbc | median | 0.9392 | fang | 0.9228 | 0.0164 | 0.9228 |
| wdbc | trimmed_mean | 0.9404 | fang | 0.9216 | 0.0187 | 0.9216 |
| wdbc | krum | 0.9287 | minsum | 0.9251 | 0.0035 | 0.9251 |
| wdbc | multi_krum | 0.9404 | agr_tailored | 0.9263 | 0.014 | 0.9263 |
| wdbc | clip_median_norm | 0.938 | agr_tailored | 0.924 | 0.014 | 0.924 |
| wdbc | anomaly_filter | 0.9404 | agr_tailored | 0.9333 | 0.007 | 0.9333 |

## Stronger untargeted attacks, synthetic K=20 f=4 (task 3.8; test accuracy)

| defense | attack | acc | acc_ci | f1 |
|---|---|---|---|---|
| mean | none | 0.7772 | [0.7715, 0.7828] | 0.6996 |
| mean | alie | 0.7772 | [0.7697, 0.7846] | 0.6994 |
| mean | minmax | 0.756 | [0.7493, 0.7627] | 0.6623 |
| mean | minsum | 0.7565 | [0.7508, 0.7622] | 0.6641 |
| mean | fang | 0.7504 | [0.7416, 0.7592] | 0.6576 |
| mean | agr_tailored | 0.3664 | [0.3052, 0.4276] | 0.4982 |
| median | none | 0.7767 | [0.7701, 0.7834] | 0.6992 |
| median | alie | 0.7751 | [0.7682, 0.7819] | 0.6947 |
| median | minmax | 0.7582 | [0.7490, 0.7674] | 0.6688 |
| median | minsum | 0.7583 | [0.7492, 0.7675] | 0.6692 |
| median | fang | 0.7564 | [0.7488, 0.7640] | 0.6663 |
| median | agr_tailored | 0.7564 | [0.7488, 0.7640] | 0.6663 |
| trimmed_mean | none | 0.7756 | [0.7682, 0.7831] | 0.6976 |
| trimmed_mean | alie | 0.7762 | [0.7685, 0.7838] | 0.6972 |
| trimmed_mean | minmax | 0.7578 | [0.7484, 0.7672] | 0.6678 |
| trimmed_mean | minsum | 0.7581 | [0.7490, 0.7672] | 0.6682 |
| trimmed_mean | fang | 0.7553 | [0.7475, 0.7631] | 0.6643 |
| trimmed_mean | agr_tailored | 0.7553 | [0.7475, 0.7631] | 0.6643 |
| krum | none | 0.7681 | [0.7619, 0.7742] | 0.6889 |
| krum | alie | 0.7736 | [0.7672, 0.7799] | 0.6912 |
| krum | minmax | 0.7491 | [0.7229, 0.7752] | 0.6524 |
| krum | minsum | 0.7649 | [0.7510, 0.7788] | 0.681 |
| krum | fang | 0.7697 | [0.7645, 0.7750] | 0.6881 |
| krum | agr_tailored | 0.7039 | [0.6822, 0.7255] | 0.5187 |
| multi_krum | none | 0.7751 | [0.7698, 0.7804] | 0.6977 |
| multi_krum | alie | 0.7764 | [0.7700, 0.7828] | 0.6986 |
| multi_krum | minmax | 0.7574 | [0.7503, 0.7646] | 0.6634 |
| multi_krum | minsum | 0.7547 | [0.7476, 0.7617] | 0.6582 |
| multi_krum | fang | 0.7768 | [0.7708, 0.7828] | 0.6996 |
| multi_krum | agr_tailored | 0.7532 | [0.7464, 0.7600] | 0.6545 |
| clip_median_norm | none | 0.7744 | [0.7684, 0.7805] | 0.6954 |
| clip_median_norm | alie | 0.7747 | [0.7672, 0.7823] | 0.6952 |
| clip_median_norm | minmax | 0.7534 | [0.7466, 0.7602] | 0.6567 |
| clip_median_norm | minsum | 0.7548 | [0.7475, 0.7622] | 0.66 |
| clip_median_norm | fang | 0.7544 | [0.7472, 0.7616] | 0.6626 |
| clip_median_norm | agr_tailored | 0.7436 | [0.7370, 0.7502] | 0.633 |
| anomaly_filter | none | 0.7769 | [0.7715, 0.7824] | 0.6989 |
| anomaly_filter | alie | 0.7768 | [0.7697, 0.7840] | 0.6989 |
| anomaly_filter | minmax | 0.7773 | [0.7712, 0.7833] | 0.6996 |
| anomaly_filter | minsum | 0.7771 | [0.7711, 0.7830] | 0.6996 |

## AGR-aware backdoor (task 3.8, synthetic)

| defense | attack | asr | asr_ci | clean_acc |
|---|---|---|---|---|
| mean | none_clean | 0.1064 | [-0.0131, 0.2258] | 0.7772 |
| mean | static_boost | 0.9978 | [0.9967, 0.9989] | 0.7378 |
| mean | minmax_backdoor | 0.6363 | [0.3805, 0.8921] | 0.7704 |
| mean | agr_backdoor | 1.0 | [1.0000, 1.0000] | 0.5978 |
| median | none_clean | 0.1133 | [-0.0136, 0.2402] | 0.7767 |
| median | static_boost | 0.2606 | [0.0152, 0.5060] | 0.7742 |
| median | minmax_backdoor | 0.2662 | [0.0293, 0.5030] | 0.7706 |
| median | agr_backdoor | 0.2497 | [0.0150, 0.4845] | 0.7692 |
| trimmed_mean | none_clean | 0.1138 | [-0.0135, 0.2411] | 0.7756 |
| trimmed_mean | static_boost | 0.2883 | [0.0319, 0.5446] | 0.7725 |
| trimmed_mean | minmax_backdoor | 0.311 | [0.0628, 0.5593] | 0.77 |
| trimmed_mean | agr_backdoor | 0.2749 | [0.0307, 0.5190] | 0.7685 |
| krum | none_clean | 0.0968 | [-0.0008, 0.1944] | 0.7681 |
| krum | static_boost | 0.1204 | [-0.0179, 0.2588] | 0.7697 |
| krum | minmax_backdoor | 0.9961 | [0.9952, 0.9971] | 0.7343 |
| krum | agr_backdoor | 0.9965 | [0.9947, 0.9984] | 0.7303 |
| multi_krum | none_clean | 0.104 | [-0.0102, 0.2182] | 0.7751 |
| multi_krum | static_boost | 0.1061 | [-0.0104, 0.2226] | 0.7768 |
| multi_krum | minmax_backdoor | 0.763 | [0.5904, 0.9356] | 0.7672 |
| multi_krum | agr_backdoor | 0.7805 | [0.6222, 0.9388] | 0.7663 |
| clip_median_norm | none_clean | 0.111 | [-0.0115, 0.2336] | 0.7744 |
| clip_median_norm | static_boost | 0.7881 | [0.6292, 0.9470] | 0.7586 |
| clip_median_norm | minmax_backdoor | 0.6391 | [0.3893, 0.8889] | 0.7671 |
| clip_median_norm | agr_backdoor | 0.7757 | [0.6148, 0.9366] | 0.7547 |
| anomaly_filter | none_clean | 0.1069 | [-0.0117, 0.2255] | 0.7769 |
| anomaly_filter | static_boost | 0.1062 | [-0.0103, 0.2227] | 0.7771 |
| anomaly_filter | minmax_backdoor | 0.6492 | [0.4084, 0.8899] | 0.7699 |
| anomaly_filter | agr_backdoor | 0.8144 | [0.6647, 0.9640] | 0.7664 |

## AGR-tailored vs strongest fixed attack, paired by seed (task 3.8; uncorrected p)

| dataset | defense | best_fixed_attack | mean_diff | ttest_p | wilcoxon_p |
|---|---|---|---|---|---|
| syn | mean | fang | -0.384 | 0.0001 | 0.0625 |
| syn | median | fang | 0.0 | 1.0 | 1.0 |
| syn | trimmed_mean | fang | 0.0 | 1.0 | 1.0 |
| syn | krum | minmax | -0.0452 | 0.0001 | 0.0625 |
| syn | multi_krum | minsum | -0.0014 | 0.007 | 0.0625 |
| syn | clip_median_norm | minmax | -0.0098 | 0.001 | 0.0625 |
| syn | anomaly_filter | fang | -0.0151 | 0.0002 | 0.0625 |
| syn_f2 | mean | minmax | -0.3771 | 0.0 | 0.0625 |
| syn_f2 | median | minmax | 0.0001 | 0.8276 | 1.0 |
| syn_f2 | trimmed_mean | minmax | -0.0011 | 0.0856 | 0.25 |
| syn_f2 | krum | minmax | -0.0341 | 0.0011 | 0.0625 |
| syn_f2 | multi_krum | minmax | -0.0058 | 0.0015 | 0.0625 |
| syn_f2 | clip_median_norm | minmax | -0.0054 | 0.013 | 0.0625 |
| syn_f2 | anomaly_filter | minmax | -0.0037 | 0.0296 | 0.125 |
| syn_f8 | mean | minmax | -0.2697 | 0.0023 | 0.0625 |
| syn_f8 | median | minmax | -0.0142 | 0.0048 | 0.0625 |
| syn_f8 | trimmed_mean | minmax | -0.0142 | 0.0048 | 0.0625 |
| syn_f8 | krum | minmax | -0.0953 | 0.0449 | 0.125 |
| syn_f8 | multi_krum | minmax | -0.0764 | 0.0034 | 0.0625 |
| syn_f8 | clip_median_norm | minmax | -0.1207 | 0.0001 | 0.0625 |
| syn_f8 | anomaly_filter | minmax | -0.025 | 0.0014 | 0.0625 |
| wdbc | mean | minmax | -0.5556 | 0.0 | 0.0625 |
| wdbc | median | fang | 0.0 | 1.0 | 1.0 |
| wdbc | trimmed_mean | fang | 0.0 | 1.0 | 1.0 |
| wdbc | krum | minsum | 0.0 | 1.0 | 0.9375 |
| wdbc | multi_krum | minsum | -0.0023 | 0.3739 | 0.5 |
| wdbc | clip_median_norm | fang | -0.0035 | 0.4676 | 0.625 |
| wdbc | anomaly_filter | fang | -0.0035 | 0.0705 | 0.25 |


### Acceptance verdicts
* Four attack families (backdoor, Byzantine, membership inference, gradient leakage) implemented with low/medium/high intensities, before/after defences: MET (Byzantine/MIA/DLG grids in `RESULTS/`).
* KPI *backdoor success < 5%*: **NOT MET** without defence (see 3.1 table); defended runs reach the clean-model trigger ASR with Multi-Krum/anomaly filtering/SMPC norm bound, but median/trimmed-mean/clipping do not in the high-intensity case (task 3.5 table).
* KPI *gradient leakage success < 10%*: **NOT MET** for plaintext small-batch updates; met only for aggregated/noised updates (task 3.4).
* KPI *membership inference AUC < 0.60*: met in the evaluated regime (synthetic, modest overfitting) – not a general claim.
* Secure aggregation and HE give **no** protection against poisoning (measured equal to the undefended ASR).

### Adaptive attacks (task 3.7; omniscient attacker, upper bound on knowledge)
* **Adaptive backdoors defeat the defences that stopped the static backdoor.** Static boosted backdoor under Multi-Krum / anomaly filter: ASR 0.106 (= clean-model level). Norm-matched backdoor: 0.557 / 0.498; MIN-MAX backdoor: 0.763 / 0.649; plain Krum: static 0.120 → MIN-MAX 0.996 (tables above). Median and trimmed mean hold ASR near 0.26-0.31 for all variants (above the 5% KPI, far below the undefended 0.99) at no measured clean-accuracy cost.
* **Untargeted ALIE and MIN-MAX did little damage in this small-model setting** (synthetic: ALIE ≈ 0 pp, MIN-MAX ≤ 2.5 pp; heart: up to 5.7 pp for Krum under MIN-MAX). The hypothesis "an adaptive untargeted attack breaks the robust aggregators" is **not supported** here; this may reflect an easy task and a small MLP, not general robustness.
* KPI *backdoor success < 5%* therefore remains **NOT MET** under adaptive attack for every defence tested.

### Task 3.8 verdicts (stronger, rule-aware attackers; computed from the tables above)
* An attacker that simulates the server rule (AGR-tailored) drops the **undefended mean** to 0.37 accuracy on synthetic data and 0.37 on WDBC (below the majority-class rate; 5 of 5 WDBC runs and 1 synthetic f=8 run ended with non-finite weights, scored as constant class-0 predictors).
* Against the robust rules at f=4 of 20 the worst-case accuracy drop is small (about 0.02-0.06); at f=8 of 20 it grows to 0.06 (median/trimmed mean) up to 0.27 (Krum): none of the tested rules is robust at 40% attackers.
* AGR-tailored is the strongest attack for most rules but not for median/trimmed mean at f=4, where Fang's directed deviation is as strong.
* AGR-aware backdoor: ASR >= 0.75 for mean, Krum, Multi-Krum, clipping and the anomaly filter; median/trimmed mean stay near 0.25-0.31 (clean-model ASR ~0.11).
* The attacker is an omniscient, rule-aware upper bound that searches a fixed candidate set; no learning-based attacker was implemented. 5 seeds only.

## Limitations
- Single trigger family, tabular data; adaptive attacks are omniscient (upper bound) and not optimised further per defence; DLG evaluated on sigmoid MLPs only.
- Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network.
