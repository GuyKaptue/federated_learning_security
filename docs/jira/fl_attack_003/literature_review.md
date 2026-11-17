# Literature review — fl_attack_003

Source status is per entry in REFERENCES_MASTER.md (most are `REFERENCE_NOT_VERIFIED`; no access to publisher hosts); statements summarise well-known results in the author's words and must be checked against the papers.

## FL-ATTACK-003.1 Backdoor attack and defenses
Backdoors embed a trigger-conditioned behaviour while keeping clean accuracy [bagdasaryan2020, sun2019, wang2020tail].

## FL-ATTACK-003.2 Byzantine attacks (random, sign-flip, scale, drift)
Median, trimmed mean and Krum offer Byzantine robustness under assumptions on the malicious fraction [blanchard2017, yin2018, fang2020].

## FL-ATTACK-003.3 Membership inference and privacy trade-off
MIA exploits generalisation gap [shokri2017, nasr2019, carlini2022]; DP-SGD/clipping reduce leakage at utility cost [abadi2016, mironov2017].

## FL-ATTACK-003.4 Gradient leakage (DLG/iDLG, analytic inversion)
Gradients leak inputs [zhu2019, zhao2020, geiping2020]; for batch size 1 and a first dense layer, inversion is analytic.

## FL-ATTACK-003.5 Countermeasure evaluation (before/after)
See tasks 3.1–3.4 and [kairouz2021].

## FL-ATTACK-003.6 Risk matrix and KPI verification
Risk = likelihood × impact is qualitative; here scores derive from measured outcomes.

## FL-ATTACK-003.7 Adaptive (defence-aware, omniscient) attacks
ALIE shifts malicious updates by a small multiple of the benign standard deviation [baruch2019]; MIN-MAX/AGR-tailored attacks keep malicious updates inside the benign distance envelope [shejwalkar2021]; backdoors can be rescaled to pass norm bounds [bagdasaryan2020, sun2019, wang2020tail].

## FL-ATTACK-003.8 Stronger adaptive attacks: rule-aware (AGR-tailored), MIN-SUM, directed deviation, AGR-backdoor
Defence-aware attackers can defeat rank- and distance-based rules [shejwalkar2021, fang2020, baruch2019]; the AGR-tailored attacker here searches a fixed candidate set under the simulated rule.

Keys used: `abadi2016`, `bagdasaryan2020`, `baruch2019`, `blanchard2017`, `bonawitz2017`, `carlini2022`, `fang2020`, `geiping2020`, `kairouz2021`, `mironov2017`, `nasr2019`, `shejwalkar2021`, `shokri2017`, `sun2019`, `wang2020tail`, `yin2018`, `zhao2020`, `zhu2019`
