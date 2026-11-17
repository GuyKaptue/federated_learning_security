# Implementation — fl_attack_003

## FL-ATTACK-003.1 Backdoor attack and defenses
**Objective.** Measure backdoor attack success (trigger + poisoning + boosting) at three intensities and defence effect.

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_1.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.1`

## FL-ATTACK-003.2 Byzantine attacks (random, sign-flip, scale, drift)
**Objective.** Quantify accuracy damage of Byzantine updates and the effect of robust aggregators.

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_2.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.2`

## FL-ATTACK-003.3 Membership inference and privacy trade-off
**Objective.** Evaluate shadow-model and loss-baseline membership inference with and without clipping/DP noise.

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_3.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.3`

## FL-ATTACK-003.4 Gradient leakage (DLG/iDLG, analytic inversion)
**Objective.** Reconstruct inputs from shared gradients and test mitigation via aggregation and noise.

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_4.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.4`

## FL-ATTACK-003.5 Countermeasure evaluation (before/after)
**Objective.** Compare defences on each attack before and after, including secure aggregation (which gives no poisoning robustness).

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_5.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.5`

## FL-ATTACK-003.6 Risk matrix and KPI verification
**Objective.** Summarise attack/defence outcomes as a risk matrix and verify the story KPIs honestly.

**Modules.** `attacks.py`, `defenses.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_6.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.6`

## FL-ATTACK-003.7 Adaptive (defence-aware, omniscient) attacks
**Objective.** Test whether the defences that worked against static attacks (task 3.2/3.5) survive attackers that see the benign updates and adapt: ALIE, MIN-MAX, norm-matched backdoor, MIN-MAX backdoor.

**Modules.** `attacks_adaptive.py`, `defenses.py`, `fl.py`, `experiments/ext.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_7.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.7`

## FL-ATTACK-003.8 Stronger adaptive attacks: rule-aware (AGR-tailored), MIN-SUM, directed deviation, AGR-backdoor
**Objective.** Test the same seven aggregation rules against attackers that additionally know the server rule and simulate it, and against MIN-SUM and Fang-style directed deviation.

**Modules.** `attacks_adaptive.py`, `experiments/ext2.py`

**Notebook.** `notebooks/sprint_tasks/fl_attack_003/FL_ATTACK_003_8.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-ATTACK-003.8`
