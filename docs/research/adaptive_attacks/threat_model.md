# Formal threat model (Gate 3, draft)

Setting: horizontal cross-silo FL, K = 20 virtual clients (K = 4 for the natural-site data), f malicious (default 4), R = 15-20 rounds, synthetic or small public tabular data, one aggregation server running rule A each round. All attackers are simulated; nothing is run against real systems.

| Attacker | Knowledge | Sees benign updates | Knows rule | History | Collusion | Status |
|---|---|---|---|---|---|---|
| A static Byzantine | none | no | no | no | no | implemented |
| B reactive | global model | no | no | yes (scalar strength) | no | not implemented |
| C aggregation-aware (AGR) | all benign updates of the round + rule | yes (omniscient, white-box) | yes | no | yes (identical rows) | implemented, fixed candidate set |
| D multi-round | global model history | no | no | yes | yes | not implemented (PoisonedFL-style) |
| E colluding / Sybil | as C, plus diversification | yes | yes | no | yes (jitter) | partly (weak jitter) |
| F learned | observation of rounds | varies | varies | yes | yes | not implemented |

Capabilities not assumed: control of the server, of the secure-aggregation keys, or of honest clients' data. Attackers C and E are upper bounds because real attackers rarely see benign updates; under secure aggregation this assumption would be violated and must be restated.

Control: malicious clients choose their submitted update freely (model poisoning) and may also train on poisoned local data (backdoor). Budget: f/K, number of rounds, optional norm constraint. Persistence: attack runs every round unless stated.

Defender: sees plaintext individual updates (no secure aggregation) in all planned experiments; knows f (Krum, trimmed mean) or not; no root dataset. Information the server sees, which party can detect, and dropouts: only defined for the plaintext case here. The secure-aggregation case is **not** analysed and no security claim is made for it.
