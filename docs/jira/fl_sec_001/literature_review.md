# Literature review — fl_sec_001

Source status is per entry in REFERENCES_MASTER.md (most are `REFERENCE_NOT_VERIFIED`; no access to publisher hosts); statements summarise well-known results in the author's words and must be checked against the papers.

## FL-SEC-001.1 Cryptographic approach selection (HE / SecAgg / SMPC)
Three families protect model updates from the aggregator: homomorphic encryption (CKKS [cheon2017], BFV [fv2012], Paillier [paillier1999]), pairwise-masking secure aggregation [bonawitz2017, bell2020] and secret-sharing MPC (SPDZ-style [damgard2012, beaver1991]). They differ in trust assumptions, dropout handling and cost; none protects against poisoning.

## FL-SEC-001.2 Homomorphic-encryption aggregation (CKKS, BFV, Paillier)
CKKS is approximate, BFV exact over integers with quantisation, Paillier additively homomorphic with one ciphertext per element [cheon2017, fv2012, paillier1999, benaloh_tenseal].

## FL-SEC-001.3 Secure aggregation with dropout recovery
Pairwise masks cancel in the sum; Shamir shares of seeds allow recovery when clients drop [bonawitz2017]; sparse neighbour graphs reduce cost [bell2020].

## FL-SEC-001.4 SMPC aggregation with authenticated shares (SPDZ-style)
SPDZ-style protocols authenticate shares with a global MAC key [damgard2012]; Beaver triples enable multiplication [beaver1991]. The offline phase here is a trusted dealer (NOT a real preprocessing protocol).

## FL-SEC-001.5 Comparative analysis of HE vs SecAgg vs SMPC
See tasks 1.1–1.4.

Keys used: `beaver1991`, `bell2020`, `benaloh_tenseal`, `bonawitz2017`, `cheon2017`, `damgard2012`, `fv2012`, `kairouz2021`, `paillier1999`, `shamir1979`
