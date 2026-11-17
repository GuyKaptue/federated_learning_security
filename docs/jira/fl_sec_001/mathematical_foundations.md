# Mathematical foundations — fl_sec_001

## FL-SEC-001.1 Cryptographic approach selection (HE / SecAgg / SMPC)
CKKS encodes a real vector into a polynomial ring element and supports additive aggregation with error ε ≈ 2^{-scale}; BFV works over Z_t with fixed-point quantisation q(x)=round(x·2^f); Paillier: Enc(m)=g^m r^n mod n², Enc(a)·Enc(b)=Enc(a+b). Shamir t-of-n sharing over a prime field. Weighted mean: w̄ = Σ p_k Δ_k, p_k = n_k/Σn.

## FL-SEC-001.2 Homomorphic-encryption aggregation (CKKS, BFV, Paillier)
Dec(Σ_k Enc(p_k Δ_k)) ≈ Σ_k p_k Δ_k; CKKS error bound ~ N·2^{-scale}; BFV fixed-point: Δ̂ = round(2^f Δ), overflow if |Σ| ≥ t/2.

## FL-SEC-001.3 Secure aggregation with dropout recovery
y_i = x_i + PRG(b_i) + Σ_{j>i} PRG(s_ij) − Σ_{j<i} PRG(s_ij) (mod 2^64); Σ y_i − Σ b-masks = Σ x_i; Shamir over GF(2^521−1) with threshold t.

## FL-SEC-001.4 SMPC aggregation with authenticated shares (SPDZ-style)
x = Σ_i x_i mod p, p=2^61−1; MAC m_j = α_j x; check Σ_i (m_j)_i − α_j x = 0 for each of K=2 keys; Beaver: z = c + εb + δa + εδ with ε=x−a, δ=y−b.

## FL-SEC-001.5 Comparative analysis of HE vs SecAgg vs SMPC
overhead% = 100·(t_secure − t_plain)/t_plain; accuracy loss (pp) = 100·(acc_plain − acc_secure).
