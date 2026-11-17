"""FoolsGoldIndicative: hand-computed toy checks. These are software validation, not experimental results."""
import numpy as np
import pytest

from federated_learning.aggregators.foolsgold_indicative import (
    FoolsGoldIndicative,
    make_foolsgold,
    output_layer_from_model,
    output_layer_weight_index,
)
from federated_learning.aggregators.literature_defenses import FoolsGold
from federated_learning.pipelines.experiments.stage2 import make_defence
from federated_learning.training.models import build_model


def _ag(idx, n_out=1, **kw):
    return FoolsGoldIndicative(None if idx is None else np.asarray(idx), n_out, **kw)


def test_output_layer_mapping_mlp_logreg_deep():
    m = build_model(40)                                   # 40*32+32 | 2*32+2 -> weight block 1312..1375
    idx, n_out = output_layer_from_model(m)
    assert (idx[0], idx[-1], len(idx), n_out) == (1312, 1375, 64, 2)
    assert output_layer_weight_index(1378, 2, 32)[0] == 1312
    i2, _ = output_layer_from_model(build_model(40, arch="logreg"))
    assert (i2[0], i2[-1]) == (0, 79)
    i3, _ = output_layer_from_model(build_model(40, arch="deep_mlp"))
    assert (i3[0], i3[-1]) == (2896, 2927)
    with pytest.raises(ValueError):
        output_layer_weight_index(10, 2, 32)


def test_importance_soft_and_hard_hand_values():
    w = np.array([1.0, -2.0, 4.0, 3.0])                   # classes: [1,-2] and [4,3] -> |W| / row max = [[.5,1],[1,.75]]
    assert np.allclose(_ag(range(4), 2, mode="soft").importance(w), [0.5, 1.0, 1.0, 0.75])
    assert np.allclose(_ag(range(4), 2, mode="hard", proportion=0.5).importance(w), [0, 1, 1, 0])      # top 2 of 4
    assert np.allclose(_ag(range(4), 2, mode="hard", proportion=0.25).importance(w), [0, 1, 1, 0])     # ties at the threshold are kept
    assert np.allclose(_ag(range(4), 2, mode="hard", proportion=1.0).importance(w), 1.0)
    assert np.allclose(_ag(range(4), 2, mode="soft").importance(np.zeros(4)), 1.0)                      # all-zero output layer -> uniform fallback


def test_alpha_interior_value_matches_legacy_by_hand():
    H = np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 1.0]])   # cs01 = 0.5, others 0 -> alpha = [.5, .5, 1] (logit(.5)+.5 = .5)
    a_new = _ag([0, 1, 2])._alpha_paper(H)
    assert np.allclose(a_new, [0.5, 0.5, 1.0])
    assert np.allclose(FoolsGold()._alpha(H), [0.5, 0.5, 1.0])


def test_alpha_exact_identical_sybils_clipped_to_zero_while_legacy_returns_one():
    H = np.array([[1, 0], [1, 0], [0, 1.0]])
    assert np.allclose(_ag([0, 1])._alpha_paper(H), [0, 0, 1])          # ln(0) = -inf is clipped to 0 (paper: out-of-range values are clipped)
    assert np.allclose(FoolsGold()._alpha(H), [1, 1, 1])                # legacy behaviour, documented in the audit; must stay unchanged


def test_noise_diversified_sybils_flagged_only_on_indicative_features():
    # params: [f0, f1 | out w (2 classes x 1 input) | f4, f5]; sybils differ hugely on non-output features, agree on the output layer
    D = np.array([[5, 0, 1, 1, 0, 0], [-5, 0, 1, 1, 0, 0], [0, 0, 1, -1, 0, 0]], float)
    g = np.array([0, 0, 1.0, 1.0, 0, 0])                                  # |W_out| = [1,1] -> S = [1,1]
    ids, w = [0, 1, 2], np.ones(3)
    ind = _ag([2, 3], 2)
    ind.set_round_context(ids, g)
    out = ind.aggregate(D, w)
    assert np.allclose(ind.last_stats["alpha"], [0, 0, 1])
    assert np.allclose(out, D[2] / 3, atol=1e-6)                          # only the honest row survives, weight 1/3
    allf = _ag(None, 2, mode="all")
    allf.set_round_context(ids, g)
    allf.aggregate(D, w)
    assert np.allclose(allf.last_stats["alpha"], [1, 1, 1])               # whole-vector similarity is fooled by the noise
    leg = FoolsGold(); leg.set_round_context(ids, g); leg.aggregate(D, w)
    assert np.allclose(leg.last_stats["alpha"], [1, 1, 1])


def test_history_accumulates_across_rounds():
    ag = _ag([0, 1], 1)                                                   # n_out=1, two inputs
    g = np.array([1.0, 1.0])
    ag.set_round_context([0, 1, 2], g)
    ag.aggregate(np.array([[1, 0], [0, 1], [1, -1.0]]), np.ones(3))
    assert not np.allclose(ag.last_stats["alpha"][:2], 0)                 # round 1: clients 0 and 1 are orthogonal
    ag.set_round_context([0, 1, 2], g)
    ag.aggregate(np.array([[0, 1], [1, 0], [1, -1.0]]), np.ones(3))
    assert np.allclose(ag.history[0], [1, 1]) and np.allclose(ag.history[1], [1, 1]) and np.allclose(ag.history[2], [2, -2])
    assert np.allclose(ag.last_stats["alpha"], [0, 0, 1])                 # histories of 0 and 1 are now identical


def test_all_mode_equals_legacy_on_generic_input_and_variants_are_reproducible():
    rng = np.random.default_rng(3)
    D = rng.normal(size=(6, 12)); w = rng.random(6) + 0.1; g = rng.normal(size=12)
    ids = list(range(6))
    a, b, leg = _ag(None, 2, mode="all"), _ag(None, 2, mode="all"), FoolsGold()
    for x in (a, b, leg):
        x.set_round_context(ids, g)
    oa, ob, ol = a.aggregate(D, w), b.aggregate(D, w), leg.aggregate(D, w)
    assert np.array_equal(oa, ob) and np.allclose(oa, ol, atol=1e-6)
    s1, s2 = _ag(range(6, 12), 2, mode="soft"), _ag(range(6, 12), 2, mode="soft")
    for x in (s1, s2):
        x.set_round_context(ids, g)
    assert np.array_equal(s1.aggregate(D, w), s2.aggregate(D, w))


def test_alpha_bounds_and_edge_cases():
    rng = np.random.default_rng(1)
    ag = _ag([0, 1, 2])
    a = ag._alpha_paper(rng.normal(size=(8, 3)))
    assert np.all((a >= 0) & (a <= 1)) and a.max() == 1.0
    assert np.allclose(ag._alpha_paper(np.ones((1, 3))), [1.0])
    assert np.allclose(ag._alpha_paper(np.ones((3, 3))), [1, 1, 1])      # everyone identical: max alpha undefined -> all ones (as legacy)
    assert np.all(np.isfinite(ag._alpha_paper(np.zeros((3, 3)))))        # zero histories: cosine 0, no NaN
    with pytest.raises(RuntimeError):
        _ag([0, 1]).aggregate(np.ones((2, 2)), np.ones(2))               # no context
    z = _ag([0, 1]); z.set_round_context([0, 1], None)
    with pytest.raises(RuntimeError):
        z.aggregate(np.ones((2, 2)), np.ones(2))                         # no global model
    big = _ag([5, 6]); big.set_round_context([0, 1], np.ones(7))
    with pytest.raises(ValueError):
        big.aggregate(np.ones((2, 3)), np.ones(2))                       # index outside the update
    for bad in (dict(mode="x"), dict(mode="hard", proportion=0.0)):
        with pytest.raises(ValueError):
            _ag([0, 1], **bad)
    with pytest.raises(ValueError):
        FoolsGoldIndicative(np.arange(5), 2)                              # 5 weights, 2 heads: unsupported layout
    with pytest.raises(ValueError):
        FoolsGoldIndicative(None, 2, "soft")


def test_make_foolsgold_variants():
    assert type(make_foolsgold("legacy")) is FoolsGold
    for v, mode in (("paper_soft", "soft"), ("paper_hard", "hard"), ("allfeat_corrected", "all")):
        assert make_foolsgold(v, np.arange(4), 2).mode == mode
    with pytest.raises(ValueError):
        make_foolsgold("nope")


def test_make_defence_foolsgold_name_still_returns_legacy_class():
    assert type(make_defence("foolsgold")) is FoolsGold


def test_make_defence_foolsgold_corrected_returns_allfeat_variant():
    d = make_defence("foolsgold_corrected")
    assert isinstance(d, FoolsGoldIndicative) and d.mode == "all" and d.out_idx is None


def test_make_defence_identical_colluders_legacy_full_weight_corrected_zero():
    H = np.array([[1, 0], [1, 0], [0, 1.0]])
    assert np.allclose(make_defence("foolsgold")._alpha(H), [1, 1, 1])
    assert np.allclose(make_defence("foolsgold_corrected")._alpha_paper(H), [0, 0, 1])
