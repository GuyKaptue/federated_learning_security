"""Attack simulations: backdoor/poisoning, Byzantine, membership inference, gradient inversion, adaptive attackers."""
from .adaptive import (
    ATTACKS,
    GAMMA_GRID,
    AdaptiveAttackAggregator,
    MedianNormClip,
    agr_tailored_choose,
    alie_vector,
    alie_z,
    fang_vectors,
    minmax_gamma,
    minmax_vector,
    minsum_gamma,
    minsum_vector,
)
from .backdoor import TRIGGER_IDX, TRIGGER_VAL, apply_trigger, backdoor_metrics, boost_hook, poison_client_data
from .byzantine import byzantine_hook
from .gradient_inversion import analytic_first_layer_inversion, dlg_attack, match_and_score, model_grad
from .membership_inference import mia_metrics, per_sample_features, shadow_attack_scores

__all__ = ['ATTACKS', 'alie_z', 'alie_vector', 'minmax_gamma', 'minmax_vector', 'minsum_gamma', 'minsum_vector', 'fang_vectors', 'GAMMA_GRID', 'agr_tailored_choose', 'AdaptiveAttackAggregator', 'MedianNormClip', 'TRIGGER_IDX', 'TRIGGER_VAL', 'apply_trigger', 'poison_client_data', 'backdoor_metrics', 'boost_hook', 'byzantine_hook', 'model_grad', 'analytic_first_layer_inversion', 'dlg_attack', 'match_and_score', 'per_sample_features', 'mia_metrics', 'shadow_attack_scores']
