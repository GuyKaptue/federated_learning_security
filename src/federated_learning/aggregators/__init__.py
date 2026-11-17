"""Server-side aggregation rules and defences (common `Aggregator` interface)."""
from .audit import Audited
from .base import Aggregator
from .committee import RuleCommittee
from .defenses import AnomalyFiltered, ClippedMean
from .literature_defenses import FLAME, FoolsGold, RobustLearningRate, SignGuard
from .mean import WeightedMean
from .robust import CoordinateMedian, Krum, TrimmedMean

__all__ = ['Audited', 'Aggregator', 'ClippedMean', 'AnomalyFiltered', 'WeightedMean', 'CoordinateMedian', 'TrimmedMean', 'Krum', 'RuleCommittee', 'FLAME', 'FoolsGold', 'RobustLearningRate', 'SignGuard']
