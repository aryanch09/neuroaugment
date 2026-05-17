"""NeuroAugment benchmark suite.

Protocols: LinearEval, FewShotFinetune, CrossSubjectLOSO
Baselines: RandomInit, SimCLRBaseline, NeuroAugBaseline, NeuroAugDRIOnly, NeuroAugInfoNCE, SupervisedBaseline
Runner:    BenchmarkRunner, ResultsTable, privacy_utility_sweep
"""
from neuroaugment.benchmarks.protocols import CrossSubjectLOSO, EvalResult, FewShotFinetune, LinearEval
from neuroaugment.benchmarks.baselines import (
    NeuroAugBaseline,
    NeuroAugDRIOnly,
    NeuroAugInfoNCE,
    RandomInit,
    SimCLRBaseline,
    SupervisedBaseline,
)
from neuroaugment.benchmarks.runner import BenchmarkRunner, ResultsTable, privacy_utility_sweep

__all__ = [
    "LinearEval", "FewShotFinetune", "CrossSubjectLOSO", "EvalResult",
    "RandomInit", "SimCLRBaseline", "NeuroAugBaseline",
    "NeuroAugDRIOnly", "NeuroAugInfoNCE", "SupervisedBaseline",
    "BenchmarkRunner", "ResultsTable", "privacy_utility_sweep",
]
