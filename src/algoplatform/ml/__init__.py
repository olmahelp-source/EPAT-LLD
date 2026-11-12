"""Machine learning: tree ensembles, classical models, unsupervised methods, ensembling and walk-forward prediction.

Leakage-safe labelling, cross-validation and the linear baselines live in ``research``; this package adds the model zoo.
"""
from .classic import KNN, GaussianNB, LassoModel
from .ensemble import StackingEnsemble, VotingEnsemble
from .metrics import (confusion, directional_accuracy, information_coefficient, precision_recall_f1, r2_score, roc_auc)
from .pipeline import forward_returns, signal_from_probability, walk_forward_predict
from .tree import DecisionTree, GradientBoosting, RandomForest
from .unsupervised import PCA, KMeans
