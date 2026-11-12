"""Governance: who may promote what, on which evidence. Strategy lifecycle (four-eyes promotion) and the model registry."""
from .lifecycle import PromotionPolicy, PromotionRequest, Stage, StrategyLifecycle, StrategyRecord
from .models import STAGES, ModelRegistry, ModelVersion, fingerprint
