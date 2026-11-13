"""Volatility: estimators (what it was), forecasting models (what it will be), evaluation, and exposure management."""
from . import estimators
from .estimators import close_to_close, garman_klass, parkinson, rogers_satchell, variance_proxy, vol_of_vol, volatility_cone, yang_zhang
from .evaluation import compare_vol_models, mincer_zarnowitz, mse, qlike, walk_forward_forecasts
from .management import VolatilityManager, VolRegime, variance_risk_premium, vol_target_scalar, vol_target_weights
from .models import MODELS, EWMAVol, GarchVol, HARVol, HistoricalVol, VolForecast, VolModel, forecast_volatility, make_model
