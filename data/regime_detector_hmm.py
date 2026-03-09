"""
HMM Volatility Regime & Confidence Estimator.

Uses a Gaussian Hidden Markov Model to classify market **volatility regimes**
(calm/normal/volatile) and provide calibrated confidence scores.

Key insight from walk-forward testing: HMM states capture volatility regimes,
NOT trend direction. SMA slope (+5.3% WF) vastly outperforms HMM (-18.1% WF)
for directional classification. Therefore:

  - SMA slope detector → direction (bull/bear/sideways) — kept as primary
  - HMM → volatility state + confidence score → adaptive sizing

The combined detector (detect_regime_combined) merges both:
  - SMA provides direction
  - HMM provides vol_state and regime_stability (confidence)
  - Low stability → reduce position size (Sprint 9: adaptive sizing)

Features:
  1. Log returns (magnitude, not direction)
  2. Realized volatility
  3. Volume ratio
  4. Absolute SMA slope (trend strength, not direction)
  5. Vol-of-vol (regime stability)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# Hyperparameters
N_STATES = 3          # calm, normal, volatile
N_ITER = 100
COV_TYPE = 'full'
RANDOM_SEED = 42
ROLLING_WINDOW = 20   # ~5h at 15m
SMA_SLOPE_WINDOW = 50
VOL_OF_VOL_WINDOW = 40

MODEL_DIR = Path(__file__).parent.parent / 'models'


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HMM observable features from OHLCV data."""
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1.0, index=df.index)

    log_ret = np.log(close / close.shift(1))
    real_vol = log_ret.rolling(ROLLING_WINDOW).std()
    vol_mean = volume.rolling(ROLLING_WINDOW).mean()
    vol_ratio = volume / vol_mean.replace(0, np.nan)
    sma = close.rolling(SMA_SLOPE_WINDOW).mean()
    abs_sma_slope = (sma.diff(5) / sma).abs()  # Absolute: strength, not direction
    vol_of_vol = real_vol.rolling(VOL_OF_VOL_WINDOW).std()

    features = pd.DataFrame({
        'log_ret': log_ret,
        'real_vol': real_vol,
        'vol_ratio': vol_ratio,
        'abs_sma_slope': abs_sma_slope,
        'vol_of_vol': vol_of_vol,
    }, index=df.index)

    # Clip extreme outliers (>5 sigma)
    for col in features.columns:
        valid = features[col].dropna()
        if len(valid) == 0:
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma > 0:
            features[col] = features[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    return features


def _map_states_to_vol_regimes(model: GaussianHMM, features: np.ndarray,
                                states: np.ndarray) -> dict:
    """
    Map HMM states to volatility regimes: calm/normal/volatile.

    Sorted by mean realized volatility (feature index 1).
    """
    n_states = model.n_components
    state_vol = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            state_vol[s] = 0.0
            continue
        state_vol[s] = float(np.mean(features[mask][:, 1]))  # real_vol

    sorted_by_vol = sorted(state_vol.keys(), key=lambda s: state_vol[s])

    labels = ['calm', 'normal', 'volatile']
    mapping = {}
    for i, s in enumerate(sorted_by_vol):
        mapping[s] = labels[min(i, len(labels) - 1)]

    logger.info("HMM vol-state mapping:")
    for s in range(n_states):
        logger.info(f"  State {s} → {mapping[s]}: mean_vol={state_vol[s]:.6f}")

    return mapping


class HMMVolatilityDetector:
    """
    HMM volatility regime detector.

    Classifies market into calm/normal/volatile states and provides
    calibrated posterior probabilities for regime stability scoring.
    """

    def __init__(self, n_states: int = N_STATES, random_seed: int = RANDOM_SEED):
        self.n_states = n_states
        self.random_seed = random_seed
        self.model: Optional[GaussianHMM] = None
        self.state_to_vol_regime: dict = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> 'HMMVolatilityDetector':
        """Train the HMM on OHLCV data."""
        features_df = _compute_features(df)
        clean = features_df.dropna()

        if len(clean) < 100:
            raise ValueError(f"Need at least 100 clean bars, got {len(clean)}")

        X = clean.values
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)
        self._feature_stds[self._feature_stds == 0] = 1.0
        X_scaled = (X - self._feature_means) / self._feature_stds

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=COV_TYPE,
            n_iter=N_ITER,
            random_state=self.random_seed,
            verbose=False,
        )
        self.model.fit(X_scaled)

        states = self.model.predict(X_scaled)
        self.state_to_vol_regime = _map_states_to_vol_regimes(
            self.model, X_scaled, states)

        logger.info(f"HMM trained: {self.n_states} states, "
                    f"{len(clean)} bars, converged={self.model.monitor_.converged}")
        return self

    def predict_vol_state(self, df: pd.DataFrame) -> pd.Series:
        """Classify each bar into calm/normal/volatile."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        features_df = _compute_features(df)
        result = pd.Series('normal', index=df.index)
        valid_mask = features_df.notna().all(axis=1)
        clean = features_df[valid_mask]

        if len(clean) == 0:
            return result

        X_scaled = (clean.values - self._feature_means) / self._feature_stds
        states = self.model.predict(X_scaled)
        result[valid_mask] = [self.state_to_vol_regime.get(s, 'normal') for s in states]
        return result

    def get_regime_stability(self, df: pd.DataFrame) -> dict:
        """
        Compute regime stability score for adaptive sizing.

        Returns:
            {
                'vol_state': 'calm'/'normal'/'volatile',
                'stability': float 0-1 (max posterior prob — high=stable regime),
                'vol_state_probs': {'calm': p, 'normal': p, 'volatile': p},
                'transition_risk': float 0-1 (1 - stability, high=likely changing),
            }
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        features_df = _compute_features(df)
        valid_mask = features_df.notna().all(axis=1)
        clean = features_df[valid_mask]

        if len(clean) == 0:
            return {'vol_state': 'normal', 'stability': 0.5,
                    'vol_state_probs': {'calm': 0.33, 'normal': 0.34, 'volatile': 0.33},
                    'transition_risk': 0.5}

        X_scaled = (clean.values - self._feature_means) / self._feature_stds
        posteriors = self.model.predict_proba(X_scaled)
        last_post = posteriors[-1]

        best_state = int(np.argmax(last_post))
        vol_state = self.state_to_vol_regime.get(best_state, 'normal')
        stability = float(last_post[best_state])

        vol_probs = {'calm': 0.0, 'normal': 0.0, 'volatile': 0.0}
        for s, prob in enumerate(last_post):
            label = self.state_to_vol_regime.get(s, 'normal')
            vol_probs[label] += float(prob)

        return {
            'vol_state': vol_state,
            'stability': round(stability, 3),
            'vol_state_probs': {k: round(v, 3) for k, v in vol_probs.items()},
            'transition_risk': round(1.0 - stability, 3),
        }

    def save(self, path: Optional[str] = None):
        """Save trained model to disk."""
        if path is None:
            MODEL_DIR.mkdir(exist_ok=True)
            path = str(MODEL_DIR / 'hmm_regime.pkl')
        data = {
            'model': self.model,
            'state_to_vol_regime': self.state_to_vol_regime,
            'feature_means': self._feature_means,
            'feature_stds': self._feature_stds,
            'n_states': self.n_states,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"HMM model saved to {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'HMMVolatilityDetector':
        """Load trained model from disk."""
        if path is None:
            path = str(MODEL_DIR / 'hmm_regime.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(n_states=data['n_states'])
        detector.model = data['model']
        detector.state_to_vol_regime = data['state_to_vol_regime']
        detector._feature_means = data['feature_means']
        detector._feature_stds = data['feature_stds']
        logger.info(f"HMM model loaded from {path}")
        return detector


# ============================================================================
# Combined detector: SMA direction + HMM confidence
# ============================================================================

def detect_regime_combined(df: pd.DataFrame,
                           hmm: Optional[HMMVolatilityDetector] = None) -> dict:
    """
    Combined regime detection: SMA for direction, HMM for confidence.

    Returns:
        {
            'regime': 'bull'/'bear'/'sideways' (from SMA slope),
            'confidence': float 0-1 (SMA confidence, adjusted by HMM stability),
            'slope': float,
            'vol_ratio': float,
            'confirmations': int,
            'vol_state': 'calm'/'normal'/'volatile' (from HMM),
            'stability': float 0-1 (HMM posterior — high=stable regime),
            'transition_risk': float 0-1 (high=regime may change),
            'sizing_mult': float 0-1 (position size multiplier for adaptive sizing),
        }
    """
    from data.regime_detector import detect_regime_with_confidence

    # SMA direction
    sma_result = detect_regime_with_confidence(df)

    # HMM stability
    if hmm is not None:
        hmm_result = hmm.get_regime_stability(df)
    else:
        hmm_result = _get_default_hmm_result(df)

    # Compute adaptive sizing multiplier
    sizing_mult = _compute_sizing_multiplier(
        sma_confidence=sma_result['confidence'],
        hmm_stability=hmm_result['stability'],
        vol_state=hmm_result['vol_state'],
    )

    return {
        # SMA direction (primary)
        'regime': sma_result['regime'],
        'confidence': sma_result['confidence'],
        'slope': sma_result['slope'],
        'vol_ratio': sma_result['vol_ratio'],
        'confirmations': sma_result['confirmations'],
        # HMM volatility (secondary)
        'vol_state': hmm_result['vol_state'],
        'stability': hmm_result['stability'],
        'transition_risk': hmm_result['transition_risk'],
        # Adaptive sizing
        'sizing_mult': sizing_mult,
    }


def _compute_sizing_multiplier(sma_confidence: float, hmm_stability: float,
                                vol_state: str) -> float:
    """
    Compute position size multiplier for adaptive sizing.

    Logic:
      - Base = average of SMA confidence and HMM stability
      - Penalty for volatile regime: -30%
      - Bonus for calm regime: +10%
      - Floor at 0.3 (never go below 30% of base size)
      - Cap at 1.0 (never exceed base size)

    Examples:
      - Bull, calm, high confidence → 1.0 (full size)
      - Bull, volatile, low stability → 0.3-0.5 (reduced)
      - Sideways, normal → 0.5-0.7 (moderate reduction)
    """
    base = 0.5 * sma_confidence + 0.5 * hmm_stability

    if vol_state == 'volatile':
        base *= 0.7  # 30% penalty
    elif vol_state == 'calm':
        base *= 1.1  # 10% bonus

    return round(max(0.3, min(1.0, base)), 2)


def _get_default_hmm_result(df: pd.DataFrame) -> dict:
    """Get HMM result, loading/training model as needed."""
    global _cached_hmm
    if _cached_hmm is None:
        default_path = MODEL_DIR / 'hmm_regime.pkl'
        if default_path.exists():
            try:
                _cached_hmm = HMMVolatilityDetector.load()
            except Exception as e:
                logger.warning(f"Failed to load HMM: {e}. Training new one.")
                _cached_hmm = HMMVolatilityDetector()
                _cached_hmm.fit(df)
                _cached_hmm.save()
        else:
            logger.info("No HMM model found. Training on provided data...")
            _cached_hmm = HMMVolatilityDetector()
            _cached_hmm.fit(df)
            _cached_hmm.save()

    return _cached_hmm.get_regime_stability(df)


_cached_hmm: Optional[HMMVolatilityDetector] = None


def train_and_save_hmm(df: pd.DataFrame, path: Optional[str] = None) -> HMMVolatilityDetector:
    """Train HMM on historical data and save to disk."""
    global _cached_hmm
    detector = HMMVolatilityDetector()
    detector.fit(df)
    detector.save(path)
    _cached_hmm = detector
    return detector
