"""
Signal generation for live trading.

Decodes strategies from genomes and evaluates them on live OHLCV data.
Reuses the same vectorized evaluation engine from backtesting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from grammar.mapper import decode
from strategy.vectorized_eval import generate_signals, IndicatorCache, compute_atr
from evolution.param_extractor import extract_params, rebuild_strategy
from live.config import StrategyConfig

logger = logging.getLogger(__name__)


class LiveSignalEngine:
    """
    Evaluates strategy signals on live data.

    Each strategy is decoded from its genome once. On each new candle,
    we re-evaluate all indicators and check the latest bar for signals.
    """

    def __init__(self, strategy_configs: List[StrategyConfig]):
        self.strategy_configs = strategy_configs
        self.decoded_strategies = {}

        # Decode all strategies at startup
        for sc in strategy_configs:
            strategy = decode(sc.genome)
            if strategy is None:
                raise ValueError(f"Failed to decode strategy {sc.key}. Genome invalid.")

            # Apply CMA-ES parameter overrides if present
            if sc.cmaes_params:
                param_specs = extract_params(strategy)
                param_vector = []
                for ps in param_specs:
                    param_vector.append(sc.cmaes_params.get(ps.name, ps.value))
                strategy = rebuild_strategy(strategy, param_vector, param_specs)
                logger.info(f"Decoded {sc.key} [CMA-ES]: {strategy.expression_raw[:80]}")
            else:
                logger.info(f"Decoded {sc.key}: {sc.expression}")

            self.decoded_strategies[sc.key] = strategy

    def evaluate(self, df: pd.DataFrame,
                 symbol: Optional[str] = None) -> Dict[str, dict]:
        """
        Evaluate strategies on the latest data.

        Args:
            df: OHLCV DataFrame (200+ closed candles)
            symbol: If provided, only evaluate strategies for this symbol.
                    If None, evaluate all strategies.

        Returns:
            Dict mapping strategy_key -> {
                'signal': bool (True if entry signal on last bar),
                'atr': float (current ATR value),
                'direction': str ('LONG' or 'SHORT'),
                'tp_atr_mult': float,
                'sl_atr_mult': float,
                'trail_atr_mult': float,
                'last_close': float,
            }
        """
        results = {}

        configs = self.strategy_configs
        if symbol:
            configs = [sc for sc in configs if sc.symbol == symbol]

        for sc in configs:
            strategy = self.decoded_strategies[sc.key]

            try:
                # Generate signals on full history
                signals = generate_signals(strategy, df)

                # Get ATR for exit calculation
                atr = compute_atr(df, 14)

                # Only look at the LAST closed bar
                last_signal = bool(signals.iloc[-1])
                last_atr = float(atr.iloc[-1])
                last_close = float(df['Close'].iloc[-1])

                # Also check if signal just turned on (wasn't on in previous bar)
                # This prevents re-entering if we already acted on this signal
                prev_signal = bool(signals.iloc[-2]) if len(signals) > 1 else False
                is_new_signal = last_signal and not prev_signal

                results[sc.key] = {
                    'signal': last_signal,
                    'new_signal': is_new_signal,
                    'atr': last_atr,
                    'direction': sc.direction,
                    'tp_atr_mult': sc.tp_atr_mult,
                    'sl_atr_mult': sc.sl_atr_mult,
                    'trail_atr_mult': sc.trail_atr_mult,
                    'last_close': last_close,
                    'last_bar_time': df.index[-1],
                }

                if last_signal:
                    logger.info(
                        f"SIGNAL {sc.key}: {sc.direction} | "
                        f"new={is_new_signal} | "
                        f"close=${last_close:,.2f} | ATR=${last_atr:,.2f} | "
                        f"TP={sc.tp_atr_mult}×ATR SL={sc.sl_atr_mult}×ATR"
                    )

            except Exception as e:
                logger.error(f"Signal evaluation failed for {sc.key}: {e}")
                results[sc.key] = {
                    'signal': False,
                    'new_signal': False,
                    'atr': 0,
                    'direction': sc.direction,
                    'error': str(e),
                }

        return results

    def compute_exit_levels(self, entry_price: float, atr_value: float,
                            direction: str, tp_mult: float,
                            sl_mult: float) -> Tuple[Optional[float], float]:
        """
        Compute TP and SL price levels.

        Returns (take_profit, stop_loss). take_profit can be None if tp_mult=0.
        """
        if direction == 'LONG':
            sl = entry_price - sl_mult * atr_value
            tp = entry_price + tp_mult * atr_value if tp_mult > 0 else None
        else:
            sl = entry_price + sl_mult * atr_value
            tp = entry_price - tp_mult * atr_value if tp_mult > 0 else None
        return tp, sl
