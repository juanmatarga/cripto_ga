"""
BNF Grammar for Grammatical Evolution of Trading Strategies.

v5b: Scale-invariant, type-safe grammar with:
  - Trailing stops and trend indicators (v4)
  - Multi-timeframe support: 15m (default), 1h, 4h
  - Alternative data REMOVED (funding rate proven noise — PBO=1.0 in all tests)

ALL indicators are normalized — no raw price/volume comparisons.
Two types: oscillators (0-100) and normalized ratios (~-3 to +3).
Comparisons are type-safe within each type.
"""

MAX_DEPTH = 8
MAX_WRAPS = 2
START_SYMBOL = "<strategy>"

GRAMMAR = {
    "<strategy>": [
        "<direction> WHEN <entry_rule> EXIT <exit_params>",
    ],

    "<direction>": [
        "LONG",
        "SHORT",
    ],

    # Entry rules: balanced structure families (25% each)
    "<entry_rule>": [
        "<condition>",                                                      # simple
        "<condition> <logical_op> <condition>",                             # binary
        "<condition> <logical_op> <condition> <logical_op> <condition>",    # ternary
        "( <condition> <logical_op> <condition> ) <logical_op> <condition>",  # grouped
    ],

    # Logical operator: evolves independently of structure
    "<logical_op>": ["AND", "OR"],

    # Conditions: TYPE-SAFE comparisons
    # Balance: 8 persistent (> / <) + 6 crossovers = 57% persistent
    # Persistent fire 5-30% of bars → enough trades for AND logic
    # Crossovers fire <0.5% → too rare for AND, but good for precision
    "<condition>": [
        # Oscillator persistent (fire while condition holds)
        "<osc> <comparator> <osc>",
        "<osc> <comparator> <osc_thresh>",
        "<osc> <comparator> <osc>",           # weighted: osc vs osc is very useful
        "<osc> <comparator> <osc_thresh>",    # weighted: osc vs threshold is core
        # Normalized persistent
        "<norm> <comparator> <norm>",
        "<norm> <comparator> <norm_thresh>",
        "<norm> <comparator> <norm>",          # weighted
        "<norm> <comparator> <norm_thresh>",   # weighted
        # Oscillator crossovers (one-bar events, with persistence window)
        "<osc> CROSSES_ABOVE <osc>",
        "<osc> CROSSES_BELOW <osc>",
        # Normalized crossovers
        "<norm> CROSSES_ABOVE <norm>",
        "<norm> CROSSES_BELOW <norm>",
        "<norm> CROSSES_ABOVE <norm_thresh>",
        "<norm> CROSSES_BELOW <norm_thresh>",
    ],

    # ================================================================
    # OSCILLATOR INDICATORS (0-100, already scale-invariant)
    # Optionally on higher timeframes
    # ================================================================
    "<osc>": [
        # Classic oscillators
        "RSI(<rsi_source>, <rsi_period>)",
        "RSI(<rsi_source>, <rsi_period>, <timeframe>)",
        "STOCH_K(<stoch_period>)",
        "STOCH_K(<stoch_period>, <timeframe>)",
        "STOCH_D(<stoch_period>)",
        "STOCH_D(<stoch_period>, <timeframe>)",
        "ADX(<adx_period>)",
        "ADX(<adx_period>, <timeframe>)",
        "MFI(<mfi_period>)",
        "MFI(<mfi_period>, <timeframe>)",
        # Pattern-based (0-100 scale, same as oscillators)
        "BREAKOUT_UP(<breakout_period>)",
        "BREAKOUT_UP(<breakout_period>, <timeframe>)",
        "BREAKOUT_DOWN(<breakout_period>)",
        "BREAKOUT_DOWN(<breakout_period>, <timeframe>)",
        "SQUEEZE(<bb_period>, <bb_std>)",
        "SQUEEZE(<bb_period>, <bb_std>, <timeframe>)",
        "DIVERGENCE_BULL(RSI, <rsi_period>, <div_period>)",
        "DIVERGENCE_BEAR(RSI, <rsi_period>, <div_period>)",
    ],

    # ================================================================
    # NORMALIZED INDICATORS (dimensionless ratios, scale-invariant)
    # ================================================================
    "<norm>": [
        # Classic normalized
        "PCT_B(<bb_period>, <bb_std>)",
        "PCT_B(<bb_period>, <bb_std>, <timeframe>)",
        "MACD_NORM(<macd_fast>, <macd_slow>, <macd_signal>)",
        "MACD_NORM(<macd_fast>, <macd_slow>, <macd_signal>, <timeframe>)",
        "PRICE_POS(<pos_period>)",
        "PRICE_POS(<pos_period>, <timeframe>)",
        "ROC(<roc_period>)",
        "ROC(<roc_period>, <timeframe>)",
        "VOL_RATIO(<vol_period>)",
        "VOL_RATIO(<vol_period>, <timeframe>)",
        "BBWIDTH(<bb_period>, <bb_std>)",
        "BBWIDTH(<bb_period>, <bb_std>, <timeframe>)",
        "ATR_PCT(<atr_period>)",
        "ATR_PCT(<atr_period>, <timeframe>)",
        # Trend-based (-100 to +100, normalized)
        "TRENDING(RSI, <rsi_period>, <trend_period>)",
        "TRENDING(RSI, <rsi_period>, <trend_period>, <timeframe>)",
        "TRENDING(MFI, <mfi_period>, <trend_period>)",
        "TRENDING(STOCH_K, <stoch_period>, <trend_period>)",
    ],

    # ================================================================
    # TIMEFRAME PARAMETER (for multi-timeframe indicators)
    # ================================================================
    "<timeframe>": ["15m", "1h", "4h"],

    # ================================================================
    # PARAMETERS
    # ================================================================
    "<rsi_source>": ["close", "open", "high", "low"],

    "<rsi_period>": ["7", "9", "14", "21"],

    "<stoch_period>": ["5", "9", "14", "21"],

    "<adx_period>": ["7", "10", "14", "21"],

    "<mfi_period>": ["7", "10", "14", "21"],

    "<bb_period>": ["10", "14", "20", "30"],

    "<bb_std>": ["1.5", "2.0", "2.5", "3.0"],

    "<macd_fast>": ["8", "12", "16"],

    "<macd_slow>": ["21", "26", "34"],

    "<macd_signal>": ["5", "9", "13"],

    "<pos_period>": ["5", "8", "13", "21", "34", "55"],

    "<roc_period>": ["3", "5", "8", "13", "21"],

    "<vol_period>": ["5", "10", "20", "50"],

    "<atr_period>": ["7", "10", "14", "21"],

    "<breakout_period>": ["10", "20", "30", "50"],

    "<div_period>": ["10", "14", "20", "30"],

    "<trend_period>": ["5", "8", "13", "21"],

    # ================================================================
    # THRESHOLDS (type-safe: separate for osc and norm)
    # ================================================================

    # Oscillator thresholds (0-100 range)
    "<osc_thresh>": [
        "10", "20", "30", "40", "50", "60", "70", "80", "90",
    ],

    # Normalized thresholds (centered ~0)
    "<norm_thresh>": [
        "-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0",
    ],

    # Comparators
    "<comparator>": [
        "_GT_",
        "_LT_",
    ],

    # ================================================================
    # EXIT PARAMETERS — now with trailing stops
    # ================================================================
    "<exit_params>": [
        "TP=<tp_mult> SL=<sl_mult>",
        "SL=<sl_mult> TRAIL=<trail_mult>",
        "TP=<tp_mult> SL=<sl_mult> TRAIL=<trail_mult>",
    ],

    # Wider TP range for trend-following
    "<tp_mult>": [
        "1.0", "1.5", "2.0", "3.0", "4.0", "5.0", "6.0", "8.0",
    ],

    # Wider SL range (trend-following needs wider stops)
    "<sl_mult>": [
        "0.5", "0.75", "1.0", "1.5", "2.0", "2.5", "3.0",
    ],

    # Trailing stop distance in ATR multiples
    "<trail_mult>": [
        "1.0", "1.5", "2.0", "2.5", "3.0", "4.0", "5.0",
    ],
}


def validate_grammar():
    """Check all non-terminals referenced in productions are defined."""
    import re
    all_nonterminals = set(GRAMMAR.keys())
    referenced = set()
    for productions in GRAMMAR.values():
        for prod in productions:
            for match in re.findall(r'<[^>]+>', prod):
                referenced.add(match)
    undefined = referenced - all_nonterminals
    if undefined:
        raise ValueError(f"Undefined non-terminals: {undefined}")
    unreachable = all_nonterminals - referenced - {START_SYMBOL}
    if unreachable:
        raise ValueError(f"Unreachable non-terminals: {unreachable}")


# Validate at import time
validate_grammar()
