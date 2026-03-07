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

    # Entry rules: biased toward multi-condition (more selective = less noise)
    "<entry_rule>": [
        "<condition>",
        "<condition> AND <condition>",
        "<condition> AND <condition>",
        "<condition> AND <condition> AND <condition>",
        "<condition> AND <condition> AND <condition>",
        "<condition> OR <condition>",
        "(<condition> AND <condition>) OR <condition>",
        "<condition> AND (<condition> OR <condition>)",
    ],

    # Conditions: TYPE-SAFE comparisons
    "<condition>": [
        # Oscillator comparisons
        "<osc> <comparator> <osc>",
        "<osc> <comparator> <osc_thresh>",
        "<osc> CROSSES_ABOVE <osc>",
        "<osc> CROSSES_BELOW <osc>",
        # Normalized comparisons
        "<norm> <comparator> <norm>",
        "<norm> <comparator> <norm_thresh>",
        "<norm> CROSSES_ABOVE <norm>",
        "<norm> CROSSES_BELOW <norm>",
        # Normalized crossing a threshold (e.g., MACD_NORM crossing 0)
        "<norm> CROSSES_ABOVE <norm_thresh>",
        "<norm> CROSSES_BELOW <norm_thresh>",
    ],

    # ================================================================
    # OSCILLATOR INDICATORS (0-100, already scale-invariant)
    # Optionally on higher timeframes
    # ================================================================
    "<osc>": [
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
    ],

    # ================================================================
    # NORMALIZED INDICATORS (dimensionless ratios, scale-invariant)
    # ================================================================
    "<norm>": [
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
