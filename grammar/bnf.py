"""
BNF Grammar for Grammatical Evolution of Trading Strategies.

v3: Scale-invariant, type-safe grammar.

ALL indicators are normalized — no raw price/volume comparisons.
Two types: oscillators (0-100) and normalized ratios (~-3 to +3).
Comparisons are type-safe: oscillators only vs oscillators/thresholds.
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

    # Entry rules: 1 to 4 conditions combined with logic
    "<entry_rule>": [
        "<condition>",
        "<condition> AND <condition>",
        "<condition> AND <condition> AND <condition>",
        "<condition> AND <condition> AND <condition> AND <condition>",
        "<condition> OR <condition>",
        "(<condition> AND <condition>) OR <condition>",
        "<condition> AND (<condition> OR <condition>)",
    ],

    # Conditions: TYPE-SAFE comparisons
    # Oscillators (0-100) only compare to oscillators or osc thresholds
    # Normalized (~-3 to +3) only compare to normalized or norm thresholds
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
    # ================================================================
    "<osc>": [
        "RSI(<rsi_source>, <rsi_period>)",
        "STOCH_K(<stoch_period>)",
        "STOCH_D(<stoch_period>)",
    ],

    # ================================================================
    # NORMALIZED INDICATORS (dimensionless ratios, scale-invariant)
    # ================================================================
    "<norm>": [
        "PCT_B(<bb_period>, <bb_std>)",         # Percent B: position within BBands (0-1)
        "MACD_NORM(<macd_fast>, <macd_slow>, <macd_signal>)",  # MACD_HIST / ATR
        "PRICE_POS(<pos_period>)",              # (Close - SMA) / ATR
        "ROC(<roc_period>)",                     # % rate of change
        "VOL_RATIO(<vol_period>)",              # Volume / SMA(Volume)
        "BBWIDTH(<bb_period>, <bb_std>)",       # Band width as % of mid
        "ATR_PCT(<atr_period>)",                # ATR / Close * 100
    ],

    # ================================================================
    # PARAMETERS
    # ================================================================
    "<rsi_source>": ["close", "open", "high", "low"],

    "<rsi_period>": ["7", "9", "14", "21"],

    "<stoch_period>": ["5", "9", "14", "21"],

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

    # Exit parameters
    "<exit_params>": [
        "TP=<tp_mult> SL=<sl_mult>",
    ],

    "<tp_mult>": [
        "0.5", "0.75", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0",
    ],

    "<sl_mult>": [
        "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0",
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
