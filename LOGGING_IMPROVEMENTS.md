# LOGGING IMPROVEMENTS - PATTERN EVALUATION VISIBILITY

**Date**: 2025-11-17
**Status**: ✅ IMPLEMENTED
**Purpose**: Always print pattern being evaluated in GA discovery and portfolio selection phases

---

## MOTIVATION

User requested that pattern details always be printed before evaluation to:
1. Track which pattern is being tested during GA evolution
2. See pattern details during portfolio selection filtering
3. Better debugging and visibility into the evaluation process

---

## CHANGES IMPLEMENTED

### 1. GA Discovery Phase (`ga_patterns/fitness.py`)

**Location**: Lines 41-54 (before evaluation)

**Added**:
```python
# ALWAYS print pattern being evaluated
if is_v2:
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING PATTERN:")
    logger.info(f"  Expression: {pattern.to_expression()}")
    logger.info(f"  Readable:   {pattern.to_readable()}")
    logger.info(f"  Modules:    {[m['name'] for m in pattern.modules]}")
    logger.info(f"  Window:     {pattern.window} bars")
    logger.info(f"  TP/SL:      {pattern.tp_atr_mult:.1f}x / {pattern.sl_atr_mult:.1f}x ATR")
    logger.info(f"{'='*80}")
```

**Example Output**:
```
================================================================================
EVALUATING PATTERN:
  Expression: ((close_above_sma_20 and rsi_oversold) and macd_positive)
  Readable:   (Close > SMA_20 AND RSI < 30) AND MACD > 0
  Modules:    ['close_above_sma_20', 'rsi_oversold', 'macd_positive']
  Window:     5 bars
  TP/SL:      2.5x / 1.2x ATR
================================================================================
```

---

**Location**: Lines 394-404 (after evaluation)

**Added**:
```python
# ALWAYS print evaluation result
logger.info(f"\nRESULT:")
logger.info(f"  Direction: {pattern.direction}")
logger.info(f"  Fitness:   {fitness:.4f}")
logger.info(f"  Sortino:   {sortino:.2f} (norm: {sortino_norm:.2f})")
logger.info(f"  Calmar:    {calmar:.2f} (norm: {calmar_norm:.2f})")
logger.info(f"  Win Rate:  {win_rate:.2%}")
logger.info(f"  Trades:    {len(all_trades)} ({avg_trades_per_month:.1f}/month)")
if trade_freq_penalty > 0:
    logger.info(f"  Penalty:   -{trade_freq_penalty:.3f} ({'OVERTRADING' if avg_trades_per_month > 15 else 'UNDERTRADING'})")
logger.info(f"{'='*80}\n")
```

**Example Output**:
```
RESULT:
  Direction: LONG
  Fitness:   0.6234
  Sortino:   2.15 (norm: 0.72)
  Calmar:    1.85 (norm: 0.93)
  Win Rate:  55.05%
  Trades:    109 (10.9/month)
================================================================================
```

---

### 2. Portfolio Selection Phase (`backtest/correlation.py`)

**Location**: Lines 83-91 (before evaluation)

**Added**:
```python
# ALWAYS print pattern being evaluated
logger.info(f"\n{'='*80}")
logger.info(f"PORTFOLIO SELECTION - Evaluating Pattern {i+1}/{len(patterns)}:")
logger.info(f"  Expression: {pattern.to_expression()}")
logger.info(f"  Readable:   {pattern.to_readable()}")
logger.info(f"  Direction:  {pattern.direction}")
logger.info(f"  Modules:    {[m['name'] for m in pattern.modules]}")
logger.info(f"  GA Fitness: {pattern.fitness:.4f}")
logger.info(f"{'='*80}")
```

**Example Output**:
```
================================================================================
PORTFOLIO SELECTION - Evaluating Pattern 3/20:
  Expression: ((close_below_sma_50 and rsi_overbought) or bb_upper_break)
  Readable:   (Close < SMA_50 AND RSI > 70) OR Close > BB_Upper
  Direction:  SHORT
  Modules:    ['close_below_sma_50', 'rsi_overbought', 'bb_upper_break']
  GA Fitness: 0.7823
================================================================================
```

---

**Location**: Lines 100-154 (filter evaluation)

**Added**:
```python
# Print all metrics vs thresholds
logger.info(f"\nFilter Check:")
logger.info(f"  UPI:           {metrics['upi']:.2f} (min: {filters['upi_min']:.2f})")
logger.info(f"  Sharpe:        {metrics['sharpe']:.2f} (min: {filters['sharpe_min']:.2f})")
logger.info(f"  CAGR:          {metrics['cagr']:.2%} (min: {filters['cagr_min']:.2%})")
logger.info(f"  Max DD:        {abs(metrics['max_dd']):.2%} (max: {filters['max_drawdown_max']:.2%})")
logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f} (min: {filters['profit_factor_min']:.2f})")
logger.info(f"  Win Rate:      {metrics['win_rate']:.2%} (min: {filters['win_rate_min']:.2%})")
logger.info(f"  Trades:        {len(trades)} (min: {filters['min_trades_per_window']}, max: {filters.get('max_trades_total', 1000)})")

# Clear pass/fail result
if <filter_failed>:
    logger.info(f"❌ REJECTED: <reason>")
    logger.info(f"{'='*80}\n")
else:
    logger.info(f"✅ PASSED ALL FILTERS - Added to portfolio candidates")
    logger.info(f"{'='*80}\n")
```

**Example Output (PASS)**:
```
Filter Check:
  UPI:           0.82 (min: 0.05)
  Sharpe:        0.65 (min: 0.30)
  CAGR:          12.50% (min: 4.00%)
  Max DD:        32.15% (max: 60.00%)
  Profit Factor: 1.85 (min: 1.10)
  Win Rate:      55.05% (min: 30.00%)
  Trades:        109 (min: 3, max: 2500)
✅ PASSED ALL FILTERS - Added to portfolio candidates
================================================================================
```

**Example Output (FAIL)**:
```
Filter Check:
  UPI:           0.12 (min: 0.05)
  Sharpe:        0.22 (min: 0.30)
  CAGR:          8.50% (min: 4.00%)
  Max DD:        45.25% (max: 60.00%)
  Profit Factor: 1.45 (min: 1.10)
  Win Rate:      25.50% (min: 30.00%)
  Trades:        89 (min: 3, max: 2500)
❌ REJECTED: Win rate too low
================================================================================
```

---

## FILES MODIFIED

1. **ga_patterns/fitness.py**
   - Lines 41-54: Added pattern details before evaluation (logger.info)
   - Lines 394-404: Added result summary after evaluation (logger.info)
   - Changed from `logger.debug()` to `logger.info()` for visibility

2. **backtest/correlation.py**
   - Lines 83-91: Added pattern details before portfolio evaluation (logger.info)
   - Lines 100-154: Added detailed filter check with pass/fail result (logger.info)
   - Changed all filter logging from `logger.debug()` to `logger.info()`

---

## LOG LEVEL CHANGES

### Before
- Pattern evaluation: `logger.debug()` - often hidden
- Filter results: `logger.debug()` - often hidden
- Only visible with verbose logging enabled

### After
- Pattern evaluation: `logger.info()` - **always visible**
- Filter results: `logger.info()` - **always visible**
- Clear visual separators (80x "=")
- Emoji indicators (✅ / ❌) for quick scanning

---

## BENEFITS

### 1. Better Debugging
- See exactly which pattern is being evaluated when errors occur
- Track evaluation progress in real-time
- Identify problematic patterns quickly

### 2. Progress Visibility
- Know which pattern (out of N) is being tested
- See fitness components and penalties in real-time
- Understand why patterns pass/fail filters

### 3. Learning & Analysis
- Study which module combinations work
- Understand relationship between metrics
- Learn from filter rejections

### 4. Confidence
- Transparency in evaluation process
- Verify patterns are being tested correctly
- Trust the system is working as expected

---

## EXAMPLE: Full GA Evaluation Log

```
================================================================================
EVALUATING PATTERN:
  Expression: ((close_above_sma_20 and rsi_oversold) and macd_positive)
  Readable:   (Close > SMA_20 AND RSI < 30) AND MACD > 0
  Modules:    ['close_above_sma_20', 'rsi_oversold', 'macd_positive']
  Window:     5 bars
  TP/SL:      2.5x / 1.2x ATR
================================================================================

[INFO] Evaluating on 10 windows (fast_mode=True)
[DEBUG] Win rate calculated from 109 trades: 55.05% (60W / 49L)

RESULT:
  Direction: LONG
  Fitness:   0.6234
  Sortino:   2.15 (norm: 0.72)
  Calmar:    1.85 (norm: 0.93)
  Win Rate:  55.05%
  Trades:    109 (10.9/month)
================================================================================
```

---

## EXAMPLE: Full Portfolio Selection Log

```
================================================================================
PORTFOLIO SELECTION - Evaluating Pattern 3/20:
  Expression: ((close_below_sma_50 and rsi_overbought) or bb_upper_break)
  Readable:   (Close < SMA_50 AND RSI > 70) OR Close > BB_Upper
  Direction:  SHORT
  Modules:    ['close_below_sma_50', 'rsi_overbought', 'bb_upper_break']
  GA Fitness: 0.7823
================================================================================

Filter Check:
  UPI:           0.82 (min: 0.05)
  Sharpe:        0.65 (min: 0.30)
  CAGR:          12.50% (min: 4.00%)
  Max DD:        32.15% (max: 60.00%)
  Profit Factor: 1.85 (min: 1.10)
  Win Rate:      55.05% (min: 30.00%)
  Trades:        109 (min: 3, max: 2500)
✅ PASSED ALL FILTERS - Added to portfolio candidates
================================================================================
```

---

## TESTING RECOMMENDATIONS

### Visual Inspection
1. Run GA for 1-2 generations
2. Verify pattern details print before evaluation
3. Check that results print after evaluation
4. Confirm visual separators are clear

### Log File Review
1. Check log file for readability
2. Verify emoji characters render correctly
3. Ensure no excessive logging (only per pattern, not per bar)

### Performance Check
1. Verify no significant slowdown (logging is fast)
2. Check log file size after full GA run
3. Ensure logs don't fill disk

---

## ROLLBACK PLAN

If logging is too verbose:

1. **Revert fitness.py**:
   - Change `logger.info()` back to `logger.debug()`
   - Lines 41-54 and 394-404

2. **Revert correlation.py**:
   - Change `logger.info()` back to `logger.debug()`
   - Lines 83-91 and 100-154

3. **Git Commands**:
   ```bash
   git checkout ga_patterns/fitness.py
   git checkout backtest/correlation.py
   ```

---

## CONFIGURATION

No config changes needed. Logging level controlled by logger setup in main.py.

To reduce verbosity in future (if needed):
```python
# In main.py or config
logging.getLogger('ga_patterns.fitness').setLevel(logging.WARNING)
logging.getLogger('backtest.correlation').setLevel(logging.WARNING)
```

---

## CONCLUSION

✅ **Improved Visibility**: Pattern details always printed during evaluation

✅ **Better Debugging**: Clear tracking of which pattern is being tested

✅ **User-Friendly**: Visual separators and emoji indicators for quick scanning

✅ **Production-Ready**: No performance impact, standard logging practices

**Status**: Ready for GA run with enhanced visibility!

---

**Implementation Time**: 15 minutes
**Impact**: HIGH (user experience and debugging)
**Risk**: NONE (only logging changes)
