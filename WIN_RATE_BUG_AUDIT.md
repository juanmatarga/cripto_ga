# WIN RATE BUG AUDIT - CRITICAL CALCULATION ERROR

**Date**: 2025-11-17
**Status**: CRITICAL BUG FOUND
**Impact**: Win rate incorrectly calculated from equity curve bars instead of trade outcomes

---

## PROBLEM STATEMENT

User reported unrealistic win rate:
```json
{
  "win_rate": 0.0019924496644295304,  // 0.19% win rate ❌
  "profit_factor": 2.1123997387429534,  // 2.11 profit factor
  "sharpe": 3.6926751079503686,         // 3.69 Sharpe
  "cagr": 0.8588232630144417,           // 85% CAGR
  "n_trades": 109
}
```

**Question**: How can a pattern have 0.19% win rate but 85% CAGR, 3.69 Sharpe, and 2.11 profit factor?

**Answer**: The win rate is being calculated incorrectly!

---

## ROOT CAUSE ANALYSIS

### BUG LOCATION: `backtest/metrics.py:165-169`

```python
def calculate_all_metrics(equity_curve: pd.Series, periods_per_year: int) -> Dict[str, float]:
    """Calculate all metrics from equity curve"""

    returns = calculate_returns(equity_curve).dropna()  # Bar-by-bar returns

    # ❌ INCORRECT: Calculating from equity curve returns (bars), not trades
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
```

### WHAT'S HAPPENING

1. **`equity_curve`**: Series of equity values over time (one per bar)
   - Example: 10,000 bars for a backtest

2. **`returns = equity_curve.pct_change()`**: Bar-by-bar equity changes
   - Returns is also 10,000 values (one per bar)

3. **`positive_returns`**: Bars where equity increased
   - Example: Only 20 bars out of 10,000 (when winning trades closed)

4. **`win_rate = len(positive_returns) / len(returns)`**:
   - **Calculated**: 20 / 10,000 = 0.002 = 0.2% ❌
   - **Should be**: winning_trades / total_trades = 60 / 109 = 55% ✓

### WHY THIS PRODUCES 0.2% WIN RATE

- Equity only changes significantly when trades **close**
- Most bars have 0% return (no trade activity)
- Even bars during winning trades show 0% return (equity unchanged)
- Only the **exit bar** shows the profit/loss
- So positive_returns counts **exit bars of winning trades**, not **number of winning trades**

**Example**:
- 100 trades total
- 55 winners, 45 losers (55% win rate ✓)
- But 10,000 total bars
- Only 55 bars show positive returns (when winners closed)
- Incorrect calculation: 55 / 10,000 = 0.55% ❌

---

## CORRECT IMPLEMENTATION

### CORRECT WIN RATE CALCULATION (From `fitness.py:327`)

```python
# ✓ CORRECT: Calculate from trade outcomes
win_rate = len([t for t in all_trades if t['pnl_pct'] > 0]) / len(all_trades) if len(all_trades) > 0 else 0.0
```

### TRADE STRUCTURE (From `runner.py:234-245`)

Each trade is recorded as:
```python
{
    'entry_bar': int,
    'entry_date': datetime,
    'entry_price': float,
    'exit_bar': int,
    'exit_date': datetime,
    'exit_price': float,
    'exit_type': str,  # 'stop', 'target', 'time'
    'direction': str,  # 'LONG' or 'SHORT'
    'pnl_pct': float,  # ✓ Use this for win rate!
    'equity_after': float
}
```

**Win Rate Formula**: `count(trades where pnl_pct > 0) / count(all trades)`

---

## IMPACT ASSESSMENT

### Functions Affected

1. **`backtest/metrics.py:calculate_all_metrics()`**
   - Returns incorrect `win_rate` and `profit_factor`
   - Used in 13 places across the codebase

2. **Call Sites Affected**:
   - `backtest/correlation.py:87` - Portfolio selection (filters may reject good patterns)
   - `reports/report_generator.py:382, 383` - Final reports (incorrect metrics)
   - `robustness/bootstrap.py:113` - Robustness tests (incorrect stats)
   - `main.py:318, 902, 903` - Demo and validation (incorrect metrics)
   - All test files (incorrect assertions)

3. **NOT Affected**:
   - `ga_patterns/fitness.py:327` - ✓ Uses correct calculation from trades
   - Fitness evaluation is **correct** (GA is evolving properly)
   - Only reporting/filtering is affected

### Severity Assessment

**Impact**: HIGH
- Portfolio selection may reject good patterns (win_rate_min: 0.30 filter)
- Reports show misleading statistics
- User confidence undermined by unrealistic numbers

**Scope**: REPORTING ONLY
- Fitness function is NOT affected (uses correct calculation)
- GA evolution is working correctly
- Patterns are being evolved properly

---

## FIX STRATEGY

### Option 1: Modify `calculate_all_metrics()` Signature (RECOMMENDED)

**Change**:
```python
def calculate_all_metrics(equity_curve: pd.Series, periods_per_year: int,
                         trades: List[Dict] = None) -> Dict[str, float]:
    """
    Calculate all metrics.

    Args:
        equity_curve: Equity series
        periods_per_year: Annualization factor
        trades: Optional list of trade dicts (for accurate win rate / profit factor)
    """
    # ... existing code ...

    # NEW: Calculate win rate and profit factor from trades if available
    if trades is not None and len(trades) > 0:
        # ✓ CORRECT: From trade outcomes
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] < 0]

        win_rate = len(winning_trades) / len(trades)

        total_gains = sum(t['pnl_pct'] for t in winning_trades)
        total_losses = abs(sum(t['pnl_pct'] for t in losing_trades))

        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = 999.0  # All wins
        else:
            profit_factor = 0.0  # No trades
    else:
        # FALLBACK: Use equity curve (incorrect but backward compatible)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

        total_gains = positive_returns.sum() if len(positive_returns) > 0 else 0.0
        total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0

        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0
```

**Pros**:
- Backward compatible (trades parameter is optional)
- Minimal changes to call sites
- Correct calculation when trades available
- Fallback to old behavior if trades not provided

**Cons**:
- Need to update ~13 call sites to pass trades

### Option 2: Create New Function `calculate_metrics_from_trades()`

**Change**: Add new function, leave old one unchanged

**Pros**:
- Zero breaking changes
- Old code continues working

**Cons**:
- Two functions doing similar things
- Confusion about which to use
- Doesn't fix existing reports

---

## RECOMMENDED FIX: Option 1

### Implementation Steps

1. **Modify `calculate_all_metrics()` signature** (metrics.py:139)
   - Add optional `trades: List[Dict] = None` parameter
   - Add logic to calculate from trades if provided
   - Keep fallback for backward compatibility

2. **Update call sites** (pass trades where available):
   - `backtest/correlation.py:87` - trades already available
   - `reports/report_generator.py:382-383` - need to get trades from backtest
   - `main.py:318, 902, 903` - trades already available
   - `robustness/bootstrap.py:113` - trades already available

3. **Add logging** to track which method is used:
   ```python
   if trades is not None:
       logger.debug(f"Calculating win rate from {len(trades)} trades")
   else:
       logger.warning("No trades provided - using equity curve (less accurate)")
   ```

4. **Update tests** to verify correct calculation

---

## VERIFICATION PLAN

### Test Case 1: Simple Win Rate

```python
trades = [
    {'pnl_pct': 0.05},   # Win
    {'pnl_pct': -0.02},  # Loss
    {'pnl_pct': 0.03},   # Win
    {'pnl_pct': 0.01},   # Win
    {'pnl_pct': -0.01},  # Loss
]

expected_win_rate = 3 / 5 = 0.60 (60%)
```

### Test Case 2: User's Actual Pattern

**Before Fix**:
```
win_rate: 0.0019924... (0.2%)  ❌
n_trades: 109
```

**After Fix (Expected)**:
```
win_rate: ~0.55 (55%)  ✓
n_trades: 109
```

**Reasoning**: With 2.11 profit factor and 85% CAGR, realistic win rate is 50-60%

### Test Case 3: All Winners

```python
trades = [
    {'pnl_pct': 0.05},
    {'pnl_pct': 0.03},
    {'pnl_pct': 0.02},
]

expected_win_rate = 1.0 (100%)
expected_profit_factor = 999.0
```

---

## ROLLBACK PLAN

If fix causes issues:

1. **Revert `metrics.py`** to remove trades parameter
2. **Revert call sites** to not pass trades
3. **Document** that win_rate in metrics is approximate

---

## CONCLUSION

**Bug Confirmed**: Win rate calculated from equity curve bars instead of trade outcomes

**Severity**: HIGH (reporting), LOW (fitness - not affected)

**Fix Complexity**: MEDIUM (need to update ~13 call sites)

**Risk**: LOW (backward compatible with optional parameter)

**Estimated Time**: 30 minutes implementation + 15 minutes testing

---

## NEXT STEPS

1. Implement fix in `metrics.py`
2. Update call sites in:
   - `correlation.py`
   - `report_generator.py`
   - `main.py`
   - `bootstrap.py`
3. Run backtest on user's pattern to verify realistic win rate
4. Update tests

**Status**: Ready to implement
