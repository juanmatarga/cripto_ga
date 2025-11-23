"""
Evolution Analytics - Post-run analysis of GA performance.

SPRINT 14: Comprehensive analytics suite for academic presentation.

Analyzes evolution snapshots and generates:
- Evolution metrics over time
- Module usage trends
- Performance convergence analysis
- Stagnation detection
- Professional visualizations
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
sns.set_palette("husl")


class EvolutionAnalyzer:
    """Analyze GA evolution from snapshots."""

    def __init__(self, snapshots_dir: str = "./evolution_snapshots"):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots = []
        self.load_snapshots()

    def load_snapshots(self):
        """Load all generation snapshots."""
        snapshot_files = sorted(self.snapshots_dir.glob("generation_*.json"))

        for file in snapshot_files:
            with open(file, 'r') as f:
                data = json.load(f)
                # Handle both old and new format
                if 'data' in data:
                    # Old format: {'generation': X, 'data': {...}}
                    snapshot = data['data']
                    snapshot['generation'] = data['generation']
                else:
                    # New format: all data at top level
                    snapshot = data
                self.snapshots.append(snapshot)

        print(f"[OK] Loaded {len(self.snapshots)} generation snapshots")
        if len(self.snapshots) == 0:
            print(f"[WARNING] No snapshots found in {self.snapshots_dir}")
            print(f"Expected files matching pattern: generation_*.json")

    def extract_metrics_timeseries(self) -> pd.DataFrame:
        """Extract key metrics over generations."""
        data = []

        for snap in self.snapshots:
            gen = snap['generation']

            # Extract data safely (handle missing keys)
            fitness_stats = snap.get('fitness_stats', {})
            population_stats = snap.get('population_stats', {})
            diversity = snap.get('diversity', {})
            performance = snap.get('performance', {})
            best_long = snap.get('best_long', {}) or {}
            best_short = snap.get('best_short', {}) or {}

            row = {
                'generation': gen,
                'best_fitness': fitness_stats.get('max', 0),
                'mean_fitness': fitness_stats.get('mean', 0),
                'median_fitness': fitness_stats.get('median', 0),
                'fitness_std': fitness_stats.get('std', 0),
                'valid_pct': population_stats.get('valid_pct', 0),
                'diversity_pct': diversity.get('unique_patterns_pct', 0),
                'avg_sharpe': performance.get('avg_sharpe', 0),
                'avg_cagr': performance.get('avg_cagr', 0),
                'avg_trades': performance.get('avg_trades', 0),
                'avg_win_rate': performance.get('avg_win_rate', 0),
                'best_long_fitness': best_long.get('fitness', 0) if isinstance(best_long, dict) else 0,
                'best_short_fitness': best_short.get('fitness', 0) if isinstance(best_short, dict) else 0
            }

            data.append(row)

        return pd.DataFrame(data)

    def plot_fitness_evolution(self, df: pd.DataFrame, output_path: str):
        """Plot fitness evolution over generations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Best and Mean Fitness
        ax = axes[0, 0]
        ax.plot(df['generation'], df['best_fitness'], 'o-', label='Best', linewidth=2, markersize=6, color='#e74c3c')
        ax.plot(df['generation'], df['mean_fitness'], 's-', label='Mean', linewidth=2, markersize=5, alpha=0.7, color='#3498db')
        ax.plot(df['generation'], df['median_fitness'], '^-', label='Median', linewidth=2, markersize=5, alpha=0.7, color='#2ecc71')
        ax.fill_between(df['generation'],
                        df['mean_fitness'] - df['fitness_std'],
                        df['mean_fitness'] + df['fitness_std'],
                        alpha=0.2, label='±1 Std Dev', color='#3498db')
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Valid Patterns %
        ax = axes[0, 1]
        ax.plot(df['generation'], df['valid_pct'], 'o-', color='#27ae60', linewidth=2, markersize=6)
        ax.axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.5, label='50% Target', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valid Patterns (%)', fontsize=12, fontweight='bold')
        ax.set_title('Pattern Quality Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # Plot 3: Diversity
        ax = axes[1, 0]
        ax.plot(df['generation'], df['diversity_pct'], 'o-', color='#9b59b6', linewidth=2, markersize=6)
        ax.axhline(y=30, color='#e74c3c', linestyle='--', alpha=0.5, label='30% Critical', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Diversity (%)', fontsize=12, fontweight='bold')
        ax.set_title('Population Diversity', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # Plot 4: LONG vs SHORT Performance
        ax = axes[1, 1]
        ax.plot(df['generation'], df['best_long_fitness'], 'o-', label='Best LONG', color='#3498db', linewidth=2, markersize=6)
        ax.plot(df['generation'], df['best_short_fitness'], 's-', label='Best SHORT', color='#e74c3c', linewidth=2, markersize=6)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title('LONG vs SHORT Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved fitness evolution plot: {output_path}")
        plt.close()

    def plot_performance_metrics(self, df: pd.DataFrame, output_path: str):
        """Plot Sharpe, CAGR, trades over time."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Sharpe Ratio
        ax = axes[0]
        ax.plot(df['generation'], df['avg_sharpe'], 'o-', color='#2c3e50', linewidth=2, markersize=6)
        ax.axhline(y=1.0, color='#27ae60', linestyle='--', alpha=0.5, label='Sharpe=1.0', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Sharpe Ratio Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: CAGR
        ax = axes[1]
        ax.plot(df['generation'], df['avg_cagr'] * 100, 'o-', color='#16a085', linewidth=2, markersize=6)
        ax.axhline(y=8, color='#e74c3c', linestyle='--', alpha=0.5, label='8% Target (S&P)', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average CAGR (%)', fontsize=12, fontweight='bold')
        ax.set_title('CAGR Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 3: Trade Count
        ax = axes[2]
        ax.plot(df['generation'], df['avg_trades'], 'o-', color='#d35400', linewidth=2, markersize=6)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Trades per Pattern', fontsize=12, fontweight='bold')
        ax.set_title('Trade Frequency Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved performance metrics plot: {output_path}")
        plt.close()

    def plot_module_trends(self, output_path: str):
        """Plot top module usage over time."""
        # Extract module counts per generation
        all_modules = set()
        for snap in self.snapshots:
            diversity = snap.get('diversity', {})
            top_modules = diversity.get('top_modules', {})
            all_modules.update(top_modules.keys())

        # Build time series for modules
        module_counts_over_time = {mod: [] for mod in all_modules}

        for snap in self.snapshots:
            diversity = snap.get('diversity', {})
            top_modules = diversity.get('top_modules', {})
            for mod in all_modules:
                module_counts_over_time[mod].append(top_modules.get(mod, 0))

        # Get top 10 by max usage
        module_max_usage = {mod: max(counts) for mod, counts in module_counts_over_time.items() if counts}
        top_10_modules = sorted(module_max_usage.items(), key=lambda x: x[1], reverse=True)[:10]

        # Plot
        plt.figure(figsize=(14, 7))
        generations = [snap['generation'] for snap in self.snapshots]

        colors = sns.color_palette("husl", 10)
        for idx, (mod, _) in enumerate(top_10_modules):
            plt.plot(generations, module_counts_over_time[mod], 'o-',
                    label=mod, linewidth=2, markersize=4, color=colors[idx])

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Module Usage Count', fontsize=12, fontweight='bold')
        plt.title('Top 10 Module Usage Trends', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved module trends plot: {output_path}")
        plt.close()

    def generate_summary_report(self, output_path: str):
        """Generate markdown summary report."""
        df = self.extract_metrics_timeseries()

        if len(df) == 0:
            print("[WARNING] No data to generate report")
            return

        # Calculate improvement metrics
        initial_best = df.iloc[0]['best_fitness']
        final_best = df.iloc[-1]['best_fitness']
        improvement = ((final_best - initial_best) / abs(initial_best) * 100) if initial_best != 0 else 0

        report = f"""# Genetic Algorithm Evolution Report

## Executive Summary

**This report presents the results of the genetic algorithm evolution for cryptocurrency trading pattern discovery.**

---

## Experiment Configuration

- **Total Generations**: {len(self.snapshots)}
- **Initial Best Fitness**: {initial_best:.4f}
- **Final Best Fitness**: {final_best:.4f}
- **Improvement**: {improvement:+.1f}%
- **Convergence Status**: {"Converged" if improvement < 5 else "Improving"}

---

## Final Generation Statistics

### Population Quality
- **Valid Patterns**: {df.iloc[-1]['valid_pct']:.1f}% ({int(df.iloc[-1]['valid_pct'])} patterns out of 100)
- **Population Diversity**: {df.iloc[-1]['diversity_pct']:.1f}%

### Fitness Metrics
- **Best Fitness**: {df.iloc[-1]['best_fitness']:.4f}
- **Mean Fitness**: {df.iloc[-1]['mean_fitness']:.4f}
- **Median Fitness**: {df.iloc[-1]['median_fitness']:.4f}
- **Fitness Std Dev**: {df.iloc[-1]['fitness_std']:.4f}

### Trading Performance Metrics
- **Average Sharpe Ratio**: {df.iloc[-1]['avg_sharpe']:.2f}
- **Average CAGR**: {df.iloc[-1]['avg_cagr']*100:.2f}%
- **Average Trades per Pattern**: {df.iloc[-1]['avg_trades']:.0f}
- **Average Win Rate**: {df.iloc[-1]['avg_win_rate']*100:.1f}%

### Direction-Specific Performance
- **Best LONG Fitness**: {df.iloc[-1]['best_long_fitness']:.4f}
- **Best SHORT Fitness**: {df.iloc[-1]['best_short_fitness']:.4f}
- **LONG/SHORT Gap**: {abs(df.iloc[-1]['best_long_fitness'] - df.iloc[-1]['best_short_fitness']):.4f}

---

## Evolution Dynamics

### Convergence Analysis
- **Stagnation Generations**: {self._count_stagnation(df)} out of {len(df)}
- **Stagnation Rate**: {(self._count_stagnation(df) / len(df) * 100):.1f}%
- **Diversity Trend**: {"Declining" if df.iloc[-1]['diversity_pct'] < df.iloc[0]['diversity_pct'] else "Stable/Increasing"}
- **Early Stopping**: {"Yes" if len(df) < 30 else "No (ran full generations)"}

### Performance Progression
| Metric | Initial (Gen 0) | Final (Gen {len(df)-1}) | Change |
|--------|----------------|------------------------|--------|
| Best Fitness | {df.iloc[0]['best_fitness']:.4f} | {df.iloc[-1]['best_fitness']:.4f} | {(df.iloc[-1]['best_fitness'] - df.iloc[0]['best_fitness']):+.4f} |
| Mean Fitness | {df.iloc[0]['mean_fitness']:.4f} | {df.iloc[-1]['mean_fitness']:.4f} | {(df.iloc[-1]['mean_fitness'] - df.iloc[0]['mean_fitness']):+.4f} |
| Valid % | {df.iloc[0]['valid_pct']:.1f}% | {df.iloc[-1]['valid_pct']:.1f}% | {(df.iloc[-1]['valid_pct'] - df.iloc[0]['valid_pct']):+.1f}% |
| Diversity % | {df.iloc[0]['diversity_pct']:.1f}% | {df.iloc[-1]['diversity_pct']:.1f}% | {(df.iloc[-1]['diversity_pct'] - df.iloc[0]['diversity_pct']):+.1f}% |

---

## Best Patterns Discovered

### Top LONG Pattern
"""

        final_snap = self.snapshots[-1]
        best_long = final_snap.get('best_long')
        if best_long and isinstance(best_long, dict) and best_long.get('fitness'):
            report += f"""
**Pattern**: `{best_long.get('readable', 'N/A')}`

**Performance Metrics**:
- Fitness: {best_long.get('fitness', 0):.4f}
- Sharpe Ratio: {best_long.get('sharpe', 0):.2f}
- CAGR: {best_long.get('cagr', 0)*100:.2f}%
- Number of Trades: {best_long.get('n_trades', 0)}
- Win Rate: {best_long.get('win_rate', 0)*100:.1f}%

**Modules**: {', '.join(best_long.get('modules', []))}
"""
        else:
            report += "\n*No valid LONG pattern found*\n"

        report += "\n### Top SHORT Pattern\n"

        best_short = final_snap.get('best_short')
        if best_short and isinstance(best_short, dict) and best_short.get('fitness'):
            report += f"""
**Pattern**: `{best_short.get('readable', 'N/A')}`

**Performance Metrics**:
- Fitness: {best_short.get('fitness', 0):.4f}
- Sharpe Ratio: {best_short.get('sharpe', 0):.2f}
- CAGR: {best_short.get('cagr', 0)*100:.2f}%
- Number of Trades: {best_short.get('n_trades', 0)}
- Win Rate: {best_short.get('win_rate', 0)*100:.1f}%

**Modules**: {', '.join(best_short.get('modules', []))}
"""
        else:
            report += "\n*No valid SHORT pattern found*\n"

        report += f"""

---

## Module Analysis

### Most Popular Modules (Final Generation)
"""

        diversity = final_snap.get('diversity', {})
        top_modules = diversity.get('top_modules', {})
        if top_modules:
            for i, (mod, count) in enumerate(list(top_modules.items())[:10], 1):
                report += f"{i}. **{mod}**: {count} uses\n"
        else:
            report += "*No module data available*\n"

        report += f"""

---

## Recommendations

### For Academic Presentation
1. **Highlight Evolution**: Show fitness_evolution.png to demonstrate algorithm convergence
2. **Performance Metrics**: Use performance_metrics.png to show trading viability
3. **Module Trends**: Use module_trends.png to show which patterns emerged

### For Trading Implementation
"""

        if df.iloc[-1]['avg_sharpe'] >= 1.0:
            report += "- **Sharpe ≥ 1.0**: Patterns show promising risk-adjusted returns\n"
        else:
            report += "- **Sharpe < 1.0**: Consider longer training periods or parameter tuning\n"

        if df.iloc[-1]['avg_cagr'] >= 0.08:
            report += "- **CAGR ≥ 8%**: Patterns exceed S&P 500 benchmark\n"
        else:
            report += "- **CAGR < 8%**: Patterns underperform S&P 500 benchmark\n"

        if df.iloc[-1]['diversity_pct'] < 30:
            report += "- **Low Diversity**: Population converged, consider restart with higher mutation\n"
        else:
            report += "- **Healthy Diversity**: Population maintains exploration capability\n"

        report += f"""

---

## Appendix: Technical Details

### Algorithm Configuration
- **Population Size**: {final_snap.get('population_stats', {}).get('total', 'N/A')}
- **Fitness Function**: 0.4×Sortino + 0.4×Calmar + 0.2×WinRate
- **TP/SL**: ATR-based adaptive stops
- **Training Windows**: {final_snap.get('n_windows', 10)} random 1-month windows

### Data Processing
- **Semantic Validation**: LONG patterns use bullish modules, SHORT use bearish
- **Trade Frequency Penalty**: Prevents overtrading (>15 trades/month penalized)
- **Redundancy Removal**: Duplicate module families filtered

---

*Report generated automatically by Evolution Analytics System*
*SPRINT 14 - Enhanced Analytics for Academic Presentation*
*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[OK] Saved summary report: {output_path}")

    def _count_stagnation(self, df: pd.DataFrame, threshold: float = 0.01) -> int:
        """Count generations where best fitness improved < threshold."""
        stagnation = 0
        for i in range(1, len(df)):
            improvement = df.iloc[i]['best_fitness'] - df.iloc[i-1]['best_fitness']
            if improvement < threshold:
                stagnation += 1
        return stagnation

    def run_full_analysis(self, output_dir: str = "./analysis_output"):
        """Run complete analysis pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("")
        print("="*80)
        print("EVOLUTION ANALYTICS - SPRINT 14")
        print("="*80)
        print("")

        if len(self.snapshots) == 0:
            print("[ERROR] No snapshots found. Cannot generate analytics.")
            print(f"Expected snapshots in: {self.snapshots_dir}")
            return None

        # Extract metrics
        df = self.extract_metrics_timeseries()

        # Generate plots
        print("Generating visualizations...")
        self.plot_fitness_evolution(df, output_dir / "fitness_evolution.png")
        self.plot_performance_metrics(df, output_dir / "performance_metrics.png")
        self.plot_module_trends(df, output_dir / "module_trends.png")

        # Generate report
        print("\nGenerating report...")
        self.generate_summary_report(output_dir / "evolution_report.md")

        print("")
        print("="*80)
        print("[OK] Analysis complete!")
        print(f"Results saved to: {output_dir.absolute()}")
        print("")
        print("Output files:")
        print(f"  [DATA] fitness_evolution.png")
        print(f"  [DATA] performance_metrics.png")
        print(f"  [DATA] module_trends.png")
        print(f"  📄 evolution_report.md")
        print("="*80)
        print("")

        return df


if __name__ == "__main__":
    import sys

    # Allow command-line override of snapshot directory
    snapshots_dir = sys.argv[1] if len(sys.argv) > 1 else "./evolution_snapshots"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./analysis_output"

    analyzer = EvolutionAnalyzer(snapshots_dir=snapshots_dir)
    df = analyzer.run_full_analysis(output_dir=output_dir)

    if df is not None:
        print("[OK] Analysis completed successfully!")
        print(f"   Open {output_dir}/evolution_report.md to view the report")
    else:
        print("[ERROR] Analysis failed - check snapshots directory")
