# Deployment & Reproducibility Checklist

## Overview

This document ensures that experiments are fully reproducible and results can be validated by others (reviewers, committee members, future researchers).

## Pre-Deployment Checklist

### ✅ Environment Setup

- [ ] Python version documented (3.9+)
- [ ] All dependencies in `requirements.txt` with pinned versions
- [ ] Virtual environment created and activated
- [ ] All tests passing (`pytest`)

### ✅ Configuration

- [ ] `config.yaml` reviewed and commented
- [ ] Random seeds set for reproducibility
- [ ] Date ranges specified explicitly
- [ ] All paths use relative references (not hardcoded)

### ✅ Data

- [ ] Data source documented (Binance API)
- [ ] Date range covers sufficient regimes (bull + bear markets)
- [ ] Data validation checks pass
- [ ] Backup of raw data stored (optional)

### ✅ Code Quality

- [ ] All functions have docstrings
- [ ] Type hints added where appropriate
- [ ] No hardcoded values (use config)
- [ ] Code formatted consistently
- [ ] Linting passes

## Reproducibility Requirements

### 1. Version Control
```bash
# Commit all code
git add .
git commit -m "Final version for thesis submission"
git tag -a v1.0.0 -m "Thesis submission version"
git push origin main --tags
```

### 2. Environment Freeze
```bash
# Exact environment
pip freeze > requirements_frozen.txt

# Python version
python --version > python_version.txt

# System info
# On Windows:
systeminfo > system_info.txt
# On Linux/Mac:
uname -a > system_info.txt
```

### 3. Random Seeds

Verify all seeds are set in `config.yaml`:
```yaml
ga:
  seed: 42           # Genetic algorithm seed

robustness:
  hansen_spa:
    seed: 42         # Hansen SPA seed
  white_rc:
    seed: 42         # White RC seed
  bootstrap:
    seed: 42         # Bootstrap seed
```

### 4. Configuration Archive
```bash
# Archive final config with timestamp
cp config.yaml config_final_20250109.yaml
```

## Running Experiments

### Standard Run
```bash
# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Run experiment
python main.py > experiment.log 2>&1
```

### With Timestamps
```bash
# Windows PowerShell:
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python main.py > "logs/experiment_$timestamp.log" 2>&1

# Linux/Mac:
timestamp=$(date +%Y%m%d_%H%M%S)
python main.py > logs/experiment_${timestamp}.log 2>&1
```

## Post-Experiment Checklist

### ✅ Outputs Generated

- [ ] `experiment_report.md` created
- [ ] All visualizations generated (5 PNG files)
- [ ] LaTeX tables exported (3 TEX files)
- [ ] Statistical test results saved (3 JSON files)
- [ ] Equity curves exported (CSV)

### ✅ Results Validation

- [ ] Fitness values are reasonable (not NaN, not -999)
- [ ] Portfolio size > 0
- [ ] Statistical tests completed successfully
- [ ] P-values are between 0 and 1
- [ ] Visualizations display correctly

### ✅ Documentation

- [ ] README.md updated with results summary
- [ ] USER_GUIDE.md reflects any config changes
- [ ] Comments added for any non-obvious code
- [ ] All docstrings complete

## Validation by Others

### For Reviewers

Provide these files in submission package:
```
submission_package/
├── code/
│   ├── *.py (all source code)
│   ├── config.yaml
│   ├── requirements.txt
│   └── README.md
├── results/
│   ├── experiment_report.md
│   ├── *.png (visualizations)
│   ├── *.tex (LaTeX tables)
│   └── *.json (statistical results)
└── REPRODUCIBILITY.md (this file)
```

### Reproduction Steps

**1. Setup environment:**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**2. Verify data:**
```bash
# Check data downloads correctly
python -c "from loader import check_binance_connection, load_binance_data; import yaml; config = yaml.safe_load(open('config.yaml')); print('Connection OK' if check_binance_connection(config) else 'Connection FAILED')"
```

**3. Run experiment:**
```bash
python main.py
```

**4. Verify results:**
```bash
# Check key files exist
ls output_reports/experiment_report.md
ls output_reports/*.png
ls output_reports/*.tex
```

## Common Issues & Solutions

### Issue: Different Results with Same Seed

**Causes**:
- Different Python version
- Different NumPy version
- Different operating system

**Solution**: Document exact environment
```bash
pip freeze > requirements_frozen.txt
python --version > python_version.txt
```

### Issue: Missing Dependencies

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Out of Memory

**Solution**: Reduce dataset size in `config.yaml`
```yaml
ga:
  population: 50  # Reduced from 100

data:
  timeframe: "1h"  # Larger timeframe = less data
  start: "2022-01-01"  # Shorter period
```

## Performance Benchmarks

Document expected runtimes on your hardware:

| Phase | Expected Time | Actual Time |
|-------|--------------|-------------|
| Data Loading | 2-5 min | ___ min |
| GA Evolution | 30-90 min | ___ min |
| Portfolio Selection | 30-60 min | ___ min |
| Statistical Tests | 10-20 min | ___ min |
| Reports | 2-5 min | ___ min |
| **Total** | **~2-3 hours** | ___ hours |

**Hardware**: [Document: CPU, RAM, OS]

## Academic Integrity

### Data Usage

- ✅ Proper attribution to data source (Binance)
- ✅ No data leakage (train/test separation validated)
- ✅ Walk-forward methodology documented
- ✅ All data transformations explained

### Code Attribution

- ✅ Third-party libraries cited in paper
- ✅ Algorithm references included (Hansen, White)
- ✅ Original contributions clearly identified

### Results Reporting

- ✅ All experiments reported (not just successful ones)
- ✅ Negative results discussed
- ✅ Limitations acknowledged
- ✅ No p-hacking (predefined significance levels)

## Archive for Thesis Submission

### Required Files
```bash
# Create submission archive
# Windows:
tar -czf thesis_submission.tar.gz *.py config.yaml requirements.txt README.md DEPLOYMENT.md docs/ tests/ output_reports/

# Linux/Mac:
tar -czf thesis_submission_$(date +%Y%m%d).tar.gz \
    *.py \
    config.yaml \
    requirements.txt \
    README.md \
    DEPLOYMENT.md \
    docs/ \
    tests/ \
    output_reports/
```

### Submission Checklist

- [ ] Code archive created
- [ ] Results archive created
- [ ] Documentation complete
- [ ] All tests passing
- [ ] Paper references code version (git tag)
- [ ] Repository made public (if allowed)

## Long-Term Preservation

### GitHub Repository

- [ ] Code pushed to GitHub
- [ ] README includes paper citation
- [ ] Releases tagged with thesis version
- [ ] License file added (MIT recommended)

### Optional: Zenodo Archive

For permanent DOI:
1. Connect GitHub to Zenodo
2. Create release on GitHub
3. Zenodo automatically creates DOI
4. Include DOI in thesis

## Contact for Reproducibility Questions

**Primary Contact**: Juan Manuel [Your Last Name]
- Email: [your.email@ucema.edu.ar]
- GitHub: [github.com/youruser]

**Thesis Committee**:
- Advisor: [Advisor Name]
- Committee Members: [Names]

---

**Version**: 1.0
**Date**: January 2025
**Last Updated**: 2025-01-09
