# S&P 500 Direction Classifier — Simple ML

**Goal:** Predict tomorrow's S&P 500 close direction (Up/Down) using simple technical features, then evaluate a naive long/flat strategy driven by the model's probability.

## Features
- Lagged returns: r1...r5
- Rolling volatility: std of r1 over 10 and 20 days
- Moving averages: MA10, MA20 + MA ratio (MA10/MA20)
- RSI(14)

## Models
- Logistic Regression (baseline)
- Random Forest (stronger non-linear baseline)

## Evaluation
- Time-aware split (70% train / 30% test by date)
- Classification metrics: Accuracy, Precision, Recall, ROC AUC
- Strategy metrics (long if P(up) > threshold): Sharpe, CAGR, Max Drawdown, Hit Rate
- Equity curve plot

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train & evaluate on SPY daily data since 2010 (LogReg)
python src/run.py --ticker SPY --start 2010-01-01 --end 2025-10-01 --model logreg --threshold 0.55

# Try RandomForest
python src/run.py --ticker SPY --start 2010-01-01 --end 2025-10-01 --model rf --threshold 0.55
```

## Notes
- Educational use only — not investment advice.
- You can switch to CAC40 proxy: `^FCHI` or ETF like `EWQ`.
