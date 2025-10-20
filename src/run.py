import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import matplotlib.pyplot as plt

from .data import download_data
from .features import add_features
from .metrics import sharpe_ratio, max_drawdown, cagr, hit_rate

def train_test_split_time(X, y, fwd, train_size=0.7):
    n = len(X)
    split = int(n * train_size)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], fwd.iloc[split:]

def make_model(name: str):
    if name == 'logreg':
        return ('logreg', LogisticRegression(max_iter=1000))
    elif name == 'rf':
        return ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=42))
    else:
        raise ValueError('Unknown model')

def main():
    parser = argparse.ArgumentParser(description='SP500 Direction Classifier')
    parser.add_argument('--ticker', type=str, default='SPY')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--model', type=str, default='logreg', choices=['logreg','rf'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--train_size', type=float, default=0.7)
    args = parser.parse_args()

    raw = download_data(args.ticker, args.start, args.end)
    X, y, fwd = add_features(raw)
    X_train, X_test, y_train, y_test, fwd_test = train_test_split_time(X, y, fwd, train_size=args.train_size)

    name, model = make_model(args.model)

    # Scale for LogReg (not necessary for RF)
    if name == 'logreg':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:,1]

    y_pred = (proba > args.threshold).astype(int)

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float('nan')

    # Strategy: long when proba>threshold
    strat_returns = pd.Series((proba > args.threshold).astype(int), index=X_test.index) * fwd_test
    equity = (1 + strat_returns).cumprod()

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc,
        'sharpe': sharpe_ratio(strat_returns),
        'hitrate': hit_rate(strat_returns),
        'cagr': cagr(equity),
        'max_drawdown': max_drawdown(equity)
    }

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path('outputs') / f"{args.ticker}_{name}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    pd.Series(proba, index=X_test.index, name='proba').to_csv(outdir / 'proba.csv')
    pd.Series(y_pred, index=X_test.index, name='y_pred').to_csv(outdir / 'y_pred.csv')
    strat_returns.to_csv(outdir / 'strategy_returns.csv')
    equity.to_csv(outdir / 'equity.csv')
    pd.Series(metrics).to_json(outdir / 'metrics.json', indent=2)

    # Plot equity
    plt.figure()
    equity.plot()
    plt.title(f'Equity Curve — {args.ticker} ({name.upper()})')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.savefig(outdir / 'equity.png', bbox_inches='tight')

    # Print a short summary
    print('Metrics:', metrics)
    print('Outputs saved to:', outdir)

if __name__ == '__main__':
    main()
