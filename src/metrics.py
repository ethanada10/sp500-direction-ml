import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sigma) if sigma != 0 else 0.0

def max_drawdown(equity_curve: pd.Series) -> float:
    cummax = equity_curve.cummax()
    dd = equity_curve / cummax - 1.0
    return float(dd.min())

def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) == 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)

def hit_rate(returns: pd.Series) -> float:
    pos = (returns > 0).sum()
    total = (returns != 0).sum()
    return float(pos / total) if total > 0 else 0.0
