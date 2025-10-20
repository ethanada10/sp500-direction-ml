import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # --- FORCER Close en Series 1D ---
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    data['r1'] = close.pct_change(1)
    data['r2'] = close.pct_change(2)
    data['r3'] = close.pct_change(3)
    data['r4'] = close.pct_change(4)
    data['r5'] = close.pct_change(5)

    data['vol_10'] = data['r1'].rolling(10).std()
    data['vol_20'] = data['r1'].rolling(20).std()
    data['ma10'] = close.rolling(10).mean()
    data['ma20'] = close.rolling(20).mean()
    data['ma_ratio'] = data['ma10'] / data['ma20']

    # RSI avec Series 1D
    rsi = RSIIndicator(close=close, window=14)
    data['rsi14'] = rsi.rsi()

    # target: next-day up/down
    data['fwd_ret'] = close.shift(-1) / close - 1.0
    data['y'] = (data['fwd_ret'] > 0).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[['r1','r2','r3','r4','r5','vol_10','vol_20','ma10','ma20','ma_ratio','rsi14']]
    y = data['y']
    fwd = data['fwd_ret']
    return X, y, fwd
