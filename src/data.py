import pandas as pd
import yfinance as yf

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No data for {ticker}. Check dates or ticker.")
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
