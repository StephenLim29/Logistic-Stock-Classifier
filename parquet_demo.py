# main.py
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","META",
    "GOOGL","JPM","WMT","HD","MCD",
    "TGT","NKE","BAC","KO","DIS",
]

START = "2018-01-01"
END = None  # None -> up to today
EARNINGS_LIMIT = 16  # ~4 years of quarterly events

# ----------------------------
# Helpers
# ----------------------------
def save_table(df: pd.DataFrame, path_without_ext: str) -> str:
    """
    Try to save as Parquet; if engine missing, fall back to CSV.
    Returns the path used.
    """
    try:
        # ensure pyarrow is present (will raise if not)
        import pyarrow  # noqa: F401
        out_path = path_without_ext + ".parquet"
        df.to_parquet(out_path)
        return out_path
    except Exception:
        out_path = path_without_ext + ".csv"
        # index often meaningful for price data, so keep it
        df.to_csv(out_path)
        return out_path

def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize price columns and index for consistency.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)  # 'Open','High','Low','Close','Adj Close','Volume'
    # Ensure Date index and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "Date"
    df = df.sort_index()
    # Drop any completely empty rows
    df = df.dropna(how="all")
    return df

# ----------------------------
# Data Fetchers
# ----------------------------
def fetch_prices(tickers, start=START, end=END):
    """
    Download daily OHLCV for each ticker and cache per-ticker file in data/raw/.
    """
    for t in tickers:
        base = os.path.join(RAW_DIR, f"{t}_prices")
        parquet_path = base + ".parquet"
        csv_path = base + ".csv"
        if os.path.exists(parquet_path) or os.path.exists(csv_path):
            print(f"[skip] prices cached: {t}")
            continue

        print(f"[download] prices: {t}")
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        df = normalize_price_frame(df)
        if df.empty:
            print(f"[warn] empty price frame for {t}; skipping save.")
            continue

        used = save_table(df, base)
        print(f"[saved] {used}  rows={len(df)}")

def fetch_earnings_dates(tickers, limit=EARNINGS_LIMIT):
    rows = []

    def _normalize_to_series(df):
        """Return a pandas Series of datetimes named 'EarningsDate' from various shapes."""
        if df is None or len(df) == 0:
            return pd.Series(dtype="datetime64[ns]", name="EarningsDate")

        # Case A: dates in DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            s = pd.Series(df.index).rename("EarningsDate")
            return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

        # Case B: common column names
        for col in ["Earnings Date", "EarningsDate", "date", "Date", "startdatetime", "startdatetime_utc"]:
            if col in df.columns:
                s = pd.to_datetime(df[col], errors="coerce")
                # drop timezone to keep everything naive
                try:
                    s = s.dt.tz_localize(None)
                except Exception:
                    pass
                s.name = "EarningsDate"
                return s

        # Case C: any column containing 'date'
        candidates = [c for c in df.columns if "date" in c.lower()]
        if candidates:
            s = pd.to_datetime(df[candidates[0]], errors="coerce")
            try:
                s = s.dt.tz_localize(None)
            except Exception:
                pass
            s.name = "EarningsDate"
            return s

        # Nothing recognizable
        return pd.Series(dtype="datetime64[ns]", name="EarningsDate")

    for t in tickers:
        print(f"[download] earnings dates: {t}")
        tk = yf.Ticker(t)

        collected = []

        # Try 1: get_earnings_dates()
        try:
            edf = tk.get_earnings_dates(limit=limit, offset=0)
            if isinstance(edf, pd.DataFrame) and not edf.empty:
                s = _normalize_to_series(edf).dropna().drop_duplicates()
                if len(s):
                    collected.append(s)
        except Exception as e:
            print(f"[info] get_earnings_dates fallback for {t}: {e}")

        # Try 2: earnings_dates property
        try:
            edp = tk.earnings_dates
            if isinstance(edp, pd.DataFrame) and not edp.empty:
                s = _normalize_to_series(edp).dropna().drop_duplicates()
                if len(s):
                    collected.append(s)
        except Exception:
            pass

        # Try 3: get_earnings_history(as_dict=True) -> list of dicts w/ 'startdatetime'
        try:
            eh = tk.get_earnings_history(as_dict=True)
            if isinstance(eh, list) and eh:
                s = pd.to_datetime([row.get("startdatetime") for row in eh], errors="coerce")
                try:
                    s = s.tz_localize(None)
                except Exception:
                    pass
                s = pd.Series(s, name="EarningsDate").dropna().drop_duplicates()
                if len(s):
                    collected.append(s)
        except Exception:
            pass

        if collected:
            s_all = pd.concat(collected).dropna().drop_duplicates().sort_values()
            out = pd.DataFrame({"Ticker": t, "EarningsDate": s_all.values})
            rows.append(out)
        else:
            print(f"[warn] no earnings dates pulled for {t}")

    if not rows:
        print("[warn] no earnings dates collected.")
        return

    out = pd.concat(rows, ignore_index=True).sort_values(["Ticker", "EarningsDate"])
    used = save_table(out, os.path.join(RAW_DIR, "earnings_dates"))
    print(f"[saved] {used}  rows={len(out)}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    print("[start] fetching Yahoo Finance data...")
    fetch_prices(TICKERS)
    fetch_earnings_dates(TICKERS)
    print("[done] fetch step. Check data/raw/ for files.")
