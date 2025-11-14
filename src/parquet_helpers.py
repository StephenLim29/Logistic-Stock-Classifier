# Goal: pipeline that takes yfinance data and turns it into parquet/csv files

import os

import pandas as pd
import yfinance as yf

# All the raw parquet/csv files
data_Dir = os.path.join("data", "raw")
os.makedirs(data_Dir, exist_ok=True)



# Table helpers

def save_table(df, base_path):
    if df is None or df.empty:
        raise ValueError("save_table(). Empty DataFrame")

    out_path = base_path + ".csv"
    df.to_csv(out_path)
    return out_path


def load_table(base_path):
    csv_path = base_path + ".csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return df

    raise FileNotFoundError(f"load_table(). No file found: {base_path}")



# Normalization 
def normalize_price_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()

    # Make column names consistent
    df = df.rename(columns=str.title)

    # Ensure Date index and sorted
    # yfinance dates look like '2018-01-02'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df.index.name = "Date"
    df = df.sort_index()
    df = df.dropna(how="all")
    return df




# Yahoo Finance fetchers

# Get daily OHLCV for each ticker
def fetch_prices(tickers, start, end=None, dataDir=data_Dir):
    os.makedirs(dataDir, exist_ok=True)

    for ticker in tickers:
        base = os.path.join(dataDir, f"{ticker}_prices")
        csv_path = base + ".csv"

        if os.path.exists(csv_path):
            print(f"Prices already cached for {ticker}")
            continue

        # Print for Debbuging
        print(f"Downloading prices for {ticker}")
        df = yf.download(ticker, start=start, end=end)
        df = normalize_price_frame(df)

        # Debbuging check again
        if df.empty:
            print(f"Price frame for {ticker} is empty")
            continue

        # Save df and print the path and row count
        file_path = save_table(df, base)
        print(f"Saved {file_path} with {len(df)} rows")



# Get earnings dates for each ticker 
def fetch_earnings_dates(tickers, limit, dataDir=data_Dir):
    os.makedirs(dataDir, exist_ok=True)

    earnings_tables = []

    for ticker in tickers:
        print(f"Downloading earnings dates for {ticker}")
        ticker_obj = yf.Ticker(ticker)

        earnings_dates = None

        # Use yfinance's "get_earnings_dates"
        try:
            earnings_df = ticker_obj.get_earnings_dates(limit=limit, offset=0)

            if isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty:
                # For "get_earnings_dates", the dates are in the index
                date_index = pd.to_datetime(earnings_df.index, errors="coerce")

                # If dates have a timezone, drop it 
                if isinstance(date_index, pd.DatetimeIndex) and date_index.tz is not None:
                    date_index = date_index.tz_localize(None)

                earnings_dates = (pd.Series(date_index, name="EarningsDate").dropna().drop_duplicates().sort_values())
        except Exception as e:
            print(f"get_earnings_dates failed for {ticker}: {e}")
            earnings_dates = None

        if earnings_dates is not None and len(earnings_dates) > 0:
            ticker_table = pd.DataFrame({"Ticker": ticker, "EarningsDate": earnings_dates.values})
            earnings_tables.append(ticker_table)
        else:
            print(f"No earnings dates pulled for {ticker}")

    if not earnings_tables:
        print("No earnings dates collected for any ticker")
        return

    combined = (pd.concat(earnings_tables, ignore_index=True).sort_values(["Ticker", "EarningsDate"]))
    used_path = save_table(combined, os.path.join(dataDir, "earnings_dates"))
    print(f"Saved earnings table to {used_path} with {len(combined)} rows")




# Loaders

# Load and normalize a single ticker's price frame from dataDir
def load_price_frame(ticker, dataDir=data_Dir):
    base = os.path.join(dataDir, f"{ticker}_prices")
    df = load_table(base)
    df = normalize_price_frame(df)
    return df


# Load the combined earnings_dates table from dataDir
def load_earnings_table(dataDir=data_Dir):
    base = os.path.join(dataDir, "earnings_dates")
    df = load_table(base)

    # Make sure EarningsDate is a clean datetime column
    df["EarningsDate"] = pd.to_datetime(df["EarningsDate"], errors="coerce")
    df = df.dropna(subset=["EarningsDate"])
    df["EarningsDate"] = df["EarningsDate"].dt.normalize()
    df = df.sort_values(["Ticker", "EarningsDate"])
    return df
