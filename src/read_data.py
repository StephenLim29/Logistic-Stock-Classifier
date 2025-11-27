# Goal: use methods from parquet_helpers.py to read data 
# This script will fetch Yahoo Finance data and build a PyTorch Dataset/DataLoader
# from the Parquet/CSV files.


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from parquet_helpers import (
    data_Dir,
    fetch_prices,
    fetch_earnings_dates,
    load_price_frame,
    load_earnings_table,
)

# Tickers of companies we want to work with
companies = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META",
    "GOOGL", "JPM", "WMT", "HD", "MCD",
    "TGT", "NKE", "BAC", "KO", "DIS",
]

# Earliest date for prices history we want to use
# yFinance  interprests None as up to date
# 16 earnings is roughly 4 years of earnings (usually 4 per year)
startDate = "2018-01-01"
endDate = None  
earningsLimit = 16       

# How many trading days before the 5 day blackout to train on?
# Changing this will presumably change accuracy and runtime
windowSize = 60


# BUILDING THE pyTorch DATA SETS

# Earnings Dataset
class EarningsWindowDataset(Dataset):
    # Each item is a window of length "windowSize" made with OHLCV history leading up to 5 days before earnings date.

    def __init__(self, dataDir, window=windowSize, tickers=None):
        super().__init__()
        self.dataDir = dataDir
        self.window = window

        # Get the earnings table from "parquet_helpers"
        earnings_df = load_earnings_table(self.dataDir)

        # If there is a specific company or companies we want to only look at, we will filter the df
        if tickers is not None:
            earnings_df = earnings_df[earnings_df["Ticker"].isin(tickers)]

        # Represents a list of the tickers in the "earnings_df" removing the duplicates due to multiple earnings for single company
        self.tickers = sorted(earnings_df["Ticker"].unique())
        if earnings_df.empty:
            print("EarningsWindowDataset(). No earnings rows found.")
            self.samples = []
            self.feature_cols = []
            return

        # Preload price frames for all tickers in "earnings_df"
        # Get the price with "load_price_frame" from "parquet_helpers"
        price_frames = {}
        for ticker in self.tickers:
            try:
                price_frames[ticker] = load_price_frame(ticker, dataDir)
            except FileNotFoundError:
                print(f"warning: no price frame file for {ticker}, skipping")
                price_frames[ticker] = pd.DataFrame()

        # Get which columns we will use as features
        # If we want to change our model's features we can change these
        self.feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        samples = []

        # Build one sample per (Ticker, EarningsDate)
        for _, row in earnings_df.iterrows():
            ticker = row["Ticker"]
            earnings_date = pd.to_datetime(row["EarningsDate"]).normalize()

            # Get the prices from the preloaded map for the current ticker
            prices = price_frames.get(ticker)
            if prices is None or prices.empty:
                continue

            # I found that the timezone can be considered in the data. We will drop this (Only care about the date)
            if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
                prices = prices.copy()
                prices.index = prices.index.tz_localize(None)
            
            prices = prices.sort_index()

            # Only use data up to "earnings_date" - 5 days
            offset = earnings_date - pd.Timedelta(days=5)

            # Gets the history of prices before the offset (5 days before this earnings dates)
            preEarnings_prices = prices[prices.index <= offset]

            # If there is not enough history, skip this event
            if preEarnings_prices.empty:
                continue

            # Grab the prices within the window (5 days before earnings and "window" out)
            input_window = preEarnings_prices.tail(self.window)
            numDays = len(input_window)

            # Get the OHLCV 
            features = input_window[self.feature_cols].to_numpy(dtype="float32")

            mean = features.mean(0)
            std = features.std(0)
            features = (features - mean) / (std + 1e-8)

            # If there are not enough days to cover the length of the window we will pad with 0s (padding 0s to the left)
            if numDays < self.window:
                numPad = self.window - numDays
                padding = np.zeros((numPad, features.shape[1]), dtype="float32")
                features = np.vstack([padding, features])

                # Build a mask to represent what is real data from the df and what is padded
                # Will be information for the transformer
                pad_mask = np.concatenate([
                        np.zeros(numPad, dtype=bool), 
                        np.ones(numDays, dtype=bool)
                    ])
            else:
                pad_mask = np.ones(self.window, dtype=bool)

            # DONE: replace this with the true classification target.
            adjClose = prices["Adj Close"]
            p1 = adjClose[adjClose.index >= earnings_date + pd.Timedelta(days=1)]
            p5 = adjClose[adjClose.index >= earnings_date + pd.Timedelta(days=5)]

            if p1.empty or p5.empty: 
                continue

            return1 = float(p1.iloc[0])
            return5 = float(p5.iloc[0])
            returnAfterEarnings = (return1 - return5) / return1
            label = 1 if returnAfterEarnings > 0 else 0

            # Save this earnings event as one sample training example 
            samples.append(
                {
                    "inputs": torch.from_numpy(features),                       
                    "pad_mask": torch.from_numpy(pad_mask),                
                    "labels": torch.tensor(label, dtype=torch.long),         
                    "ticker": ticker,
                    "earnings_date": str(earnings_date.date()),
                }
            )

        self.samples = samples
        # Print for debugging 
        print(
            f"EarningsWindowDataset: {len(self.samples)} examples, "
            f"window={self.window}, features={self.feature_cols}"
        )



    # Defining the two methods PyTorch needs so DataLoader can work
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        batch = self.samples[index]
     
        return batch



# Data Loader
def build_earnings_dataloader(
    dataDir=data_Dir,
    window=windowSize,
    batch_size=32,
    tickers=None,
    shuffle=True,
    num_workers=0,
):

    dataset = EarningsWindowDataset(dataDir=dataDir, window=window, tickers=tickers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader



# Function part of testing
def run_fetch_parquet():
    print("Getting yFinance data...")
    fetch_prices(companies, start=startDate, end=endDate)
    fetch_earnings_dates(companies, limit=earningsLimit)
    print("Done")


if __name__ == "__main__":
    # 1. Download data
    run_fetch_parquet()

    # 2. Build Dataset and DataLoader 
    dataset, loader = build_earnings_dataloader(dataDir=data_Dir, window=windowSize, batch_size=4, tickers=companies, shuffle=True)

    # Debbuging 
    print(f"Dataset length: {len(dataset)}")

    # 3. Grab a single batch and print shapes for debugging
    if len(dataset) > 0:
        batch = next(iter(loader))
        inputs = batch["inputs"]     
        pad_mask = batch["pad_mask"]  
        labels = batch["labels"]      
        tickers_batch = batch["ticker"]
        dates_batch = batch["earnings_date"]

        print("[check] batch shapes:")
        print(f"inputs: {inputs.shape}")
        print(f"pad mask: {pad_mask.shape}")
        print(f"labels: {labels.shape}")
        print(f"tickers: {tickers_batch}")
        print(f"dates: {dates_batch}")
    else:
        print("Dataset is empty")
