from pathlib import Path
import pandas as pd
import numpy as np


ETF_DIR = Path("../data/preprocessed/etfs")
NEWS_DIR = Path("../data/preprocessed/headlines/headlines_finbert.csv")
OUT_DIR = Path("../data/preprocessed/final")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sector_daily(df, ticker, min_headlines=1):
    """
    Calculates daily average sentiment scores for a selected sector.
    For each date, the function:
    - selects only the headlines where the sector flag (e.g., XLE) equals 1,
    - averages the FinBERT sentiment scores ('positive', 'neutral', 'negative'),
    - counts how many sector-related headlines were found,
    - computes a simple sentiment index (positive - negative),
    - returns one row per day with all aggregated values
    """

    # ensure datetime (basically a sanity check)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    sector_df = df.loc[df[ticker] == 1, ["Date", "positive", "neutral", "negative"]]

    # aggregate per day
    daily_sentiment_score = (
        sector_df.groupby("Date", as_index=False)
        .agg(
            **{
                f"avg_positive_{ticker}": ("positive", "mean"),
                f"avg_neutral_{ticker}":  ("neutral",  "mean"),
                f"avg_negative_{ticker}": ("negative", "mean"),
                f"n_{ticker}":            ("positive", "count"),
            }
        )
    )

    # sentiment index (positive - negative)
    daily_sentiment_score[f"sent_index_{ticker}"] = (
        daily_sentiment_score[f"avg_positive_{ticker}"] - daily_sentiment_score[f"avg_negative_{ticker}"]
    )

    # what if there are too few headlines for the day
    if min_headlines > 1:
        too_few = daily_sentiment_score[f"n_{ticker}"] < min_headlines
        daily_sentiment_score.loc[too_few, [f"avg_positive_{ticker}",
                            f"avg_neutral_{ticker}",
                            f"avg_negative_{ticker}",
                            f"sent_index_{ticker}"]] = np.nan

    return daily_sentiment_score



def _normalize_date_column(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    return df


def add_next_day_cols(ticker):
    """
    Loads preprocessed ETF CSVs, 
    calls the date normalising function and adds new columns:
    - return from the next day
    - sign from the next day
    """

    csv_path = ETF_DIR / f"{ticker}_preprocessed.csv"
    etf = pd.read_csv(csv_path)
    etf = _normalize_date_column(etf)

    etf["Return_next_day"] = etf["Return"].shift(-1)
    etf["Sign_next_day"] = etf["Sign"].shift(-1)

    return etf


def build_all_sector_datasets(tickers=("XLE", "XLF", "XLK", "XLV", "XLY"), min_headlines=1):
    df = pd.read_csv(NEWS_DIR)
    df = _normalize_date_column(df)

    for t in tickers:
        daily = compute_sector_daily(df, t, min_headlines=min_headlines)
        prices = add_next_day_cols(t)

        # daily prices + daily sector aggregates
        merged_daily = prices.merge(daily, on="Date", how="left")

        data = df.merge(merged_daily, on="Date", how="left")

        out_path = OUT_DIR / f"{t.lower()}.csv"
        data.to_csv(out_path, index=False)
        print(f"Saved {t} (headline-level) to: {out_path}")
        