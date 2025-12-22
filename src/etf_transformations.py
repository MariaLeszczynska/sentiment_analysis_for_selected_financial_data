from pathlib import Path
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal


RAW_DIR = Path("../data/preprocessed/etfs")
NEWS_DIR = Path("../data/preprocessed/final_data_with_embeddings_without_aggregations.csv")
OUT_DIR = Path("../data/preprocessed/final_etf_data")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def sign_next_day(df):
    df["Sign_next_day"] = df["Sign"].shift(-1)
    
    return df


def drop_sign_and_return(df):
    return df.drop(columns=["Sign", "Return"])


def is_trading_day_column(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    nyse = mcal.get_calendar('NYSE')

    schedule = nyse.schedule(start_date = df['Date'].min(), end_date = df['Date'].max())
    trading_days = schedule.index.normalize()

    df['is_trading_day'] = df['Date'].isin(trading_days).astype(int)
    return df



def compute_sector_daily_no_weekends(csv_path, out_dir, min_headlines=1):
    """
    Calculates daily average sentiment scores for a selected sector.

    For each date, the function:
    - selects only the headlines where the sector flag (e.g., XLE) equals 1,
    - averages the FinBERT sentiment scores ('positive', 'neutral', 'negative'),
    - counts how many sector-related headlines were found,
    - computes a simple sentiment index (positive - negative),
    - returns one row per day with all aggregated values
    """

    ticker = csv_path.stem.split("_")[0]

    news_df = pd.read_csv(NEWS_DIR)
    df = pd.read_csv(csv_path)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()

    # only headlines tagged for this sector
    sector_df = news_df.loc[news_df[ticker] == 1, ["Date", "positive", "neutral", "negative"]]

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

    daily_sentiment_score[f"sent_index_{ticker}"] = (
        daily_sentiment_score[f"avg_positive_{ticker}"]
        - daily_sentiment_score[f"avg_negative_{ticker}"]
    )

    if min_headlines > 1:
        too_few = daily_sentiment_score[f"n_{ticker}"] < min_headlines
        daily_sentiment_score.loc[too_few, [
            f"avg_positive_{ticker}",
            f"avg_neutral_{ticker}",
            f"avg_negative_{ticker}",
            f"sent_index_{ticker}",
        ]] = np.nan

    dfs_combined = df.merge(daily_sentiment_score, on="Date", how='left')

    subdir = Path(out_dir) / "no_weekends_no_embedding"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{ticker}_v1.csv"

    dfs_combined.to_csv(out_path, index=False)

    return out_path



def compute_sector_and_embeddings_daily_no_weekends(csv_path, emb_cols, min_headlines=1, prefix_emb: bool=True):
    """
    Calculates daily average sentiment scores for a selected sector.

    For each date, the function:
    - selects only the headlines where the sector flag (e.g., XLE) equals 1,
    - averages the FinBERT sentiment scores ('positive', 'neutral', 'negative'),
    - counts how many sector-related headlines were found,
    - computes a simple sentiment index (positive - negative),
    - aggregates the embeddings,
    - returns one row per day with all aggregated values
    """
    ticker = csv_path.stem.split("_")[0]

    news_df = pd.read_csv(NEWS_DIR)
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()


    columns_to_compute = ["Date", "positive", "neutral", "negative"] + list(emb_cols)

    # only headlines tagged for this sector
    sector_df = news_df.loc[news_df[ticker] == 1, columns_to_compute].copy()

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

    daily_sentiment_score[f"sent_index_{ticker}"] = (
        daily_sentiment_score[f"avg_positive_{ticker}"]
        - daily_sentiment_score[f"avg_negative_{ticker}"]
    )

    daily_emb = sector_df.groupby("Date", as_index=False)[list(emb_cols)].mean()

    if prefix_emb:
        daily_emb = daily_emb.rename(columns={c: f"{c}_{ticker}" for c in emb_cols})
        emb_out_cols = [f"{c}_{ticker}" for c in emb_cols]
    else:
        emb_out_cols = list(emb_cols)

    out = daily_sentiment_score.merge(daily_emb, on="Date", how="left")


    if min_headlines > 1:
        too_few = out[f"n_{ticker}"] < min_headlines
        cols_to_nan = [
            f"avg_positive_{ticker}",
            f"avg_neutral_{ticker}",
            f"avg_negative_{ticker}",
            f"sent_index_{ticker}",
            *emb_out_cols,
        ]
        out.loc[too_few, cols_to_nan] = np.nan


    combined_dfs = df.merge(out, on="Date", how="left")

    subdir = OUT_DIR / "no_weekends_embedding"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{ticker}_v2.csv"
    combined_dfs.to_csv(out_path, index=False)

    return out_path



def aggregate_to_next_trading_day_with_sectors(csv_path, min_headlines=1):
    ticker = csv_path.stem.split("_")[0]

    news_df = pd.read_csv(NEWS_DIR)
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date")

    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()
    news_df = news_df.sort_values("Date")


    date_flags = (news_df[["Date", "is_trading_day"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True))
    date_flags["TradingDate"] = date_flags["Date"].where(date_flags["is_trading_day"]==1)
    date_flags["TradingDate"] = date_flags["TradingDate"].bfill()

    news_df = news_df.merge(date_flags[["Date", "TradingDate"]], on="Date", how="left").dropna(subset=["TradingDate"])

    sector = news_df.loc[news_df[ticker] ==1, ["TradingDate", "positive", "neutral", "negative"]]

    out = (
        sector.groupby("TradingDate", as_index=False)
           .agg(
               **{
                   f"avg_positive_{ticker}": ("positive", "mean"),
                   f"avg_neutral_{ticker}":  ("neutral",  "mean"),
                   f"avg_negative_{ticker}": ("negative", "mean"),
                   f"n_{ticker}":            ("positive", "count"),
               }
           )
    )

    out[f"sent_index_{ticker}"] = out[f"avg_positive_{ticker}"] - out[f"avg_negative_{ticker}"]

    if min_headlines > 1:
        too_few = out[f"n_{ticker}"] < min_headlines
        out.loc[too_few, [
            f"avg_positive_{ticker}",
            f"avg_neutral_{ticker}",
            f"avg_negative_{ticker}",
            f"sent_index_{ticker}",
        ]] = np.nan

    out = out.rename(columns={"TradingDate": "Date"}).sort_values("Date").reset_index(drop=True)
    out["is_trading_day"] = 1

    combine_dfs = df.merge(out, on="Date", how='left')

    subdir = Path(OUT_DIR) / "weekends_aggregated_no_embedding"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{ticker}_v3.csv"
    combine_dfs.to_csv(out_path, index=False)

    return out_path


def aggregate_to_next_trading_day_sector_with_embeddings(csv_path, min_headlines=1):
    ticker = csv_path.stem.split("_")[0]
    
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date")

    news_df = pd.read_csv(NEWS_DIR)
    news_df = news_df.copy()
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()
    news_df = news_df.sort_values("Date")

    emb_cols = [c for c in news_df.columns if c.startswith("emb_")]

    date_flags = (news_df[["Date", "is_trading_day"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True))
    date_flags["TradingDate"] = date_flags["Date"].where(date_flags["is_trading_day"] == 1)
    date_flags["TradingDate"] = date_flags["TradingDate"].bfill()

    news_df = news_df.merge(date_flags[["Date", "TradingDate"]], on="Date", how="left")
    news_df = news_df.dropna(subset=["TradingDate"])

    cols = ["TradingDate", "positive", "neutral", "negative"] + list(emb_cols)
    sector = news_df.loc[news_df[ticker] == 1, cols]

    g = sector.groupby("TradingDate", as_index=False)

    daily_sent = g.agg(
        **{
            f"avg_positive_{ticker}": ("positive", "mean"),
            f"avg_neutral_{ticker}":  ("neutral",  "mean"),
            f"avg_negative_{ticker}": ("negative", "mean"),
            f"n_{ticker}":            ("positive", "count"),
        }
    )
    daily_sent[f"sent_index_{ticker}"] = (
        daily_sent[f"avg_positive_{ticker}"] - daily_sent[f"avg_negative_{ticker}"]
    )

    daily_emb = g[emb_cols].mean()

    out = daily_sent.merge(daily_emb, on="TradingDate", how="left")

    if min_headlines > 1:
        too_few = out[f"n_{ticker}"] < min_headlines
        cols_to_nan = [
            f"avg_positive_{ticker}",
            f"avg_neutral_{ticker}",
            f"avg_negative_{ticker}",
            f"sent_index_{ticker}",
            *emb_cols,
        ]
        out.loc[too_few, cols_to_nan] = np.nan

    out = out.rename(columns={"TradingDate": "Date"}).sort_values("Date").reset_index(drop=True)
    out["is_trading_day"] = 1

    combined_dfs = df.merge(out, on="Date", how="left")

    subdir = Path(OUT_DIR) / "weekends_aggregated_embedding"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{ticker}_v4.csv"
    combined_dfs.to_csv(out_path, index=False)

    return out_path
