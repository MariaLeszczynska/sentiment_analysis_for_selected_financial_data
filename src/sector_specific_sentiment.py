from pathlib import Path
import pandas as pd
import numpy as np


def compute_sector_daily(df: pd.DataFrame, ticker: str, min_headlines: int = 1) -> pd.DataFrame:
    """
    Calculates daily average sentiment scores for a selected sector.
    For each date, the function:
    - selects only the headlines where the sector flag (e.g., XLE) equals 1,
    - averages the FinBERT sentiment scores ('positive', 'neutral', 'negative'),
    - counts how many sector-related headlines were found,
    - computes a simple sentiment index (positive - negative),
    - returns one row per day with all aggregated values.
    """

    # ensure datetime (basically a sanity check)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # mask non-sector rows
    mask = df[ticker] == 1
    sector_df = df.loc[mask, ["Date", "positive", "neutral", "negative"]]

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

    # composite sentiment index (positive - negative)
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
