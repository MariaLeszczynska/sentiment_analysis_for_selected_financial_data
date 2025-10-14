from pathlib import Path
import pandas as pd
import numpy as np


ETF_DIR = Path("../data/preprocessed/etfs")
NEWS_DIR = Path("../data/preprocessed/headlines/headlines_finbert.csv")
OUT_DIR = Path("../data/preprocessed/final")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_date_column(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


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
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    sector_df = df.loc[df[ticker] == 1, ["Date", "positive", "neutral", "negative"]]

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
        daily_sentiment_score[f"avg_positive_{ticker}"] - 
        daily_sentiment_score[f"avg_negative_{ticker}"]
    )

    if min_headlines > 1:
        too_few = daily_sentiment_score[f"n_{ticker}"] < min_headlines
        daily_sentiment_score.loc[too_few, [
            f"avg_positive_{ticker}",
            f"avg_neutral_{ticker}",
            f"avg_negative_{ticker}",
            f"sent_index_{ticker}"
        ]] = np.nan

    return daily_sentiment_score


def add_next_day_cols_FIXED(ticker):
    """
    FIXED VERSION:
    Loads preprocessed ETF CSVs and adds next-day columns CORRECTLY.
    
    The key fix: We shift BEFORE merging, while data only contains trading days.
    This ensures each trading day maps to the NEXT trading day, not the next calendar day.
    """
    csv_path = ETF_DIR / f"{ticker}_preprocessed.csv"
    etf = pd.read_csv(csv_path)
    etf = _normalize_date_column(etf)
    
    # Sort to be safe
    etf = etf.sort_values("Date").reset_index(drop=True)
    
    # CRITICAL: Shift HERE, while we only have trading days
    # This maps each trading day to the NEXT trading day
    etf["Return_next_day"] = etf["Return"].shift(-1)
    etf["Sign_next_day"] = etf["Sign"].shift(-1)
    
    return etf


def build_all_sector_datasets(tickers=("XLE", "XLF", "XLK", "XLV", "XLY"), min_headlines=1):
    """
    Builds final datasets by combining ETF prices with sector sentiment.
    
    The pipeline:
    1. Load news with FinBERT scores
    2. For each ticker:
       a. Compute daily sector sentiment aggregates
       b. Load ETF data with CORRECTLY shifted next-day targets
       c. Merge ETF + sentiment (creates dataset with all calendar days)
       d. Merge with full news for headline-level analysis
    """
    df = pd.read_csv(NEWS_DIR)
    df = _normalize_date_column(df)

    for t in tickers:
        print(f"\nProcessing {t}...")
        
        # Compute daily sentiment for this sector
        daily = compute_sector_daily(df, t, min_headlines=min_headlines)
        print(f"  Daily sentiment aggregates: {len(daily)} days")
        
        # Load ETF data with CORRECTLY shifted targets
        prices = add_next_day_cols_FIXED(t)
        print(f"  ETF trading days: {len(prices)}")
        
        # Merge prices + daily sentiment
        # This creates a dataset with all calendar days (including weekends)
        merged_daily = prices.merge(daily, on="Date", how="outer")
        merged_daily = merged_daily.sort_values("Date").reset_index(drop=True)
        print(f"  After merge: {len(merged_daily)} total days")
        
        # Merge with full news for headline-level data
        data = df.merge(merged_daily, on="Date", how="left")
        print(f"  Final headline-level dataset: {len(data)} headlines")
        
        # Verify the fix worked
        trading_days_with_target = data[
            data["Price"].notna() & 
            data["Return_next_day"].notna()
        ]
        print(f"  Trading days with valid next-day target: {len(trading_days_with_target)}")
        
        out_path = OUT_DIR / f"{t.lower()}.csv"
        data.to_csv(out_path, index=False)
        print(f"  ✓ Saved to: {out_path}")


# Additional utility function to verify the fix
def verify_next_day_alignment(ticker="XLE"):
    """
    Verify that next-day targets are correctly aligned.
    Checks a few Fridays to ensure they map to Monday's returns.
    """
    print(f"\n=== VERIFICATION FOR {ticker} ===\n")
    
    out_path = OUT_DIR / f"{ticker.lower()}.csv"
    if not out_path.exists():
        print(f"File not found: {out_path}")
        return
    
    data = pd.read_csv(out_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data["day_of_week"] = data["Date"].dt.day_name()
    
    # Check Fridays
    fridays = data[
        (data["day_of_week"] == "Friday") & 
        (data["Price"].notna())
    ].head(5)
    
    all_correct = True
    
    for idx, friday in fridays.iterrows():
        # Find next trading day (should be Monday)
        next_trading = data[
            (data["Date"] > friday["Date"]) & 
            (data["Price"].notna())
        ]
        
        if len(next_trading) > 0:
            monday = next_trading.iloc[0]
            
            print(f"Friday {friday['Date'].strftime('%Y-%m-%d')}:")
            print(f"  Friday's Return_next_day: {friday['Return_next_day']:.6f}" if pd.notna(friday['Return_next_day']) else "  Friday's Return_next_day: NaN")
            print(f"  Monday {monday['Date'].strftime('%Y-%m-%d')}'s Return: {monday['Return']:.6f}" if pd.notna(monday['Return']) else "  Monday's Return: NaN")
            
            if pd.notna(friday['Return_next_day']) and pd.notna(monday['Return']):
                if abs(friday['Return_next_day'] - monday['Return']) < 0.0001:
                    print(f"CORRECT MATCH!\n")
                else:
                    print(f"MISMATCH!\n")
                    all_correct = False
            else:
                print(f"  ⚠ Missing data\n")
    
    if all_correct:
        print("All checks passed! Next-day alignment is correct.")
    else:
        print("Found misalignments. There may still be an issue.")


# Usage:
if __name__ == "__main__":
    # Build all datasets with the fixed pipeline
    build_all_sector_datasets()
    
    # Verify the fix worked for one ticker
    verify_next_day_alignment("XLE")