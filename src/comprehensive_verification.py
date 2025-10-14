from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("../data/preprocessed/final")


def comprehensive_verification(ticker="XLE"):
    """
    Thorough verification of next-day alignment.
    Checks multiple aspects:
    1. Different Fridays throughout the dataset
    2. Days before holidays
    3. Random trading days
    4. Statistical validation
    """
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE VERIFICATION FOR {ticker}")
    print(f"{'='*60}\n")
    
    out_path = OUT_DIR / f"{ticker.lower()}.csv"
    data = pd.read_csv(out_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["day_of_week"] = data["Date"].dt.day_name()
    
    # Get only trading days
    trading_days = data[data["Price"].notna()].copy().reset_index(drop=True)
    
    print(f"Total rows in dataset: {len(data)}")
    print(f"Trading days: {len(trading_days)}")
    print(f"Non-trading days: {len(data) - len(trading_days)}")
    print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}\n")
    
    # =====================
    # TEST 1: Check DIFFERENT Fridays
    # =====================
    print("="*60)
    print("TEST 1: Checking DIFFERENT Fridays across the dataset")
    print("="*60)
    
    fridays = trading_days[trading_days["day_of_week"] == "Friday"].copy()
    print(f"\nTotal Fridays in dataset: {len(fridays)}\n")
    
    # Sample evenly distributed Fridays
    if len(fridays) > 10:
        # Take Fridays from beginning, middle, and end
        indices = np.linspace(0, len(fridays)-1, min(10, len(fridays)), dtype=int)
        sample_fridays = fridays.iloc[indices]
    else:
        sample_fridays = fridays
    
    friday_mismatches = 0
    
    for idx, friday in sample_fridays.iterrows():
        # Find next trading day
        next_trading = trading_days[trading_days["Date"] > friday["Date"]]
        
        if len(next_trading) > 0:
            next_day = next_trading.iloc[0]
            
            match_symbol = "✓" if abs(friday["Return_next_day"] - next_day["Return"]) < 0.000001 else "❌"
            
            print(f"{match_symbol} Friday {friday['Date'].strftime('%Y-%m-%d')}:")
            print(f"   Return_next_day = {friday['Return_next_day']:.6f}")
            print(f"   Next trading day: {next_day['Date'].strftime('%Y-%m-%d')} ({next_day['day_of_week']})")
            print(f"   Actual Return = {next_day['Return']:.6f}")
            
            if abs(friday["Return_next_day"] - next_day["Return"]) > 0.000001:
                friday_mismatches += 1
                print(f"   ⚠️  MISMATCH DETECTED!")
            print()
    
    # =====================
    # TEST 2: Check days before holidays
    # =====================
    print("="*60)
    print("TEST 2: Checking days before holidays (gaps > 3 days)")
    print("="*60)
    
    trading_days["next_date"] = trading_days["Date"].shift(-1)
    trading_days["days_gap"] = (trading_days["next_date"] - trading_days["Date"]).dt.days
    
    holidays = trading_days[trading_days["days_gap"] > 3].head(5)
    
    if len(holidays) > 0:
        print(f"\nFound {len(trading_days[trading_days['days_gap'] > 3])} trading days before long breaks\n")
        
        holiday_mismatches = 0
        
        for idx, day_before in holidays.iterrows():
            next_trading = trading_days[trading_days["Date"] > day_before["Date"]]
            
            if len(next_trading) > 0:
                next_day = next_trading.iloc[0]
                
                match_symbol = "✓" if abs(day_before["Return_next_day"] - next_day["Return"]) < 0.000001 else "❌"
                
                print(f"{match_symbol} {day_before['Date'].strftime('%Y-%m-%d')} ({day_before['day_of_week']}):")
                print(f"   Gap: {day_before['days_gap']:.0f} days")
                print(f"   Return_next_day = {day_before['Return_next_day']:.6f}")
                print(f"   Next trading: {next_day['Date'].strftime('%Y-%m-%d')} ({next_day['day_of_week']})")
                print(f"   Actual Return = {next_day['Return']:.6f}")
                
                if abs(day_before["Return_next_day"] - next_day["Return"]) > 0.000001:
                    holiday_mismatches += 1
                    print(f"   ⚠️  MISMATCH DETECTED!")
                print()
    else:
        print("\nNo long holiday gaps found in dataset\n")
    
    # =====================
    # TEST 3: Random sample of consecutive trading days
    # =====================
    print("="*60)
    print("TEST 3: Random sample of consecutive trading days")
    print("="*60)
    
    # Sample 10 random trading days (excluding last one)
    if len(trading_days) > 11:
        random_indices = np.random.choice(len(trading_days)-1, size=min(10, len(trading_days)-1), replace=False)
        random_sample = trading_days.iloc[random_indices]
    else:
        random_sample = trading_days.iloc[:-1]
    
    print(f"\nChecking {len(random_sample)} random trading days\n")
    
    random_mismatches = 0
    
    for idx, current_day in random_sample.iterrows():
        next_trading = trading_days[trading_days["Date"] > current_day["Date"]]
        
        if len(next_trading) > 0:
            next_day = next_trading.iloc[0]
            
            match_symbol = "✓" if abs(current_day["Return_next_day"] - next_day["Return"]) < 0.000001 else "❌"
            
            print(f"{match_symbol} {current_day['Date'].strftime('%Y-%m-%d')} ({current_day['day_of_week']}):")
            print(f"   Return_next_day = {current_day['Return_next_day']:.6f}")
            print(f"   Next: {next_day['Date'].strftime('%Y-%m-%d')} Return = {next_day['Return']:.6f}")
            
            if abs(current_day["Return_next_day"] - next_day["Return"]) > 0.000001:
                random_mismatches += 1
                print(f"   ⚠️  MISMATCH!")
            print()
    
    # =====================
    # TEST 4: Statistical validation
    # =====================
    print("="*60)
    print("TEST 4: Statistical Validation")
    print("="*60)
    
    # For all trading days (except last), check alignment
    trading_days_with_next = trading_days[:-1].copy()
    
    # Create what SHOULD be the next day's return
    trading_days_with_next["expected_next_return"] = trading_days["Return"].shift(-1).iloc[:-1].values
    
    # Compare
    differences = abs(trading_days_with_next["Return_next_day"] - trading_days_with_next["expected_next_return"])
    
    print(f"\nTotal trading days checked: {len(trading_days_with_next)}")
    print(f"Perfect matches (diff < 1e-6): {(differences < 0.000001).sum()}")
    print(f"Mismatches: {(differences >= 0.000001).sum()}")
    print(f"Max difference: {differences.max():.10f}")
    print(f"Mean difference: {differences.mean():.10f}")
    
    if (differences >= 0.000001).sum() > 0:
        print(f"\n⚠️  WARNING: Found {(differences >= 0.000001).sum()} mismatches!")
        print("\nShowing first 5 mismatches:")
        mismatched = trading_days_with_next[differences >= 0.000001].head(5)
        for idx, row in mismatched.iterrows():
            print(f"\n  Date: {row['Date'].strftime('%Y-%m-%d')}")
            print(f"  Return_next_day: {row['Return_next_day']:.6f}")
            print(f"  Expected: {row['expected_next_return']:.6f}")
            print(f"  Difference: {abs(row['Return_next_day'] - row['expected_next_return']):.10f}")
    
    # =====================
    # FINAL SUMMARY
    # =====================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_mismatches = friday_mismatches + random_mismatches
    
    if (differences >= 0.000001).sum() == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("✅ Next-day alignment is PERFECT")
        print("✅ Your data is ready for modeling")
    else:
        print(f"\n❌ ISSUES DETECTED!")
        print(f"❌ Found {(differences >= 0.000001).sum()} misaligned samples")
        print(f"❌ This needs to be fixed before modeling")
    
    print()


def verify_all_tickers():
    """
    Run comprehensive verification on all tickers
    """
    tickers = ["XLE", "XLF", "XLK", "XLV", "XLY"]
    
    results = {}
    
    for ticker in tickers:
        comprehensive_verification(ticker)
        print("\n" + "="*80 + "\n")
        
        # Quick summary
        out_path = OUT_DIR / f"{ticker.lower()}.csv"
        data = pd.read_csv(out_path)
        data["Date"] = pd.to_datetime(data["Date"])
        trading_days = data[data["Price"].notna()].copy()
        
        if len(trading_days) > 1:
            trading_days_check = trading_days[:-1].copy()
            expected = trading_days["Return"].shift(-1).iloc[:-1].values
            differences = abs(trading_days_check["Return_next_day"].values - expected)
            mismatches = (differences >= 0.000001).sum()
            results[ticker] = "PASS" if mismatches == 0 else f"FAIL ({mismatches} errors)"
        else:
            results[ticker] = "N/A"
    
    print("\n" + "="*80)
    print("SUMMARY FOR ALL TICKERS")
    print("="*80)
    for ticker, result in results.items():
        symbol = "✅" if result == "PASS" else "❌"
        print(f"{symbol} {ticker}: {result}")
    print("="*80 + "\n")
