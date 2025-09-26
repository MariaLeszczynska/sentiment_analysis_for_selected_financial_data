from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True) # to make sure the output folder exists

def preprocess_file(csv_path: Path) -> Path:
    """
    Reads in the file with ticker prices, discards the first two rows,
    renames the columns to "Date" and "Price",
    converts data types, sorts by Date,
    calculates the daily return and its sign,
    and saves the result as <TICKER>_preprocessed.csv
    in the given output directory.
    """

    ticker = csv_path.stem.split("_")[0]

    data = pd.read_csv(csv_path, skiprows=2)
    data.columns = ["Date", "Price"]
    data["Date"] = pd.to_datetime(data["Date"])
    
    data["Return"] = data["Price"].pct_change()
    data["Sign"] = np.sign(data["Return"])
    
    out_path = OUT_DIR / f"{ticker}_preprocessed.csv"
    data.to_csv(out_path, index=False)

    return out_path
