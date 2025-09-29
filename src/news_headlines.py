from pathlib import Path
import pandas as pd
import re


RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True) # to make sure the output folder exists

# standardizing the date format
START = pd.Timestamp("2018-03-20")
END = pd.Timestamp("2020-07-17")

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True) # to make sure the output folder exists


def preprocess_news(csv_path: Path) -> Path:
    """
    Reads in the file with news headlines,
    parses 'Time' into datetime,
    drops 'Description' column if it exists,
    renames columns to ['Date', 'Headline],
    restricts the time period
    and saves the result as <NEWSPAPER>_preprocessed.csv
    in the given output directory.
    """

    newspaper = csv_path.stem.split("_")[0]

    data = pd.read_csv(csv_path)

    # fixing the time format to make sure it's the same in all the files
    time_fix = data["Time"]
    time_fix = time_fix.replace(r"\bET\b", "", regex=True) # remove "ET"
    time_fix = time_fix.replace(r"(?i)\bMon|Tue|Wed|Thu|Fri|Sat|Sun\b,?", "", regex=True)
    #time_fix = time_fix.replace(r"\s+", " ", regex=True).strip()

    data["Date"] = pd.to_datetime(time_fix, errors="coerce", dayfirst=True, format="mixed").dt.normalize() # we have to drop the hour from one of the files
    data = data.sort_values("Date")
    data = data[(data["Date"] >= START) & (data["Date"] <= END)]

    # keeping only the two main columns, dropping "Time" and "Description" if it exists
    data = data[["Date", "Headlines"]]

    out_path = OUT_DIR / f"{newspaper}_preprocessed.csv"
    data.to_csv(out_path, index=False)

    return out_path



def clean_headlines(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series of news headlines,
    converts values to string type,
    applies Unicode normalization (NFKC),
    removes zero-width/invisible characters,
    collapses all whitespace (spaces, tabs, newlines) into a single space,
    strips leading and trailing spaces,
    and returns the cleaned Series ready for FinBERT.
    """
    
    return (series.astype(str)
        .str.normalize("NFKC")                         # keep Unicode, normalize look-alikes
        .str.replace(r"[\u200B-\u200D\u2060\uFEFF]", "", regex=True)  # zero widths
        .str.replace(r"\s+", " ", regex=True)          # collapse spaces/tabs/newlines
        .str.strip()
    )

