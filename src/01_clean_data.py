"""
Step 1: Clean the raw TTC subway delay data and create our target label.

WHAT THIS SCRIPT DOES (in plain terms):
Raw data from the real world is never ready to feed into a model. This script
takes the messy CSV we downloaded and turns it into a clean table where every
row has the information a model needs, in a format it can use.

We are building a CLASSIFICATION model: given information about a subway
incident (station, line, time, day, etc.), predict whether it caused a
DELAY (Min Delay > 0) or NO DELAY (Min Delay == 0).
"""

import pandas as pd

# pandas is a Python library for working with tables of data (like Excel,
# but controlled by code). We import it under the short name "pd" -- this
# is just a very common convention, not a rule.

print("Loading raw data...")
df = pd.read_csv("data/raw_subway_delays.csv")
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.\n")

# ---------------------------------------------------------------------------
# STEP 1: Rename columns to be code-friendly.
# ---------------------------------------------------------------------------
# The original column names have spaces in them ("Min Delay"), which works
# but is annoying to type in code (df["Min Delay"] vs df.min_delay). We
# rename them to lowercase, underscore-separated names -- a very common
# Python convention called "snake_case".

df = df.rename(columns={
    "Date": "date",
    "Time": "time",
    "Day": "day_of_week",
    "Station": "station",
    "Code": "delay_code",
    "Min Delay": "delay_minutes",
    "Min Gap": "gap_minutes",
    "Bound": "bound",
    "Line": "line",
    "Vehicle": "vehicle",
})

# ---------------------------------------------------------------------------
# STEP 2: Create our TARGET LABEL -- the thing the model will learn to predict.
# ---------------------------------------------------------------------------
# We decided: predict whether an incident causes ANY measurable delay.
# This is a "binary" (two-choice) classification target: 1 means "delay
# happened", 0 means "no delay". We store it as a new column.

df["had_delay"] = (df["delay_minutes"] > 0).astype(int)

print("Target label breakdown (this is what we're trying to predict):")
print(df["had_delay"].value_counts())
print(f"-> {df['had_delay'].mean()*100:.1f}% of incidents caused a delay\n")

# ---------------------------------------------------------------------------
# STEP 3: Parse date/time into features a model can actually use.
# ---------------------------------------------------------------------------
# A model can't understand the string "12:04:00 AM" as "early morning".
# We need to extract structured pieces: hour of day, month, etc. This is
# called FEATURE ENGINEERING -- turning raw data into signals a model can
# learn patterns from.

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# Parse the time string into an hour number (0-23). errors="coerce" means:
# if a time string is broken/unreadable, turn it into a missing value (NaT)
# instead of crashing the whole script.
parsed_time = pd.to_datetime(df["time"], format="%I:%M:%S %p", errors="coerce")
df["hour"] = parsed_time.dt.hour

# A handful of times might fail to parse - drop those rows since we can't
# use them without an hour value.
before = len(df)
df = df.dropna(subset=["hour"])
print(f"Dropped {before - len(df)} rows with unparseable time values.\n")
df["hour"] = df["hour"].astype(int)

# ---------------------------------------------------------------------------
# STEP 4: Handle missing values.
# ---------------------------------------------------------------------------
# "bound" (direction) is missing in ~22% of rows. Rather than throw away
# a fifth of our data, we fill missing values with the label "Unknown" --
# this turns "missing" itself into a valid category the model can learn
# from (sometimes missingness IS a pattern worth learning).

df["bound"] = df["bound"].fillna("Unknown")

# "line" is missing in a small number of rows (506) and "delay_code" in
# just 1. These are rare enough that we simply drop those rows rather than
# invent a value for a categorical column with many possible values.
before = len(df)
df = df.dropna(subset=["line", "delay_code"])
print(f"Dropped {before - len(df)} rows with missing line/delay_code.\n")

# ---------------------------------------------------------------------------
# STEP 5: Remove clearly broken data (outliers that are data-entry errors).
# ---------------------------------------------------------------------------
# We saw a max delay of 999 minutes, which is almost certainly a placeholder
# value, not a real ~16-hour subway delay. We cap it: anything above 180
# minutes (3 hours) is treated as a data error and dropped.
before = len(df)
df = df[df["delay_minutes"] <= 180]
print(f"Dropped {before - len(df)} rows with delay > 180 min (likely data errors).\n")

# ---------------------------------------------------------------------------
# STEP 6: Keep only the columns we'll actually use for modeling.
# ---------------------------------------------------------------------------
# We keep the engineered features plus identifying columns and the label.
# We deliberately DROP "delay_minutes" and "gap_minutes" from the modeling
# columns for now, because they would leak the answer -- if the model sees
# delay_minutes, it trivially knows had_delay (delay_minutes > 0). This
# concept is called DATA LEAKAGE and it's one of the most common beginner
# mistakes in ML: accidentally giving the model the answer disguised as a
# feature.

final_columns = [
    "station", "line", "bound", "day_of_week", "hour", "month",
    "delay_code", "had_delay",
]
df_clean = df[final_columns].copy()

print("Final cleaned dataset:")
print(df_clean.head())
print(f"\nFinal shape: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")

df_clean.to_csv("data/clean_subway_delays.csv", index=False)
print("\nSaved to data/clean_subway_delays.csv")
