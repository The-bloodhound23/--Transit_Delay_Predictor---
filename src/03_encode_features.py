"""
Step 3: Turn text categories into numbers a model can use (encoding).

WHY THIS STEP EXISTS:
Machine learning models are just math underneath -- they multiply and add
numbers. A column like station = "BLOOR STATION" means nothing to that math.
We need to translate text categories into numeric columns in a way that
doesn't accidentally invent a fake meaning (like implying one station is
"twice" another station, which is nonsense).

THE TECHNIQUE WE USE: one-hot encoding.
For a column like "line" with values YU, BD, SRT, SHP, we create FOUR new
columns: is_line_YU, is_line_BD, is_line_SRT, is_line_SHP. Each row gets a
1 in exactly the column that matches it, and 0 everywhere else. This way
the model can learn "SRT has a higher delay rate" as an independent fact,
without assuming any ordering between lines.
"""

import pandas as pd

df = pd.read_csv("data/clean_subway_delays.csv")

# ---------------------------------------------------------------------------
# STEP 1: Clean up the messy "line" column first.
# ---------------------------------------------------------------------------
# We found earlier that "line" has 68 unique values because of typos and
# inconsistent entry (e.g. "YU/BD", "BD LINE", "999", bus route numbers
# accidentally logged here). Toronto really only has 4 subway lines. We
# standardize the messy variants and group anything else into "OTHER".

line_map = {
    "YU": "YU", "BD": "BD", "SRT": "SRT", "SHP": "SHP",
    "YUS": "YU", "YU LINE": "YU",
    "BD LINE": "BD",
}
df["line_clean"] = df["line"].map(line_map)
# Anything not in our known list (typos, stray combos, bus numbers, "999")
# becomes "OTHER" -- a real, valid category meaning "rare/unclear line".
df["line_clean"] = df["line_clean"].fillna("OTHER")

print("Cleaned line value counts:")
print(df["line_clean"].value_counts())
print()

# ---------------------------------------------------------------------------
# STEP 2: For high-cardinality columns (station, delay_code), keep only the
# most common categories and bucket the rest as "OTHER".
# ---------------------------------------------------------------------------
# station has 627 unique values and delay_code has 217. One-hot encoding
# ALL of them would create hundreds of mostly-empty columns, which makes
# the model slower to train and harder for a beginner to reason about,
# without adding much predictive value for the rare ones. Standard practice:
# keep the top N most frequent categories, group the rest as "OTHER".

def bucket_rare_categories(series, top_n):
    top_categories = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_categories), other="OTHER")

df["station_bucketed"] = bucket_rare_categories(df["station"], top_n=30)
df["delay_code_bucketed"] = bucket_rare_categories(df["delay_code"], top_n=20)

print(f"Station categories after bucketing: {df['station_bucketed'].nunique()} "
      f"(was {df['station'].nunique()})")
print(f"Delay code categories after bucketing: {df['delay_code_bucketed'].nunique()} "
      f"(was {df['delay_code'].nunique()})")
print()

# ---------------------------------------------------------------------------
# STEP 3: One-hot encode the categorical columns.
# ---------------------------------------------------------------------------
# pandas has a built-in function, get_dummies(), that does one-hot encoding
# for us. "prefix" controls the naming of the new columns so we can tell
# them apart (e.g. "line_YU", "day_Monday").

categorical_cols = {
    "line_clean": "line",
    "bound": "bound",
    "day_of_week": "day",
    "station_bucketed": "station",
    "delay_code_bucketed": "code",
}

encoded_parts = [df[["hour", "month", "had_delay"]]]  # numeric cols stay as-is
for col, prefix in categorical_cols.items():
    dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)
    encoded_parts.append(dummies)

df_encoded = pd.concat(encoded_parts, axis=1)

print(f"Final encoded dataset shape: {df_encoded.shape[0]:,} rows, "
      f"{df_encoded.shape[1]} columns")
print("(this is many more columns than we started with -- that's expected: "
      "one-hot encoding trades 'fewer columns of text' for 'more columns of "
      "0s and 1s')")

df_encoded.to_csv("data/encoded_subway_delays.csv", index=False)
print("\nSaved to data/encoded_subway_delays.csv")
