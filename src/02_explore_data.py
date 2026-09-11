"""
Step 2: Explore the cleaned data with simple charts.

WHY WE DO THIS BEFORE MODELING:
A model is only as good as the patterns actually present in the data. If we
skip straight to training, we're guessing blind. Looking at a few charts
first tells us: does hour of day actually seem to matter? Are some lines
worse than others? This builds intuition AND helps us sanity-check that our
cleaned data makes sense.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/clean_subway_delays.csv")

# We'll build one figure with several small charts (a "grid" of subplots),
# which is a common way to compare several views of the data at once.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- Chart 1: Delay rate by hour of day ---------------------------------
# groupby("hour") splits the data into 24 buckets (one per hour), and
# ["had_delay"].mean() computes, for each bucket, the fraction of incidents
# that had a delay. This is exactly "delay rate per hour".
delay_by_hour = df.groupby("hour")["had_delay"].mean()
axes[0, 0].bar(delay_by_hour.index, delay_by_hour.values, color="#4C72B0")
axes[0, 0].set_title("Delay rate by hour of day")
axes[0, 0].set_xlabel("Hour (24h)")
axes[0, 0].set_ylabel("Fraction of incidents that caused a delay")

# --- Chart 2: Delay rate by day of week ---------------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
delay_by_day = df.groupby("day_of_week")["had_delay"].mean().reindex(day_order)
axes[0, 1].bar(delay_by_day.index, delay_by_day.values, color="#55A868")
axes[0, 1].set_title("Delay rate by day of week")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].set_ylabel("Fraction of incidents that caused a delay")

# --- Chart 3: Delay rate by subway line ---------------------------------
delay_by_line = df.groupby("line")["had_delay"].mean().sort_values(ascending=False)
# Some "line" values are rare data glitches (e.g. very few rows) -- keep
# only lines with a meaningful number of incidents so the chart isn't noisy.
line_counts = df["line"].value_counts()
common_lines = line_counts[line_counts > 100].index
delay_by_line = delay_by_line[delay_by_line.index.isin(common_lines)]
axes[1, 0].bar(delay_by_line.index, delay_by_line.values, color="#C44E52")
axes[1, 0].set_title("Delay rate by subway line")
axes[1, 0].set_ylabel("Fraction of incidents that caused a delay")

# --- Chart 4: How many incidents happen at each hour (volume, not rate) --
# This is different from Chart 1: it's not "how often does it turn into a
# delay", it's "how many incidents get logged at all" at each hour. Useful
# context -- a high delay RATE at 3 AM on very few incidents means less
# than the same rate at 8 AM on thousands of incidents.
incidents_by_hour = df.groupby("hour").size()
axes[1, 1].bar(incidents_by_hour.index, incidents_by_hour.values, color="#8172B2")
axes[1, 1].set_title("Number of incidents logged by hour")
axes[1, 1].set_xlabel("Hour (24h)")
axes[1, 1].set_ylabel("Count of incidents")

plt.tight_layout()
plt.savefig("data/exploration_charts.png", dpi=120)
print("Saved charts to data/exploration_charts.png")

# Also print the numbers behind the charts, since a beginner should be able
# to double check what the picture is showing.
print("\nDelay rate by hour:")
print(delay_by_hour.round(3))
print("\nDelay rate by day of week:")
print(delay_by_day.round(3))
print("\nDelay rate by line:")
print(delay_by_line.round(3))
